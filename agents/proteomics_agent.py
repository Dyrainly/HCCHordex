import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, PROTEOMICS_DATA_PATH

RECURRENCE_DIR = os.path.join(_PROJECT_ROOT, "proteomics_agent_recurrence")
RECURRENCE_MODEL_DIR = os.path.join(RECURRENCE_DIR, "models")
RECURRENCE_DATA_DIR = os.path.join(RECURRENCE_DIR, "data")

MODEL_PATH = os.path.join(RECURRENCE_MODEL_DIR, "proteomics_recurrence_xgboost.pkl")
IMPORTANCE_PATH = os.path.join(RECURRENCE_MODEL_DIR, "proteomics_recurrence_importance.json")

GENE_MAPPING_PATH = os.path.join(RECURRENCE_DATA_DIR, "proteomics_gene_mapping.json")
FUNCTION_DETAILS_PATH = os.path.join(RECURRENCE_DATA_DIR, "proteomics_function_details.json")

PROTEIN_PATHWAY_KEYWORDS = {
    "drug_metabolism": ["cytochrome P450", "drug metabolism", "xenobiotic"],
    "steroid_metabolism": ["steroid", "retinol", "bile secretion"],
    "amino_acid_metabolism": ["glycine", "serine", "threonine", "tyrosine"],
    "carbohydrate_metabolism": ["gluconeogenesis", "pentose", "glucuronate"],
    "dna_replication": ["DNA replication", "mismatch repair", "MSH"],
    "cell_cycle": ["anaphase-promoting", "cell cycle", "mitotic"],
    "protein_degradation": ["ubiquitin", "proteasome", "F-box"],
    "rna_processing": ["ribonucleoprotein", "splicing", "ribosomal"],
    "mapk_signaling": ["kinase", "MAPK", "PAK"],
    "insulin_signaling": ["GRB10", "insulin", "growth regulation"],
    "hormone_signaling": ["nuclear receptor", "retinoid", "hormone"],
    "immune_response": ["HLA", "immunoglobulin", "interleukin", "IL-"],
    "inflammation": ["inflammatory", "chemokine", "complement"],
    "redox_homeostasis": ["thioredoxin", "glutathione", "peroxidase", "oxidoreductase"],
    "transcription_regulation": ["transcription", "repressor", "corepressor", "TOX"],
    "chromatin_remodeling": ["histone", "chromatin", "nucleosome"],
    "cytoskeleton": ["actin", "tubulin", "myosin", "integrin"],
    "cell_adhesion": ["cadherin", "adhesion", "extracellular matrix"],
    "lipid_metabolism": ["apolipoprotein", "cholesterol", "lipid", "fatty acid"],
    "liver_function": ["hepatocyte nuclear factor", "albumin", "fibrinogen"],
}

class ProteomicsRiskEncoder:

    def __init__(self):
        self.model = None
        self.scaler = None
        self.selected_features: List[str] = []
        self.shap_base_value: float = 0.0
        self.importance_dict: Dict[str, float] = {}
        self.uniprot_annotations: Dict = {}
        self.gene_mapping: Dict = {}
        self.pathway_keywords = PROTEIN_PATHWAY_KEYWORDS
        self._load_models()
        self._load_annotations()

    def _load_models(self):

        if os.path.exists(MODEL_PATH):
            try:
                data = joblib.load(MODEL_PATH)
                self.model = data["model"]
                self.scaler = data["scaler"]
                self.selected_features = data["selected_features"]
                self.shap_base_value = data.get("shap_base_value", 0.0)
                print(f"[ProteomicsEncoder] XGBoost模型已加载 ({len(self.selected_features)} 特征)")
            except Exception as e:
                print(f"[ProteomicsEncoder] 模型加载失败: {e}")

        if os.path.exists(IMPORTANCE_PATH):
            try:
                with open(IMPORTANCE_PATH) as f:
                    self.importance_dict = json.load(f).get("feature_importance", {})
                print(f"[ProteomicsEncoder] 特征重要性已加载 ({len(self.importance_dict)} 特征)")
            except Exception as e:
                print(f"[ProteomicsEncoder] 特征重要性加载失败: {e}")

    def _load_annotations(self):

        for path in [GENE_MAPPING_PATH]:
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        self.gene_mapping = json.load(f)
                    print(f"[ProteomicsEncoder] 基因映射已加载 ({len(self.gene_mapping)} 条目)")
                except Exception as e:
                    print(f"[ProteomicsEncoder] 基因映射加载失败: {e}")
                break

        for path in [FUNCTION_DETAILS_PATH]:
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        self.uniprot_annotations = json.load(f)
                    print(f"[ProteomicsEncoder] 功能注释已加载 ({len(self.uniprot_annotations)} 蛋白)")
                except Exception as e:
                    print(f"[ProteomicsEncoder] 功能注释加载失败: {e}")
                break

    def encode(self, patient_id: str, proteomics_data: Optional[Dict] = None) -> Dict:

        if proteomics_data is None:
            proteomics_data = self._load_patient_data(patient_id)

        if not proteomics_data:
            return self._empty_encoding(patient_id, "未找到蛋白组学数据")

        if self.model is None or len(self.selected_features) == 0:
            return self._empty_encoding(patient_id, "模型未加载")

        X = self._build_feature_vector(proteomics_data)

        valid = 0
        for f in self.selected_features:
            for key in [f, f"Tumor_{f}", f"Normal_{f}"]:
                v = proteomics_data.get(key)
                if v is not None and isinstance(v, (int, float)):
                    valid += 1
                    break
        completeness = valid / len(self.selected_features)

        X_scaled = self.scaler.transform(X.reshape(1, -1))
        risk_prob = float(self.model.predict_proba(X_scaled)[0, 1])

        shap_contribs = self._compute_shap_contributions(X_scaled[0])

        abnormal_proteins = self._detect_abnormal_proteins(proteomics_data, shap_contribs)
        core_drivers = self._extract_core_drivers(shap_contribs)
        pathway_dysregulation = self._analyze_pathways(abnormal_proteins)
        proteomic_signature = self._generate_signature(abnormal_proteins, risk_prob)
        uniprot_annots = self._get_uniprot_annotations(abnormal_proteins)

        return {
            "patient_id": patient_id,
            "encoding_success": True,
            "data_completeness": round(completeness, 3),
            "abnormal_proteins": abnormal_proteins,
            "core_risk_drivers": core_drivers,
            "pathway_dysregulation": pathway_dysregulation,
            "proteomic_signature": proteomic_signature,

            "shap_analysis": {
                "base_value": round(self.shap_base_value, 4),
                "individual_contributions": {
                    k.replace("Tumor_", "").replace("Normal_", ""): round(v, 4)
                    for k, v in sorted(shap_contribs.items(), key=lambda x: -abs(x[1]))[:15]
                },

                "top_positive_contributors": core_drivers[:5],
                "top_negative_contributors": [d for d in core_drivers if d.get("direction") == "negative"][:3],
            },
            "uniprot_annotations": uniprot_annots,
        }

    def _load_patient_data(self, patient_id: str) -> Optional[Dict]:

        if not hasattr(ProteomicsRiskEncoder, '_data_cache'):
            ProteomicsRiskEncoder._data_cache = None

        if ProteomicsRiskEncoder._data_cache is None:
            try:
                df = pd.read_excel(PROTEOMICS_DATA_PATH)
                df["就诊号"] = df["就诊号"].astype(str).str.strip()
                ProteomicsRiskEncoder._data_cache = df
                print(f"[ProteomicsEncoder] 蛋白组学数据已加载 ({len(df)} 行)")
            except Exception as e:
                print(f"[ProteomicsEncoder] 数据加载失败: {e}")
                return None

        pid = str(patient_id).strip()
        df = ProteomicsRiskEncoder._data_cache
        row = df[df["就诊号"] == pid]

        if len(row) == 0:
            return None

        data = row.iloc[0].to_dict()
        return {k: (None if pd.isna(v) else v) for k, v in data.items()}

    def _build_feature_vector(self, proteomics_data: Dict) -> np.ndarray:

        X = np.zeros(len(self.selected_features))
        for i, feat in enumerate(self.selected_features):
            for key in [feat, f"Tumor_{feat}", f"Normal_{feat}"]:
                v = proteomics_data.get(key)
                if v is not None and isinstance(v, (int, float)) and not pd.isna(v):
                    X[i] = float(v)
                    break
        return X

    def _compute_shap_contributions(self, X_scaled: np.ndarray) -> Dict[str, float]:

        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_scaled.reshape(1, -1))
            return {feat: float(shap_values[0, i]) for i, feat in enumerate(self.selected_features)}
        except Exception:

            try:
                import xgboost as xgb
                dmat = xgb.DMatrix(X_scaled.reshape(1, -1))
                contribs = self.model.get_booster().predict(dmat, pred_contribs=True)[0]
                return {feat: float(contribs[i]) for i, feat in enumerate(self.selected_features)}
            except Exception as e:
                print(f"[ProteomicsEncoder] SHAP计算失败: {e}")
                return {}

    def _detect_abnormal_proteins(self, proteomics_data: Dict, contribs: Dict[str, float]) -> List[Dict]:

        abnormal = []
        for feat, contrib in sorted(contribs.items(), key=lambda x: -abs(x[1]))[:15]:
            val = None
            for key in [feat, f"Tumor_{feat}", f"Normal_{feat}"]:
                v = proteomics_data.get(key)
                if v is not None and isinstance(v, (int, float)) and not pd.isna(v):
                    val = v
                    break
            if val is None:
                continue

            protein_name = feat.replace("Tumor_", "").replace("Normal_", "")
            abnormal.append({
                "protein": protein_name,
                "gene_symbol": self._get_gene_symbol(feat),
                "value": round(float(val), 3),
                "shap_contribution": round(contrib, 4),
                "global_importance": round(self.importance_dict.get(feat, 0), 4),
                "direction": "positive" if contrib > 0 else "negative",
                "risk_effect": "promotes_recurrence" if contrib > 0 else "protects_against_recurrence",
                "function": self._get_protein_function(feat),
            })
        return abnormal[:10]

    def _extract_core_drivers(self, contribs: Dict[str, float]) -> List[Dict]:

        drivers = []
        for feat, contrib in sorted(contribs.items(), key=lambda x: -x[1])[:10]:
            protein_name = feat.replace("Tumor_", "").replace("Normal_", "")
            drivers.append({
                "protein": protein_name,
                "gene_symbol": self._get_gene_symbol(feat),
                "shap_contribution": round(contrib, 4),
                "global_importance": round(self.importance_dict.get(feat, 0), 4),
                "direction": "positive" if contrib > 0 else "negative",
                "mechanism": self._infer_mechanism(feat),
            })
        return drivers

    def _infer_mechanism(self, protein_name: str) -> str:

        name_lower = protein_name.lower()
        for pathway, keywords in self.pathway_keywords.items():
            for kw in keywords:
                if kw.lower() in name_lower:
                    return pathway.replace("_", " ")
        return "uncertain"

    def _analyze_pathways(self, abnormal_proteins: List[Dict]) -> List[Dict]:

        pathway_data = {}
        for p in abnormal_proteins:
            name_lower = p["protein"].lower()
            for pathway, keywords in self.pathway_keywords.items():
                for kw in keywords:
                    if kw.lower() in name_lower:
                        if pathway not in pathway_data:
                            pathway_data[pathway] = {
                                "count": 0, "proteins": [],
                                "directions": {"positive": 0, "negative": 0},
                            }
                        pathway_data[pathway]["count"] += 1
                        pathway_data[pathway]["proteins"].append(p["protein"][:30])
                        pathway_data[pathway]["directions"][p["direction"]] += 1
                        break

        result = []
        for pw, data in sorted(pathway_data.items(), key=lambda x: -x[1]["count"])[:5]:
            result.append({
                "pathway": pw.replace("_", " "),
                "dysregulated_proteins": data["count"],
                "evidence": data["proteins"][:3],
                "promoting_count": data["directions"]["positive"],
                "protective_count": data["directions"]["negative"],
            })
        return result

    def _generate_signature(self, abnormal_proteins: List[Dict], risk_prob: float) -> List[str]:

        sig = []
        if risk_prob > 0.7:
            sig.append("high_risk_proteomic_profile")
        elif risk_prob > 0.5:
            sig.append("moderate_risk_proteomic_profile")
        else:
            sig.append("low_risk_proteomic_profile")

        pos = sum(1 for p in abnormal_proteins if p["direction"] == "positive")
        neg = len(abnormal_proteins) - pos
        if pos > neg * 2:
            sig.append("risk_promoting_dominant")
        elif neg > pos * 2:
            sig.append("protective_dominant")
        else:
            sig.append("mixed_signals")

        for p in abnormal_proteins[:3]:
            for pathway, keywords in self.pathway_keywords.items():
                if any(kw.lower() in p["protein"].lower() for kw in keywords):
                    sig.append(f"{pathway}_dysregulation")
                    break
        return sig

    def _get_uniprot_annotations(self, abnormal_proteins: List[Dict]) -> List[Dict]:

        annotations = []
        for p in abnormal_proteins:
            name = p["protein"]
            annot = self.uniprot_annotations.get(name, {})
            mapping = self.gene_mapping.get(name, {})
            annotations.append({
                "protein": name,
                "gene_symbol": p.get("gene_symbol", mapping.get("gene_symbol", "")),
                "uniprot_id": mapping.get("uniprot_id", ""),
                "function": annot.get("function", p.get("function", "")),
                "subcellular_location": annot.get("subcellular_location", ""),
                "pathway": annot.get("pathway", ""),
            })
        return annotations

    def _get_gene_symbol(self, protein_col_name: str) -> str:

        info = self.gene_mapping.get(protein_col_name)
        if info and info.get("gene_symbol"):
            return info["gene_symbol"]
        name = protein_col_name.replace("Tumor_", "").replace("Normal_", "")
        info = self.gene_mapping.get(name)
        if info and info.get("gene_symbol"):
            return info["gene_symbol"]
        return ""

    def _get_protein_function(self, protein_col_name: str) -> str:

        name = protein_col_name.replace("Tumor_", "").replace("Normal_", "")
        func = self.uniprot_annotations.get(name, {}).get("function", "")
        if func:
            return func[:200] + "..." if len(func) > 200 else func
        info = self.gene_mapping.get(protein_col_name, self.gene_mapping.get(name, {}))
        func = info.get("function", "")
        return func[:200] + "..." if len(func) > 200 else func

    def _empty_encoding(self, patient_id: str, reason: str) -> Dict:

        return {
            "patient_id": patient_id,
            "encoding_success": False,
            "error": reason,
            "data_completeness": 0.0,
            "abnormal_proteins": [],
            "core_risk_drivers": [],
            "pathway_dysregulation": [],
            "proteomic_signature": [],
            "shap_analysis": {},
            "uniprot_annotations": [],
        }

TASK_TEMPLATE = """
将每个异常蛋白按SHAP值+UniProt功能注释，归入以下6个功能主题之一。
对每个主题给出该患者的激活状态和临床解释。

1. 增殖信号(Proliferation): MAPK, PI3K-Akt, mTOR, Wnt, Cell cycle, Ras, IGF, 转录/翻译调控
2. 代谢重编程(Metabolism): HIF-1, Glycolysis, FAO, TCA cycle, PPP, Cholesterol, Amino acid metabolism
3. 侵袭转移(Invasion/EMT): TGF-beta, Adherens junction, Actin cytoskeleton, Focal adhesion, MMPs, Rho GTPases
4. 凋亡逃逸(Apoptosis): p53, Bcl-2 family, Caspases, Autophagy, Ferroptosis, Death receptors
5. 免疫逃逸(Immune): PD-L1, Antigen presentation, Chemokines, Complement, TCR signaling
6. 基因组不稳定(Genome): DNA repair, Replication, Telomerase, Chromatin remodeling, Mismatch repair

{protein_details}

{{
  "functional_themes": {{
    "proliferation": {{"activated": true/false, "shap_net": 数值, "key_proteins": ["蛋白1","蛋白2"], "interpretation": "临床解释"}},
    "metabolism": {{...}},
    "invasion_emt": {{...}},
    "apoptosis_evasion": {{...}},
    "immune_evasion": {{...}},
    "genome_instability": {{...}}
  }},
  "proteomic_signature": "分子表型总结",
  "dominant_theme": "主导功能主题",
  "net_risk_direction": "促复发/抑复发",
  "clinical_interpretation": "蛋白组学层面的预后意义——用临床医生能理解的语言"
}}

只输出JSON，不要额外解释。"""

class ProteomicsAgent:

    def __init__(self):
        self.encoder = ProteomicsRiskEncoder()
        self.llm = None
        self._init_llm()

    def _init_llm(self):

        try:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=LLM_MODEL,
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
                temperature=0.1,
            )
            print("[ProteomicsAgent] LLM已初始化")
        except Exception as e:
            print(f"[ProteomicsAgent] LLM初始化失败: {e}, 将使用规则引擎")

    def analyze(self, proteomics_summary: str, patient_id: str) -> Dict:

        encoded = self.encoder.encode(patient_id)

        if not encoded["encoding_success"]:
            return {
                "patient_id": patient_id,
                "status": "failed",
                "error": encoded.get("error", "编码失败"),
                "encoding_result": encoded,
            }

        prompt = self._build_prompt(encoded)

        llm_output = self._invoke_llm(prompt)
        if llm_output is None:
            llm_output = self._rule_based_interpretation(encoded)

        return {
            "patient_id": patient_id,
            "status": "success",
            "encoding_result": encoded,
            "llm_interpretation": llm_output,
            "discriminative_signature": {
                "abnormal_proteins": encoded["abnormal_proteins"],
                "core_risk_drivers": encoded["core_risk_drivers"],
                "proteomic_signature": encoded["proteomic_signature"],
                "shap_base_value": encoded["shap_analysis"]["base_value"],
            },
        }

    def _invoke_llm(self, prompt: str) -> Optional[str]:

        if self.llm is None:
            return None
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            resp = self.llm.invoke([
                SystemMessage(content=self._system_prompt()),
                HumanMessage(content=prompt),
            ])
            return resp.content
        except Exception as e:
            print(f"[ProteomicsAgent] LLM分析失败: {e}")
            return None

    def generate_backward_cot(self, report_dict: Dict, t_GT: Dict, patient_id: str = "") -> Dict:

        from agents.backward_cot_generator import ProteomicsBackwardCOTGenerator

        generator = ProteomicsBackwardCOTGenerator()
        return generator.generate_backward_cot(report_dict, t_GT, patient_id)

    def generate_cot(self, report_dict: Dict, patient_id: str = "") -> List[Dict]:

        from langchain_core.messages import SystemMessage, HumanMessage

        system_prompt = """你是蛋白组学功能分析专家。请基于SHAP值+UniProt功能注释，生成5步COT推理链。

【5步推理框架】
Step 1 - 异常蛋白识别: 按SHAP贡献值识别Top异常蛋白
Step 2 - 功能主题归类: 将蛋白归类到6大功能主题(增殖/代谢/侵袭/凋亡/免疫/基因组)
Step 3 - 通路失调分析: 分析异常蛋白参与的信号通路
Step 4 - 分子-宏观桥接: 用分子特征解释可能的临床表型
Step 5 - 风险签名生成: 综合给出蛋白组学风险方向

输出JSON格式:
{
  "cot_chain": [
    {"step": 1, "title": "异常蛋白识别", "reasoning": "...", "evidence": ["..."], "conclusion": "..."},
    {"step": 2, "title": "功能主题归类", "reasoning": "...", "evidence": ["..."], "conclusion": "..."},
    {"step": 3, "title": "通路失调分析", "reasoning": "...", "evidence": ["..."], "conclusion": "..."},
    {"step": 4, "title": "分子-宏观桥接", "reasoning": "...", "evidence": ["..."], "conclusion": "..."},
    {"step": 5, "title": "风险签名生成", "reasoning": "...", "evidence": ["..."], "conclusion": "..."}
  ]
}"""

        encoded = report_dict.get("encoding_result", {})
        abnormal = encoded.get("abnormal_proteins", [])
        pathways = encoded.get("pathway_dysregulation", [])
        signature = encoded.get("proteomic_signature", [])

        abnormal_text = "\n".join([
            f"{p['protein']}({p.get('gene_symbol','')}): SHAP={p['shap_contribution']:+.4f}, {p.get('direction','')}"
            for p in abnormal[:10]
        ])

        user_prompt = f"""患者ID: {patient_id or report_dict.get('patient_id', '未知')}

【异常蛋白】
{abnormal_text}

【通路失调】
{json.dumps(pathways, ensure_ascii=False, indent=2)}

【蛋白组学签名】
{signature}

请生成5步COT推理链，直接输出JSON。"""

        try:
            if self.llm is None:
                return []
            resp = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            text = resp.content

            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                return parsed.get("cot_chain", [])
        except Exception as e:
            print(f"[ProteomicsAgent] COT生成失败: {e}")

        return []

    def generate_report_text(self, report_dict: Dict) -> str:

        lines = []
        pid = report_dict.get("patient_id", "未知")
        lines.append(f"{'='*50}")
        lines.append(f"  蛋白组学评估报告 - 患者 {pid}")
        lines.append(f"{'='*50}")

        if report_dict.get("status") == "failed":
            lines.append(f"\n[错误] {report_dict.get('error', '分析失败')}")
            return "\n".join(lines)

        encoded = report_dict.get("encoding_result", {})

        completeness = encoded.get("data_completeness", 0)
        lines.append(f"\n数据完整度: {completeness:.1%}")

        abnormal = encoded.get("abnormal_proteins", [])
        if abnormal:
            lines.append("\n【异常蛋白分析 (个体化SHAP)】")
            for i, p in enumerate(abnormal, 1):
                gene = f"({p['gene_symbol']})" if p.get("gene_symbol") else ""
                direction = "促进复发" if p["direction"] == "positive" else "抑制复发/保护性"
                lines.append(
                    f"  {i}. {p['protein']}{gene}: "
                    f"表达值={p['value']:+.3f}, SHAP={p['shap_contribution']:+.4f}, {direction}"
                )

        drivers = encoded.get("core_risk_drivers", [])
        if drivers:
            lines.append("\n【核心风险驱动因素】")
            for d in drivers[:5]:
                gene = f" ({d['gene_symbol']})" if d.get("gene_symbol") else ""
                lines.append(f"  - {d['protein']}{gene}: SHAP={d['shap_contribution']:+.4f} ({d['mechanism']})")

        pathways = encoded.get("pathway_dysregulation", [])
        if pathways:
            lines.append("\n【通路失调分析】")
            for pw in pathways:
                promoting = pw.get("promoting_count", "?")
                protective = pw.get("protective_count", "?")
                lines.append(
                    f"  - {pw['pathway']}: {pw['dysregulated_proteins']}个蛋白异常 "
                    f"(促进={promoting}, 保护={protective})"
                )

        signature = encoded.get("proteomic_signature", [])
        if signature:
            lines.append(f"\n【蛋白组学签名】: {', '.join(signature)}")

        llm_interp = report_dict.get("llm_interpretation", "")
        if llm_interp:
            lines.append(f"\n{'='*50}")
            lines.append("【蛋白组学专家解读】")
            lines.append(llm_interp)

        return "\n".join(lines)

    def _system_prompt(self) -> str:
        return """你是蛋白组学功能分析专家。基于SHAP值+UniProt功能注释，将每个蛋白归类到功能主题。"""

    def _build_prompt(self, encoded: Dict) -> str:

        uniprot_map = {}
        for u in encoded.get("uniprot_annotations", []) or []:
            uniprot_map[u.get("protein", "")] = u

        lines = []
        for p in encoded["abnormal_proteins"]:
            gene = p.get("gene_symbol", "") or ""
            protein_name = p["protein"][:40]
            shap = p["shap_contribution"]
            expr = p["value"]
            effect = "促进复发" if shap > 0 else "抑制复发"
            ann = uniprot_map.get(p["protein"], {})
            func = (ann.get("function", "") or "")[:200]
            pathway = (ann.get("pathway", "") or "")[:100]
            func_str = f"  UniProt功能: {func}" if func else ""
            path_str = f"  KEGG通路: {pathway}" if pathway else ""

            lines.append(
                f"{protein_name} ({gene})\n"
                f"  SHAP={shap:+.4f} ({effect}) | 表达值={expr:+.3f}\n"
                f"{func_str}\n{path_str}\n"
            )

        return TASK_TEMPLATE.format(protein_details="\n".join(lines))

    def _rule_based_interpretation(self, encoded: Dict) -> str:

        lines = ["**异常情况及临床影响**："]

        for i, p in enumerate(encoded["abnormal_proteins"], 1):
            effect = "促进复发" if p["direction"] == "positive" else "抑制复发/保护性"
            gene = f"({p['gene_symbol']})" if p.get("gene_symbol") else ""
            lines.append(
                f"{i}. {p['protein']}{gene}: "
                f"表达值={p['value']:+.3f}, SHAP={p['shap_contribution']:+.4f}, {effect}"
            )

        lines.append("")
        lines.append("**需要详细分析的特定蛋白名称**：")
        for p in encoded["abnormal_proteins"][:8]:
            role = "核心风险因子" if p["direction"] == "positive" else "保护性因子"
            gene = f" ({p['gene_symbol']})" if p.get("gene_symbol") else ""
            lines.append(f"- {p['protein']}{gene} - {role} (SHAP={p['shap_contribution']:+.4f})")

        if encoded["pathway_dysregulation"]:
            lines.append("")
            lines.append("**通路异常**：")
            for pw in encoded["pathway_dysregulation"]:
                promoting = pw.get("promoting_count", "?")
                protective = pw.get("protective_count", "?")
                lines.append(
                    f"- {pw['pathway']}: {pw['dysregulated_proteins']}个蛋白异常 "
                    f"(促进={promoting}, 保护={protective})"
                )

        top_promoting = [p for p in encoded["abnormal_proteins"] if p["direction"] == "positive"][:3]
        if top_promoting:
            drivers = ", ".join(p["protein"][:25] for p in top_promoting)
            lines.append(f"- 主要驱动因素: {drivers}")

        return "\n".join(lines)

def analyze_patient(patient_id: str, use_llm: bool = True) -> Dict:

    agent = ProteomicsAgent()
    if not use_llm:
        agent.llm = None
    return agent.analyze("", patient_id)

def batch_analyze(patient_ids: List[str], use_llm: bool = True) -> List[Dict]:

    agent = ProteomicsAgent()
    if not use_llm:
        agent.llm = None
    return [agent.analyze("", pid) for pid in patient_ids]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="蛋白组学风险分析Agent")
    parser.add_argument("--patient_id", type=str, required=True)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    result = analyze_patient(args.patient_id, use_llm=not args.no_llm)
    output_str = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_str)
    else:
        print(output_str)

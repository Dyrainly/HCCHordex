import json, re, sys, os, time, traceback
from typing import Dict, List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, ENABLE_THINKING
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from analytics.clinical_extractor import ClinicalRiskExtractor

class ClinicalAgent:

    def __init__(self):
        thinking_config = {"enable_thinking": True} if ENABLE_THINKING else {"enable_thinking": False}
        self.llm = ChatOpenAI(
            model=LLM_MODEL, base_url=LLM_BASE_URL, api_key=LLM_API_KEY,
            temperature=0.2, max_retries=5, extra_body=thinking_config,
        )
        self.extractor = ClinicalRiskExtractor()
        self.system_prompt = """你是一名肝胆外科病理报告结构化专家。

你的任务不是自由评估，而是从临床数据中提取以下指定的标准化属性。
每个属性必须填写：value（实际值）、risk_contribution（促进复发/抑制复发/中性）、clinical_significance（一句话临床意义）。

【属性提取规范】

提取以下属性，从病理诊断中直接取值：
- mvi_grade: MVI分级 (M0/M1/M2)
- edmondson_grade: Edmondson分级 (I/II/III/IV)
- differentiation: 分化程度 (高分化/中分化/低分化)
- microvascular_thrombus: 镜下脉管癌栓 (有/无)
- vascular_invasion: 血管侵犯 (有/无，含侵犯血管名)
- satellite_nodules: 微卫星子灶 (有/无)
- capsule: 包膜完整性 (完整/不完整/突破)
- tumor_size_cm: 最大肿瘤直径 (数值cm)
- tumor_count: 肿瘤个数 (单发/2个/≥3个)
- immunohistochemistry: 免疫组化标志物 (CK19/CK7/GPC3等，列出阳性项目)

- margin_status: 切缘状态 (R0/R1/R2)
- margin_distance: 切缘距离 (数值mm，若未提供标注未知)
- resection_method: 手术方式 (解剖性切除/局部切除)
- resection_extent: 切除范围 (段切除/叶切除/半肝切除)
- blood_loss_ml: 术中失血量 (数值ml)

- cirrhosis: 肝硬化 (有/无，含病理类型)
- fibrosis_stage: 纤维化分期 (S0-S4)
- hepatitis_activity: 肝炎活动度 (G0-G4)
- child_pugh: Child-Pugh分级 (A/B/C)
- afp_ng_ml: AFP (数值ng/ml，>400标注高危)
- hbv_dna: HBV DNA (IU/ml，若提供)
- hcv_ab: 丙肝抗体 (阳性/阴性)
- hbs_ag: 乙肝表面抗原 (阳性/阴性)
- tnm_stage: TNM分期
- bclc_stage: BCLC分期 (0/A/B/C/D)

【缺失值处理】: 若数值缺失，value填null，risk_contribution填"未知"，clinical_significance填"数据缺失"

输出严格的JSON格式，不要markdown代码块。"""

    @staticmethod
    def _parse_json_response(text: str) -> Dict:
        cleaned = text.strip()
        if not cleaned:
            return {"parse_error": "空响应"}
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[0].startswith("```"): lines = lines[1:]
            if lines and lines[-1].startswith("```"): lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        for candidate in [cleaned,
                          re.sub(r",\s*}", "}", cleaned),
                          re.sub(r",\s*\]", "]", cleaned),
                          cleaned.replace("'", '"'),
                          cleaned.replace("\u201c", '"').replace("\u201d", '"')]:
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
        return {"parse_error": "JSON解析失败", "raw": cleaned[:500]}

    def analyze(self, clinical_summary: str, patient_id: str = "") -> Dict:
        self.extractor.extract(patient_id)

        user_msg = f"""请从以下临床数据中提取标准化属性。

患者就诊号：{patient_id}

【临床数据】
{clinical_summary}

【输出格式 — 严格按此JSON结构】
{{
  "patient_id": "{patient_id}",
  "tumor_biology": {{
    "mvi_grade": {{"value": "M0/M1/M2 or null", "risk_contribution": "促进复发/抑制复发/中性/未知", "clinical_significance": "一句话"}},
    "edmondson_grade": {{"value": "I/II/III/IV or null", "risk_contribution": "...", "clinical_significance": "..."}},
    "differentiation": {{"value": "高/中/低 or null", "risk_contribution": "...", "clinical_significance": "..."}},
    "microvascular_thrombus": {{"value": "有/无 or null", "risk_contribution": "...", "clinical_significance": "..."}},
    "vascular_invasion": {{"value": "有/无 or null", "risk_contribution": "...", "clinical_significance": "..."}},
    "satellite_nodules": {{"value": "有/无 or null", "risk_contribution": "...", "clinical_significance": "..."}},
    "capsule": {{"value": "完整/不完整/突破 or null", "risk_contribution": "...", "clinical_significance": "..."}},
    "tumor_size_cm": {{"value": "数值 or null", "risk_contribution": "...", "clinical_significance": "..."}},
    "tumor_count": {{"value": "单发/2个/≥3个 or null", "risk_contribution": "...", "clinical_significance": "..."}},
    "immunohistochemistry": {{"value": "阳性标志物列表 or null", "risk_contribution": "...", "clinical_significance": "..."}}
  }},
  "surgical_curability": {{
    "margin_status": {{"value": "R0/R1/R2 or null", "risk_contribution": "...", "clinical_significance": "..."}},
    "margin_distance": {{"value": "数值mm or null", "risk_contribution": "...", "clinical_significance": "..."}},
    "resection_method": {{"value": "解剖性切除/局部切除 or null", "risk_contribution": "...", "clinical_significance": "..."}},
    "resection_extent": {{"value": "段切除/叶切除/半肝切除 or null", "risk_contribution": "...", "clinical_significance": "..."}},
    "blood_loss_ml": {{"value": "数值 or null", "risk_contribution": "...", "clinical_significance": "..."}}
  }},
  "field_effect": {{
    "cirrhosis": {{"value": "有/无 or null", "risk_contribution": "...", "clinical_significance": "..."}},
    "fibrosis_stage": {{"value": "S0-S4 or null", "risk_contribution": "...", "clinical_significance": "..."}},
    "hepatitis_activity": {{"value": "G0-G4 or null", "risk_contribution": "...", "clinical_significance": "..."}},
    "child_pugh": {{"value": "A/B/C or null", "risk_contribution": "...", "clinical_significance": "..."}},
    "afp_ng_ml": {{"value": "数值 or null", "risk_contribution": "...", "clinical_significance": "..."}},
    "hbv_dna": {{"value": "数值IU/ml or null", "risk_contribution": "...", "clinical_significance": "..."}},
    "hcv_ab": {{"value": "阳性/阴性 or null", "risk_contribution": "...", "clinical_significance": "..."}},
    "hbs_ag": {{"value": "阳性/阴性 or null", "risk_contribution": "...", "clinical_significance": "..."}},
    "tnm_stage": {{"value": "TNM分期 or null", "risk_contribution": "...", "clinical_significance": "..."}},
    "bclc_stage": {{"value": "0/A/B/C/D or null", "risk_contribution": "...", "clinical_significance": "..."}}
  }}
}}

只输出JSON，不要markdown代码块，不要额外解释。"""

        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=user_msg)]
        last_error = None
        for attempt in range(3):
            try:
                response = self.llm.invoke(messages)
                result = self._parse_json_response(response.content)
                result.setdefault("patient_id", patient_id)
                return result
            except Exception as e:
                last_error = e
                print(f"[ClinicalAgent] retry {attempt+1}: {e}")
                time.sleep(2)
        return {"patient_id": patient_id, "error": str(last_error)}

    def generate_backward_cot(self, report_dict: Dict, t_GT: Dict, patient_id: str = "") -> Dict:

        from agents.backward_cot_generator import ClinicalBackwardCOTGenerator

        generator = ClinicalBackwardCOTGenerator()
        return generator.generate_backward_cot(report_dict, t_GT, patient_id)

    def generate_cot(self, report_dict: Dict, patient_id: str = "") -> List[Dict]:

        system_prompt = """你是资深肝胆外科病理专家。请基于临床病理数据，生成5步COT推理链。

【5步推理框架】
Step 1 - 肿瘤生物学评估: MVI分级、分化程度、Edmondson分级、免疫组化
Step 2 - 手术根治度评估: 切缘状态、切缘距离、手术方式
Step 3 - 肝脏背景评估: 肝硬化、纤维化分期、Child-Pugh分级、AFP
Step 4 - 分期整合: TNM分期、BCLC分期，与临床特征一致性
Step 5 - 复发风险评估: 综合三画像给出复发风险方向

输出JSON格式:
{
  "cot_chain": [
    {"step": 1, "title": "肿瘤生物学评估", "reasoning": "...", "evidence": ["..."], "conclusion": "..."},
    {"step": 2, "title": "手术根治度评估", "reasoning": "...", "evidence": ["..."], "conclusion": "..."},
    {"step": 3, "title": "肝脏背景评估", "reasoning": "...", "evidence": ["..."], "conclusion": "..."},
    {"step": 4, "title": "分期整合", "reasoning": "...", "evidence": ["..."], "conclusion": "..."},
    {"step": 5, "title": "复发风险评估", "reasoning": "...", "evidence": ["..."], "conclusion": "..."}
  ]
}"""

        user_prompt = f"""患者ID: {patient_id or report_dict.get('patient_id', '未知')}

【临床数据】
{json.dumps(report_dict, ensure_ascii=False, indent=2)}

请生成5步COT推理链，直接输出JSON。"""

        try:
            from langchain_core.messages import HumanMessage
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            response = self.llm.invoke(messages)

            text = response.content
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                return parsed.get("cot_chain", [])
        except Exception as e:
            print(f"[ClinicalAgent] COT生成失败: {e}")

        return []

    def generate_report_text(self, report_dict: Dict) -> str:

        lines = [f"{'='*50}", f"  临床评估报告 - 患者 {report_dict.get('patient_id','?')}", f"{'='*50}"]
        if "error" in report_dict:
            lines.append(f"\n[错误] {report_dict['error']}")
            return "\n".join(lines)

        for section, title in [
            ("tumor_biology", "肿瘤侵袭性画像"),
            ("surgical_curability", "手术根治度画像"),
            ("field_effect", "肝脏背景画像"),
        ]:
            sd = report_dict.get(section, {})
            if not sd: continue
            lines.append(f"\n【{title}】")
            for attr, info in sd.items():
                if not isinstance(info, dict): continue
                v = info.get("value", "?")
                rc = info.get("risk_contribution", "")
                cs = info.get("clinical_significance", "")
                lines.append(f"  {attr}: {v} | 风险: {rc} | {cs}")

        staging = report_dict.get("staging", {})
        fe = report_dict.get("field_effect", {})
        tnm = staging.get("tnm_stage") or (fe.get("tnm_stage", {}).get("value") if isinstance(fe.get("tnm_stage"), dict) else fe.get("tnm_stage"))
        bclc = staging.get("bclc_stage") or (fe.get("bclc_stage", {}).get("value") if isinstance(fe.get("bclc_stage"), dict) else fe.get("bclc_stage"))
        if tnm or bclc:
            lines.append(f"\n【分期】TNM={tnm or '?'} BCLC={bclc or '?'}")
        return "\n".join(lines)

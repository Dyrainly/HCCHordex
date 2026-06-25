import json
import os
import re
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import ENABLE_THINKING, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

def _extract_json(text: str) -> Optional[Dict]:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    for pattern in [r"```(?:json)?\s*\n?(.*?)\n?\s*```", r"\{.*\}"]:
        match = re.search(pattern, text or "", re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip() if "```" in pattern else match.group(0))
            except json.JSONDecodeError:
                pass
    return None

def _case_get(case: Any, key: str, default=None):
    if isinstance(case, dict):
        return case.get(key, default)
    return getattr(case, key, default)

def _safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _median(values: List[float]) -> Optional[float]:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2)

class RecurrenceTimeAgent:

    RANGE_LOW, RANGE_HIGH = 0.0, 120.0
    MAX_CASES_FOR_AGENT = 8

    COHORT_BACKGROUND = (
        "
        "- 观察到复发人群: 复发时间中位数约9.4月，p25约4.9月，p75约19.0月，约32%在6月内复发。\n"
        "- 未复发/删失人群: 随访时间中位数约46.8月，p25约33.0月，p75约68.1月。\n"
        "- 个体预测必须回到当前患者表型和相似病例差异，不能机械套用群体中位数。\n"
    )

    def __init__(self):
        thinking_config = {"enable_thinking": True} if ENABLE_THINKING else {"enable_thinking": False}
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            temperature=0.2,
            max_retries=5,
            extra_body=thinking_config,
        )
        self._retriever = None

    def predict(
        self,
        patient_reports: Dict,
        patient_id: str = "",
        use_rag: bool = True,
        similar_cases=None,
    ) -> Dict:
        patient_reports = dict(patient_reports or {})
        forbidden = ["recurrence_time", "recurrence_event", "OS_time", "OS_event", "recurrence", "recurrence_interval"]
        for field in forbidden:
            if field in patient_reports:
                raise ValueError(f"[SECURITY] 数据泄露: {field} 出现在Agent输入")

        pid = patient_id or patient_reports.get("patient_id", "unknown")
        clinical_data = patient_reports.pop("clinical_data", None)
        recurrence_label = self._extract_recurrence_label(patient_reports)
        query_text = self._build_query_text(patient_reports)
        clinical_section = self._format_clinical_pathology(clinical_data)
        modality_group = self._infer_modality_group(patient_reports)

        if similar_cases is not None:
            cases = self._exclude_self(similar_cases, pid)[: self.MAX_CASES_FOR_AGENT]
        elif use_rag:
            cases = self._retrieve_cases(patient_reports, modality_group, pid)
        else:
            cases = []

        range_low, range_high = self.RANGE_LOW, self.RANGE_HIGH
        case_summary = self._summarize_case_evidence(cases)

        print(f"[RecurrenceTimeAgent] 参考量程=[{range_low:.0f}-{range_high:.0f}月] 患者={pid}")
        print(f"[RecurrenceTimeAgent] 使用 {len(cases)} 个历史病例证据")

        if len(cases) < 2:
            print("[RecurrenceTimeAgent] 病例不足, 不生成兜底月份")
            result = self._unavailable_result(
                reason="有效历史病例不足，无法形成可靠的病例类比时间估计。",
                n_cases=len(cases),
                case_summary=case_summary,
            )
            return self._finalize_prediction(result, range_low, range_high, cases)

        result = self._predict_with_cases(
            query_text=query_text,
            cases=cases,
            clinical_section=clinical_section,
            range_low=range_low,
            range_high=range_high,
            recurrence_label=recurrence_label,
            case_summary=case_summary,
        )
        result = self._attach_stage_outputs(result)
        result = self._finalize_prediction(result, range_low, range_high, cases)
        print(f"[RecurrenceTimeAgent] 预测: {result.get('predicted_months')}月")
        return result

    def _predict_with_cases(
        self,
        query_text: str,
        cases: List,
        clinical_section: str,
        range_low: float,
        range_high: float,
        recurrence_label: str,
        case_summary: Dict,
    ) -> Dict:
        anchor_text = self._format_case_cards(cases)

        system_prompt = (
            "你是肝细胞癌(HCC)术后复发时间/无复发生存下界评估专家。\n"
            "你的任务不是执行固定规则，而是综合当前患者的多模态证据、主治医生综合报告、上游复发判断和历史病例证据，"
            "估计最符合证据的时间语义。\n\n"
            "【时间语义】\n"
            "- 如果综合判断支持复发，predicted_months表示预计复发事件时间。\n"
            "- 如果综合判断支持未复发，predicted_months表示无复发生存/随访下界倾向，不是复发时间。\n"
            "- 如果证据与上游复发判断冲突，可以保留冲突并给出最符合证据的时间假设，同时在time_type和fusion_logic中说明。\n\n"
            "【病例证据使用】\n"
            "1. 复发病例提供事件时间锚点，但必须比较其与当前患者的相似点和差异点。\n"
            "2. 未复发/删失病例只表示随访到X月仍未复发，是下界和保护性证据，不能当作X月复发事件。\n"
            "3. 不允许对病例时间做简单平均；必须解释历史病例与当前患者在侵袭性表型、风险因素组合和随访结局上的相通之处与差异。\n"
            "4. 当前患者自身证据仍是核心: MVI/血管侵犯、肿瘤大小/数目、卫星灶、多发、分化、切缘、肝功能、AFP、蛋白组学风险/保护信号都要纳入解释。\n"
            "5. 最终时间估计应体现专业医生的病程推断: 为什么这个患者更像哪些历史病例、又在哪些关键点不同。\n"
            "6. 检索偏倚审查: 若case_evidence_summary.bias_assessment标注bias_level为censored_heavy或recurrence_heavy(≥75%)，启动偏倚审查——追问少数派病例与当前患者的相似性、多数派病例的隐藏差异。偏倚下的\"多数一致性\"不能作为时间锚定的依据。\n\n"
            "【输出JSON】\n"
            "{\n"
            '  "predicted_months": 数值,\n'
            '  "risk_score": 0.0,\n'
            '  "time_type": "recurrence_time/recurrence_free_followup_lower_bound/recurrence_time_conflicts_with_upstream/time_estimate_uncertain_event",\n'
            '  "time_if_recurrence": 数值或null,\n'
            '  "rfs_lower_bound_if_no_recurrence": 数值或null,\n'
            '  "comparisons": "逐例病例对比分析",\n'
            '  "fusion_logic": "说明当前患者证据、历史病例证据、主治医生/上游判断如何融合",\n'
            '  "final_reasoning": "最终时间估计理由",\n'
            '  "evidence_conflicts": ["冲突或不确定性"]\n'
            "}"
        )

        bias_block = ""
        bias_info = case_summary.get("bias_assessment", {})
        bias_level = bias_info.get("bias_level", "")
        if bias_level in ("censored_heavy", "recurrence_heavy"):
            n_rec = case_summary.get("observed_recurrence_cases", 0)
            n_cen = case_summary.get("censored_no_recurrence_cases", 0)
            n_total = n_rec + n_cen
            ratio_pct = f"{max(n_rec, n_cen)/n_total*100:.0f}%"
            majority_label = "删失/未复发" if bias_level == "censored_heavy" else "复发"
            minority_label = "复发" if bias_level == "censored_heavy" else "删失/未复发"
            minority_pids = bias_info.get("minority_recurrence_pids", []) if bias_level == "censored_heavy" else bias_info.get("minority_censored_pids", [])
            minority_str = ", ".join(minority_pids) if minority_pids else "N/A"
            bias_block = (
                f"\n\n
                f"本次检索{n_total}个病例中{n_rec}例复发、{n_cen}例删失，"
                f"**{majority_label}占{ratio_pct}**，存在严重检索偏倚。\n"
                f"少数派病例（{minority_label}，ID: {minority_str}）是关键的反面证据——"
                f"它的存在说明该表型组合下另一种时间轨迹确实可能发生。\n"
                f"请追问：少数派病例与当前患者的差异是否被检索算法高估了？"
                f"它的时间锚点对当前患者的时间估计有什么启示？\n"
                f"**{ratio_pct}的病例一致性 ≠ 时间锚定应偏向多数派。**\n"
            )

        human_prompt = (
            f"
            f"{clinical_section}\n\n"
            f"
            f"
            f"{self.COHORT_BACKGROUND}\n"
            f"{bias_block}"
            f"
            f"
            "请给出JSON。你需要同时考虑复发事件时间和未复发随访下界两种时间语义，"
            "再选择与当前患者证据最一致的predicted_months。"
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ])
            parsed = _extract_json(response.content) or {}
            if "predicted_months" not in parsed:
                return self._unavailable_result(
                    reason="LLM输出缺少predicted_months，无法形成有效时间估计。",
                    n_cases=len(cases),
                    case_summary=case_summary,
                    evidence_conflicts=["LLM输出缺少predicted_months"],
                )
            parsed["n_cases"] = len(cases)
            parsed.setdefault("case_evidence_summary", case_summary)
            parsed.setdefault("evidence_conflicts", [])
            return parsed
        except Exception as e:
            print(f"[RecurrenceTimeAgent] LLM调用失败: {e}")
            traceback.print_exc()
            return self._unavailable_result(
                reason=f"LLM失败: {e}",
                n_cases=len(cases),
                case_summary=case_summary,
                evidence_conflicts=[str(e)],
            )

    def _retrieve_cases(self, patient_reports: Dict, modality_group: str, patient_id: str) -> List:
        try:
            query_summary = self._build_query_summary(patient_reports)
            retriever = self._get_retriever()
            retrieved = retriever.retrieve(
                query_summary=query_summary,
                modality_group=modality_group,
                top_k=self.MAX_CASES_FOR_AGENT + 1,
                dense_top_k=30,
                sparse_top_k=30,
            )
            return self._exclude_self(retrieved, patient_id)[: self.MAX_CASES_FOR_AGENT]
        except Exception as e:
            print(f"[RecurrenceTimeAgent] 病例检索失败: {e}")
            return []

    def _get_retriever(self):
        if self._retriever is None:
            from knowledge_base.case_retriever import CaseRetriever

            self._retriever = CaseRetriever()
        return self._retriever

    @staticmethod
    def _exclude_self(cases, patient_id: str) -> List:
        return [
            c for c in (cases or [])
            if str(_case_get(c, "patient_id", "")) != str(patient_id)
        ]

    @staticmethod
    def _infer_modality_group(patient_reports: Dict) -> str:
        has_mri = bool(patient_reports.get("imaging_report"))
        has_proteomics = bool(patient_reports.get("proteomics_report"))
        if has_mri and has_proteomics:
            return "all_three"
        if has_mri:
            return "clinical_mri"
        if has_proteomics:
            return "clinical_proteomics"
        return "clinical_only"

    def _build_query_summary(self, patient_reports: Dict) -> Dict:
        imaging_dict = patient_reports.get("imaging_report_dict") or {}
        comprehensive = imaging_dict.get("comprehensive_analysis") if isinstance(imaging_dict, dict) else {}
        if not isinstance(comprehensive, dict):
            comprehensive = {}
        try:
            from knowledge_base.kb_enricher import _build_structured_profile

            query_profile = _build_structured_profile({
                "clinical_report_dict": patient_reports.get("clinical_report_dict"),
                "proteomics_report_dict": patient_reports.get("proteomics_report_dict"),
                "imaging_report_dict": patient_reports.get("imaging_report_dict"),
            })
        except Exception:
            query_profile = {}
        return {
            "imaging_summary": comprehensive.get("diagnostic_conclusion") or (patient_reports.get("imaging_report") or "")[:1500],
            "clinical_summary": self._extract_clinical_summary(patient_reports.get("clinical_report_dict")) or (patient_reports.get("clinical_report") or "")[:2500],
            "proteomics_summary": self._extract_proteomics_summary(patient_reports.get("proteomics_report_dict")) or (patient_reports.get("proteomics_report") or "")[:2500],
            "discriminative_features": query_profile,
        }

    def _build_query_text(self, patient_reports: Dict) -> str:
        parts = []

        attending = self._format_attending_report(patient_reports.get("attending_report"))
        if attending:
            parts.append(f"【主治医生综合报告与上游复发判断】\n{attending}")

        clinical = patient_reports.get("clinical_report")
        if clinical:
            parts.append(f"【临床病理报告】\n{clinical}")

        imaging = patient_reports.get("imaging_report")
        if imaging:
            parts.append(f"【影像报告】\n{imaging}")

        proteomics = patient_reports.get("proteomics_report")
        if proteomics:
            parts.append(f"【蛋白组学报告】\n{proteomics}")

        if not parts:
            return "无可用的患者报告信息。"
        return "\n\n".join(parts)

    @staticmethod
    def _format_attending_report(attending_report) -> str:
        if not attending_report:
            return ""
        if isinstance(attending_report, str):
            return attending_report[:3000]
        if not isinstance(attending_report, dict):
            return str(attending_report)[:3000]

        parts = []
        for key, label in [
            ("recurrence", "上游复发判断"),
            ("recurrence_confidence", "上游置信度"),
        ]:
            value = attending_report.get(key)
            if value:
                parts.append(f"{label}: {value}")

        risk_factors = attending_report.get("recurrence_risk_factors")
        if risk_factors:
            parts.append("上游复发风险因素: " + "；".join(map(str, risk_factors[:20])))

        for step in attending_report.get("cot_chain", []) or []:
            if not isinstance(step, dict):
                continue
            title = step.get("title", "")
            conclusion = step.get("conclusion", "")
            reasoning = step.get("reasoning", "")
            text = " ".join(str(x) for x in [title, conclusion, reasoning] if x)
            if text:
                parts.append(text[:800])

        final_diagnosis = attending_report.get("final_diagnosis", {})
        if isinstance(final_diagnosis, dict):
            key_factors = final_diagnosis.get("key_factors", [])
            if key_factors:
                names = [
                    item.get("factor", str(item)) if isinstance(item, dict) else str(item)
                    for item in key_factors
                ]
                parts.append("关键因素: " + "；".join(names[:20]))

        raw = attending_report.get("raw_report")
        if raw:
            parts.append(str(raw)[:1500])

        return "\n".join(parts)[:6000]

    def _extract_recurrence_label(self, patient_reports: Dict) -> str:

        rec_pred = patient_reports.get("recurrence_prediction")
        if isinstance(rec_pred, dict):
            label = str(rec_pred.get("recurrence", "") or "")
            if label:
                return label

        attending = patient_reports.get("attending_report")
        if isinstance(attending, dict):
            for key in ("recurrence", "event_prediction", "recurrence_prediction"):
                label = str(attending.get(key, "") or "")
                if label:
                    return label
        return ""

    @staticmethod
    def _unavailable_result(
        reason: str,
        n_cases: int,
        case_summary: Dict,
        evidence_conflicts: Optional[List[str]] = None,
    ) -> Dict:
        conflicts = list(evidence_conflicts or [])
        return {
            "predicted_months": None,
            "risk_score": None,
            "time_type": "time_estimate_unavailable",
            "reasoning": reason,
            "comparisons": "",
            "fusion_logic": reason,
            "evidence_conflicts": conflicts,
            "n_cases": n_cases,
            "case_evidence_summary": case_summary,
            "time_prediction": {
                "predicted_months": None,
                "risk_score": None,
                "time_type": "time_estimate_unavailable",
                "confidence": "低",
                "reasoning": reason,
                "n_cases": n_cases,
                "evidence_conflicts": conflicts,
            },
        }

    def _finalize_prediction(
        self,
        result: Dict,
        range_low: float,
        range_high: float,
        similar_cases: List,
    ) -> Dict:

        result = dict(result or {})
        flags = list(result.get("consistency_flags", []) or [])

        predicted = _safe_float(result.get("predicted_months"))
        risk_score = _safe_float(result.get("risk_score"))
        evidence = self._summarize_case_evidence(similar_cases)

        observed = evidence.get("observed_recurrence_cases", 0)
        censored = evidence.get("censored_no_recurrence_cases", 0)
        early_observed = evidence.get("observed_recurrence_months", {}).get("early_le_24m", 0)

        if censored and censored >= observed:
            flags.append(f"历史病例以删失/未复发为主({censored}删失/{observed}复发)，删失月份解释为无复发生存下界。")
        if observed > 0 and observed >= censored:
            flags.append(f"历史病例存在复发时间信号({observed}复发/{censored}删失，24月内复发{early_observed}例)，需在推理中解释冲突。")

        time_type = result.get("time_type") or "time_estimate_uncertain_event"
        if predicted is not None:
            predicted = round(max(range_low, min(range_high, predicted)), 1)
        if risk_score is not None:
            risk_score = round(max(0.0, min(1.0, risk_score)), 4)

        result["predicted_months"] = predicted
        result["risk_score"] = risk_score
        result["time_type"] = time_type
        result["case_evidence_summary"] = evidence
        result["consistency_flags"] = flags

        time_prediction = dict(result.get("time_prediction") or {})
        time_prediction["predicted_months"] = predicted
        time_prediction["risk_score"] = risk_score
        time_prediction["time_type"] = time_type
        time_prediction["n_cases"] = result.get("n_cases", len(similar_cases or []))
        time_prediction["evidence_conflicts"] = result.get("evidence_conflicts", [])
        reasoning = (
            result.get("final_reasoning")
            or result.get("reasoning")
            or time_prediction.get("reasoning")
            or ""
        )
        if flags:
            reasoning = (reasoning + "\n\n一致性校验: " + "；".join(flags[-3:])).strip()
        time_prediction["reasoning"] = reasoning
        result["time_prediction"] = time_prediction
        return result

    def _attach_stage_outputs(
        self,
        result: Dict,
    ) -> Dict:
        result = dict(result or {})
        predicted = _safe_float(result.get("predicted_months"))
        if predicted is not None:
            result["predicted_months"] = round(predicted, 1)
        else:
            result["predicted_months"] = None

        confidence = result.get("confidence", "中")

        result.setdefault("time_prediction", {
            "predicted_months": result["predicted_months"],
            "confidence": confidence,
            "reasoning": result.get("final_reasoning") or result.get("reasoning", ""),
            "n_cases": result.get("n_cases", 0),
            "evidence_conflicts": result.get("evidence_conflicts", []),
        })
        return result

    def _summarize_case_evidence(self, cases) -> Dict:
        observed_months = []
        censored_months = []
        scores = []
        for case in cases or []:
            event = _case_get(case, "recurrence_event")
            months = _safe_float(_case_get(case, "recurrence_months", _case_get(case, "rec_months")))
            score = _safe_float(_case_get(case, "final_score", _case_get(case, "case_retrieval_score")))
            if score is not None:
                scores.append(score)
            try:
                event_i = int(event)
            except (TypeError, ValueError):
                continue
            if months is None:
                continue
            if event_i == 1:
                observed_months.append(months)
            elif event_i == 0:
                censored_months.append(months)
        return {
            "n_cases": len(cases or []),
            "observed_recurrence_cases": len(observed_months),
            "censored_no_recurrence_cases": len(censored_months),
            "observed_recurrence_months": {
                "median": round(_median(observed_months), 1) if observed_months else None,
                "min": round(min(observed_months), 1) if observed_months else None,
                "max": round(max(observed_months), 1) if observed_months else None,
                "early_le_12m": sum(1 for m in observed_months if m <= 12),
                "early_le_24m": sum(1 for m in observed_months if m <= 24),
            },
            "censored_followup_months": {
                "median": round(_median(censored_months), 1) if censored_months else None,
                "max": round(max(censored_months), 1) if censored_months else None,
                "long_followup_ge_36m": sum(1 for m in censored_months if m >= 36),
            },
            "similarity": {
                "max": round(max(scores), 3) if scores else None,
                "median": round(_median(scores), 3) if scores else None,
            },
            "interpretation_note": "复发病例提供事件时间锚点；删失病例提供无复发生存下界。该概览只帮助病例类比审查，不能替代逐例医学判断。",
        }

    def _format_case_cards(self, cases) -> str:
        if not cases:
            return "未检索到相似病例。"
        lines = []
        for i, case in enumerate(cases, 1):
            pid = _case_get(case, "patient_id", "?")
            event = _case_get(case, "recurrence_event")
            months = _safe_float(_case_get(case, "recurrence_months", _case_get(case, "rec_months")))
            score = _safe_float(_case_get(case, "final_score", _case_get(case, "case_retrieval_score")))
            context = _case_get(case, "retrieval_context", "") or ""
            try:
                event_i = int(event)
            except (TypeError, ValueError):
                event_i = None
            if event_i == 1 and months is not None:
                outcome = f"复发事件时间={months:.1f}月"
            elif event_i == 0 and months is not None:
                outcome = f"未复发/删失，随访下界={months:.1f}月"
            else:
                outcome = "结局未知"
            score_text = f"，相似度={score:.3f}" if score is not None else ""
            lines.append(f"
            if context:
                lines.append(f"- 摘要: {str(context)[:1000]}")
        return "\n".join(lines)

    @staticmethod
    def _format_clinical_pathology(clinical_data: Optional[Dict]) -> str:
        if not clinical_data:
            return ""
        lines = ["
        for key, label in [
            ("病理诊断_MVI分级", "MVI"),
            ("MVI分级", "MVI"),
            ("病理诊断_Edmondson分级", "分化"),
            ("分化程度", "分化"),
            ("肿瘤大小(cm)", "大小"),
            ("病理诊断_肿瘤最大径(cm)", "大小"),
            ("病理诊断_包膜", "包膜"),
            ("病理诊断_微卫星子灶", "卫星灶"),
            ("AFP", "AFP"),
            ("甲胎蛋白", "AFP"),
        ]:
            value = clinical_data.get(key)
            if value is not None and str(value) not in ("未知", "nan", ""):
                lines.append(f"- {label}: {value}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def _extract_clinical_summary(self, clinical_dict: Optional[Dict]) -> str:
        if not clinical_dict:
            return ""
        parts = []
        tumor_biology = clinical_dict.get("tumor_biology", {})
        if isinstance(tumor_biology, dict):
            for key in ["mvi_grade", "tumor_size_cm", "differentiation", "vascular_invasion", "satellite_nodules"]:
                value = tumor_biology.get(key, {})
                if isinstance(value, dict) and "value" in value:
                    parts.append(f"{key}={value['value']}")
                elif value:
                    parts.append(f"{key}={value}")
        return " ".join(parts)

    @staticmethod
    def _extract_proteomics_summary(proteomics_dict: Optional[Dict]) -> str:
        if not proteomics_dict:
            return ""
        encoding = proteomics_dict.get("encoding_result", {})
        signature = encoding.get("proteomic_signature", []) if isinstance(encoding, dict) else []
        if signature:
            return " ".join(map(str, signature[:5]))
        return json.dumps(proteomics_dict, ensure_ascii=False)[:1200]

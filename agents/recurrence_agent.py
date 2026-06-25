import json
import os
import re
import sys
import traceback
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

def _extract_json(text: str) -> Optional[Dict]:

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(pattern, text or "", re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    brace_match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
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

class RecurrenceAgent:

    MAX_CASES_FOR_AGENT = 8

    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            temperature=0.2,
            max_retries=5,
        )

    def predict(
        self,
        patient_reports: Dict,
        patient_id: str = "",
        use_rag: bool = True,
        similar_cases=None,
    ) -> Dict:

        patient_reports = dict(patient_reports or {})

        pid = patient_id or patient_reports.get("patient_id", "unknown")
        query_text = self._build_query_text(patient_reports)
        modality_group = self._infer_modality_group(patient_reports)

        print(f"[RecurrenceAgent] 开始对患者 {pid} 进行复发预测 (模态组: {modality_group})...")
        result = self._stage1_recurrence(
            query_text=query_text,
            modality_group=modality_group,
            patient_reports=patient_reports,
            patient_id=pid,
            use_rag=use_rag,
            provided_cases=similar_cases,
        )
        print(f"[RecurrenceAgent] 患者 {pid} 复发预测完成: {result.get('recurrence', '未知')}")
        return result

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

    def _stage1_recurrence(
        self,
        query_text: str,
        modality_group: str,
        patient_reports: Dict,
        patient_id: str = "",
        use_rag: bool = True,
        provided_cases=None,
    ) -> Dict:

        try:
            similar_cases = self._resolve_cases(
                patient_reports=patient_reports,
                query_text=query_text,
                modality_group=modality_group,
                patient_id=patient_id,
                use_rag=use_rag,
                provided_cases=provided_cases,
            )
            context = self._format_similar_cases(similar_cases)
            case_summary = self._summarize_case_evidence(similar_cases)

            print("[RecurrenceAgent] 综合证据审查...")
            system_prompt = (
                "你是一位肝细胞癌(HCC)术后预后结局预测专家，负责判断患者术后是否会发生肝内复发。\n"
                "你需要像专业医生会诊一样进行临床类比推理，不能使用固定阈值或硬性规则来做结论。\n\n"
                "【证据源】\n"
                "1. 当前患者自身多模态证据: 临床病理、影像、蛋白组学、主治医生综合报告。\n"
                "2. 历史病例证据: 知识库中与当前患者相似的真实过往病例。\n\n"
                "【Agentic case review 方法】\n"
                "A. 先抽取当前患者表型: 肿瘤负荷、MVI/血管侵犯、卫星灶/多发、分化、切缘、肝硬化/肝功能、AFP等标志物、蛋白组学高危/保护通路、缺失信息。\n"
                "B. 再逐例阅读历史病例: 每个病例都是case witness。寻找它与当前患者在肿瘤生物学、影像侵袭性、临床病理和蛋白组学信号上的相通之处，"
                "同时说明差异在哪里、结局能支持什么、不能支持什么。\n"
                "C. 对删失病例要严格解释为'随访至X月仍未复发的下界'，不能当作X月复发事件。\n"
                "D. 同时建立复发与未复发两边的证据链，显式处理冲突。高危因素提高风险，但不等于必然复发；长期删失病例也不能单独否定明显侵袭性表型。\n"
                "E. 历史病例具有很大参考意义，使用方式是临床类比和证据溯源。最终结论必须回到当前患者个体证据与病例证据的一致性、差异性和病程逻辑。\n\n"
                "【检索偏倚识别与校准】\n"
                "F. 先检查历史病例的复发/删失比例（已在case_evidence_summary.bias_assessment中标注）。若某一结局占绝大多数(≥75%)，必须启动检索偏倚审查：\n"
                "  - 多数派的一致性可能是检索算法的系统偏倚，不是真实预后分布。\n"
                "  - 少数派病例是关键的'反面证据'——它的存在证明了该表型下另一种结局确实可能发生。\n"
                "  - 追问：少数派病例与当前患者的差异是否被检索算法高估了？多数派是否在某个隐藏维度（肝硬化程度、AFP水平、蛋白组学净风险方向）上与当前患者不同？\n"
                "G. 删失病例的'一致性'不能作为排除复发的正面证据。7个删失 ≠ 87.5%不复发概率。\n"
                "H. 当bias_assessment警告bias_level为'censored_heavy'或'recurrence_heavy'时，降低历史病例锚定权重，提高当前患者自身多模态证据的权重。\n"
                "I. 历史病例用于校准和类比，不是用于投票。偏倚环境下的'多数胜出'是危险的推理捷径。\n\n"
                "【输出要求】只输出JSON，字段如下:\n"
                "{\n"
                '  "recurrence": "复发/未复发",\n'
                '  "confidence": "高/中/低",\n'
                '  "event_probability": 0.0,\n'
                '  "current_patient_phenotype": "当前患者关键表型摘要",\n'
                '  "case_evidence": ["逐例证据审查要点"],\n'
                '  "reasoning": "综合判断理由",\n'
                '  "risk_factors": ["复发风险因素"],\n'
                '  "protective_factors": ["未复发/长无复发生存支持因素"],\n'
                '  "evidence_conflicts": ["冲突或不确定性"],\n'
                '  "evidence_weighting": "说明当前患者证据、历史病例证据、主治医生报告如何被权衡"\n'
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
                    f"本次检索{ n_total }个病例中{n_rec}例复发、{n_cen}例删失，"
                    f"**{majority_label}占{ratio_pct}**，存在严重检索偏倚风险。\n\n"
                    f"在开始case review之前，请先回答三个问题：\n"
                    f"1. 唯一少数派病例（{minority_label}，ID: {minority_str}）与当前患者在哪些关键特征上不同？"
                    f"这些差异是否足以完全排除它的参考价值？如果这些差异在生物学上并非决定性，它的结局对当前患者意味着什么？\n"
                    f"2. 多数派（{majority_label}）病例中，哪些隐藏特征可能掩盖了与当前患者的真正差异？"
                    f"检索算法基于表面特征相似度排序，可能遗漏肝硬化程度、AFP水平、蛋白组学净风险方向等决定性的预后差异。\n"
                    f"3. 如果完全抛开历史病例，仅看当前患者自身多模态证据（影像、临床病理、蛋白组学），"
                    f"支持复发和支持不复发的信号各是什么？哪个更强？\n\n"
                    f"**关键提醒：{ratio_pct}的病例一致性 ≠ {ratio_pct}的复发/不复发概率。**"
                    f"历史病例用于类比和校准，不是用于投票。偏倚环境下的'多数胜出'是危险的推理捷径。\n"
                )

            human_prompt = (
                f"
                f"{bias_block}"
                f"
                f"{json.dumps(case_summary, ensure_ascii=False, indent=2)}\n\n"
                f"
                "请按Agentic case review方法进行综合判断。重点寻找当前患者与历史病例的相通之处、关键差异、删失含义和病程逻辑，"
                "再以专业医生的角度给出个体化预测。"
            )

            response = self.llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            )
            parsed = _extract_json(response.content)

            if parsed:
                recurrence = self._normalize_recurrence(str(parsed.get("recurrence", "未复发")))
                result = {
                    "recurrence": recurrence,
                    "confidence": parsed.get("confidence", "中"),
                    "reasoning": parsed.get("reasoning", ""),
                    "event_probability": self._normalize_probability(parsed.get("event_probability")),
                    "current_patient_phenotype": parsed.get("current_patient_phenotype", ""),
                    "case_evidence": parsed.get("case_evidence", []),
                    "risk_factors": parsed.get("risk_factors", []),
                    "protective_factors": parsed.get("protective_factors", []),
                    "evidence_conflicts": parsed.get("evidence_conflicts", []),
                    "evidence_weighting": parsed.get("evidence_weighting", ""),
                    "case_evidence_summary": case_summary,
                    "similar_cases_used": len(similar_cases),
                }
                return result

            return {
                "recurrence": "未复发",
                "confidence": "低",
                "reasoning": f"JSON解析失败，原始输出: {(response.content or '')[:300]}",
                "event_probability": None,
                "risk_factors": [],
                "protective_factors": [],
                "evidence_conflicts": ["LLM输出无法解析"],
                "evidence_weighting": "",
                "case_evidence_summary": case_summary,
                "similar_cases_used": len(similar_cases),
            }

        except Exception as e:
            print(f"[RecurrenceAgent] 阶段1异常: {e}")
            traceback.print_exc()
            return {
                "recurrence": "未复发",
                "confidence": "低",
                "reasoning": f"阶段1执行异常: {str(e)}",
                "event_probability": None,
                "risk_factors": [],
                "protective_factors": [],
                "evidence_conflicts": [str(e)],
                "evidence_weighting": "",
                "case_evidence_summary": {},
                "similar_cases_used": 0,
            }

    def _resolve_cases(
        self,
        patient_reports: Dict,
        query_text: str,
        modality_group: str,
        patient_id: str,
        use_rag: bool,
        provided_cases=None,
    ) -> List:
        if provided_cases is not None:
            cases = self._exclude_self(provided_cases, patient_id)[: self.MAX_CASES_FOR_AGENT]
            print(f"[RecurrenceAgent] 使用上游检索病例: {len(cases)}个")
            return cases
        if not use_rag:
            return []
        try:
            from knowledge_base.case_retriever import CaseRetriever

            query_summary = self._build_query_summary(patient_reports, query_text)
            retriever = CaseRetriever()
            retrieved = retriever.retrieve(
                query_summary=query_summary,
                modality_group=modality_group,
                top_k=self.MAX_CASES_FOR_AGENT + 1,
                dense_top_k=30,
                sparse_top_k=30,
            )
            cases = self._exclude_self(retrieved, patient_id)[: self.MAX_CASES_FOR_AGENT]
            print(f"[RecurrenceAgent] 病例检索完成: {len(cases)}个相似病例")
            return cases
        except Exception as retrieval_error:
            print(f"[RecurrenceAgent] 病例检索失败，降级为仅患者报告推理: {retrieval_error}")
            return []

    @staticmethod
    def _exclude_self(cases, patient_id: str) -> List:
        return [
            c for c in (cases or [])
            if str(_case_get(c, "patient_id", "")) != str(patient_id)
        ]

    def _build_query_summary(self, patient_reports: Dict, query_text: str) -> Dict:
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
            "clinical_summary": (patient_reports.get("clinical_report") or query_text)[:2500],
            "proteomics_summary": (patient_reports.get("proteomics_report") or "")[:2500],
            "discriminative_features": query_profile,
        }

    def _format_similar_cases(self, cases) -> str:
        if not cases:
            return "未检索到相似病例。"
        lines = ["
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
                outcome = f"观察到复发，事件时间={months:.1f}月"
            elif event_i == 0 and months is not None:
                outcome = f"未复发/删失，随访至{months:.1f}月仍未复发（无复发生存下界）"
            else:
                outcome = "真实结局未知"

            score_text = f"，相似度={score:.3f}" if score is not None else ""
            lines.append(f"
            if context:
                lines.append(f"- 摘要: {str(context)[:1000]}")
        return "\n".join(lines)

    def _summarize_case_evidence(self, cases) -> Dict:
        observed_months = []
        censored_months = []
        scores = []
        rec_pids, cen_pids = [], []
        for case in cases or []:
            pid = _case_get(case, "patient_id", "?")
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
                rec_pids.append(str(pid))
            elif event_i == 0:
                censored_months.append(months)
                cen_pids.append(str(pid))
        n_total = len(cases or [])
        n_rec = len(observed_months)
        n_cen = len(censored_months)

        rec_ratio = n_rec / n_total if n_total > 0 else 0
        cen_ratio = n_cen / n_total if n_total > 0 else 0
        if rec_ratio >= 0.75:
            bias_level = "recurrence_heavy"
            warning = (
                f"检索结果高度偏向复发病例({n_rec}/{n_total}，{rec_ratio:.0%})。"
                "表面特征相似≠生物学行为一致。请勿将复发病例的多数一致性等同于'大概率复发'。"
                "追问：少数派删失病例中是否存在当前患者可能共享的保护性特征？"
            )
        elif cen_ratio >= 0.75:
            bias_level = "censored_heavy"
            warning = (
                f"检索结果高度偏向删失病例({n_cen}/{n_total}，{cen_ratio:.0%})。"
                "这可能是检索算法的系统性偏倚——表面特征相似≠生物学行为一致。"
                "请勿将删失病例的多数一致性等同于'大概率不复发'。"
                "少数派复发病例的存在说明该表型组合下复发确实会发生。"
                "追问：唯一复发病例与当前患者的差异是否被检索算法高估了？"
                "删失病例的隐藏差异（肝硬化程度、AFP、蛋白组学净风险方向等）是否未在相似度中体现？"
            )
        elif rec_ratio >= 0.6 or cen_ratio >= 0.6:
            bias_level = "mild"
            warning = f"检索结果略偏向某一结局。请在病例审查时注意少数派病例的参考价值。"
        else:
            bias_level = "balanced"
            warning = ""

        return {
            "n_cases": n_total,
            "observed_recurrence_cases": n_rec,
            "censored_no_recurrence_cases": n_cen,
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
            "interpretation_note": "删失病例是无复发生存下界；以上概览只帮助医生式类比审查，不能替代逐例医学判断。",
            "bias_assessment": {
                "recurrence_ratio": round(rec_ratio, 3),
                "censored_ratio": round(cen_ratio, 3),
                "bias_level": bias_level,
                "minority_recurrence_pids": rec_pids if cen_ratio >= 0.75 else [],
                "minority_censored_pids": cen_pids if rec_ratio >= 0.75 else [],
                "warning": warning,
            } if n_total > 0 else {},
        }

    @staticmethod
    def _normalize_recurrence(raw: str) -> str:
        if "未复发" in raw or "无复发" in raw or "不复发" in raw:
            return "未复发"
        if "复发" in raw:
            return "复发"
        return "未复发"

    @staticmethod
    def _normalize_probability(value):
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip().replace("%", "")
            parsed = _safe_float(text)
            if parsed is None:
                return None
            value = parsed / 100.0 if parsed > 1 else parsed
        parsed = _safe_float(value)
        if parsed is None:
            return None
        return round(max(0.0, min(1.0, parsed)), 4)

    def _build_query_text(self, patient_reports: Dict) -> str:
        parts = []
        modalities = []

        clinical = patient_reports.get("clinical_report")
        if clinical:
            parts.append(f"【核心临床病理信息】\n{clinical}")
            modalities.append("临床")

        imaging = patient_reports.get("imaging_report")
        if imaging:
            parts.append(f"【影像学信息】\n{imaging}")
            modalities.append("影像")

        proteomics = patient_reports.get("proteomics_report")
        if proteomics:
            parts.append(f"【蛋白组学信息】\n{proteomics}")
            modalities.append("蛋白组学")

        attending = self._format_attending_report(patient_reports.get("attending_report"))
        if attending:
            parts.append(f"【主治医生综合报告】\n{attending}")

        if not parts:
            return "无可用的患者报告信息。"

        parts.append(f"【患者模态组成】{'、'.join(modalities) if modalities else '未知'}")
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

        return "\n".join(parts)[:5000]

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import traceback
import math
import numpy as np
from typing import Dict, Any, List

from langgraph.graph import StateGraph, START, END
from graph.state import HCCDiagnosisState
from config import *

def _preload_modules():

    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    import knowledge_base

    import transformers
    from transformers import GPT2TokenizerFast
    import langchain_openai
    import langchain_core
    import langchain_core.language_models
    from agents.imaging_agent import ImagingAgent
    from agents.clinical_agent import ClinicalAgent
    from agents.proteomics_agent import ProteomicsAgent
    from agents.attending_agent import AttendingAgent
    from agents.recurrence_agent import RecurrenceAgent
    from agents.recurrence_time_agent import RecurrenceTimeAgent

    from knowledge_base.kb_builder import KnowledgeBaseBuilder
    from knowledge_base.rag_retriever import _get_shared_chroma
    _get_shared_chroma()

_preload_modules()

def _finite_float(value, default: float = 0.0) -> float:

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result

def build_outcome_metadata(row, has_mri: bool, has_proteomics: bool) -> Dict[str, str]:
    rec_days = _finite_float(row.get("recurrence_time", row.get("recurrence_interval", 0)))
    recurrence_event = int(_finite_float(row.get("recurrence_event", row.get("recurrence", 0))))
    recurrence = int(_finite_float(row.get("recurrence", recurrence_event)))

    if has_mri and has_proteomics:
        modality_group = "all_three"
    elif has_mri:
        modality_group = "clinical_mri"
    elif has_proteomics:
        modality_group = "clinical_proteomics"
    else:
        modality_group = "clinical_only"

    return {
        "actual_recurrence_months": str(round(rec_days / 30.44, 1)),
        "actual_recurrence_event": str(recurrence_event),
        "actual_recurrence": str(recurrence),
        "has_mri": "1" if has_mri else "0",
        "has_proteomics": "1" if has_proteomics else "0",
        "modality_group": modality_group,
    }

def build_imaging_feature_metadata(discriminative_features: Dict) -> Dict[str, str]:
    if not discriminative_features or not isinstance(discriminative_features, dict):
        return {}

    if not discriminative_features.get("parse_success", False):
        return {"imaging_features_available": "0"}

    meta = {"imaging_features_available": "1"}

    core = discriminative_features.get("core_features", {})

    size_mm = core.get("size_mm", {})
    if size_mm.get("value") is not None:
        meta["tumor_size_mm"] = str(round(float(size_mm["value"]), 1))
        meta["tumor_size_confidence"] = str(round(float(size_mm.get("confidence", 0)), 2))

    aphe = core.get("aphe", {})
    if aphe.get("present") is not None:
        meta["aphe_present"] = "1" if aphe["present"] else "0"
        meta["aphe_pattern"] = str(aphe.get("pattern", "none"))
        meta["aphe_confidence"] = str(round(float(aphe.get("confidence", 0)), 2))

    washout = core.get("washout", {})
    if washout.get("present") is not None:
        meta["washout_present"] = "1" if washout["present"] else "0"
        meta["washout_timing"] = str(washout.get("timing", "none"))
        meta["washout_confidence"] = str(round(float(washout.get("confidence", 0)), 2))

    capsule = core.get("capsule", {})
    if capsule.get("present") is not None:
        meta["capsule_present"] = "1" if capsule["present"] else "0"
        meta["capsule_type"] = str(capsule.get("type", "none"))
        meta["capsule_confidence"] = str(round(float(capsule.get("confidence", 0)), 2))

    lirads = discriminative_features.get("lirads_assessment", {})
    if lirads.get("category"):
        meta["lirads_category"] = str(lirads["category"])
        meta["lirads_confidence"] = str(round(float(lirads.get("confidence", 0)), 2))

    derived = discriminative_features.get("derived_risk_flags", {})

    mvi = derived.get("mvi_high_risk", {})
    if mvi.get("flag") is not None:
        meta["mvi_high_risk"] = "1" if mvi["flag"] else "0"
        meta["mvi_confidence"] = str(round(float(mvi.get("confidence", 0)), 2))

    multifocal = derived.get("multifocal", {})
    if multifocal.get("flag") is not None:
        meta["multifocal"] = "1" if multifocal["flag"] else "0"
        meta["tumor_count"] = str(multifocal.get("count", 1))

    vasc = derived.get("vascular_invasion", {})
    if vasc.get("flag") is not None:
        meta["vascular_invasion"] = "1" if vasc["flag"] else "0"
        meta["vascular_invasion_location"] = str(vasc.get("location", "none"))

    sig = discriminative_features.get("discriminative_signature", [])
    if sig and isinstance(sig, list):
        meta["discriminative_signature"] = "|".join(sig[:10])

    uncertainty = discriminative_features.get("uncertainty", {})
    low_conf = uncertainty.get("low_confidence_features", [])
    if low_conf:
        meta["low_confidence_features"] = "|".join(low_conf[:5])

    return meta

def load_patient_data(state: HCCDiagnosisState) -> Dict:

    from data_loader import (
        get_patient_data,
        get_clinical_summary,
        get_proteomics_summary,
        load_clinical_data,
        load_proteomics_data,
    )

    patient_id = state["patient_id"]
    errors = list(state.get("errors", []))

    try:
        print(f"[workflow] 正在加载患者 {patient_id} 的多模态数据...")
        patient_data = get_patient_data(patient_id)

        mri_paths = patient_data.get("mri_paths")
        clinical_data = patient_data.get("clinical_data")
        proteomics_data = patient_data.get("proteomics_data")
        has_mri = patient_data.get("has_mri", False)
        has_clinical = patient_data.get("has_clinical", False)
        has_proteomics = patient_data.get("has_proteomics", False)

        if state.get("skip_imaging", False):
            has_mri = False
            mri_paths = None
            print(f"[workflow] skip_imaging=True，已跳过患者 {patient_id} 的影像数据")

        if state.get("skip_proteomics", False):
            has_proteomics = False
            proteomics_data = None
            print(f"[workflow] skip_proteomics=True，已跳过患者 {patient_id} 的蛋白组学数据")

        clinical_summary = None
        if has_clinical:
            clinical_df = load_clinical_data()
            clinical_summary = get_clinical_summary(patient_id, clinical_df, include_outcomes=False)

        proteomics_summary = None
        if has_proteomics:
            proteomics_df = load_proteomics_data()
            proteomics_summary = get_proteomics_summary(patient_id, proteomics_df)

        modalities = []
        if has_mri:
            modalities.append("MRI影像")
        if has_clinical:
            modalities.append("临床")
        if has_proteomics:
            modalities.append("蛋白组学")
        print(f"[workflow] 患者 {patient_id} 可用模态: {', '.join(modalities) if modalities else '无'}")

        return {
            "mri_paths": mri_paths,
            "clinical_data": clinical_data,
            "proteomics_data": proteomics_data,
            "clinical_summary": clinical_summary,
            "proteomics_summary": proteomics_summary,
            "has_mri": has_mri,
            "has_clinical": has_clinical,
            "has_proteomics": has_proteomics,
            "errors": errors,
        }

    except Exception as e:
        error_msg = f"[load_patient_data] 加载患者数据失败: {e}"
        print(error_msg)
        traceback.print_exc()
        errors.append(error_msg)
        return {
            "mri_paths": None,
            "clinical_data": None,
            "proteomics_data": None,
            "clinical_summary": None,
            "proteomics_summary": None,
            "has_mri": False,
            "has_clinical": False,
            "has_proteomics": False,
            "errors": errors,
        }

def imaging_analysis(state: HCCDiagnosisState) -> Dict:

    from agents.imaging_agent import ImagingAgent

    patient_id = state["patient_id"]
    errors = list(state.get("errors", []))

    if not state.get("has_mri", False) or not state.get("mri_paths"):
        print(f"[workflow] 患者 {patient_id} 无MRI数据，跳过影像分析")
        return {
            "imaging_report": None,
            "imaging_report_dict": None,
            "imaging_cot": None,
            "imaging_status": "missing_data",
            "errors": errors,
        }

    agent = None
    try:
        print(f"[workflow] 启动影像科智能体分析患者 {patient_id}...")
        agent = ImagingAgent()
        report_dict = agent.generate_report(state["mri_paths"], patient_id)
        report_text = agent.generate_report_text(report_dict)

        imaging_cot = agent.generate_cot(report_dict, patient_id)
        print(f"[workflow] 影像COT生成完成: {len(imaging_cot)}步")

        print(f"[workflow] 影像分析完成，报告长度: {len(report_text)} 字符")
        return {
            "imaging_report": report_text,
            "imaging_report_dict": report_dict,
            "imaging_cot": imaging_cot,
            "imaging_status": "completed",
            "errors": errors,
        }

    except Exception as e:
        error_msg = f"[imaging_analysis] 影像分析失败: {e}"
        print(error_msg)
        traceback.print_exc()
        errors.append(error_msg)
        return {
            "imaging_report": None,
            "imaging_report_dict": None,
            "imaging_cot": None,
            "imaging_status": "error",
            "errors": errors,
        }
    finally:
        if agent is not None:
            agent.release_model()

def clinical_analysis(state: HCCDiagnosisState) -> Dict:

    from agents.clinical_agent import ClinicalAgent

    patient_id = state["patient_id"]
    errors = list(state.get("errors", []))

    if not state.get("has_clinical", False) or not state.get("clinical_summary"):
        print(f"[workflow] 患者 {patient_id} 无临床数据，跳过临床分析")
        return {
            "clinical_report": None,
            "clinical_report_dict": None,
            "clinical_cot": None,
            "clinical_status": "missing_data",
            "errors": errors,
        }

    try:
        print(f"[workflow] 启动临床医生智能体分析患者 {patient_id}...")
        agent = ClinicalAgent()
        report_dict = agent.analyze(state["clinical_summary"], patient_id)
        report_text = agent.generate_report_text(report_dict)

        clinical_cot = agent.generate_cot(report_dict, patient_id)
        print(f"[workflow] 临床COT生成完成: {len(clinical_cot)}步")

        print(f"[workflow] 临床分析完成，报告长度: {len(report_text)} 字符")
        return {
            "clinical_report": report_text,
            "clinical_report_dict": report_dict,
            "clinical_cot": clinical_cot,
            "clinical_status": "completed",
            "errors": errors,
        }

    except Exception as e:
        error_msg = f"[clinical_analysis] 临床分析失败: {e}"
        print(error_msg)
        traceback.print_exc()
        errors.append(error_msg)
        return {
            "clinical_report": None,
            "clinical_report_dict": None,
            "clinical_cot": None,
            "clinical_status": "error",
            "errors": errors,
        }

def proteomics_analysis(state: HCCDiagnosisState) -> Dict:

    from agents.proteomics_agent import ProteomicsAgent

    patient_id = state["patient_id"]
    errors = list(state.get("errors", []))

    if not state.get("has_proteomics", False) or not state.get("proteomics_summary"):
        print(f"[workflow] 患者 {patient_id} 无蛋白组学数据，跳过蛋白组学分析")
        return {
            "proteomics_report": None,
            "proteomics_report_dict": None,
            "proteomics_cot": None,
            "proteomics_status": "missing_data",
            "errors": errors,
        }

    try:
        print(f"[workflow] 启动蛋白组学智能体分析患者 {patient_id}...")
        agent = ProteomicsAgent()
        report_dict = agent.analyze(state["proteomics_summary"], patient_id)
        report_text = agent.generate_report_text(report_dict)

        proteomics_cot = agent.generate_cot(report_dict, patient_id)
        print(f"[workflow] 蛋白组学COT生成完成: {len(proteomics_cot)}步")

        print(f"[workflow] 蛋白组学分析完成，报告长度: {len(report_text)} 字符")
        return {
            "proteomics_report": report_text,
            "proteomics_report_dict": report_dict,
            "proteomics_cot": proteomics_cot,
            "proteomics_status": "completed",
            "errors": errors,
        }

    except Exception as e:
        error_msg = f"[proteomics_analysis] 蛋白组学分析失败: {e}"
        print(error_msg)
        traceback.print_exc()
        errors.append(error_msg)
        return {
            "proteomics_report": None,
            "proteomics_report_dict": None,
            "proteomics_cot": None,
            "proteomics_status": "error",
            "errors": errors,
        }

def attending_synthesis(state: HCCDiagnosisState) -> Dict:

    from agents.attending_agent import AttendingAgent

    patient_id = state["patient_id"]
    errors = list(state.get("errors", []))

    try:
        print(f"[workflow] 启动主治医生综合诊断患者 {patient_id}...")
        agent = AttendingAgent()

        imaging_cot = state.get("imaging_cot") or []
        clinical_cot = state.get("clinical_cot") or []
        proteomics_cot = state.get("proteomics_cot") or []

        print(f"[workflow] 合并COT: 影像{len(imaging_cot)}步 + 临床{len(clinical_cot)}步 + 蛋白组学{len(proteomics_cot)}步")

        report = agent.synthesize(
            imaging_report=state.get("imaging_report"),
            clinical_report=state.get("clinical_report"),
            proteomics_report=state.get("proteomics_report"),
            patient_id=patient_id,
        )

        cot_text = agent.get_cot_text(report) if report else ""
        print(f"[workflow] 主治医生综合诊断完成")

        return {
            "attending_report": report,
            "attending_cot": cot_text,
            "errors": errors,
        }

    except Exception as e:
        error_msg = f"[attending_synthesis] 综合诊断失败: {e}"
        print(error_msg)
        traceback.print_exc()
        errors.append(error_msg)
        return {
            "attending_report": None,
            "attending_cot": None,
            "errors": errors,
        }

def recurrence_prediction(state: HCCDiagnosisState) -> Dict:

    from agents.recurrence_agent import RecurrenceAgent

    patient_id = state["patient_id"]
    errors = list(state.get("errors", []))

    try:
        print(f"[workflow] 启动复发预测患者 {patient_id}...")
        agent = RecurrenceAgent()

        patient_reports = {
            "imaging_report": state.get("imaging_report"),
            "clinical_report": state.get("clinical_report"),
            "proteomics_report": state.get("proteomics_report"),
            "attending_report": state.get("attending_report"),
            "imaging_report_dict": state.get("imaging_report_dict"),
            "clinical_report_dict": state.get("clinical_report_dict"),
            "proteomics_report_dict": state.get("proteomics_report_dict"),
            "clinical_data": state.get("clinical_data"),
            "patient_id": patient_id,
        }

        result = agent.predict(
            patient_reports,
            patient_id=patient_id,
            use_rag=state.get("use_rag", True),
            similar_cases=state.get("similar_patients", []),
        )
        print(f"[workflow] 复发预测完成: {result.get('recurrence', '未知')}")
        return {
            "recurrence_prediction": result,
            "errors": errors,
        }

    except Exception as e:
        error_msg = f"[recurrence_prediction] 复发预测失败: {e}"
        print(error_msg)
        traceback.print_exc()
        errors.append(error_msg)
        return {
            "recurrence_prediction": None,
            "errors": errors,
        }

def recurrence_time_prediction(state: HCCDiagnosisState) -> Dict:

    from agents.recurrence_time_agent import RecurrenceTimeAgent

    patient_id = state["patient_id"]
    errors = list(state.get("errors", []))

    try:
        print(f"[workflow] 启动复发时间预测患者 {patient_id}...")
        agent = RecurrenceTimeAgent()
        patient_reports = {
            "imaging_report": state.get("imaging_report"),
            "clinical_report": state.get("clinical_report"),
            "proteomics_report": state.get("proteomics_report"),
            "attending_report": state.get("attending_report"),
            "clinical_data": state.get("clinical_data"),
            "clinical_report_dict": state.get("clinical_report_dict"),
            "proteomics_report_dict": state.get("proteomics_report_dict"),
            "imaging_report_dict": state.get("imaging_report_dict"),
            "recurrence_prediction": state.get("recurrence_prediction"),
        }
        result = agent.predict(
            patient_reports, patient_id,
            use_rag=state.get("use_rag", True),
            similar_cases=state.get("similar_patients", []),
        )
        predicted_months = result.get("predicted_months")
        print(f"[workflow] 复发时间预测完成: {predicted_months}月")
        return {
            "recurrence_time_prediction": result,
            "errors": errors,
        }

    except Exception as e:
        error_msg = f"[recurrence_time_prediction] 复发时间预测失败: {e}"
        print(error_msg)
        traceback.print_exc()
        errors.append(error_msg)
        return {
            "recurrence_time_prediction": None,
            "errors": errors,
        }

def _build_proteomics_array_for_patient(patient_id: str):

    import pandas as pd
    from proteomics_finetune.proteomics_data import load_proteomics_raw, preprocess_proteomics

    try:
        X, protein_cols = preprocess_proteomics(use_65_key=False, min_present_rate=0.1)
        X = X.rename(columns={"就诊号": "patient_id"})
        row = X[X["patient_id"] == str(patient_id)]
        if row.empty:
            return None
        vals = row.iloc[0][protein_cols].values.astype(np.float32)
        return vals, protein_cols
    except Exception as e:
        print(f"[build_proteomics_array] 失败: {e}")
        return None

def _workflow_modality_group(state: HCCDiagnosisState) -> str:

    has_mri = bool(state.get("imaging_report") or state.get("has_mri"))
    has_proteomics = bool(state.get("proteomics_report") or state.get("has_proteomics"))
    if has_mri and has_proteomics:
        return "all_three"
    if has_mri:
        return "clinical_mri"
    if has_proteomics:
        return "clinical_proteomics"
    return "clinical_only"

def retrieve_similar_patients(state: HCCDiagnosisState) -> Dict:

    patient_id = state["patient_id"]
    errors = list(state.get("errors", []))

    if not state.get("use_rag", True):
        print(f"[workflow] use_rag=False，跳过患者 {patient_id} 的相似病例检索")
        return {"similar_patients": [], "errors": errors}

    try:
        from knowledge_base.case_retriever import CaseRetriever
        from knowledge_base.kb_enricher import _build_structured_profile

        imaging_dict = state.get("imaging_report_dict") or {}
        comprehensive = imaging_dict.get("comprehensive_analysis") if isinstance(imaging_dict, dict) else {}
        if not isinstance(comprehensive, dict):
            comprehensive = {}

        query_profile = _build_structured_profile({
            "clinical_report_dict": state.get("clinical_report_dict"),
            "proteomics_report_dict": state.get("proteomics_report_dict"),
            "imaging_report_dict": state.get("imaging_report_dict"),
        })
        query_summary = {
            "imaging_summary": comprehensive.get("diagnostic_conclusion") or (state.get("imaging_report") or "")[:1500],
            "clinical_summary": (state.get("clinical_report") or state.get("clinical_summary") or "")[:2500],
            "proteomics_summary": (state.get("proteomics_report") or state.get("proteomics_summary") or "")[:2500],
            "discriminative_features": query_profile,
        }
        modality_group = _workflow_modality_group(state)

        retriever = CaseRetriever()
        retrieved = retriever.retrieve(
            query_summary=query_summary,
            modality_group=modality_group,
            top_k=10,
            dense_top_k=30,
            sparse_top_k=30,
        )
        similar_patients = [
            case for case in retrieved
            if str(getattr(case, "patient_id", "")) != str(patient_id)
        ][:8]
        print(
            f"[workflow] 患者 {patient_id} 相似病例检索完成: "
            f"{len(similar_patients)}例 (模态组={modality_group})"
        )
        return {"similar_patients": similar_patients, "errors": errors}

    except Exception as e:
        error_msg = f"[retrieve_similar_patients] 相似病例检索失败: {e}"
        print(error_msg)
        traceback.print_exc()
        errors.append(error_msg)
        return {"similar_patients": [], "errors": errors}

def evidence_trace(state: HCCDiagnosisState) -> Dict:

    from agents.evidence_agent import EvidenceAgent

    patient_id = state["patient_id"]
    errors = list(state.get("errors", []))

    try:
        diagnosis = {
            "attending_report": state.get("attending_report"),
            "recurrence_prediction": state.get("recurrence_prediction"),
            "recurrence_time_prediction": state.get("recurrence_time_prediction"),
        }

        agent = EvidenceAgent()
        evidence_report = agent.analyze(diagnosis, patient_id)
        evidence_text = agent.generate_report_text(evidence_report)

        print(f"[workflow] 患者 {patient_id} 证据溯源完成，找到 {len(evidence_report.get('articles', []))} 篇文献")

        return {
            "evidence_report": evidence_report,
            "evidence_text": evidence_text,
            "errors": errors,
        }

    except Exception as e:
        error_msg = f"[evidence_trace] 证据溯源失败: {e}"
        print(error_msg)
        errors.append(error_msg)
        return {
            "evidence_report": None,
            "evidence_text": None,
            "errors": errors,
        }

def save_to_knowledge_base(state: HCCDiagnosisState) -> Dict:

    patient_id = state["patient_id"]
    errors = list(state.get("errors", []))

    if state.get("is_test", False):
        print(f"[workflow] 患者 {patient_id} 为测试集患者，跳过知识库存储（防止数据泄露）")
        return {"kb_saved": False, "errors": errors}

    from knowledge_base.kb_builder import KnowledgeBaseBuilder

    attending_report = state.get("attending_report")

    if not attending_report or not isinstance(attending_report, dict):
        print(f"[workflow] 患者 {patient_id} 无主治医生报告，跳过知识库存储")
        return {"kb_saved": False, "errors": errors}

    rev_result = {}
    try:
        print(f"[workflow] 正在将患者 {patient_id} 的完整诊断报告存入知识库...")

        cot_chain = attending_report.get("cot_chain", [])
        final_diagnosis = attending_report.get("final_diagnosis", {})
        raw_report = attending_report.get("raw_report", "")

        has_parse_error = "parse_error" in attending_report and attending_report.get("parse_error")
        if has_parse_error and raw_report:
            print(f"[workflow] AttendingAgent解析失败，使用raw_report全文({len(raw_report)}字符)作为KB内容")
            if not cot_chain:
                cot_chain = [{"step": 0, "title": "原始诊断报告", "reasoning": raw_report[:2000], "evidence": [], "conclusion": ""}]
            if not final_diagnosis:
                final_diagnosis = {"key_factors": [], "data_completeness": "raw_text"}

            attending_report["cot_chain"] = cot_chain
            attending_report["final_diagnosis"] = final_diagnosis
            attending_report["parse_fallback"] = True

        kb_data = {
            "cot_chain": cot_chain,
            "final_diagnosis": final_diagnosis,
            "raw_report": raw_report,
        }

        from knowledge_base.kb_enricher import enrich_report_for_kb
        enriched = enrich_report_for_kb({
            "patient_id": patient_id,
            "clinical_report_dict": state.get("clinical_report_dict"),
            "proteomics_report_dict": state.get("proteomics_report_dict"),
            "imaging_report_dict": state.get("imaging_report_dict"),
            "attending_report": state.get("attending_report"),
            "reverse_cot": rev_result.get("reverse_cot") if 'rev_result' in dir() else None,
        })
        kb_meta = enriched.get("_kb_meta", {})
        embed_text = kb_meta.get("embed_text", "")

        if not embed_text and raw_report:
            embed_text = raw_report

        builder = KnowledgeBaseBuilder()

        success = builder.add_case(patient_id, kb_data, embed_text=embed_text)

        if success:

            t_GT = None
            try:
                from data_loader import load_clinical_data
                clinical_df = load_clinical_data()
                row = clinical_df[clinical_df['就诊号'] == patient_id]
                if not row.empty:
                    row = row.iloc[0]
                    t_GT = {
                        "recurrence_months": round(float(row.get('recurrence_time', 0)) / 30.44, 1),
                        "recurrence_event": int(row.get('recurrence_event', 0)),
                    }
                    extra_meta = build_outcome_metadata(
                        row,
                        has_mri=state.get('has_mri', False),
                        has_proteomics=state.get('has_proteomics', False),
                    )
                    extra_meta["clinical_report_dict"] = json.dumps(
                        state.get("clinical_report_dict") or {}, ensure_ascii=False,
                    )
                    extra_meta["proteomics_report_dict"] = json.dumps(
                        state.get("proteomics_report_dict") or {}, ensure_ascii=False,
                    )
                    extra_meta["imaging_report_dict"] = json.dumps(
                        state.get("imaging_report_dict") or {}, ensure_ascii=False,
                    )
                    imaging_report_dict = state.get("imaging_report_dict", {})
                    disc_features = imaging_report_dict.get("discriminative_features", {})
                    imaging_meta = build_imaging_feature_metadata(disc_features)
                    extra_meta.update(imaging_meta)
                    extra_meta["structured_profile"] = json.dumps(kb_meta.get("structured_profile", {}), ensure_ascii=False)
                    extra_meta["retrieval_context"] = kb_meta.get("retrieval_context", "")
                    extra_meta["full_cot_text"] = state.get("attending_report_text", "") or json.dumps(
                        state.get("attending_report", {}).get("cot_chain", []), ensure_ascii=False
                    )
                    extra_meta["bclc_stage"] = kb_meta.get("bclc_stage", "")
                    builder.update_patient_metadata(patient_id, extra_meta)
                    print(f"[workflow] 患者 {patient_id} ChromaDB层+metadata已存入")
                else:
                    print(f"[workflow] 患者 {patient_id} 不在临床数据中，跳过outcome metadata")
            except Exception as meta_err:
                print(f"[workflow] 补充真实结局metadata失败: {meta_err}")
                traceback.print_exc()

            try:
                from knowledge_base.case_bank import save_case_document, save_profile_index, build_case_document
                rev_cot = rev_result.get("reverse_cot") or {}
                case_doc = build_case_document(state, reverse_cot=rev_cot, kb_meta=kb_meta)
                if t_GT:
                    case_doc["t_GT"] = t_GT

                backward_cots = {}
                if t_GT and (t_GT.get("recurrence_months", 0) > 0 or t_GT.get("recurrence_event", 0) == 1):
                    print(f"[workflow] 生成反向归因COT (t_GT={t_GT})...")
                    if state.get("imaging_report_dict"):
                        try:
                            from agents.imaging_agent import ImagingAgent
                            img_agent = ImagingAgent()
                            img_backward = img_agent.generate_backward_cot(
                                state["imaging_report_dict"], t_GT, patient_id
                            )
                            backward_cots["imaging_backward_cot"] = img_backward.get("backward_cot")
                            img_agent.release_model()
                        except Exception as be:
                            print(f"[workflow] 影像反向COT失败: {be}")
                    if state.get("clinical_report_dict"):
                        try:
                            from agents.clinical_agent import ClinicalAgent
                            clin_agent = ClinicalAgent()
                            clin_backward = clin_agent.generate_backward_cot(
                                state["clinical_report_dict"], t_GT, patient_id
                            )
                            backward_cots["clinical_backward_cot"] = clin_backward.get("backward_cot")
                        except Exception as be:
                            print(f"[workflow] 临床反向COT失败: {be}")
                    if state.get("proteomics_report_dict"):
                        try:
                            from agents.proteomics_agent import ProteomicsAgent
                            prot_agent = ProteomicsAgent()
                            prot_backward = prot_agent.generate_backward_cot(
                                state["proteomics_report_dict"], t_GT, patient_id
                            )
                            backward_cots["proteomics_backward_cot"] = prot_backward.get("backward_cot")
                        except Exception as be:
                            print(f"[workflow] 蛋白反向COT失败: {be}")

                try:
                    from agents.self_critique import SelfCritique
                    reports_for_validation = {
                        "imaging_report": state.get("imaging_report") or "",
                        "clinical_report": state.get("clinical_report") or "",
                        "proteomics_report": state.get("proteomics_report") or "",
                    }
                    critique = SelfCritique()
                    revised_cot, validation_result = critique.validate_and_fix(
                        case_doc["CoT_refined"], reports_for_validation
                    )
                    case_doc["CoT_refined"] = revised_cot
                    case_doc["validation_result"] = validation_result
                    print(f"[workflow] SelfCritique 完成: quality_level={validation_result.get('quality_level', '?')}")
                except Exception as sc_err:
                    print(f"[workflow] SelfCritique 失败: {sc_err}")

                save_case_document(patient_id, case_doc)
                save_profile_index(patient_id, {
                    "clinical": kb_meta.get("structured_profile", {}).get("clinical", {}),
                    "proteomic": kb_meta.get("structured_profile", {}).get("proteomic", {}),
                    "imaging": kb_meta.get("structured_profile", {}).get("imaging", {}),
                })
                print(f"[workflow] Case Bank文档+画像已存储")
            except Exception as cb_err:
                print(f"[workflow] Case Bank存储失败: {cb_err}")
                traceback.print_exc()

            if t_GT:
                try:
                    from knowledge_base.reverse_cot import ReverseCoTGenerator
                    rev_gen = ReverseCoTGenerator()
                    rec_days = float(row.get('recurrence_time', 0))
                    rec_months = round(rec_days / 30.44, 1)
                    rec_event = int(row.get('recurrence_event', 0))
                    rev_cot = rev_gen.generate(
                        mri_report=state.get("imaging_report") or "",
                        clinical_report=state.get("clinical_report") or "",
                        proteomics_report=state.get("proteomics_report") or "",
                        recurrence_months=rec_months,
                        recurrence_event=rec_event,
                        patient_id=patient_id,
                    )
                    if rev_cot and "error" not in rev_cot:
                        rev_summary = rev_cot.get("integrated_cot", "")[:500]
                        builder.update_patient_metadata(patient_id, {
                            "reverse_cot": json.dumps(rev_cot, ensure_ascii=False),
                            "reverse_cot_summary": rev_summary,
                        })
                        rev_result["reverse_cot"] = rev_cot
                        rev_result["reverse_cot_summary"] = rev_summary
                        print(f"[workflow] Reverse CoT 已存储")
                except Exception as rev_err:
                    print(f"[workflow] Reverse CoT生成失败: {rev_err}")
                    traceback.print_exc()

            print(f"[workflow] 患者 {patient_id} 已成功存入知识库")
        else:
            errors.append("[save_to_kb] 知识库存储返回失败")

        return {"kb_saved": success, "errors": errors, **rev_result}

    except Exception as e:
        error_msg = f"[save_to_kb] 知识库存储异常: {e}"
        print(error_msg)
        traceback.print_exc()
        errors.append(error_msg)
        return {"kb_saved": False, "errors": errors}

def route_after_load(state: HCCDiagnosisState) -> List[str]:

    routes = []
    if state.get("has_mri", False):
        routes.append("imaging_analysis")
    if state.get("has_clinical", False):
        routes.append("clinical_analysis")
    if state.get("has_proteomics", False):
        routes.append("proteomics_analysis")
    if not routes:
        routes.append("attending_synthesis")
    return routes

def build_workflow():

    workflow = StateGraph(HCCDiagnosisState)

    workflow.add_node("load_patient_data", load_patient_data)
    workflow.add_node("imaging_analysis", imaging_analysis)
    workflow.add_node("clinical_analysis", clinical_analysis)
    workflow.add_node("proteomics_analysis", proteomics_analysis)
    workflow.add_node("attending_synthesis", attending_synthesis)
    workflow.add_node("retrieve_similar_patients", retrieve_similar_patients)
    workflow.add_node("recurrence_prediction", recurrence_prediction)
    workflow.add_node("recurrence_time_prediction", recurrence_time_prediction)
    workflow.add_node("evidence_trace", evidence_trace)
    workflow.add_node("save_to_kb", save_to_knowledge_base)

    workflow.add_edge(START, "load_patient_data")

    workflow.add_conditional_edges(
        "load_patient_data",
        route_after_load,
        ["imaging_analysis", "clinical_analysis", "proteomics_analysis", "attending_synthesis"],
    )

    workflow.add_edge("imaging_analysis", "attending_synthesis")
    workflow.add_edge("clinical_analysis", "attending_synthesis")
    workflow.add_edge("proteomics_analysis", "attending_synthesis")

    workflow.add_conditional_edges(
        "attending_synthesis",
        route_after_attending,
        ["save_to_kb", "retrieve_similar_patients"],
    )

    workflow.add_edge("retrieve_similar_patients", "recurrence_prediction")
    workflow.add_edge("recurrence_prediction", "recurrence_time_prediction")
    workflow.add_edge("recurrence_time_prediction", "evidence_trace")
    workflow.add_edge("evidence_trace", "save_to_kb")
    workflow.add_edge("save_to_kb", END)

    return workflow.compile()

def _route_after_attending(state: HCCDiagnosisState) -> List[str]:

    is_test = state.get("is_test", False)
    if not is_test:
        print(f"[workflow] 路由: attending → save_to_kb (训练集模式，不调用复发/时间/证据Agent)")
        return ["save_to_kb"]
    print(f"[workflow] 路由: attending → retrieve_similar_patients (测试集模式)")
    return ["retrieve_similar_patients"]

def route_after_attending(state: HCCDiagnosisState) -> List[str]:

    return _route_after_attending(state)

def run_diagnosis(patient_id: str, skip_imaging: bool = False, skip_proteomics: bool = False, is_test: bool = False, use_rag: bool = True) -> Dict:

    app = build_workflow()
    initial_state = {
        "patient_id": patient_id,
        "skip_imaging": skip_imaging,
        "skip_proteomics": skip_proteomics,
        "is_test": is_test,
        "use_rag": use_rag,
        "mri_paths": None,
        "clinical_data": None,
        "proteomics_data": None,
        "clinical_summary": None,
        "proteomics_summary": None,
        "has_mri": False,
        "has_clinical": False,
        "has_proteomics": False,
        "imaging_report": None,
        "imaging_report_dict": None,
        "imaging_cot": None,
        "imaging_status": "missing_data",
        "clinical_report": None,
        "clinical_report_dict": None,
        "clinical_cot": None,
        "clinical_status": "missing_data",
        "proteomics_report": None,
        "proteomics_report_dict": None,
        "proteomics_cot": None,
        "proteomics_status": "missing_data",
        "debate_result": None,
        "supervisor_result": None,
        "reverse_cot": None,
        "reverse_cot_summary": None,
        "attending_report": None,
        "attending_report_text": None,
        "cot_review_result": None,
        "retrieved_patients": None,
        "similar_patients": [],
        "recurrence_prediction": None,
        "recurrence_time_prediction": None,
        "evidence_report": None,
        "kb_saved": False,
        "errors": [],
    }
    result = app.invoke(initial_state)
    return result

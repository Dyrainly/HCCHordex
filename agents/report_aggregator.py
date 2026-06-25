from typing import Dict, List, Optional

SUMMARIZED_REPORT_TEMPLATE = """【影像学摘要】
{imaging_summary}

【临床病理摘要】
{clinical_summary}

【蛋白组学摘要】
{proteomics_summary}

【风险指示器汇总】
高危因素: {risk_indicators}
保护因素: {protective_indicators}"""

def extract_imaging_summary(imaging_report: Dict) -> str:

    summary_parts = []

    disc_features = imaging_report.get("discriminative_features", {})
    lirads = disc_features.get("lirads_assessment", {})
    if lirads.get("category"):
        summary_parts.append(f"LI-RADS {lirads['category']}级")
        if lirads.get("rationale"):
            summary_parts.append(f"依据:{lirads['rationale']}")

    core_features = disc_features.get("core_features", {})

    aphe = core_features.get("aphe", {})
    if aphe.get("present"):
        pattern = aphe.get("pattern", "")
        summary_parts.append(f"动脉期高强化(APHE):{pattern if pattern else '存在'}")

    washout = core_features.get("washout", {})
    if washout.get("present"):
        timing = washout.get("timing", "")
        summary_parts.append(f"廓清(Washout):{timing if timing else '存在'}")

    capsule = core_features.get("capsule", {})
    if capsule.get("present"):
        cap_type = capsule.get("type", "")
        summary_parts.append(f"包膜:{cap_type if cap_type else '存在'}")

    t2_signal = core_features.get("t2_signal", {})
    if t2_signal.get("intensity") and t2_signal["intensity"] != "unclear":
        summary_parts.append(f"T2信号:{t2_signal['intensity']}")

    margin = core_features.get("margin", {})
    if margin.get("type") and margin["type"] != "unclear":
        summary_parts.append(f"边缘:{margin['type']}")

    risk_flags = disc_features.get("derived_risk_flags", {})

    mvi_risk = risk_flags.get("mvi_high_risk", {})
    if mvi_risk.get("flag"):
        evidence = mvi_risk.get("evidence", "")
        summary_parts.append(f"MVI高风险:{evidence if evidence else '是'}")

    vascular = risk_flags.get("vascular_invasion", {})
    if vascular.get("flag"):
        location = vascular.get("location", "")
        summary_parts.append(f"血管侵犯:{location if location else '存在'}")

    multifocal = risk_flags.get("multifocal", {})
    if multifocal.get("flag"):
        count = multifocal.get("count", 0)
        summary_parts.append(f"多发病灶:{count}个")

    comp_analysis = imaging_report.get("comprehensive_analysis", {})
    if comp_analysis.get("diagnostic_conclusion"):

        conclusion = comp_analysis["diagnostic_conclusion"]
        if len(conclusion) < 100:
            summary_parts.append(f"结论:{conclusion}")

    size = core_features.get("size_mm", {})
    if size.get("value"):
        summary_parts.append(f"大小:{size['value']}mm")

    if not summary_parts:
        return "影像学特征提取失败"

    return "; ".join(summary_parts)

def extract_clinical_summary(clinical_report: Dict) -> str:

    summary_parts = []

    fe = clinical_report.get("field_effect", {})
    staging = clinical_report.get("staging", {})
    tnm_v = fe.get("tnm_stage")
    tnm = tnm_v.get("value") if isinstance(tnm_v, dict) else tnm_v or staging.get("tnm_stage")
    bclc_v = fe.get("bclc_stage")
    bclc = bclc_v.get("value") if isinstance(bclc_v, dict) else bclc_v or staging.get("bclc_stage")
    if tnm:
        summary_parts.append(f"TNM:{tnm}")
    if bclc:
        summary_parts.append(f"BCLC:{bclc}")

    tumor_bio = clinical_report.get("tumor_biology", {})

    mvi = tumor_bio.get("mvi_grade", {})
    if mvi.get("value"):
        risk = mvi.get("risk_contribution", "")
        summary_parts.append(f"MVI分级:{mvi['value']}({risk})" if risk else f"MVI分级:{mvi['value']}")

    diff = tumor_bio.get("differentiation", {})
    if diff.get("value"):
        summary_parts.append(f"分化:{diff['value']}")

    edmondson = tumor_bio.get("edmondson_grade", {})
    if edmondson.get("value"):
        summary_parts.append(f"Edmondson:{edmondson['value']}")

    vasc_inv = tumor_bio.get("vascular_invasion", {})
    if vasc_inv.get("value") and vasc_inv["value"] not in ["无", "null", None]:
        summary_parts.append(f"血管侵犯:{vasc_inv['value']}")

    micro_thrombus = tumor_bio.get("microvascular_thrombus", {})
    if micro_thrombus.get("value") == "有":
        summary_parts.append("镜下脉管癌栓")

    satellite = tumor_bio.get("satellite_nodules", {})
    if satellite.get("value") == "有":
        summary_parts.append("卫星灶")

    tumor_size = tumor_bio.get("tumor_size_cm", {})
    if tumor_size.get("value"):
        summary_parts.append(f"肿瘤大小:{tumor_size['value']}cm")

    tumor_count = tumor_bio.get("tumor_count", {})
    if tumor_count.get("value"):
        summary_parts.append(f"肿瘤数量:{tumor_count['value']}")

    surgical = clinical_report.get("surgical_curability", {})

    margin = surgical.get("margin_status", {})
    if margin.get("value"):
        risk = margin.get("risk_contribution", "")
        summary_parts.append(f"切缘:{margin['value']}({risk})" if risk else f"切缘:{margin['value']}")

    margin_dist = surgical.get("margin_distance", {})
    if margin_dist.get("value"):
        summary_parts.append(f"切缘距离:{margin_dist['value']}mm")

    resection_method = surgical.get("resection_method", {})
    if resection_method.get("value"):
        summary_parts.append(f"手术方式:{resection_method['value']}")

    field_effect = clinical_report.get("field_effect", {})

    cirrhosis = field_effect.get("cirrhosis", {})
    if cirrhosis.get("value") == "有":
        summary_parts.append("肝硬化")

    fibrosis = field_effect.get("fibrosis_stage", {})
    if fibrosis.get("value"):
        summary_parts.append(f"纤维化:{fibrosis['value']}")

    child_pugh = field_effect.get("child_pugh", {})
    if child_pugh.get("value"):
        summary_parts.append(f"Child-Pugh:{child_pugh['value']}")

    afp = field_effect.get("afp_ng_ml", {})
    if afp.get("value"):
        afp_val = afp["value"]
        if isinstance(afp_val, (int, float)):
            if afp_val > 400:
                summary_parts.append(f"AFP:{afp_val}ng/ml(高危)")
            else:
                summary_parts.append(f"AFP:{afp_val}ng/ml")
        else:
            summary_parts.append(f"AFP:{afp_val}")

    hbv_dna = field_effect.get("hbv_dna", {})
    if hbv_dna.get("value"):
        summary_parts.append(f"HBV DNA:{hbv_dna['value']}")

    if not summary_parts:
        return "临床病理信息提取失败"

    return "; ".join(summary_parts)

def extract_proteomics_summary(proteomics_report: Dict) -> str:

    summary_parts = []

    encoding_result = proteomics_report.get("encoding_result", {})

    abnormal_proteins = encoding_result.get("abnormal_proteins", [])

    if abnormal_proteins:

        top_proteins = abnormal_proteins[:5]
        shap_list = []
        for p in top_proteins:
            gene = p.get("gene_symbol", p.get("protein", "未知"))
            shap = p.get("shap_contribution", 0)
            direction = "+" if shap > 0 else ""
            shap_list.append(f"{gene}({direction}{shap:.2f})")

        summary_parts.append(f"Top异常蛋白(SHAP): {', '.join(shap_list)}")

    core_drivers = encoding_result.get("core_risk_drivers", [])
    if core_drivers:
        driver_list = []
        for d in core_drivers[:3]:
            gene = d.get("gene_symbol", d.get("protein", "未知"))
            shap = d.get("shap_contribution", 0)
            direction = "+" if shap > 0 else ""
            driver_list.append(f"{gene}({direction}{shap:.2f})")

        summary_parts.append(f"核心风险驱动: {', '.join(driver_list)}")

    proteomic_sig = encoding_result.get("proteomic_signature", [])
    if proteomic_sig and isinstance(proteomic_sig, list):

        summary_parts.append(f"蛋白签名: {', '.join(proteomic_sig[:5])}")

    shap_analysis = encoding_result.get("shap_analysis", {})
    if shap_analysis:
        individual_contribs = shap_analysis.get("individual_contributions", {})
        if individual_contribs:

            positive = sum(1 for v in individual_contribs.values() if v > 0)
            negative = sum(1 for v in individual_contribs.values() if v < 0)
            summary_parts.append(f"SHAP分布: {positive}正向贡献, {negative}负向贡献")

    if not summary_parts:
        return "蛋白组学特征提取失败"

    return "; ".join(summary_parts)

def extract_risk_indicators(
    imaging_report: Dict, 
    clinical_report: Dict, 
    proteomics_report: Dict
) -> List[str]:

    risk_indicators = []

    disc_features = imaging_report.get("discriminative_features", {})
    risk_flags = disc_features.get("derived_risk_flags", {})

    if risk_flags.get("mvi_high_risk", {}).get("flag"):
        risk_indicators.append("影像学MVI高风险征象")

    if risk_flags.get("vascular_invasion", {}).get("flag"):
        risk_indicators.append("血管侵犯")

    if risk_flags.get("multifocal", {}).get("flag"):
        risk_indicators.append("多发病灶")

    if risk_flags.get("aggressive_pattern", {}).get("flag"):
        risk_indicators.append("侵袭性影像模式")

    lirads = disc_features.get("lirads_assessment", {})
    if lirads.get("category") in ["5", "5类", "LR-5"]:
        risk_indicators.append("LI-RADS 5类(确定HCC)")

    tumor_bio = clinical_report.get("tumor_biology", {})

    mvi = tumor_bio.get("mvi_grade", {})
    if mvi.get("value") in ["M1", "M2"]:
        risk_indicators.append(f"MVI分级{mvi['value']}")

    diff = tumor_bio.get("differentiation", {})
    if diff.get("value") == "低分化":
        risk_indicators.append("低分化")

    if tumor_bio.get("microvascular_thrombus", {}).get("value") == "有":
        risk_indicators.append("镜下脉管癌栓")

    if tumor_bio.get("satellite_nodules", {}).get("value") == "有":
        risk_indicators.append("卫星灶")

    surgical = clinical_report.get("surgical_curability", {})
    if surgical.get("margin_status", {}).get("value") in ["R1", "R2"]:
        risk_indicators.append("切缘阳性")

    field_effect = clinical_report.get("field_effect", {})
    afp = field_effect.get("afp_ng_ml", {})
    if afp.get("value"):
        try:
            afp_val = float(afp["value"])
            if afp_val > 400:
                risk_indicators.append(f"AFP高危(>{afp_val:.0f}ng/ml)")
        except (ValueError, TypeError):
            pass

    encoding_result = proteomics_report.get("encoding_result", {})

    core_drivers = encoding_result.get("core_risk_drivers", [])
    for driver in core_drivers[:2]:
        if driver.get("direction") == "positive":
            gene = driver.get("gene_symbol", driver.get("protein", "未知"))
            risk_indicators.append(f"异常蛋白{gene}(促复发)")

    return risk_indicators[:5]

def extract_protective_indicators(
    imaging_report: Dict, 
    clinical_report: Dict, 
    proteomics_report: Dict
) -> List[str]:

    protective_indicators = []

    tumor_bio = clinical_report.get("tumor_biology", {})
    surgical = clinical_report.get("surgical_curability", {})
    field_effect = clinical_report.get("field_effect", {})

    mvi = tumor_bio.get("mvi_grade", {})
    if mvi.get("value") == "M0":
        protective_indicators.append("MVI阴性(M0)")

    diff = tumor_bio.get("differentiation", {})
    if diff.get("value") == "高分化":
        protective_indicators.append("高分化")

    if surgical.get("margin_status", {}).get("value") == "R0":
        protective_indicators.append("R0根治性切除")

    if tumor_bio.get("tumor_count", {}).get("value") == "单发":
        protective_indicators.append("单发肿瘤")

    if field_effect.get("cirrhosis", {}).get("value") == "无":
        protective_indicators.append("无肝硬化背景")

    if field_effect.get("child_pugh", {}).get("value") == "A":
        protective_indicators.append("Child-Pugh A级")

    encoding_result = proteomics_report.get("encoding_result", {})

    abnormal_proteins = encoding_result.get("abnormal_proteins", [])
    for protein in abnormal_proteins:
        if protein.get("direction") == "negative" and protein.get("risk_effect") == "protects_against_recurrence":
            gene = protein.get("gene_symbol", protein.get("protein", "未知"))
            protective_indicators.append(f"保护性蛋白{gene}")
            break

    return protective_indicators[:3]

def generate_summarized_report(
    imaging_report: Dict, 
    clinical_report: Dict, 
    proteomics_report: Dict
) -> Dict:

    imaging_summary = extract_imaging_summary(imaging_report)
    clinical_summary = extract_clinical_summary(clinical_report)
    proteomics_summary = extract_proteomics_summary(proteomics_report)

    risk_indicators = extract_risk_indicators(
        imaging_report, clinical_report, proteomics_report
    )
    protective_indicators = extract_protective_indicators(
        imaging_report, clinical_report, proteomics_report
    )

    full_text = SUMMARIZED_REPORT_TEMPLATE.format(
        imaging_summary=imaging_summary,
        clinical_summary=clinical_summary,
        proteomics_summary=proteomics_summary,
        risk_indicators="、".join(risk_indicators) if risk_indicators else "无",
        protective_indicators="、".join(protective_indicators) if protective_indicators else "无"
    )

    return {
        "imaging_summary": imaging_summary,
        "clinical_summary": clinical_summary,
        "proteomics_summary": proteomics_summary,
        "risk_indicators": risk_indicators,
        "protective_indicators": protective_indicators,
        "full_text": full_text
    }

if __name__ == "__main__":

    test_imaging = {
        'discriminative_features': {
            'lirads_assessment': {'category': '4', 'rationale': 'APHE+washout'},
            'core_features': {
                'aphe': {'present': True, 'pattern': '均匀强化'},
                'washout': {'present': True, 'timing': '门脉期'},
                'capsule': {'present': True, 'type': '完整'},
                't2_signal': {'intensity': '轻度高信号'},
                'margin': {'type': '光滑'},
                'size_mm': {'value': 45}
            },
            'derived_risk_flags': {
                'mvi_high_risk': {'flag': True, 'evidence': '边缘不光滑'},
                'aggressive_pattern': {'flag': False},
                'multifocal': {'flag': False, 'count': 1},
                'vascular_invasion': {'flag': False}
            }
        },
        'comprehensive_analysis': {
            'diagnostic_conclusion': '肝右叶HCC，大小4.5cm'
        }
    }

    test_clinical = {
        'staging': {'tnm_stage': 'II', 'bclc_stage': 'A'},
        'tumor_biology': {
            'mvi_grade': {'value': 'M1', 'risk_contribution': '促进复发'},
            'differentiation': {'value': '中分化'},
            'microvascular_thrombus': {'value': '无'},
            'satellite_nodules': {'value': '无'},
            'tumor_size_cm': {'value': 4.5},
            'tumor_count': {'value': '单发'}
        },
        'surgical_curability': {
            'margin_status': {'value': 'R0'},
            'margin_distance': {'value': 5},
            'resection_method': {'value': '解剖性切除'}
        },
        'field_effect': {
            'cirrhosis': {'value': '有'},
            'child_pugh': {'value': 'A'},
            'afp_ng_ml': {'value': 89}
        }
    }

    test_proteomics = {
        'encoding_result': {
            'abnormal_proteins': [
                {'protein': 'ALB', 'gene_symbol': 'ALB', 'shap_contribution': 0.15, 'direction': 'positive'},
                {'protein': 'AFP', 'gene_symbol': 'AFP', 'shap_contribution': 0.12, 'direction': 'positive'},
                {'protein': 'GGT1', 'gene_symbol': 'GGT1', 'shap_contribution': -0.08, 'direction': 'negative', 'risk_effect': 'protects_against_recurrence'}
            ],
            'core_risk_drivers': [
                {'gene_symbol': 'ALB', 'shap_contribution': 0.15, 'direction': 'positive'},
                {'gene_symbol': 'AFP', 'shap_contribution': 0.12, 'direction': 'positive'}
            ],
            'proteomic_signature': {
                'functional_themes': ['代谢重编程', '氧化应激'],
                'pathway_enrichment': ['药物代谢', '补体通路']
            },
            'shap_analysis': {
                'individual_contributions': {
                    'ALB': 0.15,
                    'AFP': 0.12,
                    'GGT1': -0.08
                }
            }
        }
    }

    result = generate_summarized_report(test_imaging, test_clinical, test_proteomics)

    print("=" * 60)
    print("【Report Aggregator 测试结果】")
    print("=" * 60)
    print(f"\n影像摘要: {result['imaging_summary']}")
    print(f"\n临床摘要: {result['clinical_summary']}")
    print(f"\n蛋白摘要: {result['proteomics_summary']}")
    print(f"\n高危因素: {result['risk_indicators']}")
    print(f"\n保护因素: {result['protective_indicators']}")
    print(f"\n全文:\n{result['full_text']}")

    print("\n" + "=" * 60)
    print("【约束验证】")
    print("=" * 60)

    has_risk_prob = any(keyword in result['proteomics_summary'] for keyword in ['风险概率', 'risk_probability', 'prediction', '复发概率'])
    print(f"⚠️ 蛋白摘要不含风险概率: {'✓ 通过' if not has_risk_prob else '✗ 失败'}")

    has_shap = 'SHAP' in result['proteomics_summary']
    print(f"✓ 蛋白摘要包含SHAP信息: {'通过' if has_shap else '失败'}")

    print("\n测试完成!")

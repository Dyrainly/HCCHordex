import os
import sys
import json
import traceback
import numpy as np
from typing import Dict, Optional, List

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import HULUMED_MODEL_PATH, IMAGING_GPU

NII_NUM_SLICES = 72
NII_AXIS = 2

SEQUENCE_PROMPTS = {
    "T1_pre": (
        "这是一例肝细胞癌(HCC)患者的3D MRI T1平扫期影像。请用中文详细描述以下内容：\n"
        "1. 肝脏整体形态（大小、轮廓、有无肝硬化表现）；\n"
        "2. 病灶的位置、大小（估计最大径）、数量（单发/多发）；\n"
        "3. 病灶T1信号特点（低信号/等信号/高信号/混杂信号）；\n"
        "4. 病灶边界是否清晰、形态是否规则；\n"
        "5. 有无卫星灶或子灶；\n"
        "6. 肝内胆管及门静脉主干有无异常。\n"
        "请直接输出影像描述："
    ),
    "T1_arterial": (
        "这是一例肝细胞癌(HCC)患者的3D MRI T1动脉期影像。请用中文详细描述以下内容：\n"
        "1. 病灶在动脉期的强化模式（均匀强化/不均匀强化/环状强化/结节状强化/无明显强化）；\n"
        "2. 是否存在非边缘动脉期高强化（APHE），这是HCC的关键特征；\n"
        "3. 强化程度与周围肝实质的对比；\n"
        "4. 病灶内有无坏死、囊变等无强化区域；\n"
        "5. 肿瘤供血动脉有无增粗；\n"
        "6. 有无其他异常强化灶。\n"
        "请直接输出影像描述："
    ),
    "T1_portal": (
        "这是一例肝细胞癌(HCC)患者的3D MRI T1门脉期影像。请用中文详细描述以下内容：\n"
        "1. 病灶在门脉期的信号变化——是否出现廓清（washout），即信号低于周围肝实质；\n"
        "2. 是否存在\"快进快出\"的典型HCC强化模式；\n"
        "3. 有无延迟期包膜强化（enhancing capsule）；\n"
        "4. 包膜是否完整，边缘是否光滑（评估微血管侵犯MVI的关键指标）；\n"
        "5. 门静脉及其分支有无癌栓（充盈缺损或异常强化）；\n"
        "6. 肝静脉、下腔静脉有无受侵。\n"
        "请直接输出影像描述："
    ),
    "T2": (
        "这是一例肝细胞癌(HCC)患者的3D MRI T2加权像。请用中文详细描述以下内容：\n"
        "1. 病灶T2信号特点（低信号/等信号/轻度高信号/明显高信号/混杂信号）；\n"
        "2. 轻度T2高信号为HCC典型表现，明显高信号需考虑血管瘤等鉴别；\n"
        "3. 病灶内有无液-液平面或出血信号；\n"
        "4. 肝脏背景信号——有无弥漫性肝硬化表现（结节状改变、脾大等）；\n"
        "5. 腹腔有无积液；\n"
        "6. 有无肿大淋巴结。\n"
        "请直接输出影像描述："
    ),
}

COMPREHENSIVE_PROMPT_TEMPLATE = """你是一位资深肝细胞癌影像专家。以下是同一患者的MRI四序列影像分析结果。请综合所有序列信息，按LI-RADS标准进行评估，生成结构化诊断报告。

【各序列影像描述】

{sequence_descriptions}

【报告要求】
请严格按以下JSON格式输出综合分析结果：
{{
    "tumor_location": "肿瘤位置（肝段）",
    "tumor_size": "肿瘤最大径估计",
    "tumor_morphology": "肿瘤形态描述",
    "enhancement_pattern": "强化模式（如快进快出、动脉期高强化伴门脉期廓清等）",
    "t2_signal": "T2信号特征",
    "liver_background": "肝脏背景（肝硬化程度等）",
    "vascular_involvement": "血管侵犯情况（门静脉/肝静脉癌栓等）",
    "lirads_grade": "LI-RADS分级及依据",
    "mvi_risk": "微血管侵犯风险评估（高/中/低）及影像依据",
    "satellite_nodules": "卫星灶情况",
    "diagnostic_conclusion": "影像诊断结论（定性诊断+大小+关键危险因素）"
}}

请直接输出JSON结果："""

DISCRIMINATIVE_FEATURE_PROMPT = """你是一位肝细胞癌影像专家。请分析MRI影像并提取**结构化判别特征**。

【核心原则】
1. 只输出影像中**清晰可见**的特征，禁止编造
2. 每个特征必须附带置信度（0.0-1.0）
3. 无法判断的特征标记为 "unclear"
4. 优先输出差异性强的特征（如 APHE、Washout、MVI风险）

【各序列影像描述】
{sequence_descriptions}

【输出格式】严格按以下JSON结构输出：
{{
  "core_features": {{
    "size_mm": {{"value": 数值, "confidence": 0.0-1.0, "source": "T1_pre/T1_arterial"}},
    "aphe": {{
      "present": true/false,
      "pattern": "non-rim" | "rim" | "none",
      "confidence": 0.0-1.0
    }},
    "washout": {{
      "present": true/false,
      "timing": "portal" | "delayed" | "none",
      "confidence": 0.0-1.0
    }},
    "capsule": {{
      "present": true/false,
      "type": "smooth" | "irregular" | "none",
      "confidence": 0.0-1.0
    }},
    "t2_signal": {{
      "intensity": "mildly_high" | "markedly_high" | "iso" | "low" | "mixed",
      "confidence": 0.0-1.0
    }},
    "margin": {{
      "type": "smooth" | "irregular" | "lobulated",
      "confidence": 0.0-1.0
    }}
  }},
  "derived_risk_flags": {{
    "mvi_high_risk": {{
      "flag": true/false,
      "evidence": "irregular_margin" | "corona_enhancement" | "peritumoral_hypointensity" | "none",
      "confidence": 0.0-1.0
    }},
    "aggressive_pattern": {{
      "flag": true/false,
      "evidence": "APHE+washout+capsule" | "infiltrative" | "none",
      "confidence": 0.0-1.0
    }},
    "multifocal": {{
      "flag": true/false,
      "count": 数值,
      "confidence": 0.0-1.0
    }},
    "vascular_invasion": {{
      "flag": true/false,
      "location": "portal_vein" | "hepatic_vein" | "none",
      "confidence": 0.0-1.0
    }}
  }},
  "lirads_assessment": {{
    "category": "LR-1" | "LR-2" | "LR-3" | "LR-4" | "LR-5" | "LR-M" | "LR-NC",
    "major_features": ["APHE", "Washout", "Capsule"],
    "size_category": "<10mm" | "10-19mm" | "≥20mm",
    "confidence": 0.0-1.0
  }},
  "discriminative_signature": [
    "特征1（如 APHE_non-rim）",
    "特征2（如 Washout_portal）",
    "特征3（如 LIRADS5）",
    "..."
  ],
  "uncertainty": {{
    "missing_sequences": ["序列名"],
    "low_confidence_features": ["特征名"],
    "ambiguous_findings": ["描述"]
  }},
  "narrative_summary": "一句话总结：肿瘤大小+位置+关键特征+LI-RADS分级"
}}

【判别签名生成规则】
discriminative_signature 必须包含：
- APHE 状态（如 "APHE_non-rim" 或 "APHE_absent"）
- Washout 状态（如 "Washout_portal" 或 "Washout_none"）
- LI-RADS 分级（如 "LIRADS5"）
- MVI 风险（如 "MVI_high" 或 "MVI_low"）
- 其他高置信度特征

【置信度校准指南】
- 0.9-1.0: 特征非常清晰，无争议
- 0.7-0.9: 特征可见但需仔细观察
- 0.5-0.7: 特征可能存在，但有不确定性
- <0.5: 标记为 unclear，不纳入判别签名

【防幻觉约束】
- 不允许编造未观察到的特征
- 不允许使用模糊描述（如"可能"、"大概"）
- 如果无法判断 → 明确写 "unclear"
- 响应长度控制在500字以内

请直接输出JSON结果："""

class ImagingAgent:

    def __init__(self, device: str = None):

        self.model = None
        self.processor = None
        self.device = device or f"cuda:{IMAGING_GPU}"
        self._load_model()

    def _load_model(self):

        import torch
        import time

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                from transformers import AutoModelForCausalLM, AutoProcessor
                break
            except ImportError as e:
                print(f"[ImagingAgent] transformers导入失败 (第{attempt}次): {e}")
                if attempt < max_retries:

                    import sys as _sys
                    mods_to_remove = [k for k in _sys.modules if k.startswith('transformers')]
                    for mod_name in mods_to_remove:
                        del _sys.modules[mod_name]
                    time.sleep(1)
                else:
                    raise

        print("[ImagingAgent] 正在加载 HuluMed-7B 模型...")

        torch.cuda.empty_cache()

        self.model = AutoModelForCausalLM.from_pretrained(
            HULUMED_MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map={"": self.device},
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(
            HULUMED_MODEL_PATH,
            trust_remote_code=True,
        )
        print(f"[ImagingAgent] HuluMed-7B 加载完成，设备: {self.device}")

    def release_model(self):

        import torch
        import gc

        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        torch.cuda.empty_cache()
        gc.collect()
        print(f"[ImagingAgent] 模型已释放，GPU显存已清理")

    def _build_single_sequence_conversation(self, nii_path: str, sequence_name: str) -> list:

        prompt = SEQUENCE_PROMPTS.get(sequence_name, SEQUENCE_PROMPTS["T1_pre"])
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "3d",
                        "3d": {
                            "image_path": nii_path,
                            "nii_num_slices": NII_NUM_SLICES,
                            "nii_axis": NII_AXIS,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return conversation

    def _build_multi_sequence_conversation(self, mri_paths: Dict[str, str], prompt: str) -> list:

        content = []
        sequence_order = ["T1_pre", "T1_arterial", "T1_portal", "T2"]
        for seq_name in sequence_order:
            path = mri_paths.get(seq_name)
            if path and os.path.exists(path):
                content.append({
                    "type": "3d",
                    "3d": {
                        "image_path": path,
                        "nii_num_slices": NII_NUM_SLICES,
                        "nii_axis": NII_AXIS,
                    },
                })
        content.append({"type": "text", "text": prompt})
        conversation = [{"role": "user", "content": content}]
        return conversation

    def _run_inference(self, conversation: list, max_new_tokens: int = 2048) -> str:

        import torch
        import gc

        model_inputs = self.processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        model_device = next(self.model.parameters()).device

        model_inputs = {
            k: v.to(device=model_device) if isinstance(v, torch.Tensor) else v
            for k, v in model_inputs.items()
        }

        if "pixel_values" in model_inputs:
            model_inputs["pixel_values"] = model_inputs["pixel_values"].to(
                device=model_device, dtype=torch.float16
            )

        with torch.inference_mode():
            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                temperature=None,
                top_p=None,
                top_k=None,
            )

        outputs = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        del model_inputs, output_ids
        torch.cuda.empty_cache()
        gc.collect()

        return outputs

    def analyze_sequence(self, nii_path: str, sequence_name: str) -> str:

        if not os.path.exists(nii_path):
            return f"[{sequence_name}] 文件不存在: {nii_path}"

        print(f"[ImagingAgent] 正在分析序列: {sequence_name} -> {os.path.basename(nii_path)}")

        import torch
        import gc

        try:
            conversation = self._build_single_sequence_conversation(nii_path, sequence_name)
            try:
                output = self._run_inference(conversation, max_new_tokens=1024)
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    gc.collect()
                    import time; time.sleep(1)
                    try:
                        output = self._run_inference(conversation, max_new_tokens=512)
                    except Exception as e2:
                        return f"[{sequence_name}] 影像分析失败: GPU显存不足 - {str(e2)}"
                else:
                    raise

            prompt_tail = "请直接输出影像描述："
            if prompt_tail in output:
                output = output.split(prompt_tail)[-1].strip()

            return output
        except Exception as e:
            error_msg = f"[{sequence_name}] 分析失败: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return error_msg

    def generate_report(self, mri_paths: Dict[str, str], patient_id: str = "") -> Dict:

        print(f"[ImagingAgent] 开始为患者 {patient_id} 生成影像报告...")

        report = {
            "patient_id": patient_id,
            "sequences": {},
            "comprehensive_analysis": {},
            "raw_report": "",
        }

        sequence_descriptions = []
        sequence_order = ["T1_pre", "T1_arterial", "T1_portal", "T2"]
        sequence_labels = {
            "T1_pre": "T1平扫期",
            "T1_arterial": "T1动脉期",
            "T1_portal": "T1门脉期",
            "T2": "T2加权像",
        }

        import torch
        import gc

        for seq_name in sequence_order:
            nii_path = mri_paths.get(seq_name, "")
            if nii_path and os.path.exists(nii_path):
                description = self.analyze_sequence(nii_path, seq_name)
                report["sequences"][seq_name] = {
                    "description": description,
                    "findings": description,
                }
                label = sequence_labels.get(seq_name, seq_name)
                sequence_descriptions.append(f"【{label}】\n{description}")

                torch.cuda.empty_cache()
                gc.collect()
            else:
                report["sequences"][seq_name] = {
                    "description": "未获取",
                    "findings": "该序列影像未提供，无法分析",
                }
                label = sequence_labels.get(seq_name, seq_name)
                sequence_descriptions.append(f"【{label}】\n未获取")

        print("[ImagingAgent] 正在生成综合诊断报告...")

        available_paths = {k: v for k, v in mri_paths.items() if v and os.path.exists(v)}
        combined_desc_text = "\n\n".join(sequence_descriptions)

        try:
            if len(available_paths) >= 2:
                comprehensive_prompt = COMPREHENSIVE_PROMPT_TEMPLATE.format(
                    sequence_descriptions=combined_desc_text
                )
                conversation = self._build_multi_sequence_conversation(
                    available_paths, comprehensive_prompt
                )
                try:
                    comprehensive_output = self._run_inference(conversation, max_new_tokens=2048)
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    if "CUDA" in str(e) or "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        gc.collect()
                        import time; time.sleep(2)

                        print("[ImagingAgent] 多序列同时输入OOM，退化为纯文本综合分析...")
                        text_conversation = [
                            {"role": "user", "content": [{"type": "text", "text": comprehensive_prompt}]}
                        ]
                        comprehensive_output = self._run_inference(text_conversation, max_new_tokens=2048)
                    else:
                        raise
            else:

                comprehensive_prompt = COMPREHENSIVE_PROMPT_TEMPLATE.format(
                    sequence_descriptions=combined_desc_text
                )
                text_conversation = [
                    {"role": "user", "content": [{"type": "text", "text": comprehensive_prompt}]}
                ]
                comprehensive_output = self._run_inference(text_conversation, max_new_tokens=2048)

            report["comprehensive_analysis"] = self._parse_comprehensive_json(comprehensive_output)
            report["raw_report"] = comprehensive_output

        except Exception as e:
            print(f"[ImagingAgent] 综合诊断生成失败: {e}")
            traceback.print_exc()
            report["comprehensive_analysis"] = self._build_fallback_analysis(combined_desc_text)
            report["raw_report"] = combined_desc_text

        print("[ImagingAgent] 正在提取结构化判别特征...")
        try:
            discriminative_prompt = DISCRIMINATIVE_FEATURE_PROMPT.format(
                sequence_descriptions=combined_desc_text
            )

            if len(available_paths) >= 2:
                disc_conversation = self._build_multi_sequence_conversation(
                    available_paths, discriminative_prompt
                )
                try:
                    discriminative_output = self._run_inference(disc_conversation, max_new_tokens=1024)
                except RuntimeError as e:
                    if "CUDA" in str(e) or "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        gc.collect()
                        text_disc_conversation = [
                            {"role": "user", "content": [{"type": "text", "text": discriminative_prompt}]}
                        ]
                        discriminative_output = self._run_inference(text_disc_conversation, max_new_tokens=768)
                    else:
                        raise
            else:
                text_disc_conversation = [
                    {"role": "user", "content": [{"type": "text", "text": discriminative_prompt}]}
                ]
                discriminative_output = self._run_inference(text_disc_conversation, max_new_tokens=1024)

            report["discriminative_features"] = self._parse_discriminative_features(discriminative_output)
            report["discriminative_features"]["raw_output"] = discriminative_output[:500]

            if report["discriminative_features"]["parse_success"]:
                sig = report["discriminative_features"]["discriminative_signature"]
                print(f"[ImagingAgent] 判别特征提取成功: {sig[:5]}...")
            else:
                print("[ImagingAgent] 判别特征解析失败，使用默认值")

        except Exception as e:
            print(f"[ImagingAgent] 判别特征提取失败: {e}")
            traceback.print_exc()
            report["discriminative_features"] = self._parse_discriminative_features("")

        available_sequences = sum(
            1 for s in report["sequences"].values()
            if s.get("description") and s["description"] != "未获取"
        )

        radiomics_features = {}
        try:
            from analytics.imaging_extractor import ImagingRadiomicsExtractor
            rad_ext = ImagingRadiomicsExtractor()
            radiomics_features = rad_ext.extract_features_from_report(combined_desc_text)
            if radiomics_features:
                print(f"[ImagingAgent] 影像组学特征提取完成: {radiomics_features}")
        except Exception as rad_err:
            print(f"[ImagingAgent] 影像组学特征提取失败: {rad_err}")

        report["quantitative_risk"] = self._compute_imaging_risk_score(
            report["comprehensive_analysis"], 
            available_sequences, 
            radiomics_features,
            report.get("discriminative_features")
        )

        torch.cuda.empty_cache()
        gc.collect()

        print(f"[ImagingAgent] 患者 {patient_id} 影像报告生成完成"
              f" (影像风险评分: {report['quantitative_risk']['risk_score']:.2f})")
        return report

    def _parse_discriminative_features(self, text: str) -> Dict:
        default_result = {
            "core_features": {
                "size_mm": {"value": None, "confidence": 0.0},
                "aphe": {"present": None, "pattern": "none", "confidence": 0.0},
                "washout": {"present": None, "timing": "none", "confidence": 0.0},
                "capsule": {"present": None, "type": "none", "confidence": 0.0},
                "t2_signal": {"intensity": "unclear", "confidence": 0.0},
                "margin": {"type": "unclear", "confidence": 0.0},
            },
            "derived_risk_flags": {
                "mvi_high_risk": {"flag": False, "evidence": "none", "confidence": 0.0},
                "aggressive_pattern": {"flag": False, "evidence": "none", "confidence": 0.0},
                "multifocal": {"flag": False, "count": 1, "confidence": 0.0},
                "vascular_invasion": {"flag": False, "location": "none", "confidence": 0.0},
            },
            "lirads_assessment": {
                "category": "LR-NC",
                "major_features": [],
                "size_category": "unclear",
                "confidence": 0.0,
            },
            "discriminative_signature": [],
            "uncertainty": {
                "missing_sequences": [],
                "low_confidence_features": [],
                "ambiguous_findings": [],
            },
            "narrative_summary": "未能解析判别特征",
            "parse_success": False,
        }

        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start == -1 or json_end <= json_start:
            return default_result

        json_str = text[json_start:json_end]
        try:
            parsed = json.loads(json_str)
            result = default_result.copy()

            if "core_features" in parsed:
                result["core_features"] = self._merge_nested_dict(
                    default_result["core_features"], parsed["core_features"]
                )

            if "derived_risk_flags" in parsed:
                result["derived_risk_flags"] = self._merge_nested_dict(
                    default_result["derived_risk_flags"], parsed["derived_risk_flags"]
                )

            if "lirads_assessment" in parsed:
                result["lirads_assessment"] = self._merge_nested_dict(
                    default_result["lirads_assessment"], parsed["lirads_assessment"]
                )

            result["discriminative_signature"] = parsed.get("discriminative_signature", [])
            result["uncertainty"] = parsed.get("uncertainty", default_result["uncertainty"])
            result["narrative_summary"] = parsed.get("narrative_summary", "判别特征已提取")
            result["parse_success"] = True

            return result
        except json.JSONDecodeError:
            return default_result

    def _merge_nested_dict(self, default: Dict, override: Dict) -> Dict:
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = {**result[key], **value}
            elif key in result:
                result[key] = value
        return result

    def _parse_comprehensive_json(self, text: str) -> Dict:

        expected_keys = [
            "tumor_location", "tumor_size", "tumor_morphology",
            "enhancement_pattern", "t2_signal", "liver_background",
            "vascular_involvement", "lirads_grade", "diagnostic_conclusion",
        ]

        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = text[json_start:json_end]
            try:
                parsed = json.loads(json_str)
                result = {}
                for key in expected_keys:
                    result[key] = parsed.get(key, "未评估")
                for key in ["mvi_risk", "satellite_nodules"]:
                    if key in parsed:
                        result[key] = parsed[key]
                return result
            except json.JSONDecodeError:
                pass

        result = {k: "未评估" for k in expected_keys}
        result["diagnostic_conclusion"] = text[:500] if text else "综合分析未完成"
        return result

    def _build_fallback_analysis(self, combined_desc: str) -> Dict:

        return {
            "tumor_location": "未评估",
            "tumor_size": "未评估",
            "tumor_morphology": "未评估",
            "enhancement_pattern": "未评估",
            "t2_signal": "未评估",
            "liver_background": "未评估",
            "vascular_involvement": "未评估",
            "lirads_grade": "未评估",
            "diagnostic_conclusion": f"各序列描述已获取，但综合分析未能自动完成。各序列原始描述：\n{combined_desc[:500]}",
        }

    @staticmethod
    def _compute_imaging_risk_score(
        comp: Dict, 
        available_sequences: int, 
        radiomics_features: Dict = None,
        discriminative_features: Dict = None
    ) -> Dict:
        score = 0.35
        lirads = ""
        mvi = ""
        sat = ""
        vasc = ""
        enh = ""

        use_discriminative = (
            discriminative_features 
            and isinstance(discriminative_features, dict)
            and discriminative_features.get("parse_success", False)
        )

        if use_discriminative:
            core = discriminative_features.get("core_features", {})
            derived = discriminative_features.get("derived_risk_flags", {})
            lirads_assess = discriminative_features.get("lirads_assessment", {})

            lirads_category = lirads_assess.get("category", "LR-NC")
            lirads_conf = lirads_assess.get("confidence", 0.5)

            if "LR-5" in lirads_category or "LR5" in lirads_category:
                score += 0.35 * lirads_conf
            elif "LR-4" in lirads_category or "LR4" in lirads_category:
                score += 0.20 * lirads_conf
            elif "LR-3" in lirads_category or "LR3" in lirads_category:
                score += 0.05 * lirads_conf
            elif "LR-2" in lirads_category or "LR2" in lirads_category:
                score -= 0.10 * lirads_conf
            elif "LR-1" in lirads_category or "LR1" in lirads_category:
                score -= 0.15 * lirads_conf
            lirads = lirads_category

            mvi_flag = derived.get("mvi_high_risk", {})
            if mvi_flag.get("flag", False):
                mvi_conf = mvi_flag.get("confidence", 0.5)
                score += 0.12 * mvi_conf
                mvi = "高"
            else:
                mvi = "低"

            multifocal = derived.get("multifocal", {})
            if multifocal.get("flag", False):
                sat_conf = multifocal.get("confidence", 0.5)
                score += 0.08 * sat_conf
                sat = "多发"
            else:
                sat = "单发"

            vasc_invasion = derived.get("vascular_invasion", {})
            if vasc_invasion.get("flag", False):
                vasc_conf = vasc_invasion.get("confidence", 0.5)
                score += 0.10 * vasc_conf
                vasc = vasc_invasion.get("location", "血管侵犯")
            else:
                vasc = "无"

            aphe = core.get("aphe", {})
            washout = core.get("washout", {})
            if aphe.get("present", False) and washout.get("present", False):
                aphe_conf = aphe.get("confidence", 0.5)
                washout_conf = washout.get("confidence", 0.5)
                combined_conf = (aphe_conf + washout_conf) / 2
                score += 0.05 * combined_conf
                enh = f"APHE_{aphe.get('pattern', 'unknown')}_Washout_{washout.get('timing', 'unknown')}"
            else:
                enh = "非典型强化"

            aggressive = derived.get("aggressive_pattern", {})
            if aggressive.get("flag", False):
                agg_conf = aggressive.get("confidence", 0.5)
                score += 0.03 * agg_conf

            sig = discriminative_features.get("discriminative_signature", [])
            high_risk_keywords = ["LR-5", "LR5", "MVI_high", "vascular_invasion"]
            high_risk_count = sum(1 for s in sig if any(kw in str(s) for kw in high_risk_keywords))
            if high_risk_count >= 3:
                score += 0.08
            elif high_risk_count >= 2:
                score += 0.04

        else:
            lirads = str(comp.get("lirads_grade", "")).upper()
            if "LR-5" in lirads or "LR5" in lirads or "5" in lirads:
                score += 0.35
            elif "LR-4" in lirads or "LR4" in lirads or "4" in lirads:
                score += 0.20
            elif "LR-3" in lirads or "LR3" in lirads or "3" in lirads:
                score += 0.05
            elif "LR-2" in lirads or "LR2" in lirads or "2" in lirads:
                score -= 0.10
            elif "LR-1" in lirads or "LR1" in lirads or "1" in lirads:
                score -= 0.15

            mvi = str(comp.get("mvi_risk", ""))
            if "高" in mvi or "high" in mvi.lower():
                score += 0.12
            elif "中" in mvi or "medium" in mvi.lower():
                score += 0.06
            elif "低" in mvi or "low" in mvi.lower():
                score += 0.00

            sat = str(comp.get("satellite_nodules", ""))
            if "有" in sat or "多发" in sat or "yes" in sat.lower() or "multiple" in sat.lower():
                score += 0.08

            vasc = str(comp.get("vascular_involvement", ""))
            if "有" in vasc or "阳性" in vasc or "癌栓" in vasc or "侵犯" in vasc:
                score += 0.10

            enh = str(comp.get("enhancement_pattern", ""))
            if "快进快出" in enh or "APHE" in enh or "廓清" in enh or "washout" in enh.lower():
                score += 0.05

            liver = str(comp.get("liver_background", ""))
            if "肝硬化" in liver and ("重" in liver or "明显" in liver or "severe" in liver.lower()):
                score += 0.03

            high_risk_count = sum([
                1 if ("LR-5" in lirads or "LR5" in lirads or "5" in lirads) else 0,
                1 if ("高" in mvi or "high" in mvi.lower()) else 0,
                1 if ("有" in sat or "多发" in sat or "yes" in sat.lower() or "multiple" in sat.lower()) else 0,
                1 if ("有" in vasc or "阳性" in vasc or "癌栓" in vasc or "侵犯" in vasc) else 0,
            ])
            if high_risk_count >= 3:
                score += 0.08
            elif high_risk_count >= 2:
                score += 0.04

        if radiomics_features and isinstance(radiomics_features, dict):
            try:
                from analytics.imaging_extractor import ImagingRadiomicsExtractor
                rad_delta = ImagingRadiomicsExtractor.compute_radiomics_risk_score(radiomics_features)
                score += rad_delta
                print(f"[ImagingAgent] 影像组学风险增量: {rad_delta}")
            except Exception:
                pass

        score = max(0.05, min(0.95, score))

        if available_sequences >= 4:
            quality = "high"
        elif available_sequences >= 2:
            quality = "medium"
        else:
            quality = "low"

        predicted_months = max(3.0, 48.0 - score * 45.0)

        result = {
            "risk_score": round(score, 4),
            "predicted_months": round(predicted_months, 1),
            "model_quality": quality,
            "data_completeness": round(available_sequences / 4.0, 2),
            "imaging_risk_factors": {
                "lirads_grade": lirads,
                "mvi_risk": mvi,
                "satellite_nodules": sat,
                "vascular_involvement": vasc,
                "enhancement_pattern": enh,
            },
            "used_discriminative_features": use_discriminative,
        }

        if use_discriminative:
            result["discriminative_signature"] = discriminative_features.get("discriminative_signature", [])
            result["uncertainty"] = discriminative_features.get("uncertainty", {})

        return result

    def generate_backward_cot(self, report_dict: Dict, t_GT: Dict, patient_id: str = "") -> Dict:

        from agents.backward_cot_generator import ImagingBackwardCOTGenerator

        generator = ImagingBackwardCOTGenerator()
        return generator.generate_backward_cot(report_dict, t_GT, patient_id)

    def generate_cot(self, report_dict: Dict, patient_id: str = "") -> List[Dict]:

        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, ENABLE_THINKING

        thinking_config = {"enable_thinking": True} if ENABLE_THINKING else {"enable_thinking": False}
        llm = ChatOpenAI(
            model=LLM_MODEL, base_url=LLM_BASE_URL, api_key=LLM_API_KEY,
            temperature=0.2, max_retries=3, extra_body=thinking_config,
        )

        sequences = report_dict.get("sequences", {})
        comp = report_dict.get("comprehensive_analysis", {})
        disc = report_dict.get("discriminative_features", {})
        quant = report_dict.get("quantitative_risk", {})

        seq_desc = []
        for seq_name in ["T1_pre", "T1_arterial", "T1_portal", "T2"]:
            seq_data = sequences.get(seq_name, {})
            desc = seq_data.get("description", "未获取")
            seq_desc.append(f"【{seq_name}】{desc}")
        seq_text = "\n".join(seq_desc)

        system_prompt = """你是资深肝细胞癌影像诊断专家。请基于MRI四序列分析，生成5步COT推理链。

【5步推理框架】
Step 1 - 序列特征提取: 从T1平扫、动脉期、门脉期、T2提取关键影像特征
Step 2 - LI-RADS特征识别: 识别APHE、Washout、包膜等LI-RADS主要特征
Step 3 - 侵袭性评估: 基于影像推断MVI风险、肿瘤边界、卫星灶
Step 4 - 多序列一致性验证: 验证各序列发现是否一致，排除伪影/技术因素
Step 5 - 综合结论: 给出LI-RADS分级、影像风险评分、关键发现总结

输出JSON格式:
{
  "cot_chain": [
    {"step": 1, "title": "序列特征提取", "reasoning": "...", "evidence": ["..."], "conclusion": "..."},
    {"step": 2, "title": "LI-RADS特征识别", "reasoning": "...", "evidence": ["..."], "conclusion": "..."},
    {"step": 3, "title": "侵袭性评估", "reasoning": "...", "evidence": ["..."], "conclusion": "..."},
    {"step": 4, "title": "多序列一致性验证", "reasoning": "...", "evidence": ["..."], "conclusion": "..."},
    {"step": 5, "title": "综合结论", "reasoning": "...", "evidence": ["..."], "conclusion": "..."}
  ]
}"""

        user_prompt = f"""患者ID: {patient_id or report_dict.get('patient_id', '未知')}

【各序列描述】
{seq_text}

【综合分析】
{json.dumps(comp, ensure_ascii=False, indent=2)}

【判别特征】
{json.dumps(disc, ensure_ascii=False, indent=2)[:1500]}

请生成5步COT推理链，直接输出JSON。"""

        try:
            import json as _json
            resp = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            text = resp.content

            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = _json.loads(text[start:end])
                return parsed.get("cot_chain", [])
        except Exception as e:
            print(f"[ImagingAgent] COT生成失败: {e}")

        return []

    def generate_report_text(self, report_dict: Dict) -> str:

        lines = []
        pid = report_dict.get("patient_id", "未知")
        lines.append(f"{'='*60}")
        lines.append(f"  影像诊断报告 — 患者 {pid}")
        lines.append(f"{'='*60}")

        lines.append("\n【各序列影像描述】")
        sequence_labels = {
            "T1_pre": "T1平扫期",
            "T1_arterial": "T1动脉期",
            "T1_portal": "T1门脉期",
            "T2": "T2加权像",
        }
        sequences = report_dict.get("sequences", {})
        for seq_name in ["T1_pre", "T1_arterial", "T1_portal", "T2"]:
            seq_data = sequences.get(seq_name, {})
            label = sequence_labels.get(seq_name, seq_name)
            desc = seq_data.get("description", "未获取")
            lines.append(f"\n  [{label}]")
            lines.append(f"  {desc}")

        lines.append(f"\n{'─'*60}")
        lines.append("【综合分析】")
        comp = report_dict.get("comprehensive_analysis", {})

        field_labels = {
            "tumor_location": "肿瘤位置",
            "tumor_size": "肿瘤大小",
            "tumor_morphology": "肿瘤形态",
            "enhancement_pattern": "强化模式",
            "t2_signal": "T2信号",
            "liver_background": "肝脏背景",
            "vascular_involvement": "血管侵犯",
            "lirads_grade": "LI-RADS分级",
            "mvi_risk": "MVI风险",
            "satellite_nodules": "卫星灶",
            "diagnostic_conclusion": "诊断结论",
        }

        for key, label in field_labels.items():
            val = comp.get(key)
            if val and val != "未评估":
                lines.append(f"  - {label}: {val}")

        lines.append(f"\n{'='*60}")
        return "\n".join(lines)

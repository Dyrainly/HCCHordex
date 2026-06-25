import json
import os
import re
import sys
import time
import traceback
from typing import Dict, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

class AttendingAgent:

    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            temperature=0.3,
            max_tokens=16384,
            max_retries=5,
        )
        self.system_prompt = """你是一位资深肝胆外科主治医生，拥有20年以上肝细胞癌(HCC)多学科诊疗(MDT)经验。

你将收到来自三位专科医生的诊断报告：
1. 影像科医生：基于3D MRI多序列的影像诊断报告
2. 临床医生：基于临床信息的风险评估报告
3. 分子病理医生：基于蛋白组学的分子特征报告

你需要综合三方报告，运用Chain-of-Thought(COT)思维链进行系统性推理，给出最终的综合诊断报告。

【重要】你必须严格按照以下5步COT进行推理：
Step 1 - 影像学证据综合：提取MRI报告中的关键发现（肿瘤特征、强化模式、LI-RADS等）
Step 2 - 临床因素评估：提取临床报告中的主要风险因素和保护因素
Step 3 - 分子层面分析：提取蛋白组学报告中的关键分子标志物
Step 4 - 多模态证据融合：综合三个维度，分析一致性和矛盾点，进行复发时间预后判断
Step 5 - 最终结论：给出预后状态、预估复发时间、复发风险、关键决定因素

如果某个模态的数据缺失，请在对应步骤中明确标注"该模态数据缺失"，并降低该维度的权重。

输出格式必须为JSON。"""

    def synthesize(
        self,
        imaging_report: str = None,
        clinical_report: str = None,
        proteomics_report: str = None,
        patient_id: str = "",
    ) -> Dict:

        img_text = imaging_report if imaging_report else "【该模态数据缺失】影像科报告未提供。"
        clin_text = clinical_report if clinical_report else "【该模态数据缺失】临床评估报告未提供。"
        prot_text = proteomics_report if proteomics_report else "【该模态数据缺失】蛋白组学报告未提供。"

        user_msg = f"""请对以下肝细胞癌患者进行多学科综合诊断。

患者就诊号：{patient_id}

===== 影像科报告 =====
{img_text}

===== 临床评估报告 =====
{clin_text}

===== 蛋白组学报告 =====
{prot_text}

请严格按5步COT思维链推理，并以如下JSON格式输出（直接输出JSON，不要markdown代码块）：
{{
    "patient_id": "{patient_id}",
    "cot_chain": [
        {{
            "step": 1,
            "title": "影像学证据综合",
            "reasoning": "详细推理过程",
            "evidence": ["关键证据1", "关键证据2"],
            "conclusion": "该步骤结论"
        }},
        {{
            "step": 2,
            "title": "临床因素评估",
            "reasoning": "详细推理过程",
            "evidence": ["关键证据1"],
            "conclusion": "该步骤结论"
        }},
        {{
            "step": 3,
            "title": "分子层面分析",
            "reasoning": "详细推理过程",
            "evidence": ["关键证据1"],
            "conclusion": "该步骤结论"
        }},
        {{
            "step": 4,
            "title": "多模态证据融合",
            "reasoning": "综合三个维度的推理",
            "evidence": ["融合后的关键发现"],
            "conclusion": "融合结论"
        }},
        {{
            "step": 5,
            "title": "最终结论",
            "reasoning": "最终推理",
            "evidence": ["决定性证据"],
            "conclusion": "最终结论"
        }}
    ],
    "final_diagnosis": {{
        "prognosis_status": "良好/中等/不良",
        "recurrence_time_prediction": "预计复发时间描述",
        "recurrence_months_estimate": 18.0,
        "recurrence_risk": "高/中/低",
        "key_factors": [
            {{"factor": "因素名", "impact": "正面/负面", "weight": "高/中/低", "source": "影像/临床/蛋白组学"}}
        ],
        "treatment_recommendations": ["建议1", "建议2"],
        "confidence_level": "高/中/低",
        "data_completeness": {{
            "imaging": true,
            "clinical": true,
            "proteomics": true
        }}
    }}
}}"""

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_msg),
        ]

        last_error = None
        for attempt in range(3):
            try:
                response = self.llm.invoke(messages)
                raw_text = response.content
                result = self._parse_json_response(raw_text)

                result.setdefault("patient_id", patient_id)
                result.setdefault("raw_report", raw_text)

                fd = result.get("final_diagnosis", {})
                if isinstance(fd, dict):
                    dc = fd.setdefault("data_completeness", {})
                    dc["imaging"] = imaging_report is not None and len(imaging_report or "") > 0
                    dc["clinical"] = clinical_report is not None and len(clinical_report or "") > 0
                    dc["proteomics"] = proteomics_report is not None and len(proteomics_report or "") > 0

                return result

            except Exception as e:
                last_error = e
                print(f"[AttendingAgent] 第{attempt + 1}次调用失败: {e}")
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))

        return self._build_error_report(patient_id, last_error, imaging_report, clinical_report, proteomics_report)

    def get_cot_text(self, report: Dict) -> str:

        lines = []
        patient_id = report.get("patient_id", "未知")
        lines.append(f"=== 患者 {patient_id} 综合诊断COT推理 ===\n")

        cot_chain = report.get("cot_chain", [])
        for step in cot_chain:
            if isinstance(step, dict):
                step_num = step.get("step", "?")
                title = step.get("title", "")
                reasoning = step.get("reasoning", "")
                evidence = step.get("evidence", [])
                conclusion = step.get("conclusion", "")

                lines.append(f"【Step {step_num} - {title}】")
                lines.append(f"推理: {reasoning}")
                if evidence:
                    lines.append(f"证据: {'; '.join(str(e) for e in evidence)}")
                lines.append(f"结论: {conclusion}")
                lines.append("")
            elif isinstance(step, str):
                lines.append(step)
                lines.append("")

        fd = report.get("final_diagnosis", {})
        if isinstance(fd, dict):
            lines.append("【最终诊断】")
            lines.append(f"预后状态: {fd.get('prognosis_status', '未知')}")
            lines.append(f"复发时间预测: {fd.get('survival_prediction', '未知')}")
            lines.append(f"复发风险: {fd.get('recurrence_risk', '未知')}")
            lines.append(f"置信度: {fd.get('confidence_level', '未知')}")

            key_factors = fd.get("key_factors", [])
            if key_factors:
                lines.append("关键因素:")
                for kf in key_factors:
                    if isinstance(kf, dict):
                        lines.append(
                            f"  - {kf.get('factor','')}: "
                            f"影响={kf.get('impact','')}, "
                            f"权重={kf.get('weight','')}, "
                            f"来源={kf.get('source','')}"
                        )
                    else:
                        lines.append(f"  - {kf}")

            recs = fd.get("treatment_recommendations", [])
            if recs:
                lines.append("治疗建议: " + "; ".join(recs))
        elif isinstance(fd, str):
            lines.append(f"最终诊断: {fd}")

        return "\n".join(lines)

    def _parse_json_response(self, text: str) -> Dict:

        cleaned = text.strip()

        md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
        if md_match:
            cleaned = md_match.group(1).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        try:
            return self._repair_truncated_json(cleaned)
        except (json.JSONDecodeError, ValueError):
            pass

        return {"raw_report": cleaned, "parse_error": "无法解析JSON，返回原始文本"}

    def _repair_truncated_json(self, text: str) -> Dict:

        brace_match = re.search(r"\{.*", text, re.DOTALL)
        if not brace_match:
            raise ValueError("No JSON object found")

        partial = brace_match.group(0)

        stack = []
        i = 0
        in_string = False
        escape = False
        last_valid_pos = 0

        while i < len(partial):
            ch = partial[i]
            if escape:
                escape = False
                i += 1
                continue
            if ch == '\\' and in_string:
                escape = True
                i += 1
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                i += 1
                continue
            if in_string:
                i += 1
                continue
            if ch in '{[':
                stack.append(ch)
            elif ch == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
                else:

                    pass
            elif ch == ']':
                if stack and stack[-1] == '[':
                    stack.pop()
                else:
                    pass

            if not stack and not in_string:

                pass
            i += 1

        if not stack and not in_string:

            raise ValueError("JSON appears complete but failed to parse")

        repaired = partial.rstrip()

        if in_string:
            repaired += '"'

        for bracket in reversed(stack):
            if bracket == '{':
                repaired += '}'
            elif bracket == '[':
                repaired += ']'

        return json.loads(repaired)

    def _build_error_report(
        self, patient_id: str, error, imaging_report, clinical_report, proteomics_report
    ) -> Dict:

        return {
            "patient_id": patient_id,
            "error": f"API调用3次均失败: {error}",
            "cot_chain": [
                {
                    "step": i + 1,
                    "title": t,
                    "reasoning": "API调用失败，无法完成推理",
                    "evidence": [],
                    "conclusion": "无法得出结论",
                }
                for i, t in enumerate(
                    ["影像学证据综合", "临床因素评估", "分子层面分析", "多模态证据融合", "最终结论"]
                )
            ],
            "final_diagnosis": {
                "prognosis_status": "未知",
                "recurrence_time_prediction": "无法预测",
                "recurrence_months_estimate": 0.0,
                "recurrence_risk": "未知",
                "key_factors": [],
                "treatment_recommendations": [],
                "confidence_level": "低",
                "data_completeness": {
                    "imaging": imaging_report is not None and len(imaging_report or "") > 0,
                    "clinical": clinical_report is not None and len(clinical_report or "") > 0,
                    "proteomics": proteomics_report is not None and len(proteomics_report or "") > 0,
                },
            },
            "raw_report": "",
        }

if __name__ == "__main__":
    print("[AttendingAgent] Import 测试...")
    agent = AttendingAgent()
    print(f"[AttendingAgent] LLM模型: {agent.llm.model_name}")
    print("[AttendingAgent] Import OK")

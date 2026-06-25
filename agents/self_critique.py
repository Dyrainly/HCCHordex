import json
import re
import sys
import os
from typing import Dict, Optional, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, ENABLE_THINKING
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

def _extract_json(text: str) -> Optional[Dict]:

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    patterns = [
        r"```(?:json)?\s*\n?(.*?)\n?\s*```",
        r"\{.*\}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:

                content = match.group(1).strip() if "```" in pattern else match.group(0)
                return json.loads(content)
            except json.JSONDecodeError:
                continue

    return None

VALIDATE_PROMPT = """你是肝癌MDT质量审核专家。请对以下CoT（思维链）诊断报告进行质量评估。

⚠️ 重要前提：你不知道患者的真实生存结局（t_GT），因此只能评估推理逻辑本身的质量，
不能预知或反向推理得到最终结局。

请从以下四个维度评估CoT质量：

1. **证据溯源 (Evidence Grounding)**
   - 每个关键判断是否有明确的影像/临床/蛋白组学证据支撑？
   - 是否存在"凭空断言"或"跳过证据直接结论"的情况？

2. **因果一致性 (Causal Consistency)**
   - 推理链是否前后一致？结论是否能从前提自然推出？
   - 是否存在逻辑跳跃或自相矛盾？

3. **不确定性处理 (Uncertainty Handling)**
   - 当证据矛盾或不足时，是否诚实地承认不确定性？
   - 置信度是否与证据强度匹配？（证据弱却高置信=过度自信）

4. **无反向推理泄露 (No Backward Reasoning Leakage)**
   - 推理过程是否从数据→结论，而非从结论→证据？
   - 是否有"因为结果是X，所以证据Y很重要"的嫌疑？

输出JSON格式：
{
  "quality_level": "high" 或 "low",
  "critique": "详细的批评意见，指出具体问题和改进方向",
  "confidence_score": 0.0-1.0之间的浮点数，表示对质量评估的置信度
}

判定标准：
- high: 推理逻辑清晰、证据充分、不确定性处理得当、无反向推理
- low: 存在明显的逻辑缺陷、证据缺失、过度自信或反向推理嫌疑"""

REVISE_PROMPT = """你是资深肝胆外科主治医生。你的CoT诊断报告经审核后需要修订。

请根据以下批评意见，重新生成改进后的CoT思维链。

要求：
1. 保留原有结构（Step 1-5的推理链）
2. 针对批评中指出的具体问题进行修改
3. 不要添加新的未经证实的判断
4. 如实反映不确定性

⚠️ 重要：你不知道患者的真实生存结局，只能基于现有证据进行推理。

请直接输出修订后的完整CoT文本（非JSON）。"""

class SelfCritique:

    def __init__(self):

        thinking_config = {"enable_thinking": True} if ENABLE_THINKING else {"enable_thinking": False}
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            temperature=0.1,
            max_retries=3,
            extra_body=thinking_config,
        )

    def validate(self, cot: str, reports: Dict) -> Dict:

        context_parts = []

        imaging = reports.get("imaging_report") or reports.get("imaging")
        if imaging:
            context_parts.append(f"【影像报告】\n{self._truncate(imaging, 1000)}")

        clinical = reports.get("clinical_report") or reports.get("clinical")
        if clinical:
            context_parts.append(f"【临床报告】\n{self._truncate(clinical, 1000)}")

        proteomics = reports.get("proteomics_report") or reports.get("proteomics")
        if proteomics:
            context_parts.append(f"【蛋白组学报告】\n{self._truncate(proteomics, 800)}")

        reports_context = "\n\n".join(context_parts) if context_parts else "无报告上下文"

        validate_user_prompt = f"""

{reports_context}

{cot}

请评估上述CoT的质量，输出JSON格式的评估结果。"""

        try:
            response = self.llm.invoke([
                SystemMessage(content=VALIDATE_PROMPT),
                HumanMessage(content=validate_user_prompt)
            ])

            parsed = _extract_json(response.content)

            if parsed and "quality_level" in parsed:

                return {
                    "quality_level": parsed.get("quality_level", "low"),
                    "critique": parsed.get("critique", ""),
                    "confidence_score": float(parsed.get("confidence_score", 0.5))
                }
            else:

                return {
                    "quality_level": "low",
                    "critique": f"验证结果解析失败: {response.content[:200]}",
                    "confidence_score": 0.3
                }

        except Exception as e:
            return {
                "quality_level": "low",
                "critique": f"验证过程出错: {str(e)}",
                "confidence_score": 0.0
            }

    def revise(self, cot: str, critique: str) -> str:

        revise_user_prompt = f"""

{cot}

{critique}

请根据批评意见修订CoT，直接输出修订后的完整文本。"""

        try:
            response = self.llm.invoke([
                SystemMessage(content=REVISE_PROMPT),
                HumanMessage(content=revise_user_prompt)
            ])
            return response.content
        except Exception as e:

            print(f"[SelfCritique] 修订失败: {e}")
            return cot

    def validate_and_fix(self, cot: str, reports: Dict, max_iterations: int = 5) -> Tuple[str, Dict]:

        current_cot = cot
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            validation = self.validate(current_cot, reports)
            validation["iteration"] = iteration

            if validation["quality_level"] == "high":
                print(f"[SelfCritique] 第{iteration}次验证通过，quality_level=high")
                validation["revised"] = iteration > 1
                validation["iterations"] = iteration
                return current_cot, validation

            print(f"[SelfCritique] 第{iteration}次验证quality_level=low，执行修订...")
            current_cot = self.revise(current_cot, validation["critique"])

        print(f"[SelfCritique] 达到最大迭代次数{max_iterations}，quality_level仍为low")
        validation["revised"] = True
        validation["iterations"] = iteration
        validation["max_iterations_reached"] = True

        return current_cot, validation

    @staticmethod
    def _truncate(text: str, max_length: int) -> str:

        if not text:
            return ""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "...(已截断)"

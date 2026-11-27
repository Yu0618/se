import json
import os
from google import genai
from google.genai import types

# 1. 定義 Agent 可以使用的工具（Python 函式）
def calculate(expression: str) -> str:
    """安全地計算數學算式，例如 '1200 * 32.5' 或 '(100 + 50) / 3'。"""
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression).issubset(allowed_chars):
        return json.dumps({"error": "算式包含不允許的字元"})
    try:
        # 限制全域與區域命名空間，降低 eval 風險
        result = eval(expression, {"__builtins__": {}}, {})
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": f"計算失敗: {e}"})


def get_exchange_rate(currency_pair: str) -> str:
    """查詢指定幣別對的匯率，例如 'USD_TWD' 或 'JPY_TWD'。"""
    # 實務上這裡會串接外部匯率 API；此處以範例資料示範
    mock_rates = {
        "USD_TWD": 32.5,
        "JPY_TWD": 0.21,
        "EUR_TWD": 35.2,
    }
    pair = currency_pair.upper()
    if pair in mock_rates:
        return json.dumps({"currency_pair": pair, "rate": mock_rates[pair]})
    return json.dumps({"error": f"找不到匯率資料: {pair}"})


TOOLS_MAP = {
    "calculate": calculate,
    "get_exchange_rate": get_exchange_rate,
}


# 2. 建構 Agent Loop
def run_agent(user_prompt: str, max_steps: int = 6) -> str:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    config = types.GenerateContentConfig(
        system_instruction=(
            "你是一個嚴謹的助理 Agent。遇到需要查匯率或數學計算時，"
            "請呼叫提供的工具取得精確結果，不要心算或猜測。"
            "拿到工具結果後，再清楚回答使用者。"
        ),
        # 直接傳入 Python 函式，SDK 會自動生成對應的 Tool Schema
        tools=[calculate, get_exchange_rate],
        temperature=0.2,
    )

    # 初始化對話歷史（短期記憶）
    contents: list[types.Content | str] = [user_prompt]

    for step in range(1, max_steps + 1):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config,
        )

        # 檢查模型是否要求呼叫工具
        function_calls = response.function_calls
        if not function_calls:
            # 沒有工具呼叫，代表模型已完成推理並產出最終回覆
            return response.text or ""

        # 將模型的「工具呼叫意圖」加入對話歷史
        if response.candidates and response.candidates[0].content:
            contents.append(response.candidates[0].content)

        # 依序執行所有被要求的工具，並收集執行結果
        tool_response_parts: list[types.Part] = []
        for fc in function_calls:
            tool_name = fc.name
            tool_args = dict(fc.args) if fc.args else {}
            print(f"[Step {step}] 呼叫工具: {tool_name}({tool_args})")

            func = TOOLS_MAP.get(tool_name)
            if func is None:
                raw_output = json.dumps({"error": f"未知的工具: {tool_name}"})
            else:
                raw_output = func(**tool_args)

            print(f"[Step {step}] 工具回傳: {raw_output}")

            # 嘗試解析 JSON 作為 function_response payload
            try:
                parsed_output = json.loads(raw_output)
            except json.JSONDecodeError:
                parsed_output = {"output": raw_output}

            tool_response_parts.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response=parsed_output,
                )
            )

        # 將工具執行結果以 user/tool 回應內容塞回對話歷史，進入下一輪迴圈
        contents.append(types.Content(role="user", parts=tool_response_parts))

    return "已達最大執行步數，任務尚未完成。"


if __name__ == "__main__":
    question = "我有 1,250 美元，再加上 30,000 日圓，換算成台幣總共是多少？請幫我算清楚。"
    answer = run_agent(question)
    print("\n=== 最終回覆 ===")
    print(answer)

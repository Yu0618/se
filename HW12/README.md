# 原生 Gemini Agent 範例（Function Calling + Agent Loop）

本專案示範如何不依賴重型框架，僅使用官方 `google-genai` SDK，將純粹的 LLM API 建構成具備工具呼叫與多步推理能力的 **Agent**。

## 核心概念

純粹的 LLM API 只負責根據輸入預測下一個字元；要成為 **Agent**，需要加上控制迴圈與外部工具，讓模型能夠依照目標反覆進行：

`推理決策 → 呼叫工具 → 取得結果 → 回灌上下文 → 產出最終回覆`

本範例涵蓋五個基礎要素：
1. **LLM（推理引擎）**：使用 `gemini-2.5-flash` 判斷何時需要工具以及如何彙整結果。
2. **Tools（工具集）**：提供 `calculate` 與 `get_exchange_rate` 兩個本地函式。
3. **Agent Loop（執行迴圈）**：攔截 `response.function_calls`，執行工具並將結果回傳模型，直到模型輸出純文字。
4. **Memory / Context（對話上下文）**：透過 `contents` 串列累積每一輪的提問、工具呼叫與工具回傳內容。
5. **Guardrails（基本護欄）**：透過 `max_steps` 限制最大迴圈次數，並在工具中加入字元檢查與例外捕捉。

---

## 環境準備與安裝

1. 安裝官方 SDK：

```bash
pip install google-genai
```

2. 設定 API 金鑰：

```bash
# Linux / macOS
export GEMINI_API_KEY="你的_GEMINI_API_KEY"

# Windows (PowerShell)
$env:GEMINI_API_KEY="你的_GEMINI_API_KEY"
```

---

## 程式碼結構說明

* **`calculate(expression: str) -> str`**：驗證算式字元後進行數學運算，回傳 JSON 格式的計算結果或錯誤訊息。
* **`get_exchange_rate(currency_pair: str) -> str`**：查詢幣別匯率（範例內建 `USD_TWD`、`JPY_TWD`、`EUR_TWD` 模擬資料，可改接真實匯率 API）。
* **`TOOLS_MAP`**：將工具名稱映射到對應的可執行函式。
* **`run_agent(user_prompt: str, max_steps: int = 6) -> str`**：核心 Agent Loop，負責呼叫 `generate_content`、處理 `function_calls`、包裝 `Part.from_function_response` 並推進下一輪推理。

---

## 執行方式

將主程式存為 `agent.py` 後執行：

```bash
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
```

### 預期執行流程

1. 程式送出多幣別換算台幣的問題。
2. 模型回傳 `get_exchange_rate` 呼叫請求（查詢 `USD_TWD` 與 `JPY_TWD`）。
3. 程式執行工具並將匯率回傳給模型。
4. 模型視需要呼叫 `calculate` 進行加總與換算，最後輸出完整計算說明。

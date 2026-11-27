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
Agent.py
```

### 預期執行流程

1. 程式送出多幣別換算台幣的問題。
2. 模型回傳 `get_exchange_rate` 呼叫請求（查詢 `USD_TWD` 與 `JPY_TWD`）。
3. 程式執行工具並將匯率回傳給模型。
4. 模型視需要呼叫 `calculate` 進行加總與換算，最後輸出完整計算說明。

---

## 進階擴充方向

* **串接真實外部 API**：將 `get_exchange_rate` 改為呼叫即時匯率 API，或新增搜尋、讀寫檔案、查資料庫等工具。
* **多輪對話記憶**：將 `contents` 抽離成 session 狀態管理，支援跨多輪使用者提問。
* **人工審核（Human-in-the-Loop）**：針對寫入資料、發送通知或交易等高風險工具，在執行前加入確認步驟。
* **可觀測性與錯誤處理**：整合結構化日誌、Token 統計與重試機制，提升正式環境穩定度。

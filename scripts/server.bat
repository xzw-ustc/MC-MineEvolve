@echo off
REM Start the MineEvolve FastAPI server on Windows.
REM
REM Default LLM backend is Qwen Plus via DashScope's OpenAI-compatible endpoint.
REM Set the matching API key first, e.g.:
REM
REM   set DASHSCOPE_API_KEY=sk-...     (DEFAULT)
REM   set ZHIPUAI_API_KEY=xxx.yyy      (then set MINEEVOLVE_LLM_PROVIDER=glm)
REM   set GOOGLE_API_KEY=AIza...       (then set MINEEVOLVE_LLM_PROVIDER=gemini)
REM   set OPENAI_API_KEY=sk-...        (then set MINEEVOLVE_LLM_PROVIDER=openai)

if "%MINEEVOLVE_PORT%"=="" set MINEEVOLVE_PORT=9000
if "%MINEEVOLVE_HOST%"=="" set MINEEVOLVE_HOST=0.0.0.0
if "%CUDA_VISIBLE_DEVICES%"=="" set CUDA_VISIBLE_DEVICES=0

if "%DASHSCOPE_API_KEY%"=="" if "%OPENAI_API_KEY%"=="" if "%ZHIPUAI_API_KEY%"=="" if "%GOOGLE_API_KEY%"=="" (
  echo [warn] No LLM API key in environment; planner calls will fail.
  echo [hint] Default backend is Qwen: set DASHSCOPE_API_KEY=sk-...
)

uvicorn app:app --host %MINEEVOLVE_HOST% --port %MINEEVOLVE_PORT%

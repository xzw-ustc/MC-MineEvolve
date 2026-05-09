@echo off
REM Run a single benchmark group with a given LLM backend on Windows.
REM
REM Usage:
REM   scripts\run_eval.bat                          # default: wooden + qwen_plus
REM   scripts\run_eval.bat iron qwen_plus           # iron tier with qwen plus
REM   scripts\run_eval.bat diamond gpt_5_5          # diamond tier with gpt
REM
REM The server (scripts\server.bat) MUST already be running.

setlocal
if "%~1"=="" (set BENCHMARK=wooden) else (set BENCHMARK=%~1)
if "%~2"=="" (set LLM=qwen_plus) else (set LLM=%~2)
if "%MINEEVOLVE_PORT%"=="" set MINEEVOLVE_PORT=9000

python -m mineevolve.main benchmark=%BENCHMARK% llm=%LLM% server.port=%MINEEVOLVE_PORT%
endlocal

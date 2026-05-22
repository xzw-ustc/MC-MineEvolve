# MC-MineEvolve

Open-source implementation of **MineEvolve** — a knowledge-driven self-evolution framework for long-horizon embodied Minecraft agents.

This repository contains a minimal end-to-end implementation of the four-stage MineEvolve pipeline (Monitor → Inducer → Curator → Adaptor) on top of MineRL + STEVE-1

---

## What is here

```
MC-MineEvolve/
├── app.py                       # FastAPI server entry (loads STEVE-1 + LLM planner)
├── scripts/
│   ├── server.{sh,bat}          # start the server on port 9000
│   └── run_eval.{sh,bat}        # start the env-side client and run a benchmark
├── checkpoints/                 # placeholder; download STEVE-1/VPT/MineCLIP weights here
├── docs/
│   ├── architecture.md          # 4-stage data-flow diagram and module map
│   └── tasks.md                 # the 70-task split (paper Table 3)
└── src/mineevolve/
    ├── conf/                    # Hydra configs (benchmark/, llm/, evaluate.yaml)
    ├── env/                     # MineRL env spec + wrapper (dynamic ore + auto pickaxe)
    ├── monitor/                 # paper §3.2: TypedFeedback + ProgressScore + stagnation
    ├── inducer/                 # paper §3.2: BuildSkill / BuildRemedy from feedback
    ├── curator/                 # paper §3.3: validate / merge / retrieve external KB
    ├── adaptor/                 # paper §3.3: knowledge-conditioned local plan repair
    ├── planner/                 # high-level LLM planner + 5 OpenAI-compatible backends
    ├── executor/                # STEVE-1 loader + runner + craft helper
    ├── server/                  # FastAPI app (plan/action/monitor/induce/repair/...)
    ├── client/                  # HTTP client used by the env-side process
    ├── monitors/                # SuccessMonitor / StepMonitor for evaluation
    └── main.py                  # Hydra @main: Algorithm 1 main loop
```

This implementation is **completely independent**: no source files are copied from any other Minecraft agent repository. All third-party dependencies (MineRL, STEVE-1 / MineStudio, OpenAI SDK) are installed via `pip`.

---

## Installation

### 1. System prerequisites

- Linux or Windows (WSL2 / native both work).
- Python **3.10+**.
- Java **OpenJDK 8** (required by MineRL's MCP-Reborn build).
- An NVIDIA GPU is recommended for STEVE-1 inference.
- `git`, `clang` (Linux) or MSVC build tools (Windows).

### 2. Create environment

```bash
conda create -n mineevolve python=3.10 -y
conda activate mineevolve
pip install -r requirements.txt
pip install -e .
```

### 3. Install MineRL

We do not vendor MineRL. Install the official package from the MineRL Labs (the version that supports STEVE-1 weights, e.g. MineRL 1.0.x):

```bash
pip install minerl
# or, if you need the gym-based 0.4.x API:
# pip install "minerl>=0.4,<1.0"
```

If you need to (re)build the Java backend yourself, follow MineRL's official docs:
<https://minerl.readthedocs.io/>.

### 4. Install STEVE-1

Choose **one** of the two paths:

**(a) Recommended: MineStudio**

```bash
pip install MineStudio
```

The first call to `mineevolve.executor.steve_loader.load_steve_policy()` will pull `CraftJarvis/MineStudio_STEVE-1.official` from HuggingFace automatically; nothing needs to live in `checkpoints/`.

**(b) Original STEVE-1 package**

```bash
pip install "git+https://github.com/Shalev-Lifshitz/STEVE-1.git"
```

Then download:

| Weight                | Place at                             |
| --------------------- | ------------------------------------ |
| VPT 2x model          | `checkpoints/vpt/2x.model`           |
| STEVE-1 weights       | `checkpoints/steve1/steve1.weights`  |
| STEVE-1 prior         | `checkpoints/steve1/steve1_prior.pt` |
| MineCLIP attn weights | `checkpoints/mineclip/attn.pth`      |

See `checkpoints/README.md` for source URLs.

### 5. LLM API keys

The default backend is **Qwen Plus** via DashScope's OpenAI-compatible endpoint
(no source change needed — both `conf/evaluate.yaml` and the FastAPI server
boot in Qwen mode):

```bash
# DEFAULT: Qwen via DashScope
export DASHSCOPE_API_KEY=sk-...
```

Other supported backends (set the matching key + pass `llm=<name>` on the
command line):

```bash
export ZHIPUAI_API_KEY=...    # then: llm=glm_4_7
export GOOGLE_API_KEY=...     # then: llm=gemini_flash
export OPENAI_API_KEY=...     # then: llm=gpt_5_5
```

---

## Running

### Start the FastAPI server (one-time, GPU)

```bash
bash scripts/server.sh
# Windows:
scripts\server.bat
```

The server:
- loads STEVE-1 once,
- holds the external knowledge store K and feedback buffer B,
- executes Monitor / Inducer / Curator / Adaptor on each `/chat` request.

### Run a benchmark group (env-side process)

```bash
# DEFAULT: wooden tier with Qwen Plus (just needs DASHSCOPE_API_KEY)
bash scripts/run_eval.sh

# iron tier with Qwen Flash (cheaper)
bash scripts/run_eval.sh iron qwen_flash

# diamond tier with Qwen Plus
bash scripts/run_eval.sh diamond qwen_plus

# any tier with another vendor
bash scripts/run_eval.sh iron glm_4_7        # ZHIPUAI_API_KEY required
bash scripts/run_eval.sh iron gemini_flash   # GOOGLE_API_KEY required
bash scripts/run_eval.sh iron gpt_5_5        # OPENAI_API_KEY required
```

Per-task and aggregate results are printed via `rich.Table`. Per-episode logs and (optionally) videos go to `logs/eval/<date>/<time>/` and `videos/<date>/`.

### Evaluate a custom task subset

`conf/benchmark/<group>.yaml::evaluate` selects task ids; leave it `[]` for all 70 tasks. To run iron tasks #2 and #5 only:

```bash
python -m mineevolve.main benchmark=iron benchmark.evaluate='[2, 5]'
```

(Default `llm=qwen_plus` is taken from `conf/evaluate.yaml`; override with
`llm=qwen_flash` etc. if needed.)

---

## Method overview

MineEvolve converts each subgoal execution into typed feedback, induces skills (from successful segments) and remedies (from failed/stagnant segments), validates and retrieves them under a prompt budget, and repairs the unfinished plan suffix when failures repeat.

```
                  Algorithm 1 main loop
+-----------------------------------------------------+
| reset task -> initial plan                          |
| for each subgoal i:                                  |
|   (a) STEVE-1 executes z_i                           |
|   (b) Monitor builds typed feedback e_i (Eq. 1-3)    |
|   (c) Inducer -> skill | remedy candidates           |
|   (d) Curator validates (Eq. 6) and stores K         |
|   (e) if repeated failure or stagnation:             |
|         Adaptor freezes prefix, repairs suffix (Eq. 8)|
+-----------------------------------------------------+
```

See [docs/architecture.md](docs/architecture.md) for a per-module breakdown.

---

## License

MIT.

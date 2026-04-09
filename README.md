---
title: Dataops Env
emoji: 🧼
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# 🏋️ DataOps GYM

### *The Benchmark That Punishes Overconfidence — Not Just Wrong Answers*

**A semantic, step-based reinforcement learning environment for evaluating data-cleaning agents on tabular datasets**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Pydantic](https://img.shields.io/badge/Pydantic-Schema_Validation-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Spaces_Compatible-FFD21E)](https://huggingface.co/spaces)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

<br/>

> **"Any model can clean data. Only a smart one knows when *not* to."**
>
> DataOps GYM is an interactive gym environment for training and benchmarking LLM-based data-cleaning agents —
> with dense per-step rewards, structured action protocols, and deliberate adversarial traps
> designed to expose hallucination, overcorrection, and overconfidence.

<br/>

</div>

---

## 📌 Table of Contents

- [Why DataOps GYM Exists](#-why-dataops-gym-exists)
- [Core Philosophy](#-core-philosophy)
- [Architecture Overview](#-architecture-overview)
- [Repository Layout](#-repository-layout)
- [The Environment Model](#-the-environment-model)
- [Action Protocol](#-action-protocol)
- [Task Difficulty Tiers](#-task-difficulty-tiers)
- [Scoring & Reward System](#-scoring--reward-system)
- [HTTP API Reference](#-http-api-reference)
- [Quick Start](#-quick-start)
- [Docker Deployment](#-docker-deployment)
- [Contributing](#-contributing)

---

## 🔍 Why DataOps GYM Exists

Real-world data pipelines fail silently. Automated cleaners and LLM agents frequently:

- **Hallucinate corrections** — inventing plausible-sounding values with no evidentiary basis
- **Over-correct valid data** — mistaking unusual-but-correct formats as errors *(e.g., `q.xu+vip@example.com` is a valid plus-address — don't touch it)*
- **Flatten genuine ambiguity** — making irreversible decisions where `cannot_determine` was the right call
- **Ignore cross-record consistency** — fixing one row while silently creating a new constraint violation in another

**DataOps GYM was built to measure all of these failure modes simultaneously**, forcing agents to balance **precision, restraint, and consistency** — not just produce a tidy-looking output table.

---

## 🧠 Core Philosophy

| Traditional Benchmark | DataOps GYM |
|---|---|
| Compares final table to ground truth | Evaluates **every step** semantically |
| Rewards correct fixes | Also **penalizes hallucination** and **rewards appropriate abstention** |
| Single-pass evaluation | Multi-turn, stateful episode loop |
| No cross-record awareness | Tracks **consistency across related rows** |
| Ignores agent confidence | **Confidence calibration** affects reward directly |
| `cannot_determine` = failure | `cannot_determine` = **first-class correct action** |

> DataOps GYM is purpose-built around the insight that **knowing when not to act is as important as knowing how to act.**

---

## 🏗 Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                          DataOps GYM                             │
│                                                                  │
│   ┌──────────┐      ┌──────────────┐      ┌──────────────┐      │
│   │ task.py  │─────▶│    env.py    │─────▶│  grader.py   │      │
│   │          │      │              │      │              │      │
│   │  Task    │      │   Episode    │      │  Per-step    │      │
│   │ Factory  │      │  Lifecycle   │      │  Reward  +   │      │
│   │ 3 tiers  │      │  + State     │      │  Final Score │      │
│   │ 2 vars   │      │  Tracking    │      │              │      │
│   └──────────┘      └──────────────┘      └──────────────┘      │
│                            │                                     │
│                            ▼                                     │
│                     ┌──────────────┐                             │
│                     │  models.py   │                             │
│                     │  Action /    │                             │
│                     │  Observation │                             │
│                     │  (Pydantic)  │                             │
│                     └──────────────┘                             │
│                            │                                     │
│                            ▼                                     │
│                     ┌──────────────┐      ┌──────────────────┐  │
│                     │   server/    │◀─────│  inference.py    │  │
│                     │   app.py     │      │  Reference Agent │  │
│                     │  (FastAPI)   │      │  / Evaluator     │  │
│                     └──────────────┘      └──────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

Every layer is cleanly separated — the environment knows nothing about the HTTP layer; the grader knows nothing about environment internals. Each component is independently testable and swappable.

---

## 📁 Repository Layout

```
DataOps-GYM/
│
├── env.py                       # Core RL environment: reset / step / observe / metrics
├── task.py                      # Task factories: easy / medium / hard (2 variants each)
├── grader.py                    # Per-step reward math + final task score formula
├── models.py                    # Pydantic schemas: Action, Observation
├── inference.py                 # Reference baseline agent + evaluator script
│
├── server/
│   └── app.py                   # FastAPI HTTP server (/reset, /step, /state, /health)
│
├── utils/                       # Shared helper utilities
├── .dataops_policy_cache.json   # Cached policy artifacts
│
├── Dockerfile                   # Container definition (port 7860, HF Spaces-ready)
├── .dockerignore
├── openenv.yaml                 # HuggingFace Spaces metadata
├── pyproject.toml               # Project metadata & build configuration
├── requirements.txt             # Python dependencies
└── uv.lock                      # Reproducible lock file for uv package manager
```

---

## ⚙️ The Environment Model

### Episode Lifecycle

Every interaction follows the standard gym pattern:

```python
# 1. Initialize a task episode (easy / medium / hard, seeded for reproducibility)
obs = env.reset(task_name="hard", seed=42)

# 2. Agent acts step-by-step until done
while not done:
    action = agent.decide(obs)
    obs, reward, done, info = env.step(action)

# 3. Retrieve terminal score in range (0, 1)
final_score = info["final_task_score"]
```

When `task_name` is not fixed, the environment randomly samples a difficulty tier and variant (both seeded), making the benchmark resistant to test-set memorization.

---

### What the Agent Sees — `Observation`

The observation gives the agent everything it needs to reason — without ever revealing the hidden answer key:

| Field | Description |
|---|---|
| `dataset.original` | Immutable snapshot of the table at episode start |
| `dataset.modified` | Current working table reflecting all accepted fixes so far |
| `action_history` | Full sequence of all past actions taken this episode |
| `per_record_scores` | Cumulative score contribution per row ID |
| `current_iteration_score` | Score delta from the most recent step |
| `previous_iteration_score` | Score delta from the prior step (for trend awareness) |
| `steps_remaining` | Hard cap on remaining interactions |

> ⚠️ The agent **never** sees `hidden_issues`. All semantic evaluation is performed internally.

---

### Hidden Issues — What's Lurking in the Data

Each task defines a set of typed hidden issues the agent must discover and resolve:

| Issue Type | Description | Fixable? |
|---|---|---|
| `duplicate` | Two rows represent the same real entity | ❌ Not by `fix_value` alone |
| `missing_value` | A required field is null | ✅ Yes |
| `invalid_format` | Email / phone / date doesn't match expected pattern | ✅ Yes |
| `inconsistent_casing` | Name or city uses wrong casing convention | ✅ Yes |
| `conflict` | Same customer has contradictory field values across rows | ❌ Irreconcilable |
| `constraint_violation` | Two distinct rows violate a uniqueness constraint (e.g., same email) | ❌ Requires judgment |
| `valid_trap` | Row looks suspicious but is actually correct — **do not touch** | N/A |

---

## 🎮 Action Protocol

Agents interact through a strict, typed JSON protocol validated by Pydantic:

```json
{
  "action_type": "fix_value",
  "record_id": "C201",
  "field": "email",
  "value": "evan.cole@example.com",
  "confidence": 0.92
}
```

### Action Types

| Action | When to Use | Reward Signal |
|---|---|---|
| `detect_issue` | Flag a problem without yet resolving it | Low positive — passive identification only |
| `fix_value` | Apply a concrete correction to a specific field | High positive if correct; severe penalty if hallucinated |
| `cannot_determine` | Abstain when conflict is genuinely irreconcilable | Rewarded when `fixable: false`; penalized otherwise |
| `skip` | Explicitly pass on a record/field | Penalized if a real issue existed there |

### Protocol Validation Rules

- `value` is **required** for `fix_value` and **forbidden** for all other action types
- `record_id` and `field` must be non-empty strings
- `confidence` must be a float in `[0.0, 1.0]`

### Behavioral Discipline

The environment enforces **follow-through discipline** across steps:

- After `detect_issue`, the agent must follow up on that same record/field before moving on — or receive a `passive_penalty`
- Handling a duplicate/conflict pair inconsistently (different strategies for related rows) triggers `inconsistent_handling` penalty
- Re-flagging an already-detected issue yields `repeated_detection` penalty

---

## 📊 Task Difficulty Tiers

### 🟢 Easy — `easy_cleaning_task`

**Scenarios:** `easy_customer_master`, `easy_vendor_onboarding`

**Goal:** Foundational hygiene — deduplicate obvious duplicate rows and fill required missing values without deleting rows just because they are incomplete.

**Issues planted:**
- Exact duplicate rows (identical across all fields)
- Missing required values (`city`, `email`)

**Agent strategy:** Detect duplicates → deduplicate → fill missing fields. No traps. No ambiguity.

**Max steps:** 7 &nbsp;|&nbsp; **Variants:** 2

---

### 🟡 Medium — `medium_normalization_task`

**Scenarios:** `medium_customer_normalization`, `medium_partner_directory`

**Goal:** Normalize — consistent casing, valid email shapes, deduplication where needed.

**Issues planted:**
- Duplicate rows
- Inconsistent casing on `name` and `city` (e.g., `"OMAR HASSAN"` → `"Omar Hassan"`)
- Invalid email tokens (e.g., `[at]` instead of `@`, missing `@` entirely)

**Agent strategy:** Normalize casing to `title_case`, repair malformed emails, deduplicate. Validators check format correctness, not just non-null values.

**Max steps:** 9 &nbsp;|&nbsp; **Variants:** 2

---

### 🔴 Hard — `hard_conflict_resolution_task`

**Scenarios:** `hard_customer_conflicts`, `hard_account_merges`

**Goal:** Multi-way reasoning under adversarial traps — deduplicate, handle irreconcilable conflicts, enforce unique constraints, fix formats, and **leave valid-looking unusual rows completely untouched**.

**Issues planted:**
- Exact duplicates
- **Irreconcilable conflicts** — same customer ID with contradictory `age` values (e.g., `250` vs `45`). Correct answer: `cannot_determine`
- Invalid email and phone formats
- **Unique constraint violations** — two distinct customers sharing the same email address
- **`valid_trap` rows** — rows that look suspicious but are correct:
  - `q.xu+vip@example.com` — a valid RFC-compliant plus-address
  - `A. J. Brown` — a valid abbreviated name

**Agent strategy:** Nuanced multi-step reasoning, cross-record constraint checking, confident abstention, and deliberate non-intervention on valid traps.

**Max steps:** 14 &nbsp;|&nbsp; **Variants:** 2

---

## 🏆 Scoring & Reward System

### Per-Step Reward — `grade_step_details`

Each step produces a composite scalar reward (no clamping — scores can go negative):

| Component | Condition | Δ Score |
|---|---|---|
| **Classification** | Correct action type for the situation | `+0.1` (detect) / `+0.2` (fix or cd) |
| **Classification** | Wrong action type | `−0.20` |
| **Issue Detection** | Correctly identified real issue | `+0.05` (detect) / `+0.15` (fix or cd) |
| **Issue Detection** | Missed a real issue | `−0.15` |
| **Issue Detection** | False positive (no issue there) | `−0.05` |
| **Decision** | Correct fix (passes `validate_fix`) | `+0.25` |
| **Decision** | Correct `cannot_determine` on non-fixable issue | `+0.25` |
| **Decision** | Hallucinated fix (no matching issue) | `−0.50` |
| **Decision** | Wrong fix (fails validation) | `−0.40` |
| **Decision** | Wrong `cannot_determine` (abstained when fixable) | `−0.20` |
| **Cross-record Consistency** | Consistent handling of related row pair | `+0.20` |
| **Cross-record Consistency** | Inconsistent handling of related row pair | `−0.30` |
| **Confidence Calibration** | confidence > 0.7 AND correct | `+0.05` |
| **Confidence Calibration** | confidence > 0.7 AND wrong | `−0.10` |
| **Confident Hallucination** | confidence > 0.8 AND hallucinated fix | `−0.20` (amplifier) |
| **Resolution Reward** | Previously detected issue now resolved | `+0.15` |
| **Passive Penalty** | Unresolved detection + off-topic action | `−0.05` |
| **Overcorrection** | Extra fields modified unintentionally | `−0.05 × N` |
| **Repeated Detection** | Same issue flagged again | `−0.10` |

> The returned step reward also adjusts by **±0.1** based on whether the sum of `per_record_scores` improved over the previous iteration.

---

### Final Task Score — `grade_task_result`

Terminal score is a weighted composite guaranteed in the open interval **(0, 1)**:

```
Final Score =  0.50 × normalized_record_score
             + 0.20 × (1 − hallucination_rate)
             + 0.15 × uncertainty_accuracy
             + 0.15 × consistency_score
```

| Metric | What It Measures |
|---|---|
| `normalized_record_score` | Average per-record cumulative reward, mapped to [0, 1] |
| `hallucination_rate` | Fraction of `fix_value` actions that had no evidentiary basis |
| `uncertainty_accuracy` | Fraction of `cannot_determine` actions that were genuinely correct |
| `consistency_score` | Fraction of related-row pair decisions handled with consistent strategy |

> 📌 **Scoring above 0.85 on the hard tier requires genuine multi-step reasoning — not pattern matching.**

### Failure Telemetry

The `task_failure_messages` function surfaces structured, human-readable failure logs from the episode — making it straightforward to diagnose specific agent failure modes during evaluation and iteration.

---

## 🌐 HTTP API Reference

The FastAPI server exposes a clean REST interface for agent integration:

| Endpoint | Method | Body / Params | Description |
|---|---|---|---|
| `/reset` | `POST` | `{ "seed": 42, "task_name": "hard" }` | Start a new episode |
| `/step` | `POST` | JSON matching `Action` schema | Submit one agent action |
| `/state` | `GET` | — | Full internal state snapshot (debugging) |
| `/health` | `GET` | — | Liveness probe |
| `/docs` | `GET` | — | Interactive Swagger UI |

### Example: Full Episode via REST

```python
import requests

BASE = "http://localhost:7860"

# 1. Reset — start a seeded hard episode
obs = requests.post(f"{BASE}/reset", json={"seed": 0, "task_name": "hard"}).json()

# 2. Step loop
done = False
while not done:
    action = {
        "action_type": "detect_issue",
        "record_id": "21",
        "field": "email",
        "confidence": 0.88
    }
    result = requests.post(f"{BASE}/step", json=action).json()
    print(f"Reward: {result['reward']:.3f} | Done: {result['done']}")
    done = result["done"]

# 3. Final score
state = requests.get(f"{BASE}/state").json()
print(f"Final Score: {state['final_task_score']:.4f}")
```

---

## 🚀 Quick Start

### Option 1 — Standard pip

```bash
# Clone the repository
git clone https://github.com/graheetphartyal23/DataOps-GYM.git
cd DataOps-GYM

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
python -m server.app

# In a separate terminal — run the reference baseline agent
python inference.py
```

The API server starts at **`http://localhost:7860`**
Interactive docs available at **`http://localhost:7860/docs`**

---

### Option 2 — Using `uv` (faster installs)

```bash
uv sync
uv run python -m server.app
```

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t dataops-gym .

# Run the container
docker run -p 7860:7860 dataops-gym
```

The `Dockerfile` and `openenv.yaml` are pre-configured for **one-click deployment to HuggingFace Spaces**.

---

## 🤝 Contributing

Contributions are welcome — especially new task domains, improved validation logic, or smarter baseline agents.

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/DataOps-GYM.git

# 2. Create a feature branch
git checkout -b feature/new-task-domain

# 3. Make your changes, then open a Pull Request
```

**When adding new tasks**, please follow the `TaskDefinition` TypedDict schema in `task.py` and ensure every new task includes:
- `initial_table` with realistic domain data
- `hidden_issues` with typed, structured issue definitions
- `expected_outcome` with validation rules
- At least 2 variants (`variant_count = 2`)

---

## 📄 License

This project is open source. See [LICENSE](LICENSE) for complete terms.

---

<div align="center">

<br/>

**Built to make data-cleaning agents honest — not just accurate.**

*The only benchmark where doing nothing is sometimes the right answer.*

<br/>

⭐ **Star this repo** if DataOps GYM helped your research or evaluation work!

<br/>

</div>
 

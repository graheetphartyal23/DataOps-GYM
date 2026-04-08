---
title: Dataops Env
emoji: 🧼
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# 🧼 DataOps Gym

> Semantic, step-based benchmark environment for data cleaning agents.  
> Agents interact with tabular datasets through a strict JSON action protocol, receive dense per-step rewards, and are scored on per-record behavior, hallucination rate, appropriate abstention, and cross-record consistency.

---

## 📌 Why this project exists

Automated cleaners and LLM agents often “fix” data without evidence, invent values, or flatten ambiguity. DataOps Gym evaluates whether an agent knows when not to change data—not only whether the final table looks tidy.

---

## 🗂️ Repository layout (core)

Path Role

env.py DataOpsEnv: reset/step loop, issue matching, fix validation, metrics, observations

task.py Task factories: easy_cleaning_task, medium_normalization_task, hard_conflict_resolution_task (+ variants)

grader.py grade_step_details (per-step reward), grade_task_result (final score in
[
0
,
1
]
[0,1])

models.py Pydantic Action, Observation schemas

server/app.py HTTP API: /reset, /step, /state, /health

inference.py Reference / baseline agent that talks to the environment


---

## ⚙️ Environment model

### Episode lifecycle


reset() — Instantiates a task from the registry, copies initial_table into dataset_modified, and initializes counters (steps_remaining, per_record_scores, failure logs, etc.).

step(action) — Validates the action, evaluates it against hidden ground-truth issues, updates the working table when appropriate, updates per-record cumulative scores and global metrics, returns an Observation, a scalar reward, done, and info (including final_task_score snapshot).


On reset, if task_name is not fixed, the environment picks easy / medium / hard at random (seeded). Within each difficulty, a variant index in {0, 1} is chosen randomly (each factory exposes variant_count = 2).

---

## 👀 What the agent sees (Observation)


From 433:444:g:\DataOps Gym\dataops-env\env.py

def _build_observation(self) -> Observation:
return Observation(
dataset={
"original": deepcopy(self._state_data["dataset_original"]),
"modified": deepcopy(self._state_data["dataset_modified"]),
},
action_history=deepcopy(self._state_data["action_history"]),
per_record_scores=deepcopy(self._state_data["per_record_scores"]),
current_iteration_score=float(self._state_data["current_iteration_score"]),
previous_iteration_score=float(self._state_data["previous_iteration_score"]),
steps_remaining=int(self._state_data["steps_remaining"]),
)


The agent does **not** receive `hidden_issues`; evaluation uses them internally.

---

## 🧩 Hidden issues (`task.py`)

Each task defines:

- **`initial_table`**: rows with `row_id` and domain fields  
- **`hidden_issues`**: typed issues (duplicate, missing_value, invalid_format, inconsistent_casing, conflict, constraint_violation, valid_trap, etc.)  
- **`expected_outcome`**: human-readable success criteria (the environment’s **mechanical** grading is driven by issue matching + `validate_fix`, not by parsing this prose)  
- **`max_steps`**: hard cap on interactions  
- **`fixable`**: inferred when absent—issues of types `duplicate`, `conflict`, and `constraint_violation` default to **not fixable** by a single `fix_value` (see `_with_fixable_flags`)  

---

## 🔄 Action protocol


class Action(BaseModel):
...
action_type: Literal[
"detect_issue",
"fix_value",
"cannot_determine",
"skip",
]
record_id: str
field: str
value: Optional[str]
confidence: float


Rules: value is required for fix_value and forbidden for other types. record_id and field are non-empty strings.

---

## 🧠 How each action type is interpreted

### Action Behavior

**skip**  
If a real issue exists for that (record_id, field), counts as missed issue + passive penalty; unresolved tracking may apply.

**detect_issue**  
If an issue matches: marks correct detection but is treated as passive (you identified but did not resolve). Re-detecting the same issue triggers repeated detection penalty. False positives: classification incorrect / false issue.

**cannot_determine**  
Correct only when the matched issue exists and fixable is false (e.g. irreconcilable conflict). Otherwise wrong_cannot_determine.

**fix_value**  
Hallucinated fix if row/field invalid, no matching issue, issue already resolved, or fix would break cross-record rules.

---

## 📊 Tasks in detail

### Easy — easy_cleaning_task

Goal: foundational hygiene—deduplicate obvious duplicate rows and fill required missing values without deleting rows only because they are incomplete.

Representative issues: duplicate pair + missing city / email on other rows.  
Variants: e.g. easy_customer_master, easy_vendor_onboarding (row IDs and entity names change; structure is the same).  
Typical agent moves: detect_issue on duplicates and nulls, then fix_value to populate required fields; handle duplicate rows via fixes that leave one representative row (exact table edits depend on agent policy).

---

### Medium — medium_normalization_task

Goal: normalization—consistent casing, valid email shapes, deduplication where needed.

Representative issues: lower/upper case names and cities, invalid email tokens (e.g. [at] instead of @), duplicate rows under same business key.  
Validator highlights: invalid_format for email/phone/date fields; inconsistent_casing expects .title() on the corrected field.

---

### Hard — hard_conflict_resolution_task

Goal: multi-way reasoning under traps—dedupe, non-fixable conflicts (e.g. contradictory ages for same customer), unique email violations across rows, invalid formats, plus valid_trap rows that must not be “corrected” for looking odd (plus-address email, abbreviated name).

cannot_determine is first-class here: e.g. conflict issues with fixable: False reward proper abstention.  
Structural issues (duplicate, constraint_violation) are not honestly closed by a single naive fix_value; the environment treats them as non-fixable by default—agents should align with cannot_determine or a coherent multi-step strategy that respects consistency checks.

---

## 📈 Scoring

Per-step reward (grade_step_details)

Each step produces a raw score contribution (can be negative) from labeled outcomes: classification, detection, decision quality, passive behavior, repeated detection, overcorrection, cross-record consistency, confidence calibration, amplification for confident hallucinations, and resolution when a prior detection is actually resolved.

---

## 🚀 Quick start


pip install -r requirements.txt
python -m server.app
python inference.py


---

## 🧠 Design principles (operational)

- Strict schema  
- Penalize invented edits  
- Reward calibrated abstention  
- Cross-record integrity  
- Transparent decomposition  

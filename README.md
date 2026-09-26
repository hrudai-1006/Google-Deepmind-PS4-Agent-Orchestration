# Aegis: Autonomous Orchestration with Managed Agents

**Hackathon Entry:** Google DeepMind Hyderabad Hackathon  
**Problem Statement 4:** Autonomous Orchestration with Managed Agents  
**Base Model:** `antigravity-preview-09-2026` (Fallback: `antigravity-preview-05-2026`)  
**SDK / API:** `google-genai` (Vertex AI Agents API & Interactions API)

---

## 1. System Architecture

Aegis is an autonomous multi-agent mission control system that demonstrates genuine multi-agent collaboration, stateful session persistence, tool execution in sandboxed environments, long-horizon state tracking, and two fundamentally distinct autonomous recovery loops.

```
                      +---------------------------------------+
                      |       AEGIS MISSION CONTROL UI        |
                      |     (Rich Live Terminal Dashboard)    |
                      +-------------------+-------------------+
                                          |
                      +-------------------+-------------------+
                      |          ORCHESTRATOR         |
                      |     (State & Ledger Engine)           |
                      +-------------------+-------------------+
                                          |
         +--------------------------------+--------------------------------+
         |                                |                                |
         v                                v                                v
+------------------+             +------------------+             +------------------+
|  research-agent  |             |  analyst-agent   |             |   writer-agent   |
| (Search, Context)|             |(Code Exec, Files)|             |   (Filesystem)   |
+--------+---------+             +--------+---------+             +--------+---------+
         |                                |                                |
         | Findings                       | Quantitative Angle             | Report Draft
         +--------------------------------+--------------------------------+
                                          |
                                          v
                                 +------------------+
                                 |  auditor-agent   |
                                 | (Pure Reasoning) |
                                 +--------+---------+
                                          |
                        +-----------------+-----------------+
                        |                                   |
                  [If Approved]                       [If Rejected]
                        v                                   v
               Final /report.md               Loop 2: Writer Session Revision
                                                (Same environment_id & prev_ix)
```

### The Four Managed Agents

| Agent ID | Base Model | Enabled Tools | Operational Role |
| :--- | :--- | :--- | :--- |
| **`research-agent`** | `antigravity-preview-09-2026` | `google_search`, `url_context` | Investigates the specified topic; returns 5–8 concise, factual, well-sourced bullet findings with verified citations. |
| **`analyst-agent`** | `antigravity-preview-09-2026` | `code_execution`, `filesystem` | Consumes research text (not a pre-baked file) and runs Python code in an isolated sandbox to compute growth metrics, CAGRs, and ratios. |
| **`writer-agent`** | `antigravity-preview-09-2026` | `filesystem` | Synthesizes research and quantitative metrics into a ~250-word executive briefing; writes `/report.md` via filesystem tool. |
| **`auditor-agent`** | `antigravity-preview-09-2026` | *None* (Pure Reasoning) | Compares every claim in the report against ground-truth findings. Rejects ungrounded claims with specific evidentiary critique. |

---

## 2. Two Distinct Recovery Loops

Aegis deliberately distinguishes between execution tool failures and quality/verification failures. Both loops operate independently and surface distinct telemetry:

### Loop 1: Tool / Data Failure (`analyst-agent`)
* **Trigger:** The orchestrator deliberately instructs the analyst to read `/data/findings.txt` (a non-existent file path).
* **Detection:** The agent’s code execution environment returns `FileNotFoundError`. The orchestrator inspects the response for failure signatures (`error`, `failed`, `not found`, `no such file`).
* **Telemetry & Ledger:** Marked as `FAILED`. Mission control displays:
  ```
  FAILURE DETECTED -> diagnosing -> recovering
  ```
* **Recovery Mechanism:** Rather than abandoning the session or creating a fresh instance, the orchestrator issues a corrective follow-up inside the **SAME session** by reusing `environment_id` and `previous_interaction_id`, passing the real findings inline.
* **Outcome:** The analyst executes Python code over the inline data and transitions to `RECOVERED`.

### Loop 2: Quality & Verification Failure (`auditor-agent` -> `writer-agent`)
* **Trigger:** The writer-agent’s initial draft contains an ungrounded or exaggerated assertion (e.g., an unverified `150% YoY adoption rate`).
* **Detection:** The zero-tool `auditor-agent` compares the draft against the ground-truth research bullets, identifies the unsupported claim, and outputs `REJECTED` with the specific reason.
* **Telemetry & Ledger:** Marked as `FAILED`. Mission control displays:
  ```
  CLAIM REJECTED -> revising -> re-auditing
  ```
* **Recovery Mechanism:** The orchestrator delegates the objection back to the **writer-agent’s persistent session** (`environment_id`, `previous_interaction_id`) with the exact auditor critique. The writer removes the unsupported metric, updates `/report.md`, and resubmits to the auditor.
* **Outcome:** The auditor re-evaluates the revised draft, verifies all claims against findings, and returns `APPROVED`, transitioning the step to `RECOVERED` (or `NEEDS_HUMAN_REVIEW` if rejected twice).


## 4. Setup & Installation

### Prerequisites
* Python 3.10+
* Google Cloud SDK (`gcloud`) or Gemini API credentials (for live mode)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Environment Configuration
Configure your target Google Cloud Project and Vertex AI region:
```bash
# Required for live Vertex AI mode
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"

# Optional overrides (defaults shown)
export GOOGLE_CLOUD_LOCATION="global"
export ANTIGRAVITY_MODEL="antigravity-preview-09-2026"  # or antigravity-preview-05-2026
```

---

## 5. How to Run

### Step 1: Provision the Managed Agents (Idempotent)
Provisions the 4 managed agents on Vertex AI. If an agent already exists, it reports the existing ID and continues safely:
```bash
# Dry-run validation (no credentials required)
python3 create_agents.py --dry-run

# Live Vertex AI provisioning
python3 create_agents.py --project "$GOOGLE_CLOUD_PROJECT"
```

### Step 2: Run the Mission Control Orchestrator
Launches the full live terminal dashboard, executes the pipeline, tracks transitions in `ledger.json`, and outputs `/report.md`:
```bash
# Dry-run simulation mode (demonstrates live UI, both loops, and ledger generation offline)
python3 orchestrator.py --dry-run

# Live Vertex AI execution
python3 orchestrator.py --project "$GOOGLE_CLOUD_PROJECT"

# Custom topic execution
python3 orchestrator.py --project "$GOOGLE_CLOUD_PROJECT" --topic "Next-Gen Quantum Error Correction: Commercial Benchmarks & 2026 Timelines"
```

---

## 6. Project Artifacts
* **`create_agents.py`**: Idempotent agent provisioning script utilizing `google-genai` Agents API.
* **`orchestrator.py`**: Main coordinator, terminal UI, interactions client, and recovery loop manager.
* **`ledger.json`**: Machine-readable JSON ledger tracking every agent transition and recovery milestone.
* **`report.md`**: Verified executive investor briefing generated and validated by the agent swarm.
* **`requirements.txt`**: Project dependencies (`google-genai`, `rich`).

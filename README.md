# HCCHordex

Interpretable prediction of hepatocellular carcinoma recurrence and recurrence timing via case-grounded multimodal multi-agent reasoning.

## Overview

HCCHordex is a case-grounded multi-agent framework that combines modality-specific specialist agents, retrieval-augmented historical case evidence, and cross-modal reasoning to predict both recurrence status and time-to-recurrence for HCC patients.

The system processes three complementary modalities — clinical-pathology records, proteomic expression profiles, and multi-phase MRI — through independent specialist agents. An attending agent synthesizes their reports into a patient-level integrated assessment. During inference, a recurrence-classification agent and a recurrence time agent retrieve analogous historical cases from a multimodal case bank and combine them with the current patient's attending report to produce evidence-grounded predictions.

## Agents

| Agent | File | Role |
|-------|------|------|
| Clinical Agent | `agents/clinical_agent.py` | Extracts structured pathological and laboratory variables into three profiles: tumor biology, surgical curability, and hepatic background |
| Proteomic Agent | `agents/proteomics_agent.py` | Processes tissue proteomic profiles via XGBoost + SHAP, with UniProt/KEGG annotation interpreted by LLM |
| Imaging Agent | `agents/imaging_agent.py` | Generates per-sequence MRI descriptions using HuluMed-7B vision-language model |
| Attending Agent | `agents/attending_agent.py` | Synthesizes modality-specific traces into a patient-level integrated report via structured 5-step reasoning |
| Self-Critique | `agents/self_critique.py` | Validates attending reports and triggers revision when quality criteria are not met |
| Recurrence Agent | `agents/recurrence_agent.py` | Evidence-grounded case review: retrieves analogous cases and predicts recurrence status with supporting evidence |
| Recurrence Time Agent | `agents/recurrence_time_agent.py` | Estimates time-to-recurrence using the same retrieved case set, independent of the binary decision |
| Report Aggregator | `agents/report_aggregator.py` | Aggregates multi-modal reports into structured summaries for case bank storage and retrieval |

## Workflow

| File | Role |
|------|------|
| `graph/state.py` | LangGraph state schema defining the full diagnostic pipeline state |
| `graph/workflow.py` | LangGraph workflow orchestrating agent execution, case bank construction, and retrieval-augmented inference |

## Key Features

- **Modality-robust**: Each specialist agent operates independently; the attending agent uses available evidence without requiring imputation when a modality is missing
- **Evidence-grounded**: Predictions are accompanied by retrieved historical case analogues, not just scores
- **Inspectable**: Every prediction includes a chain-of-thought reasoning trace from each specialist agent and the attending synthesis
- **Case bank**: Multimodal historical cases stored with structured profiles, chain-of-thought traces, and ground-truth outcomes

## Dependencies

```
langchain >= 0.2.0
langchain-openai >= 0.1.0
langchain-core >= 0.2.0
langgraph >= 0.1.0
pandas >= 2.0.0
numpy >= 1.24.0
scikit-learn >= 1.3.0
xgboost >= 2.0.0
shap >= 0.44.0
chromadb >= 0.4.0
openai >= 1.30.0
transformers >= 4.40.0
torch >= 2.0.0
```

## Configuration

The agent code expects a `config.py` module providing:

- `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` — backbone LLM (default: Qwen3.6-Plus via DashScope)
- `EMBEDDING_API_KEY`, `EMBEDDING_MODEL` — embedding model for case bank retrieval
- `HULUMED_MODEL_PATH` — path to HuluMed-7B for imaging agent
- Data and output directory paths

## License

This repository contains the agent architecture and workflow code. Data and model weights are not included.

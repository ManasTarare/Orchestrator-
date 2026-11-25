# Mornork rebuild plan

## Product direction

Rebuild Mornork as a local-first platform for creating, evaluating, and composing language models. Users can choose a pretrained checkpoint for fine-tuning or create a randomly initialized Transformer. Expert training runs on the user's PC and reports progress in the UI. Workflows combine only evaluated, deployable expert versions.

The three supplied module specs are the product requirements for this rebuild:

- **Experts** owns expert profiles, datasets, training approach and architecture, training jobs, evaluation, model versions, and deploy readiness.
- **Orchestrator** is the control plane and observability page for build/evaluation/deployment runs, their stages, logs, metrics, and health. It is distinct from the AI orchestrator model.
- **Workflows** composes an orchestrator model, registry-selected experts, routing policy, optional escalation model, aggregator, and response configuration into a versioned pipeline.

## Terminology and model choices

"Build from scratch" means initialize model weights randomly. It does not reuse the Qwen checkpoint currently in `artifacts/mornork-0.6b`. It requires a tokenizer and substantial, legally usable training text. A prompt/response file alone is not a general pretraining corpus.

The AI orchestrator has two selectable approaches:

1. **Fine-tune a pretrained model**: choose a compatible base checkpoint and adapt it to routing/planning examples.
2. **Train from scratch**: train a compact Transformer on a text corpus, then teach its structured routing output with labeled examples.

Experts have the same two training approaches. The architecture selector is **Transformer** first; **Mixture of Experts** is a later architecture option after the training, checkpoint, inference, and memory paths for dense Transformers are stable. This preserves the spec's MoE choice without pretending it is already implemented.

Training from scratch has two distinct data stages:

- **Pretraining corpus**: clean, deduplicated text for next-token prediction. Use data the user owns or data with a license that permits this use.
- **Instruction/evaluation examples**: domain prompts and reviewed answers, with train/validation/test partitions kept separate. These teach the model the response behavior and measure task performance.

For Indian finance/tax experts, the data workflow should retain source URLs, section/page references, tax period, and reviewer/approval fields. The corpus must preserve legal time context: the Income Tax Department says the Income-tax Act, 2025 applies from 1 April 2026 / Tax Year 2026–27 onward, while earlier periods continue under the 1961 Act. Public statutes, rules, circulars, and RBI material are source material, not automatically reviewed answer labels. Do not mark a model deployable merely because it trained on those documents.

## Architecture

### Local application

- Keep the current FastAPI service as the local API and UI host, but split its responsibilities into application, storage, training, inference, workflow, and orchestration modules.
- Use a local SQLite database as the source of truth for experts, datasets, model versions, jobs, stages, metrics, workflow versions, test runs, and audit events. Keep large datasets and checkpoint weights as files under configured local data/artifact directories; store metadata and relative paths in SQLite.
- Run long training jobs in a dedicated local worker process, separate from the web request handler. The browser can close and reopen while a job continues. A single-GPU scheduler prevents simultaneous jobs from exhausting the 6 GB RTX 3050. Stream persisted job events to the UI using server-sent events or polling.
- Keep secrets in `.env` for this single-user local setup and never put tokens or full message text into routine run logs. Provide `.env.example` with safe defaults and ignore `.env` and uploaded data in version control.

### Model registry and routing

- Save every model version with approach, architecture, tokenizer, training-data version, config, evaluation report, and artifact path.
- Runtime selection uses the active orchestrator version from the registry. If a routed expert is ready, load/call that version. Otherwise use the configured escalation model or Mornork fallback and label the source of the answer.
- The AI orchestrator returns a validated structured route (expert IDs, rationale, and score); deterministic policy checks remain in the service so an unconstrained model cannot call arbitrary paths or unready experts.

### Dataset and training lifecycle

1. Upload supported CSV, JSONL, TXT, and PDF files into a per-project local data directory; show parsing and schema errors before training.
2. Preview records, remove duplicates/empty rows, validate required fields, and create reproducible train/validation/test splits. Never evaluate on training rows.
3. Show a preflight summary: selected GPU/CPU, available VRAM, parameter count, sequence length, batch/accumulation settings, expected disk output, and warnings. Block configurations that cannot fit; offer smaller settings.
4. Queue a versioned run and show stages: ingest, clean/split, tokenizer, model initialization or base load, training, evaluation, registry save.
5. Persist step/epoch, loss, validation loss, elapsed time, ETA, GPU memory, logs, and checkpoints. Support cancel and resume from checkpoints.
6. Mark a version Ready to Deploy only after evaluation passes and required review gates are satisfied. Keep the documented 85% threshold configurable, but define and report the scoring method by task; an arbitrary scalar is not evidence of expert quality.

## UI modules

### Experts

- List/search/filter profiles and versions; create an expert wizard with name, domain, description, dataset upload, training approach, architecture, and advanced settings.
- Let the user inspect data, select splits, review a VRAM/time estimate, and start training.
- Provide a persistent job view with stage, percentage, loss charts, recent logs, elapsed time/ETA, GPU usage, cancel, and resume.
- Show evaluation report and readiness reason; block workflow use until eligible.

### Orchestrator

- Show runs from expert and orchestrator-model builds, evaluations, and deployments.
- Filter on training approach (fine-tune/scratch), architecture, expert, version, type, status, and date.
- Display stage timeline, streaming logs, loss/evaluation charts, resource usage, health, and failure summaries. This is read-only for jobs; users start/cancel/retry from the owning builder.

### Workflows

- List and version workflows; use a builder for Request → Orchestrator → Registry/Experts → optional Escalation → Aggregator → Response.
- Support semantic, rule-based, broadcast, and hybrid routing; sequential/parallel execution; top-K, timeouts/retries; confidence thresholds; aggregation and response format.
- Test with one prompt or an uploaded held-out dataset and show a node-by-node trace. Publish only a valid version whose experts are still eligible and whose required test gates pass. Editing a published workflow creates a draft version.

### Chat

- Send one request through the published/local workflow. Show routing, each expert's answer and source/version, any escalation, and the final answer with rendered Markdown. Clearly label prototype, fine-tuned, and from-scratch models.

## Implementation sequence

1. **Foundation and migration**: inventory the current app, add SQLite storage/migrations, model/version schemas, `.env` configuration, and preserve the current chat while moving state to repositories.
2. **Training slice**: dataset upload/validation/splitting, background worker, dense Transformer scratch training, compatible base-model fine-tuning, saved checkpoints, cancellation/resume, and live progress UI.
3. **Evaluation and registry**: held-out evaluation, reports, configurable readiness gate, model version selection, and runtime loading. Define the scoring method before treating the 85% gate as meaningful.
4. **Orchestrator observability**: persistent run/stage/event/metric records, filters, streamed logs, health rules, and exports.
5. **Experts UI**: replace placeholder profile editing with the creation/training/evaluation lifecycle and readiness indicators.
6. **Workflow builder and execution**: versioned workflow graph/config, registry integration, router, expert calls, escalation, aggregation, test traces, publish/deactivate lifecycle.
7. **Release and data governance**: dataset/license metadata, source references, redaction, artifact cleanup/retention, checkpoint compatibility, and deployment documentation.

## Current implementation in this workspace

- The Training Studio now supports a randomly initialized dense Transformer with a trained tokenizer, or LoRA fine-tuning of a compatible pretrained causal LM. Expert and AI orchestrator are separate job targets.
- Dataset ingestion supports TXT, CSV, JSONL, and selectable-text PDF; the UI can preview records. `approved` filters reviewed rows. CSV/JSONL `split` values keep train, validation, and test partitions separate, and exact duplicate text rows are removed within each worker partition. A fixed-seed 80/20 split remains available when explicit partitions are absent.
- The local worker serializes CUDA jobs, streams persisted progress and resource metrics, supports cancellation and checkpoint resume, and stores every completed build as an evaluation-required model version. A held-out answer review and qualified-reviewer confirmation gate activation.
- The Orchestrator page tracks model jobs and workflow test runs. The Workflows page creates editable drafts, versions edits after publication, runs real local tests with node traces, and blocks publication until the current version passes a run and its selected experts remain Ready to Deploy. Chat can execute a selected published workflow.

This is a local first implementation, not an operational release. The prompt-role experts remain prototypes; scratch training needs licensed corpora at a scale appropriate to the intended capability; routing and answer quality need domain-specific review; and the 85% human-rating gate is a workflow check, not a guarantee of correctness. Per the user's instruction, no smoke or integration tests were run. The edited Python files passed AST parsing and the edited inline JavaScript passed Node syntax parsing.

## Local hardware boundary

The user's reported RTX 3050 has 6 GB VRAM. The UI must inspect actual free memory and offer small configurations rather than assume a large model fits. Mixed precision and activation checkpointing can reduce memory use, but they trade precision/compute and do not make a 6 GB card equivalent to a multi-GPU pretraining system. A compact scratch model is appropriate for a local proof of concept; a broadly capable domain LLM trained from random initialization needs far more clean tokens and compute. Larger scratch runs should be resumable and can be continued on a larger machine by copying the versioned checkpoint and run metadata.

## Requirements to resolve during implementation

- Decide the first supported training file formats and maximum upload size.
- Define task-specific evaluation metrics and the exact aggregation behind the 85% readiness threshold.
- Confirm whether PDF extraction is in the first release or whether text/CSV/JSONL come first.
- Confirm whether workflow publishing is local-only initially or exposed to other applications.
- Add MoE architecture only after dense Transformer jobs, evaluation, and loading are stable.

## References

- Hugging Face's from-scratch training guide explains that scratch training needs a lot of data and considerably more compute than fine-tuning: <https://huggingface.co/docs/course/en/chapter7/6>
- Hugging Face documents training a tokenizer and keeping large corpora iterator-based: <https://huggingface.co/learn/llm-course/chapter6/2>
- PyTorch documents automatic mixed precision and activation checkpointing as memory/performance techniques: <https://docs.pytorch.org/docs/stable/amp.html> and <https://docs.pytorch.org/docs/stable/checkpoint.html>
- FastAPI recommends a separate job tool/process for heavy background computation: <https://fastapi.tiangolo.com/tutorial/background-tasks/>
- The Income Tax Department explains the Act transition and tax-year applicability: <https://www.incometax.gov.in/iec/foportal/help/all-topics/e-filing-services/General%20Questions-faqs>

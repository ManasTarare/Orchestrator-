# Training data formats and held-out splits

Mornork has two distinct training targets. The expert model produces a domain answer. The orchestrator predicts a route label and never learns answer text as its target.

## Expert answer models

Accepted inputs are TXT, selectable-text PDF, CSV, or JSONL. TXT/PDF are corpus text for language-model training. CSV/JSONL can contain a `text` column or these answer fields:

```csv
prompt,response,context,split,approved
"Calculate the margin","Gross profit / revenue = 25%","Revenue 200; gross profit 50",train,true
```

`prompt` and `response` are required together for task evaluation and fine-tuning. `context`, `source`, `tax_year`, `source_section`, `official_source_excerpt`, and `source_url` are optional context/citation metadata. Fine-tuning requires at least two approved prompt/response rows. Corpus-only data is suitable for scratch language-model pretraining but does not supply answer-accuracy test labels.

## Orchestrator route classifier

Orchestrator CSV/JSONL must provide a request and its desired destination:

```csv
message,route,split,approved
"What was revenue growth?",finance-expert-id,train,true
"Hello",mornork,validation,true
```

Request column names may be `message`, `query`, or `text`. Destination names may be `route`, `label`, `expert_id`, or `target`. Use `mornork` when Mornork should answer directly; otherwise use the expert ID (or domain name) expected to exist when the orchestrator is activated. The train split must include at least two distinct route labels. Held-out labels must also appear in train.

## Split rules

Use `split=train`, `split=validation`, and `split=test` to keep the partitions explicit. The validation split is used for training-time held-out loss; test is reserved for task metrics and reviewer evaluation. A separate evaluation file with no split column is treated as test data. If an unsplit file is used without a separate evaluation file, Mornork creates deterministic 80/10/10 train/validation/test partitions using the configured seed. Approved-row formats may include an `approved` column; if present, only `true`, `yes`, `1`, or `approved` rows are included.

Do not reuse test examples in the training file. Evaluation and reviewer approval are required before a trained model can be activated.

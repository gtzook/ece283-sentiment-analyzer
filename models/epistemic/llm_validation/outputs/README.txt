Drop LLM output files here.

Expected format
---------------
One JSONL file per model, containing ALL 250 labeled sentences
(across all batches) concatenated together:

    {"id": 1, "label": "asserted"}
    {"id": 2, "label": "hedged"}
    ...

Naming convention:  <model_name>.jsonl
Examples:
    gpt4o.jsonl
    claude_sonnet.jsonl
    gemini_pro.jsonl

If you have per-batch output files, concatenate them:
    cat batch_01_output.txt batch_02_output.txt ... > gpt4o.jsonl

Then run:
    python compute_agreement.py

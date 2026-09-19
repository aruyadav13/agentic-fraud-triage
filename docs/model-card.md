# Model and system card

| Field | Description |
|---|---|
| Intended use | Educational fraud investigation and reproducible offline evaluation |
| Decision classes | Fraud suggestion, legitimate suggestion, human review |
| Baseline | scikit-learn histogram gradient boosting, 100 iterations, 15 leaves, learning rate 0.08, seed 42 |
| LLM integration | LangChain ChatHuggingFace with a configurable tool-capable model |
| Recorded agent | Deterministic rule protocol fixture; not an LLM |
| Data | First 100,000 rows of a pinned public Sparkov simulated CSV, sorted chronologically |
| Inputs | Amount, timestamp and strictly prior numeric account aggregates |
| Excluded inputs | Fraud labels, personal identity fields, raw card number, future transactions |
| Abstention | Insufficient evidence/history, low reported confidence, validation/retry failure, provider failure |
| Side effects | Local reports and review queue only |

The sample is simulated and drawn from a short prefix of one source. It cannot establish performance on real institutions, jurisdictions, fraud strategies, or changing customer behavior. Historical mean and velocity patterns can be misleading for infrequent or changing accounts. No demographic fairness study was performed; no protected attributes are sent to the model.

The recorded replay results underperform the numeric baseline. No LLM superiority claim is supported. A live LLM may hallucinate rationales, miss fraud, overflag legitimate transactions or mishandle provider tool protocols. Reported confidence is not calibrated. Automatic gates reduce unsupported decisions but do not prove that accepted decisions are correct.

Do not deploy this research pipeline to move funds or restrict access to financial services without separate operational, security, model-risk and human-process validation. No real customer data were required to develop or evaluate the included code.

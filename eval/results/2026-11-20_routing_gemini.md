# Eval: routing

| | |
|---|---|
| date | 2026-11-20 |
| git | be29dbd |
| provider | gemini |
| split | test |
| n | 60 |
| models | {'router': 'gemini:gemini-3.5-flash-lite'} |
| prompt versions | {'charlie': 'charlie-v1', 'scout': 'scout-v1', 'chase': 'chase-v1'} |
| datasets | {'routing.jsonl': '0ff35efb7cdfe4ef25592c47eb7bc467eb3cdae1970753c3c948eae0fed475db'} |
| config overrides | none |
| cache | none |

## Metrics

| metric | value |
|---|---|
| n | 60 |
| intent_accuracy | 1.000 |
| macro_f1 | 1.000 |
| per_intent.discovery.precision | 1.000 |
| per_intent.discovery.recall | 1.000 |
| per_intent.discovery.f1 | 1.000 |
| per_intent.cpg_qa.precision | 1.000 |
| per_intent.cpg_qa.recall | 1.000 |
| per_intent.cpg_qa.f1 | 1.000 |
| per_intent.outreach.precision | 1.000 |
| per_intent.outreach.recall | 1.000 |
| per_intent.outreach.f1 | 1.000 |
| per_intent.clarify.precision | 1.000 |
| per_intent.clarify.recall | 1.000 |
| per_intent.clarify.f1 | 1.000 |
| per_intent.out_of_scope.precision | 1.000 |
| per_intent.out_of_scope.recall | 1.000 |
| per_intent.out_of_scope.f1 | 1.000 |
| confusion.discovery.discovery | 21 |
| confusion.cpg_qa.cpg_qa | 15 |
| confusion.outreach.outreach | 9 |
| confusion.clarify.clarify | 9 |
| confusion.out_of_scope.out_of_scope | 6 |
| slot_accuracy | 0.952 |
| slot_n | 21 |
| outreach_channel_accuracy | 0.667 |
| over_clarify_rate | 0.000 |

## Notes

- split: test

## Items that failed (4 of 60)

```json
{"id": "r036", "query": "show me bakery supply wholesalers in Chicago", "gold": "discovery", "predicted": "discovery", "ok": false, "slots_ok": false, "gold_channel": null, "predicted_channel": null, "errors": []}
{"id": "r063", "query": "ring the LA broker", "gold": "outreach", "predicted": "outreach", "ok": false, "slots_ok": null, "gold_channel": "voice", "predicted_channel": "email", "errors": []}
{"id": "r064", "query": "call the top 2", "gold": "outreach", "predicted": "outreach", "ok": false, "slots_ok": null, "gold_channel": "voice", "predicted_channel": "email", "errors": []}
{"id": "r065", "query": "phone the last lead and ask for a meeting", "gold": "outreach", "predicted": "outreach", "ok": false, "slots_ok": null, "gold_channel": "voice", "predicted_channel": "email", "errors": []}
```

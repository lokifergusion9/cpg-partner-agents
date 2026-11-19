# Eval: extraction

| | |
|---|---|
| date | 2026-11-20 |
| git | be29dbd |
| provider | gemini |
| embedder | BAAI/bge-small-en-v1.5 |
| split | test |
| n | 30 |
| models | {'extractor': 'gemini:gemini-3.5-flash-lite'} |
| prompt versions | {'charlie': 'charlie-v1', 'scout': 'scout-v1', 'chase': 'chase-v1'} |
| datasets | {'extraction.jsonl': '19bcb7fd31079d233de2f7761d1f6095fa52608f5aef576f974b711a60c1d392'} |
| config overrides | none |
| cache | {'search_misses': 60, 'embed_misses': 41, 'embed_hits': 54} |

## Metrics

| metric | value |
|---|---|
| n | 30 |
| email_precision | 1.000 |
| email_recall | 0.737 |
| email_f1 | 0.848 |
| phone_precision | 1.000 |
| phone_recall | 0.710 |
| phone_f1 | 0.830 |
| raw_hallucination_rate | 0.062 |
| eligible_hallucination_rate | 0.000 |
| abstention_accuracy | 1.000 |
| abstention_n | 4 |
| company_name_match | 1.000 |
| proposed_emails | 16 |
| eligible_emails | 14 |

## Notes

- every page is read with a generic intent for its partner type

## Items that failed (9 of 30)

```json
{"id": "e002", "case": "standard", "gold_emails": ["info@centralcoop.coop"], "out_emails": ["info@centralcoop.coop"], "gold_phones": ["+12063291545", "+12538884288"], "out_phones": ["+12063291545"], "page_status": ["ok"], "statuses": ["verified"], "errors": [], "ok": false}
{"id": "e003", "case": "standard", "gold_emails": ["customerservice@olympiafood.coop"], "out_emails": ["customerservice@olympiafood.coop"], "gold_phones": ["+13603571106", "+13607547666", "+13609563870"], "out_phones": ["+13609563870"], "page_status": ["ok"], "statuses": ["verified"], "errors": [], "ok": false}
{"id": "e006", "case": "standard", "gold_emails": ["customerservice@saxelbycheese.com", "orders@saxelbycheese.com"], "out_emails": ["orders@saxelbycheese.com"], "gold_phones": ["+16468923077", "+17182608660"], "out_phones": ["+16468923077"], "page_status": ["ok"], "statuses": ["verified"], "errors": [], "ok": false}
{"id": "e008", "case": "standard", "gold_emails": ["hotline@melissas.com"], "out_emails": [], "gold_phones": [], "out_phones": [], "page_status": ["ok"], "statuses": [], "errors": [], "ok": false}
{"id": "e009", "case": "standard", "gold_emails": ["info@formaggionewyork.com"], "out_emails": ["info@formaggionewyork.com"], "gold_phones": ["+12129828200", "+18882123224"], "out_phones": ["+18882123224"], "page_status": ["ok"], "statuses": ["verified"], "errors": [], "ok": false}
{"id": "e011", "case": "standard", "gold_emails": ["sales@marisafoods.com"], "out_emails": [], "gold_phones": ["+15624355629"], "out_phones": ["+15624355629"], "page_status": ["ok"], "statuses": ["not_in_source"], "errors": [], "ok": false}
{"id": "e017", "case": "standard", "gold_emails": [], "out_emails": [], "gold_phones": ["+16128148000", "+19524730292", "+19526973366"], "out_phones": ["+19526973366"], "page_status": ["ok"], "statuses": [], "errors": [], "ok": false}
{"id": "e019", "case": "obfuscated", "gold_emails": ["info@harvestbrokerage.com"], "out_emails": [], "gold_phones": ["+12816906707", "+18323681234", "+18325865562"], "out_phones": ["+18323681234"], "page_status": ["ok"], "statuses": ["invalid_format"], "errors": [], "ok": false}
{"id": "e020", "case": "obfuscated", "gold_emails": ["hello@superasia.ca"], "out_emails": [], "gold_phones": ["+14167465550"], "out_phones": ["+14167465550"], "page_status": ["ok"], "statuses": [], "errors": [], "ok": false}
```

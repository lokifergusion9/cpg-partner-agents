# Eval: retrieval

| | |
|---|---|
| date | 2026-11-20 |
| git | cf835f8 |
| provider | none |
| embedder | BAAI/bge-small-en-v1.5 |
| split | test |
| n | 26 |
| models | none called |
| prompt versions | {'charlie': 'charlie-v1', 'scout': 'scout-v1', 'chase': 'chase-v1'} |
| datasets | {'extraction.jsonl': '19bcb7fd31079d233de2f7761d1f6095fa52608f5aef576f974b711a60c1d392'} |
| config overrides | {'evidence_k': '8', 'evidence_max': '8'} |
| cache | {'search_misses': 52, 'embed_misses': 36, 'embed_hits': 98} |

## Metrics

| metric | value |
|---|---|
| vector.recall | 1.000 |
| vector.pages | 26 |
| vector.no_gold_chunk | 0 |
| contact_pattern.recall | 0.962 |
| contact_pattern.pages | 26 |
| contact_pattern.no_gold_chunk | 0 |
| hybrid.recall | 1.000 |
| hybrid.pages | 26 |
| hybrid.no_gold_chunk | 0 |
| recall | 1.000 |

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
| config overrides | {'chunk_tokens': '100', 'chunk_overlap': '20', 'evidence_k': '1', 'evidence_max': '2'} |
| cache | {'search_misses': 52, 'embed_misses': 253, 'embed_hits': 103} |

## Metrics

| metric | value |
|---|---|
| vector.recall | 0.538 |
| vector.pages | 26 |
| vector.no_gold_chunk | 0 |
| contact_pattern.recall | 0.962 |
| contact_pattern.pages | 26 |
| contact_pattern.no_gold_chunk | 0 |
| hybrid.recall | 0.962 |
| hybrid.pages | 26 |
| hybrid.no_gold_chunk | 0 |
| recall | 0.962 |

## Items that failed (1 of 26)

```json
{"id": "e026", "case": "standard", "chunks": 11, "gold_chunks": [10], "selected": {"vector": [4], "contact_pattern": [], "hybrid": [4]}, "ok": false}
```

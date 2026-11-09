# serving benchmark · local-small · ollama cpu, shared prefix

- model `qwen2.5-3b-instruct-8k` revision `sha256:155251c4b20a9a96f844f6ed01e45338b06fe4ad5c456660b58eee40d7dd7cea` · max_tokens 48
- shared prefix: 3000 chars (prefix-cache test when > 0)

| concurrency | req | err | throughput tok/s | TTFT p50 / p95 s | ITL p50 / p95 ms | latency p50 / p95 s | cached input |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 0 | 5.4 | 0.871 / 14.196 | 88.9 / 89.1 | 5.04 / 18.38 | 74% |
| 4 | 4 | 0 | 9.6 | 7.904 / 14.736 | 89.8 / 94.9 | 12.12 / 19.19 | 97% |

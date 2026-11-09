# serving benchmark · local-small · ollama cpu, no shared prefix

- model `qwen2.5-3b-instruct-8k` revision `sha256:155251c4b20a9a96f844f6ed01e45338b06fe4ad5c456660b58eee40d7dd7cea` · max_tokens 48
- shared prefix: 0 chars (prefix-cache test when > 0)

| concurrency | req | err | throughput tok/s | TTFT p50 / p95 s | ITL p50 / p95 ms | latency p50 / p95 s | cached input |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 0 | 10.9 | 0.596 / 0.798 | 80.3 / 80.7 | 4.36 / 4.58 | 53% |
| 2 | 4 | 0 | 12.2 | 3.939 / 4.077 | 82.1 / 84.6 | 7.77 / 8.04 | 98% |
| 4 | 4 | 0 | 12.1 | 6.088 / 11.536 | 82.9 / 85.0 | 10.02 / 15.29 | 98% |

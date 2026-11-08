# tasks eval · local-small · system.v1

- source: `2026-11-08T0907_tasks_local-small_system.v1.json`
- created: 2026-11-08T09:07:25+00:00 · git `72f69e9-dirty`
- model: `qwen2.5-3b-instruct-8k` · revision `sha256:155251c4b20a9a96f844f6ed01e45338b06fe4ad5c456660b58eee40d7dd7cea`
- sampling: `{"temperature": 0.7, "top_p": 0.8, "top_k": 20}`

> **Warning:** 2 of 36 runs ended in a model error. Those runs count as failures above, so the pass rates understate the agent. Check the endpoint.

12 tasks × 3 runs = 36 runs. Intervals: 95% percentile bootstrap over tasks.

| metric | value |
|---|---|
| pass@1 | 11.1% [2.8%, 19.4%] |
| pass^3 | 0.0% [0.0%, 0.0%] |
| pass@3 | 33.3% |
| mean turns | 7.4 |
| mean tokens in / out | 11046 / 788 |
| mean wall time | 120.2 s |
| cost per task | $0.0000 |
| cost per successful task | $0.0000 |
| invalid tool call rate | 0.8% of 240 calls |
| runs that edited tests | 3 |
| infrastructure errors | 0 |
| model errors | 2 |

Stop reasons: completed 28, loop_detected 5, max_time 1, model_error 2

| tool | calls | error rate |
|---|---|---|
| bash | 40 | 95% |
| edit_file | 24 | 50% |
| glob | 16 | 6% |
| grep | 14 | 0% |
| list_dir | 75 | 53% |
| read_file | 67 | 27% |
| write_file | 4 | 0% |

| task | passed |
|---|---|
| binary-search-bounds | 0/3 |
| discount-units-multifile | 0/3 |
| fizzbuzz-order | 1/3 |
| leap-year-feature | 1/3 |
| mutable-default-arg | 0/3 |
| off-by-one-moving-average | 0/3 |
| palindrome-normalize | 1/3 |
| parse-duration-units | 0/3 |
| safe-divide-none | 0/3 |
| slugify-trailing-dash | 0/3 |
| word-count-whitespace | 0/3 |
| wrong-operator-add | 1/3 |

Failed runs (open with `agent trace show <trace>`):

| task | # | stop reason | turns | trace |
|---|---|---|---|---|
| binary-search-bounds | 1 | model_error | 8 | `1282aa2b88bc7431fe4edadec39e6ad0` |
| binary-search-bounds | 2 | loop_detected | 11 | `791316039c21ef931ac9c78952100129` |
| binary-search-bounds | 3 | completed | 5 | `c133423ec19421625189fd4a35b3adbe` |
| discount-units-multifile | 1 | completed | 8 | `91a48b30ba60c1fdd4ae61333ca5ea7a` |
| discount-units-multifile | 2 | completed | 5 | `f9d4160f86da8bad236ee77b7cb80a47` |
| discount-units-multifile | 3 | completed | 6 | `445582f1b7b2bfada16b864fccf19c4f` |
| fizzbuzz-order | 2 | max_time | 14 | `635c2aa77c06f85167e6201e11c0a5d4` |
| fizzbuzz-order | 3 | completed | 7 | `44a851b817d5ac41f7f23e2a36c7386d` |
| leap-year-feature | 1 | completed | 3 | `d64140d9fbf00ce254ddf6531bd59663` |
| leap-year-feature | 2 | completed | 4 | `e7c06c112c333c518a9148de8da3fb22` |
| mutable-default-arg | 1 | completed | 4 | `af66c9b296be188ef20ab8de9d52a18e` |
| mutable-default-arg | 2 | completed | 8 | `bf0d3ed63ecb693f81daa3c7b83f4112` |
| mutable-default-arg | 3 | completed | 4 | `97a8533dd9e1846f010285667f0be116` |
| off-by-one-moving-average | 1 | completed | 4 | `a708d62b7daf916d9accaf2753f83b53` |
| off-by-one-moving-average | 2 | completed | 5 | `28e2ecd13655b93d82016735981ef47d` |
| off-by-one-moving-average | 3 | completed | 9 | `efb7deab77dcad7599377147b68785da` |
| palindrome-normalize | 1 | completed | 3 | `29ec2d7f703b8957e79f95b23f30b9e0` |
| palindrome-normalize | 3 | loop_detected | 14 | `ed0b59b004a86886d11aa76ecd613ec7` |
| parse-duration-units | 1 | completed | 8 | `68e404d016c1d050962d5aca864742aa` |
| parse-duration-units | 2 | completed | 10 | `47fab61a2c986a195e8190a329556148` |
| parse-duration-units | 3 | completed | 9 | `c68d37cef0650a911d218eb891358cfc` |
| safe-divide-none | 1 | completed | 5 | `eb3170213b1628ed4c73dda695b25cff` |
| safe-divide-none | 2 | completed | 3 | `f656128d59a37b7ea62333dd57cfd95b` |
| safe-divide-none | 3 | completed | 17 | `1fc7ee29236e883124256eca5538ade2` |
| slugify-trailing-dash | 1 | completed | 4 | `af53038ede1902dbbbd27c4171f589ce` |
| slugify-trailing-dash | 2 | completed | 8 | `23edf6cb75a76a337a40a25edc60c397` |
| slugify-trailing-dash | 3 | loop_detected | 8 | `72bdc2fe25c1a353598773e0117778ac` |
| word-count-whitespace | 1 | completed | 8 | `6d4372ac5f124d986283ebecb73b7014` |
| word-count-whitespace | 2 | completed | 7 | `92c93b514ca265fff003fd4a50634555` |
| word-count-whitespace | 3 | completed | 9 | `5dd7ea11b0c9cbdf0d7ee2be6b76652d` |
| wrong-operator-add | 2 | loop_detected | 7 | `4cec75caf21506ba255a07ca20052c62` |
| wrong-operator-add | 3 | completed | 2 | `9193411fd66ff1fbde630b0e63363b99` |

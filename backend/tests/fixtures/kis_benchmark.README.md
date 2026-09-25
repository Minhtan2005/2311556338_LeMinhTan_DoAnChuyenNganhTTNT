# KIS Benchmark Fixture

`kis_benchmark.json` intentionally starts empty. Add manually verified labels only after inspecting the videos and confirming acceptable frame ranges.

Schema:

```json
[
  {
    "query": "Tìm cảnh một người đứng cạnh xe máy",
    "expected_video": "L01_V028",
    "acceptable_frame_start": 3400,
    "acceptable_frame_end": 3500
  }
]
```

The benchmark runner reports Recall@1, Recall@5, Recall@10, MRR, and average query latency.

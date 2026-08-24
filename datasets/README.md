# Dataset layout

Download the datasets from their official sources. Raw videos, annotations, extracted features, and generated keyframe files are not distributed in this repository.

```text
datasets/
├── videomme/
│   ├── data/                  # Video-MME videos
│   ├── videomme_json_file.json
│   ├── features_and_scores/
│   └── keyframe_dir/
├── longvideobench/
│   ├── videos/                # LongVideoBench videos
│   ├── lvb_val.json
│   ├── features_and_scores/
│   └── keyframe_dir/
├── mlvu/
│   ├── video/                 # MLVU videos
│   ├── mlvu_dev.json
│   ├── features_and_scores/
│   └── keyframe_dir/
└── LVbench/
    ├── videos/                # LVBench videos
    ├── lvbench.json
    ├── features_and_scores/
    └── keyframe_dir/
```

If your local paths differ, pass the corresponding locations to the preprocessing, RIDGE selection, and `lmms-eval` commands.

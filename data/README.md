# Local data layout

This directory is intentionally excluded from Git except for this guide.

```
raw/naver_shopping.txt       # source reviews
interim/                     # preprocessing and Gemini filtering outputs
final/train_for_kfold.csv    # 80% stratified training/CV split
final/test_final_holdout.csv # 20% stratified evaluation split
```

Before public release, publish any redistributable dataset through an archive
with its source, licence, version, and checksum.

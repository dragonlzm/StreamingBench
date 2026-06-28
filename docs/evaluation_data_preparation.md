# Evaluation Data Preparation Notes

These notes record how the StreamingBench evaluation data was prepared on the
Pluto node used for the Qwen2-VL real-task run.

## Source Data

The raw Hugging Face dataset snapshot was staged under:

```bash
data_raw/StreamingBench/
```

The staged snapshot contained the annotation CSVs plus the video archives:

- `Real-Time Visual Understanding_*.zip`
- `Sequential Question Answering_*.zip`
- `Proactive Output_*.zip`
- `Anomaly Context Understanding.zip`
- `Emotion Recognition.zip`
- `Misleading Context Understanding.zip`
- `Multimodal Alignment.zip`
- `Scene Understanding_*.zip`
- `Source Discrimination.zip`

## Extraction Layout

Archives were extracted into the repository's expected `data/` layout:

```text
data/
  real/        # Real-Time Visual Understanding archives
  omni/        # Omni-source/contextual archives
  sqa/         # Sequential Question Answering archives
  proactive/   # Proactive Output archives
```

For the real and omni tasks, the extraction commands used were:

```bash
mkdir -p data/real data/omni

for z in data_raw/StreamingBench/Real-Time\ Visual\ Understanding_*.zip; do
  unzip -n -q "$z" -d data/real
done

for z in \
  data_raw/StreamingBench/Anomaly\ Context\ Understanding.zip \
  data_raw/StreamingBench/Emotion\ Recognition.zip \
  data_raw/StreamingBench/Misleading\ Context\ Understanding.zip \
  data_raw/StreamingBench/Multimodal\ Alignment.zip \
  data_raw/StreamingBench/Scene\ Understanding_1-25.zip \
  data_raw/StreamingBench/Scene\ Understanding_26-50.zip \
  data_raw/StreamingBench/Source\ Discrimination.zip; do
  unzip -n -q "$z" -d data/omni
done
```

The `sqa` and `proactive` directories were already extracted on this node.

## Video Staging

StreamingBench expects all videos to be under `src/data/videos/` with names like
`sample_348_real.mp4`, `sample_12_sqa.mp4`, or `sample_7_omni.mp4`.

The repository helper was used to move and rename videos:

```bash
python src/data/move_video.py --src data/real --dest src/data/videos
python src/data/move_video.py --src data/omni --dest src/data/videos
python src/data/move_video.py --src data/sqa --dest src/data/videos
python src/data/move_video.py --src data/proactive --dest src/data/videos
```

`move_video.py` skips `__MACOSX` directories and renames each `video.mp4` using
the sample folder and task folder.

## Annotation Paths

The checked-in annotation JSON files under `src/data/` already used absolute
paths pointing at this checkout:

```text
/mnt/localssd/project2/zm_temp/StreamingBench/src/data/videos/<video>.mp4
```

After staging videos, the real-task references were validated with:

```bash
python - <<'PY'
import json
import pathlib

data = json.load(open("src/data/questions_real.json"))
paths = set()

def walk(value):
    if isinstance(value, dict):
        if "video_path" in value:
            paths.add(value["video_path"])
        for child in value.values():
            walk(child)
    elif isinstance(value, list):
        for child in value:
            walk(child)

walk(data)
missing = [path for path in paths if not pathlib.Path(path).exists()]
print("unique_real_videos", len(paths))
print("missing_real_videos", len(missing))
PY
```

Expected validation result for the real split:

```text
unique_real_videos 499
missing_real_videos 0
```

## Generated Artifacts

The extracted videos, temporary clips, model outputs, and stats are generated
runtime artifacts. They were intentionally left out of the documentation commit.

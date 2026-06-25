# SussexDrone

A small-drone detection dataset with bird annotations, for the drone-versus-bird
discrimination task.

**Status:** under preparation. The dataset will be released on Sussex Figshare
alongside thesis submission (autumn 2026); a DOI will be added here on release.
This repository hosts the documentation and the YOLO conversion script. The
video data and annotations are distributed via Figshare, not through GitHub.

## Overview

SussexDrone contains 211 annotated video clips (78,896 frames, 47.5 minutes)
captured with three camera types, with bounding-box annotations for drones and
birds. It is designed for two tasks:

- **Detection** — object detection of small drones across varied backgrounds.
- **Trajectory** — temporal analysis of drone and bird flight behaviour.

The two tasks are served by two subsets with their own train/val/test splits,
assigned by recording session so that clips from one session never span
multiple splits (no session leakage).

## At a glance

| | |
|---|---|
| Videos | 211 |
| Frames | 78,896 |
| Annotated boxes | 84,618 (78,820 drone, 5,798 bird) |
| Cameras | GoPro, Fujifilm, DJI |
| Resolutions | 1920x1080 and 3840x2160 |
| Environments | grass, urban, sky, trees, mixed |
| Perspectives | static ground, dynamic ground, aerial (drone-to-drone) |

A notable property: the median bird and median drone occupy almost the same
apparent size (around 40 and 38 pixels respectively), so the two classes
overlap in appearance and are separated mainly by motion. This motivates the
inclusion of bird annotations and trajectory information.

## Contents of the published dataset (Figshare)

```
SussexDrone/
    videos/                 211 MP4 clips
    annotations/            211 annotation files (Drone-vs-Bird format)
    splits/                 detection_{train,val,test}.txt
                            trajectory_{train,val,test}.txt
    manifest.csv            per-video metadata
```

See [docs/annotation_format.md](docs/annotation_format.md) for the annotation
format.

## Using the dataset

Download the dataset from Figshare, then convert it to YOLO format with the
provided script:

```
pip install -r scripts/requirements.txt
python scripts/extract_yolo.py --dataset /path/to/SussexDrone \
                               --output  /path/to/SussexDrone_yolo \
                               --subset  detection
```

See [scripts/README.md](scripts/README.md) for all options.

## Citation

See [CITATION.cff](CITATION.cff). The dataset DOI will be added on release.

## Authors

- Elif Ucurum, University of Sussex
- Dr. Phil Birch, University of Sussex (supervisor)

## License

- Code in this repository: MIT (see [LICENSE](LICENSE)).
- The dataset (hosted on Figshare): CC BY 4.0.

# Annotation format

SussexDrone uses the Drone-vs-Bird Challenge annotation format: one plain-text
file per video, sharing the video's name (e.g. `dji_grass_001.mp4` is annotated
by `dji_grass_001.txt`).

## Per-frame lines

Each line describes one frame of the video:

```
frame_index  num_objects  [x_left  y_top  width  height  class]  ...
```

- `frame_index` — zero-based frame number.
- `num_objects` — number of annotated objects in that frame.
- Each object is then given as five values: the top-left corner (`x_left`,
  `y_top`), the box `width` and `height` in absolute pixels, and the `class`
  name (`drone` or `bird`).

A frame with no objects is written as:

```
12 0
```

## Example

```
0 1 2469 6 30 15 drone
1 2 2468 13 30 15 drone 3659 842 47 26 drone
2 0
3 1 2438 219 228 147 drone
```

Frame 0 has one drone; frame 1 has two drones; frame 2 is empty; frame 3 has
one drone.

## Notes

- Coordinates are absolute pixels in the video's native resolution. The dataset
  contains both 1920x1080 and 3840x2160 videos, so normalise using each video's
  own dimensions (the `extract_yolo.py` script does this automatically).
- Bounding boxes are clipped to the image bounds; no negative coordinates or
  out-of-frame boxes appear in the published annotations.
- Annotations were produced in CVAT using track mode with keyframe
  interpolation, then exported to this per-frame format.

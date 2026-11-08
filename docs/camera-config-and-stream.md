# Camera config v2 and the cropped stream

This covers what changed when calibration gained a crop step, what the config file now contains, and how the video is processed. The code lives in `notebooks/notebook1.ipynb`, in the cells **Camera calibration v2** and **Entry / exit detection v2**.

## Why crop

The cam1 frame includes the main road. Cars on the road get detected and tracked even though they never enter or leave the premises. The gate-zone rule already stops them being counted, but they still:

- cost inference time on every frame
- create tracks that can be confused with gate traffic (ID switches, stitching onto the wrong car)
- clutter the output video and the `ignored_tracks` list

Cropping to the driveway and gate removes them before YOLO runs. A smaller input also makes each car larger relative to the image YOLO sees, which helps detection.

## Calibration flow

Running the calibration cell opens an OpenCV window and walks through five steps:

| Step | Shown on | You do | Keys |
|---|---|---|---|
| 1. Crop | Full frame | Drag a rectangle around the area to keep (driveway + gate, no road) | Enter = use crop, F = full frame |
| 2. Camera type | Cropped image | Choose passageway or parking zone | Click, or P / K |
| 3. Entry arrow | Cropped image | Drag along the path an entering car takes | Backspace = back |
| 4. Exit arrow | Cropped image | Drag along the path an exiting car takes | Backspace = redo entry |
| 5. Gate zone | Cropped image | Click the corners of the area every entering/exiting car drives through | Right-click / Enter = finish, Backspace = undo point |
| Review | Cropped image | Check it | S / Enter = save, R = redo arrows + gate, G = redo gate, C = redo crop |

N skips one second ahead in the video at any step before review, for when the first frame is a bad view. Small crops are shown enlarged (up to 2×) so they are easy to draw on; saved coordinates are always real pixels.

Changing the crop clears the type, arrows and gate zone, because they are drawn relative to the crop.

## Config file: `cameraConfigs/<camera_id>.json`

### What changed from v1

| | v1 | v2 |
|---|---|---|
| `config_version` | absent | `2` |
| `crop` | absent | crop rectangle in full-frame pixels and normalised 0–1 |
| `coordinate_space` | full frame (implicit) | `"crop"` |
| `entry`, `exit`, `gate_zone` | relative to the full frame | relative to the **crop** |

v1 and v2 configs are not interchangeable. The v2 detection cell refuses a config without `config_version: 2`, and the v1 detection cell refuses a v2 config. Re-running the v2 calibration overwrites `cam1.json`.

### Example

```json
{
  "config_version": 2,
  "camera_id": "cam1",
  "camera_type": "passageway",
  "source": "data/videos/6_11_26_morning_cam1.mp4",
  "frame_size": [960, 1080],
  "crop": {
    "x1": 0, "y1": 300, "x2": 960, "y2": 1080,
    "size": [960, 780],
    "norm": [0.0, 0.2778, 1.0, 1.0]
  },
  "coordinate_space": "crop",
  "calibration_frame": 0,
  "entry": {
    "start": [620, 240], "end": [69, 742],
    "start_norm": [0.6458, 0.3077], "end_norm": [0.0719, 0.9513],
    "unit_vector": [-0.7392, 0.6735]
  },
  "exit": { "...": "same fields as entry" },
  "gate_zone": {
    "points": [[10, 586], [190, 468], [405, 776], [12, 778]],
    "points_norm": [[0.0104, 0.7513], [0.1979, 0.6], [0.4219, 0.9949], [0.0125, 0.9974]]
  },
  "created_at": "2026-11-08T10:00:00"
}
```

(Values are illustrative.)

### Field reference

| Field | Meaning |
|---|---|
| `frame_size` | Full frame size of the calibration video, `[width, height]` |
| `crop.x1, y1, x2, y2` | Crop rectangle in full-frame pixels (`x2`, `y2` exclusive) |
| `crop.size` | Crop `[width, height]` in pixels |
| `crop.norm` | Crop rectangle as fractions of the full frame. **This is what the pipeline uses**, so a stream at a different resolution still crops the same area |
| `entry` / `exit` `.start`, `.end` | Arrow endpoints in crop pixels |
| `entry` / `exit` `.start_norm`, `.end_norm` | Same, as fractions of the crop. Used by the pipeline |
| `entry` / `exit` `.unit_vector` | Arrow direction, length 1. The decision compares a car's net movement against this |
| `gate_zone.points` / `.points_norm` | Gate polygon corners, crop pixels / fractions of the crop |
| `calibration_frame` | Frame index the calibration was drawn on |

Calibration also saves two images next to the config:

- `<camera_id>_calibration.png`: the cropped view with arrows and gate zone
- `<camera_id>_crop.png`: the full frame, outside the crop dimmed, crop outlined

## Stream setup

### v1

```
video file ──► model.track(source=video) ──► results on the full frame
```

Ultralytics opened the video itself and every full frame went to YOLO.

### v2

```
cv2.VideoCapture ──► read frame ──► crop to config rectangle ──► model.track(crop, persist=True)
                                                                       │
                     write output ◄── draw overlays on crop ◄── tracks / events (crop coords)
```

1. **Read**: frames are read one at a time with OpenCV.
2. **Crop**: the crop rectangle is computed once from `crop.norm` × this video's resolution, then each frame is sliced to it.
3. **Track**: the crop is passed to `model.track(..., persist=True)`. `persist=True` keeps ByteTrack's state between calls, so IDs carry across frames exactly as when Ultralytics read the file.
4. **Decide**: point recording, the gate zone, net displacement vs. arrows, the order check and the persistence logic are unchanged from v1. They all run in crop coordinates, and "touching the edge" now means the crop edge.
5. **Write**: `OUTPUT_VIEW = "crop"` writes a video of the cropped area only. `"full"` writes the full frame with the outside dimmed and the crop outlined.

Reading frames ourselves, rather than handing Ultralytics a file path, is also what a live camera needs: replace the file path with an RTSP URL in `cv2.VideoCapture` and the rest of the loop stays the same.

### Coordinates in the output JSON

Every point in `<video>_yolo26s_cropped_events.json` (`start_point`, `end_point`, `path`) is in **crop** pixels. To get full-frame pixels, add the crop offset stored at the top of the file:

```
full_x = crop_x + crop.x1
full_y = crop_y + crop.y1
```

## Duplicate protection

A car that has already produced an entry or exit event is never recorded again. A finished track is skipped, and listed in `ignored_tracks` as `duplicate of event N`, when:

| Case | How it is caught |
|---|---|
| The tracker revives an ID after its car was already counted | Every tracker ID in a counted track is remembered (`counted_ids`) |
| The same car is split into two tracks that both qualify | Same event type, starts within `DEDUP_S` (5 s) of the previous event ending, and starts within `DEDUP_DIST_FRAC` (15% of the crop diagonal) of where that event ended |
| Two boxes on one car at the same time | Same type, overlapping in time, and both start and end points within the stitch distance |

Live boxes for cars that are already counted are labelled `(counted)` in the output video.

## Outputs

| File | Contents |
|---|---|
| `data/outputs/<video>_yolo26s_cropped_entry_exit.mp4` | Annotated video (crop or full frame, per `OUTPUT_VIEW`) |
| `data/outputs/<video>_yolo26s_cropped_events.json` | `counts`, `events`, `ignored_tracks` (with reasons, including duplicates), the crop offset, and every setting used |
| `trackers/<camera_id>_bytetrack.yaml` | Tracker config generated from the cell's settings on each run |

## Tuning

All settings are at the top of the detection cell.

| Group | Settings | When to change |
|---|---|---|
| Output | `OUTPUT_VIEW` | `"full"` to see the crop in context |
| Detector + tracker | `CONF`, `TRACK_HIGH_THRESH`, `TRACK_LOW_THRESH`, `NEW_TRACK_THRESH`, `MATCH_THRESH`, `TRACK_BUFFER_S` | Tracks dropping or flickering: lower the thresholds, raise the buffer. Junk tracks: raise `NEW_TRACK_THRESH` |
| Track records | `COAST_S`, `FINALIZE_AFTER_S`, `EDGE_FINALIZE_S`, `STITCH_DIST_FRAC`, `SMOOTH_ALPHA`, `VEL_ALPHA` | One car split across several IDs: raise `STITCH_DIST_FRAC` / `FINALIZE_AFTER_S` |
| Duplicates | `DEDUP_S`, `DEDUP_DIST_FRAC` | Real back-to-back cars merged into one: lower them. One car counted twice: raise them |
| Decision | `MIN_COS`, `MIN_DISP_FRAC`, `ORDER_SPLIT`, `POINT_MODE` | Real entries missed: check the reasons in `ignored_tracks` first |

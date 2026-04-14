# Singularity Umbrella

This repository explores a guided-chaotic-trajectory pipeline:

1. extract a planar target curve,
2. build a custom 3D dissipative vector field whose orbit contains a guided segment matching that curve,
3. visualize the guided segment against the original target,
4. for complex SVG drawings, split them into manageable sub-curves and batch-compose the results back into one image.

The code is organized around small, single-purpose scripts so that each stage stays inspectable.

## Files

- [preprocess.py](./preprocess.py): extract a single curve from a white-background black-line raster image.
- [preprocess_svg.py](./preprocess_svg.py): extract a single curve from an SVG path.
- [core.py](./core.py): build the custom 3D dissipative vector field and sample a reference orbit.
- [visual.py](./visual.py): overlay one target curve, one orbit projection, and one guided segment.
- [dissect_svg.py](./dissect_svg.py): split a complex SVG into filtered sub-curves and export per-part `CurveFit` files.
- [batch_compose.py](./batch_compose.py): run the guided-field pipeline over all dissected parts and merge the results into one image.

## Environment

This project uses `uv` and the local interpreter at:

```bash
.venv/bin/python
```

## Single-Curve Pipeline

### 1. Raster image to `CurveFit`

```bash
.venv/bin/python preprocess.py test.jpg --output curve_fit.json
```

This expects a white background and a dark foreground stroke. The result is a JSON file containing a resampled, centered, and scaled planar curve.

### 2. SVG to `CurveFit`

```bash
.venv/bin/python preprocess_svg.py experiment_outputs/test.svg --output curve_fit_svg.json
```

This reads SVG `<path>` geometry directly, samples it into a polyline, normalizes it by the SVG `viewBox`, centers it, and rescales it to a fixed local geometric extent.

### 3. Build the guided chaotic field

In Python:

```python
from preprocess import load_curve_fit
from core import build_guided_chaotic_field

fit = load_curve_fit("curve_fit.json")
model = build_guided_chaotic_field(fit.as_array(), num_curve_samples=fit.num_samples)
```

The returned `model` includes:

- the 3D vector field itself,
- one sampled orbit,
- the target curve samples,
- metadata for the orbit interval whose 2D projection follows the target.

### 4. Visualize one curve

```python
from visual import render_comparison_image

render_comparison_image(model, output_path="comparison.png")
```

The image contains:

- the full orbit projection,
- the target curve,
- the guided segment,
- sparse error links between the guided segment and the target samples.

## Complex SVG Pipeline

Large line-art SVGs are usually not one simple curve. They are better treated as many sub-curves.

### 1. Dissect a complex SVG

```bash
.venv/bin/python dissect_svg.py target_noshadow.svg --output-dir experiment_outputs/canary_1
```

This stage:

- flattens SVG `<path>` commands into polylines,
- splits very long subpaths by arclength,
- rejects tiny or extremely short pieces,
- exports one standalone SVG per accepted part,
- exports one `CurveFit` JSON per accepted part,
- writes an overview image and a manifest.

Typical output layout:

```text
experiment_outputs/canary_1/
  manifest.json
  overview.png
  parts/
    part_000.svg
    part_001.svg
    ...
  curve_fits/
    part_000.json
    part_001.json
    ...
```

### 2. Batch-run the guided-field model and compose one image

```bash
.venv/bin/python batch_compose.py experiment_outputs/canary_1/manifest.json \
  --output experiment_outputs/canary_1/batch_comparison.png \
  --report experiment_outputs/canary_1/batch_report.json
```

The composed image contains:

- target strokes in dark gray,
- guided segments in blue,
- full attractors in faint translucent blue.

This is the preferred view for complex drawings, because it keeps the fitted segment visible while still showing the surrounding attractor structure.

## Reading the Output Files

### `curve_fit.json` or `curve_fit_svg.json`

These files serialize one local target curve in the shared `CurveFit` format.

Important fields:

- `image_path`: source image or SVG file used to build this curve.
- `image_size`: source canvas size.
- `bounding_box`: bounding box of the extracted source geometry in source coordinates.
- `num_samples`: number of planar samples exported.
- `target_extent`: local geometric scale after centering and normalization.
- `samples`: the actual sampled `(x, y)` points used by `core.py`.

Important detail:

- `samples` are not stored in original image or SVG coordinates.
- they are stored in a local centered coordinate system suitable for the guided-chaotic model.

### `manifest.json`

This is the main index produced by [dissect_svg.py](./dissect_svg.py). It describes how a complex SVG was split and how each accepted part should be mapped back to the original drawing.

Top-level fields:

- `svg_path`: original complex SVG.
- `output_dir`: directory containing the dissection results.
- `viewbox`: original SVG coordinate box as `[min_x, min_y, width, height]`.
- `image_size`: raster-like size inferred from the SVG canvas.
- `num_samples`: number of local samples exported for each part.
- `target_extent`: local scale used for each part's `CurveFit`.
- `min_svg_length`: reject parts shorter than this source arclength.
- `min_normalized_extent`: reject parts whose normalized bounding-box max side is smaller than this.
- `max_svg_length`: split parts longer than this source arclength.
- `source_subpath_count`: number of subpaths found before filtering.
- `accepted_part_count`: number of exported parts kept for downstream fitting.
- `rejected_part_count`: number of dropped parts after filtering.
- `overview_path`: overview image showing accepted parts in original SVG coordinates.

`parts` is the most important section. Each entry corresponds to one exported sub-curve:

- `part_id`: stable local identifier such as `part_012`.
- `source_subpath_index`: which original SVG subpath this part came from.
- `split_index`: if one long source path was split into several pieces, this is the piece index.
- `source_length`: source arclength in original SVG units.
- `source_points`: number of sampled polyline points in the exported part SVG.
- `source_closed`: whether the source polyline was closed when dissected.
- `source_bbox`: bounding box in original SVG coordinates.
- `normalized_center`: center of this part after normalization into the SVG unit box.
- `fit_scale`: scale factor used to map between the local `CurveFit` coordinates and normalized SVG coordinates.
- `svg_path`: standalone SVG containing only this part.
- `curve_fit_path`: local `CurveFit` JSON used by `core.py`.
- `num_samples`: number of local samples in that `CurveFit`.
- `target_extent`: local extent used for that `CurveFit`.

Why `normalized_center` and `fit_scale` matter:

- `core.py` works in a local centered coordinate system.
- batch composition must map each local orbit back into the original SVG coordinate system.
- these two values provide the inverse transform needed for that reconstruction.

`rejected` lists parts that were dropped before export. Each item contains:

- `source_subpath_index`
- `split_index`
- `reason`
- `source_length`
- `normalized_extent`

This is useful when tuning dissection thresholds.

### `overview.png`

This is a quick diagnostic image produced by [dissect_svg.py](./dissect_svg.py). It shows only the accepted parts, drawn in the original SVG coordinate system, with lightweight labels.

Use it to answer:

- Did the splitter keep the strokes I care about?
- Did it drop small decorative fragments?
- Did one long stroke get split too aggressively or not enough?

### `batch_report.json`

This is the main report produced by [batch_compose.py](./batch_compose.py). It summarizes the final batched fitting run.

Top-level fields:

- `manifest_path`: which dissected manifest was used.
- `output_path`: final composed image path.
- `image_size`: final composite canvas size.
- `accepted_part_count`: number of parts actually fitted in this run.
- `mean_projection_error`: average of the per-part guided-segment mean errors.
- `max_projection_error`: worst per-part maximum error in the run.
- `parts`: detailed per-part fitting diagnostics.
- `model_kwargs`: simulation settings passed into `core.py`.

Each item in `parts` contains:

- `part_id`: which dissected part this report row refers to.
- `curve_fit_path`: local curve input used for fitting.
- `svg_path`: standalone SVG for the part.
- `max_projection_error`: worst samplewise guided-segment deviation for that part.
- `mean_projection_error`: average guided-segment deviation for that part.
- `start_time`: start of the guided interval on the stored orbit.
- `end_time`: end of the guided interval on the stored orbit.
- `separation_peak_ratio`: simple finite-time nearby-orbit separation diagnostic.
- `separation_final_ratio`: final nearby-orbit separation diagnostic over the sampled window.

How to interpret the error fields:

- lower is better,
- `mean_projection_error` is usually the best overall quality indicator,
- `max_projection_error` helps identify localized mismatches,
- if a few parts are clearly worse than the rest, inspect those parts individually first.

## Recommended Workflow for Complex SVGs

1. Run `dissect_svg.py`.
2. Inspect `overview.png`.
3. Check `manifest.json` to see how many parts were accepted or rejected.
4. Run `batch_compose.py`.
5. Inspect `batch_comparison.png`.
6. Open `batch_report.json` and sort or scan by `max_projection_error` to find the worst parts.

## Notes and Limitations

- `preprocess_svg.py` currently supports `M/L/H/V/C/S/Q/T/Z` SVG path commands.
- elliptical arc commands `A/a` are not supported yet.
- the guided-field model is intended for finite-time guided matching, not for making the entire attractor equal to the target drawing.
- simple toy line segments may fit worse than the smoother medium-length curves this pipeline was tuned for.

## Tests

Run the current regression tests with:

```bash
.venv/bin/python -m unittest -v \
  test_core.py \
  test_preprocess.py \
  test_preprocess_svg.py \
  test_dissect_svg.py \
  test_batch_compose.py
```

## Example Canary Output

The current canary experiment for `target_noshadow.svg` lives in:

- [experiment_outputs/canary_1](./experiment_outputs/canary_1)

At the time of writing, that run produced:

- `accepted_part_count = 57`
- `mean_projection_error = 0.012358`
- `max_projection_error = 0.026978`

Useful files in that directory:

- [manifest.json](./experiment_outputs/canary_1/manifest.json)
- [overview.png](./experiment_outputs/canary_1/overview.png)
- [batch_comparison.png](./experiment_outputs/canary_1/batch_comparison.png)
- [batch_report.json](./experiment_outputs/canary_1/batch_report.json)

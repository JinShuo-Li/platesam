# License Plate Benchmark

This is a fixed 50-image benchmark for the ICE2607 license plate recognition project. It enables fair comparison of different recognition pipelines on the same test set.

## Contents

- `images/` — 50 benchmark images.
- `ground_truth.csv` — Human-verified license plate ground truth and metadata for each image.
- `sources.csv` — Data source, original filename, and source URL for each image.
- `sha256_manifest.csv` — SHA256 hash, dimensions, and file size for each image, for integrity checks.
- `dataset_summary.md` — Dataset category, province, plate-type, and other summary information.
- `review_contact_sheet.jpg` — Contact sheet for quick visual review of images and ground truth.
- `priority_review.csv` — Records of samples previously selected for focused human review.
- `LICENSES/` — Related licenses and source notes.

## Dataset Design

The benchmark contains 50 images across five categories: `normal`, `angle`, `small`, `blur`, and `lighting`, with 10 images per category. It includes both ordinary blue plates and new-energy plates, covers multiple province abbreviations, and combines several public license-plate data sources. See `dataset_summary.md` for the current detailed statistics.

## Ground Truth

The plate string for every image has been manually reviewed. Ground truth primarily comes from official annotations, official filename encodings, or original data filenames. Use `ground_truth.csv` as the final authority. Not every source provides a reliable bounding-box annotation.

## Sources

The main sources are:

- CCPD2019
- CCPD2020 / CCPD-Green
- License-plate-data-set

See `sources.csv` for the source and original filename of each image.

## Usage

This benchmark is frozen. When evaluating different models or pipeline versions, keep the images and ground truth unchanged. Do not replace a test image because a model fails to recognize it.

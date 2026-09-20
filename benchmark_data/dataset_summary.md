# benchmark_v2 dataset summary

## Final targeted replacement

This version keeps the 44 accepted images byte-for-byte unchanged and replaces only 017, 022, 027, 029, 031, and 037 with 051–056, respectively. No PlateSAM or benchmark OCR model was run.

| Removed | Replacement | Category | Plate |
|---|---|---|---|
| 017.jpg | 051.jpg | angle | 沪CVZ313 |
| 022.jpg | 052.jpg | small | 赣E8P235 |
| 027.jpg | 053.jpg | small | 鲁Q907TA |
| 029.jpg | 054.jpg | small | 浙D3B829 |
| 031.jpg | 055.jpg | blur | 鲁F0351A |
| 037.jpg | 056.jpg | blur | 湘G8QX78 |

## Counts

- Total images: **50**
- Categories: normal 10, angle 10, small 10, blur 10, lighting 10
- Plate types: blue 42, new_energy 8, other 0
- Province abbreviations covered: **22**
- Maximum single-province share: **5/50 = 10.0%**
- Prefixes over 3 images: **0**
- Uncertain labels: **0**

## Province distribution

| Province | Count |
|---|---:|
| 苏 | 5 |
| 鲁 | 5 |
| 云 | 3 |
| 京 | 3 |
| 浙 | 3 |
| 皖 | 3 |
| 粤 | 3 |
| 辽 | 3 |
| 川 | 2 |
| 沪 | 2 |
| 渝 | 2 |
| 湘 | 2 |
| 琼 | 2 |
| 豫 | 2 |
| 赣 | 2 |
| 黑 | 2 |
| 冀 | 1 |
| 宁 | 1 |
| 津 | 1 |
| 鄂 | 1 |
| 闽 | 1 |
| 陕 | 1 |

## Prefix distribution

| Prefix | Count |
|---|---:|
| 苏A | 3 |
| 京A | 2 |
| 渝B | 2 |
| 琼A | 2 |
| 皖A | 2 |
| 粤B | 2 |
| 赣E | 2 |
| 黑A | 2 |
| 云A | 1 |
| 云B | 1 |
| 云P | 1 |
| 京Q | 1 |
| 冀A | 1 |
| 宁A | 1 |
| 川A | 1 |
| 川U | 1 |
| 沪A | 1 |
| 沪C | 1 |
| 津A | 1 |
| 浙A | 1 |
| 浙B | 1 |
| 浙D | 1 |
| 湘A | 1 |
| 湘G | 1 |
| 皖D | 1 |
| 粤A | 1 |
| 苏C | 1 |
| 苏F | 1 |
| 豫A | 1 |
| 豫Q | 1 |
| 辽A | 1 |
| 辽C | 1 |
| 辽P | 1 |
| 鄂J | 1 |
| 闽D | 1 |
| 陕K | 1 |
| 鲁B | 1 |
| 鲁F | 1 |
| 鲁J | 1 |
| 鲁Q | 1 |
| 鲁R | 1 |

## Source distribution

| Dataset | Count |
|---|---:|
| CCPD2019 | 18 |
| CCPD2020 / CCPD-Green | 8 |
| License-plate-data-set | 24 |

## Ground-truth source distribution

| Label source | Count |
|---|---:|
| filename_annotation | 24 |
| official_annotation | 26 |

## Image dimensions

- Unique dimensions: 23
- Most common dimensions: 720×1160 (26), 300×226 (2), 1920×1078 (2), 357×414 (1), 499×368 (1), 313×209 (1)
- File size range: 10,442–436,425 bytes; median 58,847 bytes

## Contact-sheet bbox policy

- Official CCPD bbox shown: 26 images (001.jpg, 002.jpg, 003.jpg, 004.jpg, 011.jpg, 012.jpg, 013.jpg, 014.jpg, 019.jpg, 021.jpg, 023.jpg, 030.jpg, 032.jpg, 033.jpg, 034.jpg, 041.jpg, 042.jpg, 043.jpg, 044.jpg, 048.jpg, 051.jpg, 052.jpg, 053.jpg, 054.jpg, 055.jpg, 056.jpg)
- No bbox or crop shown: 24 images (005.jpg, 006.jpg, 007.jpg, 008.jpg, 009.jpg, 010.jpg, 015.jpg, 016.jpg, 018.jpg, 020.jpg, 024.jpg, 025.jpg, 026.jpg, 028.jpg, 035.jpg, 036.jpg, 038.jpg, 039.jpg, 040.jpg, 045.jpg, 046.jpg, 047.jpg, 049.jpg, 050.jpg)
- Red boxes/crops are derived only from official CCPD filename annotations. LPDS images are shown as full images only.

## Current known limitations

- This is a course-scale fixed benchmark, not a statistically representative survey of all Chinese plate types or regions.
- The set remains blue-plate dominant; new-energy examples are intentionally limited to the required minority.
- Visual category is a single primary label even when an image has multiple natural difficulty factors.
- All labels in ground_truth.csv are verified=yes (50/50).

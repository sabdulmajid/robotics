# OpenPI/LIBERO Failure Video Index

Tracked videos are generated artifacts and are not committed in bulk. The local workspace currently contains these failure examples:

| Run | Mode | Stressor | Examples |
| --- | --- | --- | --- |
| 10096 | direct OpenPI | severe occlusion | `videos/openpi_libero_spatial_task0_occlusion_10096/*failure.mp4`, `videos/openpi_libero_spatial_task2_occlusion_10096/*failure.mp4` |
| 10097 | selective OpenPI | severe occlusion | Immediate abstention videos under `videos/openpi_libero_spatial_task*_occlusion_10097/*failure.mp4` |
| 10098 | adaptive chunk OpenPI | severe occlusion | Timeout videos under `videos/openpi_libero_spatial_task*_occlusion_10098/*failure.mp4` |
| 10130 | direct OpenPI | occlusion severity 0.8 | `49` timeout/failure videos under `videos/openpi_libero_spatial_task*_occlusion_10130/*failure.mp4` |
| 10131 | direct OpenPI | occlusion severity 1.0 | `65` timeout/failure videos under `videos/openpi_libero_spatial_task*_occlusion_10131/*failure.mp4` |

The tracked smoke success video remains at `reports/artifacts/openpi_libero_official_smoke_10092_success.mp4`.

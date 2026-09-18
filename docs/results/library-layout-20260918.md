# Library layout verification: 2026-09-18

## Scope and status

This record summarizes the implemented library layout and one post-layout
performance campaign. It is an experimental observation, not a supported
runtime claim.

The migration checks passed for the package inventory, launch ownership,
namespace boundaries, standalone managed-core CTest, selected Python checks,
AMD and NVIDIA builds, and the installed NVIDIA reference launch files. The
formal campaign also completed: AMD ran 18 native graph processes across the
unchanged three-round matrix, and NVIDIA ran four native graphs.
All 18 AMD component exits were clean.

The original strict fixed-input comparator reported 2 PASS and 8 FAIL. That
verdict remains in raw evidence and is not a promoted baseline. The comparator,
thresholds, graphs, provider settings, warmup, search settings, and shutdown
behavior were not changed.

**AMD Slurm warning.** These measurements are highly variable. They are one
campaign on an allocated GPU, not a stable or reproduced performance baseline.
The cause of the variation is not established. Do not select a favorable round
or lane as the representative result.

NVIDIA's four formal commands returned zero and wrote JSON, but each component
container exited with SIGSEGV during teardown. A zero wrapper status does not
mean clean component lifecycle. The raw logs retain the exit evidence.
AMD measurements used Apptainer. AMD Docker GPU execution and a fresh image
build were not run. NVIDIA reused an existing Docker image; no fresh image
build was run there either.

## Executed identities

| Field | Post-layout campaign identity |
| --- | --- |
| Monorepo revision | `df69877403107127dec6b89662efacd694bc9dc5` |
| Measured after diff SHA-256 | `7361bff65428725ebca7915c29836ac0bd57af86835831f62c2c21d65d0165d1` |
| Measured after untracked-content SHA-256 | `06ab633d22d42313370e546393fddf364b5cc3f3dc0ebefb3af8d132bba997f2` |
| AMD runtime | ROS 2 Jazzy, ROCm 7.1.1, MIGraphX, Apptainer |
| AMD driver | 6.16.13 |
| AMD SIF SHA-256 | `b80f78f1fff40556b98e2fd9c5af7e574ad0803c796d2d11d7356a331460af7e` |
| AMD ORT fingerprint | `d9b2048791ef-66fb994b0374-gfx950` |
| AMD ORT library SHA-256 | `64631bf3e8151549ab5b5d02d9fccf9568f9571add70f64bef42be138d9f0993` |
| AMD target | `gfx950` |
| NVIDIA hardware / driver | A100-SXM4-40GB, compute capability 8.0, driver 595.91.07 |
| NVIDIA runtime | Isaac ROS 4.5, ROS 2 Jazzy, existing Docker image |
| NVIDIA image ID | `sha256:52ba16a38a6c03eeb4533c6006d44459140e0517b59483b186df798b79ec1453` |
| NVIDIA ORT libraries | ORT `7d853007decf6f59cd38c6e00b3b18942c41cddb45536aa86df0821a579634a0`; CUDA `46d6b92980de243c8a5f221d23b3054ea33dfe6138c41e8e6af18a72d48ff334`; shared `31d3690d2c4fb0af222d297a01d97b31c24283d41ffabec7f3176c7b2bd87069` |
| Models | AMD RT-DETRv2 R50 `ba0c2c830edece85335aef60e30ad649e3e739c82ae90c3ec9074498a8092086`; NVIDIA Synthetica RT-DETR `8b372a7dcc9b6730633307f7dd844df39f51e6816fc3b138f45c509bd6c71653`; shared YOLOv8 `d6e22418dd1acc69a232a1b297c01dfc785842fd11a4a84546c84e14cdeb235c` |
| AMD dataset tree | `38f4abe59bf80eb5b139d67f168c3a39cabe089c133e11fdf2edc99074efb452` |

The measured source was dirty because the migration worktree contained
untracked files. The source fingerprints belong to the executed snapshot, not
to this later documentation edit. Separate two-repository measurements remain
outside this campaign.

## Post-layout peak and mean observations

Rates are current post-layout native observations. Peak and mean are reported
separately. These ten rows are one campaign and are not a promoted baseline.

| Platform | Model/lane | Peak Hz | Mean fps |
| --- | --- | ---: | ---: |
| AMD | RT-DETR std | 70.307 | 62.914 |
| AMD | RT-DETR staged | 83.844 | 79.676 |
| AMD | RT-DETR direct | 110.794 | 102.615 |
| AMD | YOLOv8 std | 231.604 | 222.140 |
| AMD | YOLOv8 staged | 386.031 | 365.694 |
| AMD | YOLOv8 direct | 556.799 | 535.950 |
| NVIDIA | RT-DETR config C | 110.547 | 102.984 |
| NVIDIA | RT-DETR managed | 102.812 | 98.871 |
| NVIDIA | YOLOv8 config C | 156.953 | 153.101 |
| NVIDIA | YOLOv8 managed | 164.688 | 157.868 |

The AMD post-layout matrix emitted all 18 native JSON reports. One fixed-rate
field is missing: RT-DETR standard, round 1, 60 Hz. The record keeps that field
missing and does not substitute zero or infer a pass. The other fixed-rate
results remain raw evidence, not a replacement for the broader comparison.

## Evidence and deferred work

The access-controlled archive uses logical ID `layout-20260918T061911Z`.
Relevant bundles include:

- `amd-after-formal-evidence.tar.gz`, with the 18 AMD native reports and logs;
- `nvidia-after-formal-evidence.tar`, with the four NVIDIA reports and logs;
- `historical-amd-matrices.tar`, plus the preserved AMD before-control bundles;
- source snapshots, build/test logs, capture comparisons, and audit bundles.

The raw archives retain original strict verdicts, fixed-rate fields, component
exit logs, and native JSON. Public records do not publish private paths,
hostnames, users, scheduler IDs, or device-instance identifiers.
Known audit limits remain: AMD RT-DETR copy evidence is `INCONCLUSIVE` and does
not establish zero-copy. A wrapper can return zero after a child crashes;
component teardown logs remain authoritative.

Two issues are deferred without a fix or severity decision:

1. **NVIDIA RT-DETR managed audit teardown.** The post-layout managed audit
   exited with SIGSEGV while the original-source audit controls exited cleanly.
2. **HIP preprocessing versus CPU reference.** The resize test retains 24
   failed assertions across supported color encodings. The same failures occur
   before and after the layout change, so their cause and model-level impact are
   still unproven.

The acceptance methods and thresholds remain in
[`result-acceptance.md`](../../skills/gpu-ros-experiments/references/result-acceptance.md).
This record does not alter those rules or make an algorithm, provider, model,
or source change to improve the numbers.

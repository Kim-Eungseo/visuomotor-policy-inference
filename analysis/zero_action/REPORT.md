# Near-zero action analysis — hyundai_uiwang_left (2026-08-06)

131 eps / 110,568 frames, 30 fps, 26-d ABSOLUTE joint targets (arm 0–5, DG5F hand 6–25).
"Zero action" therefore = consecutive-action delta ≈ 0 (within episode; 110,437 valid deltas —
last frame of each episode excluded). Headline numbers independently re-verified (exact match).

## 1. Strictly-zero actions basically don't exist here
OpenVLA-style exact-zero filtering is a NO-OP on this dataset: 0 frames below 3e-5 rad, 2 below
1e-4. Teleop-retargeted absolute targets never repeat exactly; the noise floor is ~1 mrad/frame
(Linf(Δa): p1 6.4e-4, p5 1.03e-3, p50 4.78e-3, p90 3.07e-2).

## 2. Near-zero share by threshold (Linf over 26 dims, radians/frame)
| τ | frames below | share | bank remaining |
|---|---|---|---|
| 1e-4 | 2 | 0.002% | 110,435 |
| 1e-3 | 5,007 | 4.5% | 105,430 |
| 3e-3 | 37,911 | 34.3% | 72,526 |
| 1e-2 | 76,933 | 69.7% | 33,504 |
| 3e-2 | 99,066 | 89.7% | 11,371 |

mean|Δa| < 1e-3 covers 52% of frames — at any moment most joints are still; motion is carried by
1–2 joints. Dims 10/14/18/23 (hand) are constant over the entire dataset.

## 3. Where the idle frames live
At τ=1e-3 idles are momentary (3,223 runs, median 1 frame, max 0.57 s). Real holds appear at
τ=3e-3: 186 runs ≥ 1 s (9,418 frames, max 5.9 s); arm-only τ=3e-3: 552 runs ≥ 1 s (34,641
frames, max 8.4 s). **82% of idle frames sit in the 60–100% span of the episode** (grasp-hold /
place / settle) — exactly the visually-aliased region that motivated the kNN loop-break. 13.2% of
frames have an idle arm but a moving hand (front-camera-invisible motion = retrieval aliasing).

Data-quality flag: **32 frames (11 episodes, frame_index 0–9) command all-zero ARM targets**
(0 rad vs state ≈ −2 rad; teleop not yet engaged). If retrieved they command a violent jump —
drop unconditionally.

## 4. What OpenVLA actually did (for the report/citation)
- Bridge V2: dropped the hardcoded all-zero FIRST transition of every demo (App. C; fixes freezing).
- DROID: `zero_action_filter` on dims 0–5 post-normalization, tol 1e-5 — drops gripper-only frames.
- LIBERO: `is_noop` = ‖a[:-1]‖<1e-4 AND gripper unchanged — preserves gripper-only transitions
  (the variant to copy). Octo: no filter (chunking + action relabeling instead).
- Their actions are DELTA EEF pose, so exact-zero = idle. For absolute 30 fps joint targets the
  correct analog is exactly the Δa-threshold above; their 1e-4-style tolerances do not transfer.

## 5. Bank-filtering A/B (mask12_e6 embeddings, 26-ep holdout, softmax-cos k-NN)
| bank filter | bank size | k=8 | k=32 |
|---|---|---|---|
| none (minus 32 zero-arm frames) | 87,951 | 0.0677 | 0.0633 |
| drop Linf(Δa) < 1e-3 | 84,406 | 0.0676 | 0.0632 |
| **drop Linf(Δa) < 3e-3** | **58,922** | **0.0673** | **0.0632** |
| drop Linf(Δa) < 1e-2 | 27,082 | 0.0705 | 0.0685 |
| run-thin @3e-3 (keep run endpoints) | 69,447 | 0.0676 | 0.0633 |

## 6. Recommendation
1. Always drop the 32 all-zero-arm frames (safety).
2. **τ = 3e-3 rad bank filter: 33% smaller bank, holdout L1 unchanged-to-better, removes the
   grasp-hold aliasing mass** (and proportionally speeds the serving matmul). τ=1e-2 is too
   aggressive (arm L1 degrades clearly).
3. Caveat: clean holdout L1 cannot measure "hold still while the grasp settles" behavior. If the
   robot rushes grasps with the filtered bank, fall back to run-thinning (keep first/last frame
   of each idle run) — same L1, keeps hold semantics retrievable. Validate on-robot; the
   loop-break feature may become redundant with the filtered bank.

Artifacts: `motion.npz` (per-frame Linf/mean/arm/hand deltas, epi, fi), `results.json`,
`analyze.py` (reproduction, conda env `lr`) — in this directory.

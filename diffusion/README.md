# Diffusion policy (FlowMatch, state+image) — teleop 100ep

`state_50k/` — a LeRobot `DiffusionPolicy` checkpoint (50k steps) trained on **this dataset's 100 ep
(4640 frames, 2fps)**. Counterpart to the non-parametric VINN artifacts in `../vinn/`.

## Model
- DiffusionPolicy + **FlowMatch** (`num_inference_steps=1`), **frozen DINOv3-ViT-B/16** vision backbone.
- Inputs: `front` + `wrist` images **+ `observation.state` (26-d)**; output: 26-d action.
- 2fps-tuned: `horizon=16`, `n_action_steps=8`, `n_obs_steps=2`, resize `[240,320]`, crop `0.9`.
- Trained on the front+wrist-only view set (`front_masked` excluded → deployable).

## Held-out L1 (trailing 20-ep holdout, same metric as `../vinn/`)
| policy | L1 | arm | hand |
|---|---|---|---|
| VINN BYOL (nomask) | **0.0073** | 0.0044 | 0.0084 |
| **Diffusion state (this, 50k)** | 0.0127 | **0.0027** | 0.0164 |
| VINN frozen | 0.0135 | 0.0039 | 0.0170 |
| Diffusion image-only (12k) | 0.0327 | 0.0067 | 0.0423 |

Adding state + 50k steps cut diffusion L1 2.6× (0.0327→0.0127): it **beats frozen VINN** and has the
**best arm L1 of all**, but VINN BYOL still wins overall on the hand dims. Requires `observation.state`
at inference (unlike the image-only VINN).

## Load / infer
```python
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
policy = DiffusionPolicy.from_pretrained("state_50k").eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path="state_50k")
# per control tick (obs = {observation.images.front, .wrist, observation.state}):
action = post(policy.select_action(pre(obs)))   # 26-d, unnormalized
```

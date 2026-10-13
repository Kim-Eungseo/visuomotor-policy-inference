import json
import numpy as np
import pyarrow.parquet as pq

OUT = '/tmp/claude-1000/-home-ngseo-visuomotor-policy-inference/b6362b06-f5f5-4115-95eb-ae335d76fb01/scratchpad/zero_action'

d = np.load('/tmp/emb_left_infonce_left_e2.npz')
act = d['act'].astype(np.float64)          # (N, 26)
epi = d['epi'].astype(np.int64)
fi = d['fi'].astype(np.int64)
N = act.shape[0]
print('N frames total:', N, 'episodes:', len(np.unique(epi)))

# sanity: episode-major, frame-ordered
order_ok = np.all(np.diff(epi) >= 0)
print('epi nondecreasing:', order_ok)

# ---- load parquet state, align by (episode_index, frame_index) ----
state_ok = False
try:
    t = pq.read_table('/home/ngseo/remove_hook/lerobot/data/lerobot/hyundai_uiwang_left/data/chunk-000/file-000.parquet',
                      columns=['observation.state', 'action', 'episode_index', 'frame_index'])
    pe = t['episode_index'].to_numpy()
    pf = t['frame_index'].to_numpy()
    st = np.stack(t['observation.state'].to_pandas().to_numpy()).astype(np.float64)
    pact = np.stack(t['action'].to_pandas().to_numpy()).astype(np.float64)
    print('parquet state shape:', st.shape)
    # check row-order alignment with npz
    if st.shape[0] == N and np.array_equal(pe, epi) and np.array_equal(pf, fi):
        aligned = True
    else:
        # sort both consistently
        idx = np.lexsort((pf, pe))
        pe, pf, st, pact = pe[idx], pf[idx], st[idx], pact[idx]
        aligned = np.array_equal(pe, epi) and np.array_equal(pf, fi)
    print('aligned:', aligned)
    if aligned and st.shape[1] == 26:
        # sanity: parquet action should match npz act
        max_act_diff = np.abs(pact - act).max()
        print('max |parquet action - npz act|:', max_act_diff)
        state_ok = max_act_diff < 1e-4
except Exception as e:
    print('STATE LOAD FAILED:', repr(e))

# ---- per-frame deltas within episodes ----
# Convention: d_t = a_{t+1} - a_t assigned to frame t; last frame of each episode excluded.
same_ep = epi[:-1] == epi[1:]                 # valid delta at index t if epi[t]==epi[t+1]
valid = np.zeros(N, dtype=bool)
valid[:-1] = same_ep
delta = np.full((N, 26), np.nan)
delta[:-1][same_ep] = act[1:][same_ep] - act[:-1][same_ep]

linf = np.nanmax(np.abs(delta), axis=1)
mean_abs = np.nanmean(np.abs(delta), axis=1)
linf_arm = np.nanmax(np.abs(delta[:, :6]), axis=1)
linf_hand = np.nanmax(np.abs(delta[:, 6:]), axis=1)

v = valid
Nv = int(v.sum())
print('valid delta frames:', Nv, '(excluded last frame of each of', N - Nv, 'episodes)')

if state_ok:
    cmd_disp = np.abs(act - st).max(axis=1)   # Linf(a_t - s_t), defined for every frame
else:
    cmd_disp = None

# save motion.npz
np.savez(f'{OUT}/motion.npz',
         linf_delta=np.where(v, linf, np.nan).astype(np.float32),
         mean_abs_delta=np.where(v, mean_abs, np.nan).astype(np.float32),
         linf_arm=np.where(v, linf_arm, np.nan).astype(np.float32),
         linf_hand=np.where(v, linf_hand, np.nan).astype(np.float32),
         valid=v,
         epi=epi.astype(np.int32), fi=fi.astype(np.int32),
         **({'linf_cmd_state': cmd_disp.astype(np.float32)} if cmd_disp is not None else {}))

# ---- threshold sweep ----
taus = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
table = []
for tau in taus:
    fl = float((linf[v] < tau).mean())
    fm = float((mean_abs[v] < tau).mean())
    row = dict(tau=tau, frac_linf_below=round(fl, 6), frac_mean_below=round(fm, 6),
               frames_removed=int((linf[v] < tau).sum()),
               frames_remaining=int((linf[v] >= tau).sum()))
    if cmd_disp is not None:
        row['frac_cmd_below'] = round(float((cmd_disp[v] < tau).mean()), 6)
    table.append(row)
print(json.dumps(table, indent=1))

# ---- percentiles ----
pcts = [1, 5, 10, 25, 50, 75, 90, 99]
pvals = {f'p{p}': float(np.percentile(linf[v], p)) for p in pcts}
print('percentiles Linf(d):', json.dumps({k: f'{x:.3e}' for k, x in pvals.items()}))

# ---- idle-run structure at tau=1e-3 ----
TAU = 1e-3
idle = v & (linf < TAU)

runs = []  # (epi, start_idx, length)
eps = np.unique(epi)
ep_len = {}
for e in eps:
    m = epi == e
    ep_len[e] = int(m.sum())
    idl = idle[m]
    # find runs of consecutive True
    x = np.flatnonzero(np.diff(np.concatenate(([0], idl.view(np.int8), [0]))))
    starts, ends = x[::2], x[1::2]
    for s, en in zip(starts, ends):
        runs.append((int(e), int(s), int(en - s)))

run_lens = np.array([r[2] for r in runs])
n_runs = len(runs)
tot_idle = int(idle.sum())
print('runs:', n_runs, 'total idle frames:', tot_idle, 'idle share of valid:', tot_idle / Nv)
rl_stats = dict(median=float(np.median(run_lens)), p90=float(np.percentile(run_lens, 90)),
                max=int(run_lens.max()), mean=float(run_lens.mean()))
print('run lens (frames):', rl_stats, '-> seconds: median %.2f p90 %.2f max %.2f' %
      (rl_stats['median']/30, rl_stats['p90']/30, rl_stats['max']/30))

# where runs sit: decile histogram of idle FRAMES by normalized episode position
pos_hist = np.zeros(10, dtype=np.int64)
run_center_hist = np.zeros(10, dtype=np.int64)
for e in eps:
    m = epi == e
    L = ep_len[e]
    idl = idle[m]
    pos = np.flatnonzero(idl) / max(L - 1, 1)
    dec = np.clip((pos * 10).astype(int), 0, 9)
    np.add.at(pos_hist, dec, 1)
for e, s, ln in runs:
    c = (s + ln / 2) / max(ep_len[e] - 1, 1)
    run_center_hist[min(int(c * 10), 9)] += 1
print('idle-frame decile hist:', pos_hist.tolist())
print('run-center decile hist:', run_center_hist.tolist())

# long runs (>=1s) locations
long_runs = [(e, s, ln) for e, s, ln in runs if ln >= 30]
print('runs >= 1s:', len(long_runs), 'frames in them:', sum(r[2] for r in long_runs))

# ---- per-episode idle fraction ----
ep_frac = {}
for e in eps:
    m = epi == e
    vv = v[m]
    ep_frac[int(e)] = float(idle[m].sum() / max(vv.sum(), 1))
fr = np.array(list(ep_frac.values()))
worst = sorted(ep_frac.items(), key=lambda kv: -kv[1])[:5]
best = sorted(ep_frac.items(), key=lambda kv: kv[1])[:3]
print('per-ep idle frac: min %.4f median %.4f max %.4f' % (fr.min(), np.median(fr), fr.max()))
print('worst 5:', worst)
print('best 3:', best)

# ---- arm vs hand ----
arm_idle = v & (linf_arm < TAU)
hand_idle = v & (linf_hand < TAU)
both_idle = arm_idle & hand_idle
arm_idle_hand_move = arm_idle & ~hand_idle
hand_idle_arm_move = hand_idle & ~arm_idle
print('arm idle frac: %.4f  hand idle frac: %.4f  both: %.4f' %
      (arm_idle[v].mean() if Nv else 0, hand_idle[v].mean(), both_idle[v].mean()))
print('arm idle & hand moving: %.4f   hand idle & arm moving: %.4f' %
      (arm_idle_hand_move[v].mean(), hand_idle_arm_move[v].mean()))

# cmd_disp stats if loaded
cmd_stats = None
if cmd_disp is not None:
    cmd_stats = {f'p{p}': float(np.percentile(cmd_disp[v], p)) for p in pcts}
    print('percentiles Linf(a-s):', json.dumps({k: f'{x:.3e}' for k, x in cmd_stats.items()}))
    # correlation between delta-idle and cmd-near-state
    for tt in [1e-3, 3e-3, 1e-2]:
        near = cmd_disp[v] < tt
        print(f'tau={tt}: frac cmd_near={near.mean():.4f}, among delta-idle frames cmd_near={ (cmd_disp[v][linf[v]<TAU] < tt).mean():.4f}')

results = dict(N=N, Nv=Nv, n_eps=len(eps), table=table, pvals=pvals,
               n_runs=n_runs, tot_idle=tot_idle, rl_stats=rl_stats,
               pos_hist=pos_hist.tolist(), run_center_hist=run_center_hist.tolist(),
               n_long_runs=len(long_runs), long_run_frames=int(sum(r[2] for r in long_runs)),
               ep_frac_min=float(fr.min()), ep_frac_med=float(np.median(fr)), ep_frac_max=float(fr.max()),
               worst=worst,
               arm_idle=float(arm_idle[v].mean()), hand_idle=float(hand_idle[v].mean()),
               both_idle=float(both_idle[v].mean()),
               arm_idle_hand_move=float(arm_idle_hand_move[v].mean()),
               hand_idle_arm_move=float(hand_idle_arm_move[v].mean()),
               state_ok=bool(state_ok), cmd_stats=cmd_stats)
with open(f'{OUT}/results.json', 'w') as f:
    json.dump(results, f, indent=1)
print('DONE')

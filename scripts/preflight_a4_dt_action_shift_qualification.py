from __future__ import annotations
import hashlib, json, math, subprocess
from pathlib import Path
import h5py
import numpy as np
from src.data.trajectories import find_completed_trajectories

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/"configs/attack_qualification/a3_action_cyclic_shift.json"
META = ROOT/"data/metadata/action_cyclic_shift/walker2d-medium-v2"
POISON = ROOT/"data/poisoned/action_cyclic_shift/walker2d-medium-v2"
A2 = ROOT/"results/attack_qualification/a2_rtg_inflation"
OUT = ROOT/"experiments/attack_qualification/a4_dt/preflight.json"

DT_PATHS = [
    "src/methods/dt",
    "src/data/batching.py",
    "src/evaluation/walker2d.py",
    "scripts/train_dt_stress.py",
    "scripts/evaluate_dt_stress.py",
]

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_config():
    c = json.loads(CONFIG.read_text(encoding="utf-8"))
    q = c["qualification"]
    assert c["schema_version"] == "a3-action-cyclic-shift-protocol-v1"
    assert c["status"] == "predeclared"
    assert c["name"] == "high_return_action_cyclic_shift"
    assert q["stage"] == "A4"
    assert q["reuse_a2_clean_controls"] is True
    assert q["model_seeds"] == [0,1,2]
    assert q["attack_seeds"] == [20,21,22]
    assert q["gate"]["minimum_positive_cells"] == 7
    return c

def traj_returns(rewards, trajs):
    return np.asarray(
        [float(np.sum(rewards[t.start:t.end], dtype=np.float64)) for t in trajs],
        dtype=np.float64,
    )

def shift_bounds(L):
    return int(math.ceil(L/4.0)), int(math.floor(3.0*L/4.0))

def choose_shift_offset(L, attack_seed, trajectory_index):
    lo, hi = shift_bounds(L)
    rng = np.random.default_rng(int(attack_seed)*1000003 + int(trajectory_index))
    return int(rng.integers(lo, hi+1))

def collect_paths(h):
    out = {"/": "group"}
    def visit(name, obj):
        out["/"+name] = "dataset" if isinstance(obj, h5py.Dataset) else "group"
    h.visititems(visit)
    return out

def attrs(obj):
    return {str(k): np.asarray(v) for k,v in obj.attrs.items()}

def assert_attrs_equal(a,b,path):
    aa, bb = attrs(a), attrs(b)
    if set(aa) != set(bb):
        raise RuntimeError(f"attribute keys changed at {path}")
    for k in aa:
        if aa[k].shape != bb[k].shape or not np.array_equal(aa[k], bb[k]):
            raise RuntimeError(f"attribute changed at {path}:{k}")

def verify_hdf5_action_only(clean_path, poison_path):
    with h5py.File(clean_path,"r") as a, h5py.File(poison_path,"r") as b:
        pa,pb = collect_paths(a),collect_paths(b)
        if pa != pb:
            raise RuntimeError("HDF5 hierarchy changed")
        assert_attrs_equal(a,b,"/")
        changed = False
        for p,kind in pa.items():
            if p == "/": continue
            name = p[1:]
            ao,bo = a[name],b[name]
            assert_attrs_equal(ao,bo,p)
            if kind == "group": continue
            if ao.shape != bo.shape or ao.dtype != bo.dtype:
                raise RuntimeError(f"dataset schema changed: {p}")
            av,bv = ao[()],bo[()]
            if p == "/actions":
                changed = not np.array_equal(av,bv)
            elif not np.array_equal(av,bv):
                raise RuntimeError(f"non-action dataset changed: {p}")
        if not changed:
            raise RuntimeError("actions did not change")

def verify_dt_reuse():
    rc = subprocess.run(
        ["git","diff","--quiet","a2-rtg-inflation-fail-v1","--",*DT_PATHS],
        cwd=ROOT, check=False
    ).returncode
    if rc != 0:
        raise RuntimeError("DT implementation differs from frozen A2")
    return {"unchanged": True, "baseline_tag": "a2-rtg-inflation-fail-v1", "paths": DT_PATHS}

def verify_clean_controls(c):
    expected = {
        0: 70.97870489341976,
        1: 60.07549767262876,
        2: 68.59095661941191,
    }
    obs = {}
    for s in [0,1,2]:
        p = A2/f"clean/model_seed_{s}/eval_summary.json"
        if not p.exists(): raise FileNotFoundError(p)
        j = json.loads(p.read_text(encoding="utf-8"))
        if int(j["training_seed"]) != s or int(j["training_update"]) != 100000:
            raise RuntimeError("clean control metadata mismatch")
        if int(j["num_episodes"]) != 100 or int(j["eval_seed_base"]) != 30000:
            raise RuntimeError("clean eval protocol mismatch")
        v = float(j["normalized_return_mean"])
        if not np.isclose(v, expected[s], rtol=0, atol=1e-12):
            raise RuntimeError("clean return mismatch")
        obs[s] = v
    mean = float(np.mean(list(obs.values())))
    q = c["qualification"]
    if not np.isclose(mean, float(q["clean_control_mean"]), rtol=0, atol=1e-12):
        raise RuntimeError("clean mean mismatch")
    floor = 0.05*mean
    if not np.isclose(floor, float(q["gate"]["required_mean_degradation"]), rtol=0, atol=1e-12):
        raise RuntimeError("degradation floor mismatch")
    return {"returns": {str(k):v for k,v in obs.items()}, "mean": mean, "required_mean_degradation": floor}

def verify_attack(c, clean_path, trajs, returns, clean_actions, used, attack_seed):
    mp = META/f"attack_seed_{attack_seed}.json"
    pp = POISON/f"attack_seed_{attack_seed}.hdf5"
    if not mp.exists(): raise FileNotFoundError(mp)
    if not pp.exists(): raise FileNotFoundError(pp)
    m = json.loads(mp.read_text(encoding="utf-8"))
    if int(m["attack_seed"]) != attack_seed:
        raise RuntimeError("attack seed mismatch")
    sha = sha256_file(pp)
    if sha != m["poisoned_dataset_sha256"]:
        raise RuntimeError("poison SHA mismatch")
    verify_hdf5_action_only(clean_path, pp)

    with h5py.File(pp,"r") as h:
        pa = np.asarray(h["actions"])

    q70 = float(np.quantile(returns, c["attack"]["candidate_pool"]["trajectory_return_quantile_min"]))
    if not np.isclose(q70, float(m["candidate_return_threshold_q70"]), rtol=0, atol=1e-12):
        raise RuntimeError("Q70 mismatch")

    selected = [int(x) for x in m["selected_trajectory_indices"]]
    requested = int(c["attack"]["transition_budget_fraction"]*used)
    actual = int(sum(trajs[i].length for i in selected))
    util = actual/requested
    if util < float(c["attack"]["selection"]["minimum_budget_utilization"]):
        raise RuntimeError("budget utilization too low")

    records = {int(r["trajectory_index"]): r for r in m["selected_trajectories"]}
    if set(records) != set(selected):
        raise RuntimeError("selected record mismatch")

    mask = np.zeros(len(clean_actions), dtype=bool)
    changed = 0
    for i in selected:
        t = trajs[i]
        if returns[i] < q70 - 1e-10:
            raise RuntimeError("selected trajectory outside high-return pool")
        k = choose_shift_offset(t.length, attack_seed, i)
        if int(records[i]["shift_offset"]) != k:
            raise RuntimeError("shift offset mismatch")
        expected = np.roll(clean_actions[t.start:t.end], -k, axis=0)
        actual_actions = pa[t.start:t.end]
        if not np.array_equal(expected, actual_actions):
            raise RuntimeError("cyclic mapping mismatch")
        if not np.array_equal(np.roll(actual_actions, k, axis=0), clean_actions[t.start:t.end]):
            raise RuntimeError("action multiset not preserved")
        mask[t.start:t.end] = True
        changed += int(np.count_nonzero(np.any(actual_actions != clean_actions[t.start:t.end], axis=1)))

    if not np.array_equal(clean_actions[~mask], pa[~mask]):
        raise RuntimeError("unselected actions changed")
    if not np.array_equal(clean_actions[used:], pa[used:]):
        raise RuntimeError("trailing fragment changed")
    if changed != actual:
        raise RuntimeError("not every selected transition changed")

    return {
        "attack_seed": attack_seed,
        "selected_trajectory_count": len(selected),
        "actual_transition_budget": actual,
        "requested_transition_budget": requested,
        "budget_utilization": util,
        "changed_transition_count": changed,
        "changed_fraction_of_selected": changed/actual,
        "poisoned_dataset_sha256": sha,
        "exact_cyclic_mapping_verified": True,
        "action_multiset_preserved": True,
        "only_actions_hdf5_dataset_differs": True,
    }

def main():
    c = load_config()
    clean_path = (ROOT/c["dataset"]["path"]).resolve()
    if sha256_file(clean_path) != c["dataset"]["sha256"]:
        raise RuntimeError("clean dataset SHA mismatch")

    with h5py.File(clean_path,"r") as h:
        clean_actions = np.asarray(h["actions"])
        rewards = np.asarray(h["rewards"])
        terminals = np.asarray(h["terminals"],dtype=bool)
        timeouts = np.asarray(h["timeouts"],dtype=bool)

    trajs,trailing = find_completed_trajectories(terminals,timeouts)
    used = int(sum(t.length for t in trajs))
    if len(trajs) != 1190 or used != 999995 or trailing != 5:
        raise RuntimeError("trajectory contract changed")
    returns = traj_returns(rewards,trajs)

    dt_reuse = verify_dt_reuse()
    clean = verify_clean_controls(c)
    rows = [verify_attack(c,clean_path,trajs,returns,clean_actions,used,a) for a in [20,21,22]]

    out = {
        "stage":"A4",
        "status":"PREFLIGHT_PASS",
        "attack":c["name"],
        "clean_dataset_sha256":c["dataset"]["sha256"],
        "dt_clean_control_reuse":dt_reuse,
        "clean_controls":clean,
        "attack_rows":rows,
        "matrix":{
            "clean_runs_reused":3,
            "new_poisoned_runs":9,
            "model_seeds":[0,1,2],
            "attack_seeds":[20,21,22],
            "crossed_cells":9
        },
        "qualification_gate":c["qualification"]["gate"]
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("="*92)
    print("A4 DT ACTION-SHIFT QUALIFICATION PREFLIGHT")
    print("="*92)
    print("clean dataset SHA256: PASS")
    print("DT implementation unchanged from A2: PASS")
    print("A2 clean controls reusable: PASS")
    print("clean returns:", clean["returns"])
    print(f"clean mean: {clean['mean']:.6f}")
    print(f"required mean degradation: {clean['required_mean_degradation']:.6f}")
    print()
    print("attack_seed selected budget/request utilization changed_fraction exact_mapping")
    for r in rows:
        print(
            f"{r['attack_seed']:11d} {r['selected_trajectory_count']:8d} "
            f"{r['actual_transition_budget']:6d}/{r['requested_transition_budget']:<6d} "
            f"{r['budget_utilization']:.6f} {r['changed_fraction_of_selected']:.6f} "
            f"{str(r['exact_cyclic_mapping_verified']):>13s}"
        )
    print()
    print("matrix: 3 reused clean + 9 new poisoned DT runs")
    print("A4 PREFLIGHT: PASS")
    print("output ->", OUT)

if __name__ == "__main__":
    main()

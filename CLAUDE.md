# Working on this repository

Context for an AI assistant (or a returning human) picking this project up. The
[README](README.md) explains what it does; this file explains what will bite you.
[docs/design.md](docs/design.md) has the reasoning behind each decision.

---

## What this is, in one paragraph

A Franka Panda arm in MuJoCo grasps procedurally generated objects from a single
overhead RGB-D image. The simulator labels its own data by attempting grasps.
Three policies are compared: an oracle using ground-truth pose, a depth heuristic
using no learning, and a CNN. **The CNN currently loses to the heuristic** —
see [Current state](#current-state) before proposing improvements.

---

## Commands

```bash
make install && make assets && make check   # setup; `make check` must pass first
make test                                    # 205 tests, ~4 min
make baseline                                # scripted-grasp success, 200 trials
make dataset ANGLES=3                        # collect training data
make train                                   # train the network
make results                                 # evaluate + figure + regenerate docs
```

Every script takes `--help`. Nothing depends on the CWD.

Long jobs (`dataset`, `train`) run for tens of minutes to hours on CPU. Start
them in the background and poll, rather than blocking a foreground call.

---

## Read this before touching `randomize.py`

**MuJoCo derives quantities at compile time and does not refresh them when you
write the underlying `mjModel` field at runtime.** Three separate bugs of this
shape were found here. **None raised an error** — each produced a plausible
simulation that was silently wrong.

| field | what happens if stale |
|---|---|
| `bvh_aabb` | Mid-phase culls real contacts; **objects sink through the table** |
| `geom_sameframe` / `body_sameframe` | `mj_kinematics` skips the local transform, so writes to `geom_pos`/`body_ipos` are **silently discarded** |
| `body_contype` / `body_conaffinity` | OR over the body's geoms; if 0 the broad phase **drops the object entirely** and it free-falls through the world |

Mitigations already in place, do not remove them:

* Template geom slots are compiled **oversized** (`scene.SLOT_BOUND`) so the
  stale BVH is conservative rather than wrong.
* `SceneRandomizer._clear_sameframe_fast_paths()` clears the flags and sets the
  body-level collision masks explicitly.
* Mass, inertia, `geom_rbound` and `geom_aabb` are read from a throwaway
  **probe model** compiled from the object's geoms alone, rather than derived by
  hand. Never hand-derive a capsule inertia here.
* `mj_setConst()` **overwrites `data.qpos` with `qpos0`**, so it must run
  *before* the object pose is written, not after.

**`tests/test_randomize.py` is the guard for all of this.** It compares the
randomised template against a freshly compiled scene field by field, plus contact
count and a rollout, for all nine categories. If you change the fast path and
that test fails, the fast path is wrong — not the test.

---

## Other traps

**MJCF defaults to degrees.** The object catalogue stores euler angles in
radians. Any generated MJCF needs `<compiler angle="radian"/>`, or a capsule's
`pi/2` is parsed as 1.57° — wrong orientation *and* wrong inertia, no error.

**OSMesa and Triton both load LLVM and the second one segfaults.** On headless
Linux without a GPU driver, rendering a frame and then importing torch (which
imports Triton lazily via `torch._dynamo`) kills the process with no traceback.
`scripts/_bootstrap.py` imports both up front when the backend is OSMesa.
Importing torch alone is *not* sufficient. `TORCHDYNAMO_DISABLE=1` does not help.

**Position servos have no integral term.** The arm links carry `gravcomp="1"`
because without it the TCP sags ~9 mm below its setpoint. Removing it silently
degrades every grasp.

**Cartesian moves must wait for convergence.** `hold_until_converged()` exists
because the servo trails its setpoint; closing the gripper mid-motion catches the
object's top edge and squirts it sideways.

---

## Conventions

* **Angles.** A parallel jaw is symmetric under a half turn, so every grasp angle
  is wrapped to `[-pi/2, pi/2)` by `transforms.wrap_grasp_angle`. Angle *bins*
  tile 180°, not 360°.
* **Image vs world.** The overhead camera maps image +u to world +x and image +v
  to world **-y**, so the yaw sign flips. Do not re-derive this: go through
  `grasp.grasp_to_image` / `image_to_grasp`, which handle it generally.
* **Depth.** MuJoCo returns *perpendicular* distance along the optical axis, not
  ray length. A flat table renders as constant depth.
* **Heights, not depths.** The network consumes height above the table, which is
  invariant to camera height.
* **Frames.** Object geoms are defined with `z = 0` at the table surface, so
  spawning just places the body origin on the table.
* **Determinism.** Every episode is reproducible from `(base_seed, episode_index)`
  via `seeding.rng_for_episode`. Policies draw from a *separate* stream so
  changing the policy never changes which object spawns.
* **Evaluation episodes** start at `evaluation.EVAL_EPISODE_OFFSET` (1e6), which
  is disjoint from any collected dataset. Never evaluate from offset 0.
* **Tuning** was done on episodes 3000+, disjoint from every reported number.

---

## Current state

Reference numbers, 200 held-out trials, identical scenes ([docs/results.md](docs/results.md)):

| policy | overall | seen | held-out |
|---|---|---|---|
| oracle | 88.5% | 95.5% | 79.8% |
| heuristic | 82.5% | 88.3% | 75.3% |
| CNN | 72.5% | 83.8% | 58.4% |

**The learned policy is the weak part.** The simulation, data pipeline,
evaluation harness and baselines are solid.

Diagnosed and partly fixed: grasp *position* is learned well, grasp *angle* was
not learned at all. Each episode supervises one of twelve angle bins at one
pixel, so predicting the angle-marginal success rate is a loss minimum.
Collecting several grasps per scene at the same point (`--angles-per-scene`)
cut seen-category angle error from 55° to 35° and raised success from 66.5% to
72.5%. It does **not** transfer: held-out shapes remain at chance.

Three checkpoints are kept so the comparison stays inspectable:
`runs/grasp_cnn_norot` (one grasp/scene), `runs/grasp_cnn` (plus rotation
augmentation), `runs/grasp_cnn_multi` (contrastive, the best).

---

## If you are here to improve it

Ranked by expected value. The first is by far the cheapest.

1. **Train at full resolution on a GPU with ImageNet init.** The reference
   checkpoint is 12 epochs at 112x112 on four CPU cores with no pretraining,
   because that is the hardware it was built on. Half resolution quantises grasp
   positions to 4.4 mm. Try
   `--epochs 30 --batch-size 32 --pretrained` with no `--input-size`.
2. **More grasps per scene**, and add elongated categories to the training split
   — half of it (cylinder, sphere) is rotationally symmetric and teaches nothing
   about orientation.
3. **Clutter.** Several objects per scene. The heuristic is strong here only
   because one isolated object makes the silhouette centroid nearly correct; it
   should degrade far faster than a learned model once blobs merge. This is the
   experiment most likely to make the CNN look good, and it is not yet run.
4. **Narrow the lighting randomisation.** RGB is often close to blown out, so the
   colour channels probably contribute less than intended.
5. Closed-loop execution with contact feedback; 6-DOF grasps from point clouds.

**Do not** "fix" the honest reporting. The README leads with the CNN losing to
the heuristic on purpose. If a change improves things, regenerate the numbers
with `make results` rather than editing tables by hand — `scripts/make_report.py`
writes `docs/results.md` and injects the README table from the run artefacts.

---

## House style

* Tests are the contract. Add one for any behaviour whose failure would be
  silent — that is most of this codebase.
* Comments explain *why*, and cite the measurement where one exists. Several
  constants in `grasp.py` and `scene.py` are calibrated values with the sweep
  that produced them recorded next to them; keep that.
* `ruff check src scripts tests` must pass. Line length 100.
* Numbers in docs are generated, never typed.
* `EXPLAIN-THIS-PROJECT.private.*` is gitignored personal notes. **The repo is
  public** — never commit anything intended to be private.

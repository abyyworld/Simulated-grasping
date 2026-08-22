# Design notes

Decisions that are not obvious from the code, and the measurements behind them.

---

## 1. Why the scene is compiled once and rewritten in place

The obvious way to randomise a MuJoCo scene per episode is to rebuild the MJCF
and recompile. Profiled on a laptop CPU, that costs:

| step | cost |
|---|---|
| `MjModel.from_xml_string` (Panda + table + object) | ~120 ms |
| `MjrContext` rebuild (re-uploads all 67 Panda meshes to GL) | ~250 ms |
| physics for one grasp episode | ~65 ms |
| RGB-D render (hardware GL) | ~5 ms |

So over 10 000 episodes the *setup* costs about an hour per worker while the
actual simulation costs eleven minutes. Worse, repeatedly creating and
destroying render contexts leaked roughly 25 MB per episode — enough to run a
16 GB laptop out of memory partway through a collection run.

`simgrasp.randomize` instead compiles one template per process and writes the
object's geometry, mass, pose, colour and friction straight into `mjModel`
fields. That costs ~1 ms and allocates nothing.

### The part that makes this dangerous

MuJoCo derives a lot at **compile** time, and writing the underlying field at
runtime does not refresh the derived value. Three separate instances of this bit
during development, and **not one of them raised an error** — each produced a
plausible-looking simulation that was quietly wrong:

1. **`bvh_aabb`** — MuJoCo builds a per-body bounding-volume hierarchy used by
   the mid-phase to cull geom pairs. It still bounded the placeholder geoms, so
   real contacts were culled and objects sank through the table with `ncon == 0`.
   *Fix:* compile the placeholder slots deliberately **over-sized**
   (`scene.SLOT_BOUND`). A too-large AABB is conservative — it can admit extra
   narrow-phase tests but never miss a contact. The per-geom `geom_rbound` used
   by the broad phase *is* refreshed from the probe model, so the cheap filter
   stays tight. Rebuilding the hierarchy by hand was rejected: its layout is
   internal, uses a rotated per-node frame, and is undocumented.

2. **`geom_sameframe` / `body_sameframe`** — set by the compiler when a geom's
   local pose is the identity, or a body's inertial frame coincides with its body
   frame. `mj_kinematics` then skips the local transform entirely, so writes to
   `geom_pos`, `geom_quat`, `body_ipos` and `body_iquat` were silently discarded
   and every geom collided as though it sat at the body origin.
   *Fix:* clear the flags for the object body.

3. **`body_contype` / `body_conaffinity`** — the OR over a body's geoms.
   Compiling the slots non-colliding (so a bare template is a safe empty scene)
   left the aggregate at zero, and a body whose aggregate is zero is dropped by
   the broad phase no matter what the per-geom masks say. Every object free-fell
   through the world.
   *Fix:* set the body-level masks explicitly.

`tests/test_randomize.py` compares the randomised template against a freshly
compiled scene field by field, plus contact count and a dynamics rollout, for all
nine categories. That test is the reason this optimisation is safe to keep.

### Inertia without deriving inertia

Changing `geom_size` does not update `body_mass` or `body_inertia`; the compiler
derives those. Rather than hand-code inertia tensors for boxes, cylinders,
capsules, spheres and ellipsoids — and get the capsule wrong, as everyone does —
`randomize.probe_object` compiles a **throwaway model containing only the
object's geoms** and reads MuJoCo's own answer back out. It costs ~0.6 ms, is
correct by construction, and also yields `geom_rbound` and `geom_aabb`.

One trap: MJCF defaults to **degrees**. The probe model omitted
`<compiler angle="radian"/>`, so a capsule's `pi/2` euler was parsed as 1.57
degrees. The result was a near-upright capsule with the wrong rotational
inertia — again, no error.

---

## 2. Why gravity compensation, and what it exposed

The Menagerie Panda actuators are position servos with an affine bias and no
integral term, so at steady state the joint torque must balance gravity and the
arm settles *below* its setpoint. Measured: **8.8 mm** of TCP droop and 1.9 mrad
per joint.

A real Panda does not behave this way — the Franka control interface compensates
gravity in firmware — so the arm links carry `gravcomp="1"`. TCP error drops to
0.50 mm and joint error to 0.08 mrad.

Turning it on initially *lowered* oracle success from 85% to 80%. That was
diagnostic, not a regression: the droop had been masking a grasp-height rule that
aimed too high. See §4.

The IK null-space term had a related problem. It was built from the *damped*
pseudo-inverse, which is not exactly `I - J⁺J`, so it leaked into task space and
stalled the solver at a 0.78 mm floor regardless of iteration count. Fading the
term out as the error shrinks keeps its branch-selection benefit and removes the
floor: IK now reaches 99 µm in 13 iterations with no failures over 80 targets.

---

## 3. Why procedural objects instead of a mesh dataset

The headline result is "how far does grasp success fall on object *categories*
the network never saw?" With procedural shapes the split is a design decision
rather than an accident of someone's folder structure: the held-out set contains
non-convex and multi-part shapes (L, T, mug, dumbbell) that share no topology
with the convex training shapes. The dataset is also regenerable from one
integer, and primitives need no convex decomposition, so a scene compiles in
~120 ms and steps at ~38 000 steps/s.

The cost is real and worth stating in any write-up: primitives are less visually
and geometrically realistic than scanned meshes, so **absolute** success rates
here are optimistic relative to a real robot. The seen-versus-held-out **gap** is
the number that carries meaning.

---

## 4. Grasp height

The rule is "grip at the object's mid-height, minus a 5 mm bias", bounded by
three limits: at least 12 mm below the top surface (so the pads bite the side,
not the top face), at most 35 mm below it (so the hand body clears a tall
object), and at least 11 mm above the table.

The bias is not arbitrary. Measuring the gripper directly — rather than reading
it off the MJCF — shows the fingertip pad spans TCP−8 mm to TCP+9 mm and the
lowest collision point sits 8.86 mm below the TCP. Two effects then both favour
gripping low:

* On a short object, centring the TCP at mid-height puts half the pad above the
  object and wastes it.
* On a curved object, contact above the widest cross-section has normals tilting
  outward, so a downward slip *loosens* the grip. Below it, the object wedges
  into a widening section and the grip tightens.

A sweep on episodes 3000+ — disjoint from every seed range used for reported
results — picked 5 mm; success saturates from 5 mm to 12 mm.

Fingertip clearance above the table is 2.1 mm. Flat objects want it as small as
possible (oracle success on L-shapes rises monotonically from 46% at 5 mm to 71%
at 0.1 mm), but 0.1 mm only works because the simulated arm tracks to 0.5 mm. A
real robot would drag its fingertips across the table, so 2.1 mm is the honest
choice and thin non-convex objects remain the hardest case.

---

## 5. Why the grasp sampler is a mixture

Executing only the oracle grasp would give a dataset that is ~90% positive and
carries no information about *where not to grasp* — which is exactly what a
grasp-quality network has to learn. Episodes are drawn from a mixture:

| mode | weight | what it teaches | measured positive rate |
|---|---|---|---|
| `near_oracle` | 0.45 | precision; near-misses | ~63% |
| `on_object` | 0.25 | the bulk of the signal | ~44% |
| `edge` | 0.15 | hard cases at the silhouette | ~33% |
| `off_object` | 0.15 | unambiguous negatives | 0% |

The result is a roughly balanced dataset (~45% positive) rather than a
degenerate one. The weights are a design choice, not a tuned hyper-parameter,
and they are recorded in `dataset_meta.json`.

---

## 6. Why per-angle quality bins instead of GG-CNN's angle regression

GG-CNN predicts `(quality, cos 2θ, sin 2θ, width)` per pixel and is trained from
annotated grasp rectangles, where every label is positive. Our labels come from
*executing* a sampled grasp, so a failure carries no information about whether
the position or the angle was wrong: regressing an angle from a failure is
meaningless, and discarding failures throws away more than half the dataset.

Binning the angle into 12 bins over 180° (a parallel jaw is symmetric under a
half turn) turns this into classification over `(u, v, angle_bin)`, where a
failure is a clean negative for exactly the cell that was tried. Inference is an
argmax over the whole volume in one forward pass.

### Free negatives

Each episode yields one executed label — 10 000 supervised cells out of
`10⁴ × 224 × 224 × 12`. Bare table far from any object is a *certain* failure at
every angle, so those cells are labelled for free from the height map, without
running the simulator. "Far" matters: the jaws span up to 68 mm (~15 px), so a
table pixel close to an object can still produce a success. Only pixels more than
20 px from any object are used, which is why there is a distance transform in the
data loader rather than a plain mask inversion.

---

## 7. Storage format, and the 16 GB constraint

Shards are plain `.npy`: `uint8` RGB and `uint16` height in units of 0.1 mm.
`.npy` is the only common option `numpy` can **memory-map**, which is the whole
point — a 10k-episode dataset is ~2.5 GB and must not be resident on a 16 GB
laptop. With `mmap_mode='r'` the training loop touches only the pages for the
current batch. Compressed archives or HDF5 would force a decode per sample and
defeat this.

Height rather than raw depth: the two are interconvertible given the camera pose
(`depth = cam_z − table_z − height`), height is what every consumer wants, and
`uint16` at 0.1 mm resolution is far finer than the camera's 2.19 mm ground
sampling distance at half the size of `float32`.

Memory maps are opened **lazily per worker process** — one created in the parent
and inherited through `fork` shares a file offset and degrades badly with several
DataLoader workers.

---

## 8. Known limitations

* **Single object, clean table.** No clutter, no occlusion, no bin. This is why
  the depth-only heuristic baseline is strong: with one isolated object, the
  silhouette centroid is nearly the right answer. The learned policy's advantage
  should be larger in clutter, and that is the natural next experiment.
* **Top-down 4-DOF grasps only.** Full 6-DOF grasping needs point-cloud methods
  and is out of scope.
* **Primitive geometry**, as discussed in §3.
* **No sensor noise.** Real depth cameras have edge artefacts, missing returns
  and quantisation that this simulation does not model, so sim-to-real transfer
  of these numbers should not be assumed.
* **Open-loop execution.** The controller does not react to contact during the
  grasp, which is what makes thin non-convex objects the failure mode they are.

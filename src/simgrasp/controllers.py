"""Joint servo wrapper, damped-least-squares IK, and Cartesian move primitives.

Control model
-------------
The Menagerie Panda actuators are ``general`` actuators with ``biastype="affine"``
configured as position servos (``gainprm=kp``, ``biasprm=[0, -kp, -kv]``). So
``data.ctrl[0:7]`` is a **commanded joint position**, not a torque, and the
gains are already tuned. There is no reason to write a torque controller for
this project: everything the scripted grasp needs is expressible as a smooth
sequence of joint-position setpoints.

Actuator 8 drives the ``split`` tendon. Solving the affine bias for steady state
gives ``tendon_length = 1.568e-4 * ctrl``, and the tendon length equals one
finger's travel, so ``ctrl = 255`` is a fully open 0.08 m gripper.

IK
--
Damped least squares (Levenberg-Marquardt) on the ``tcp`` site, iterated on a
throwaway ``MjData`` so the live simulation is never disturbed. Damping keeps the
solution finite near singularities, and a null-space term biases the redundant
7th DOF toward a nominal elbow-up posture so repeated solves stay in the same IK
branch instead of flipping the arm between waypoints.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from .scene import ARM_JOINTS, FINGER_JOINTS, HOME_QPOS, TCP_SITE
from .transforms import rotation_error

GRIPPER_CTRL_MAX = 255.0
GRIPPER_MAX_WIDTH = 0.08


@dataclass
class IKResult:
    qpos: np.ndarray
    success: bool
    pos_err: float
    rot_err: float
    iterations: int


class ArmInterface:
    """Index bookkeeping for the Panda's arm/gripper inside a scene model."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.joint_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS]
        )
        if np.any(self.joint_ids < 0):
            raise KeyError("arm joints missing from model")
        self.qpos_adr = model.jnt_qposadr[self.joint_ids]
        self.dof_adr = model.jnt_dofadr[self.joint_ids]
        # Look the fingers up by name rather than assuming they follow joint7 in
        # qpos: the object's free joint changes the address layout.
        finger_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in FINGER_JOINTS]
        )
        if np.any(finger_ids < 0):
            raise KeyError("finger joints missing from model")
        self.finger_qpos_adr = model.jnt_qposadr[finger_ids]
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
        if self.site_id < 0:
            raise KeyError(f"site {TCP_SITE!r} missing from model")
        self.joint_range = model.jnt_range[self.joint_ids].copy()
        # Actuators 0..6 drive the arm, actuator 7 drives the finger tendon.
        self.arm_act = np.arange(7)
        self.gripper_act = 7

    def get_arm_qpos(self, data: mujoco.MjData) -> np.ndarray:
        return data.qpos[self.qpos_adr].copy()

    def set_arm_qpos(self, data: mujoco.MjData, q: np.ndarray) -> None:
        data.qpos[self.qpos_adr] = q
        data.qvel[self.dof_adr] = 0.0

    def set_arm_ctrl(self, data: mujoco.MjData, q: np.ndarray) -> None:
        data.ctrl[self.arm_act] = np.clip(q, self.joint_range[:, 0], self.joint_range[:, 1])

    def set_gripper_ctrl(self, data: mujoco.MjData, width_m: float) -> None:
        frac = float(np.clip(width_m / GRIPPER_MAX_WIDTH, 0.0, 1.0))
        data.ctrl[self.gripper_act] = frac * GRIPPER_CTRL_MAX

    def gripper_width(self, data: mujoco.MjData) -> float:
        """Current *total* opening in metres (sum of both finger travels)."""
        return float(np.sum(data.qpos[self.finger_qpos_adr]))

    def set_gripper_qpos(self, data: mujoco.MjData, width_m: float) -> None:
        data.qpos[self.finger_qpos_adr] = np.clip(width_m, 0.0, GRIPPER_MAX_WIDTH) / 2.0

    def tcp_pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        return data.site_xpos[self.site_id].copy(), data.site_xmat[self.site_id].reshape(3, 3).copy()


def solve_ik(
    model: mujoco.MjModel,
    arm: ArmInterface,
    target_pos: np.ndarray,
    target_mat: np.ndarray,
    q_init: np.ndarray | None = None,
    *,
    max_iters: int = 120,
    pos_tol: float = 1e-3,
    rot_tol: float = 1e-2,
    damping: float = 5e-2,
    step_scale: float = 0.7,
    null_gain: float = 0.05,
    scratch: mujoco.MjData | None = None,
) -> IKResult:
    """Damped-least-squares IK for the TCP site.

    Parameters mirror the usual LM knobs. ``damping`` is on the *squared* scale,
    i.e. we solve ``(J J^T + damping^2 I) y = e``.
    """
    data = scratch if scratch is not None else mujoco.MjData(model)
    q = np.array(HOME_QPOS if q_init is None else q_init, dtype=np.float64)
    q_nom = np.asarray(HOME_QPOS, dtype=np.float64)
    lo, hi = arm.joint_range[:, 0], arm.joint_range[:, 1]

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    pos_err = rot_err = np.inf
    it = 0

    for it in range(1, max_iters + 1):
        data.qpos[arm.qpos_adr] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

        cur_pos = data.site_xpos[arm.site_id]
        cur_mat = data.site_xmat[arm.site_id].reshape(3, 3)
        e_pos = np.asarray(target_pos, dtype=np.float64) - cur_pos
        e_rot = rotation_error(cur_mat, np.asarray(target_mat, dtype=np.float64))
        pos_err = float(np.linalg.norm(e_pos))
        rot_err = float(np.linalg.norm(e_rot))
        if pos_err < pos_tol and rot_err < rot_tol:
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, arm.site_id)
        jac = np.vstack([jacp[:, arm.dof_adr], jacr[:, arm.dof_adr]])  # 6 x 7
        err = np.concatenate([e_pos, e_rot])

        jjt = jac @ jac.T + (damping**2) * np.eye(6)
        y = np.linalg.solve(jjt, err)
        dq = jac.T @ y

        # Null-space posture bias: keeps the redundant DOF near the nominal pose
        # so consecutive waypoints stay on the same IK branch.
        if null_gain > 0.0:
            j_pinv = jac.T @ np.linalg.solve(jjt, np.eye(6))
            dq += (np.eye(7) - j_pinv @ jac) @ (null_gain * (q_nom - q))

        # Trust region: cap the per-iteration joint step so large targets do not
        # take a wild first step through a singularity.
        max_step = 0.35
        norm = np.linalg.norm(dq)
        if norm > max_step:
            dq *= max_step / norm

        q = np.clip(q + step_scale * dq, lo, hi)

    return IKResult(qpos=q, success=bool(pos_err < pos_tol and rot_err < rot_tol),
                    pos_err=pos_err, rot_err=rot_err, iterations=it)


def minimum_jerk(t: np.ndarray | float) -> np.ndarray | float:
    """Smooth 0->1 interpolation with zero velocity and acceleration at both ends."""
    s = np.clip(t, 0.0, 1.0)
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


StepHook = Callable[[mujoco.MjModel, mujoco.MjData], None]


class CartesianController:
    """Executes smooth joint-space and Cartesian moves by streaming servo setpoints."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, arm: ArmInterface | None = None):
        self.model = model
        self.data = data
        self.arm = arm or ArmInterface(model)
        self._scratch = mujoco.MjData(model)
        self.dt = float(model.opt.timestep)
        self.ik_failures = 0
        self.last_convergence_error = 0.0

    # -- low level ---------------------------------------------------------- #
    def step(self, n: int = 1, hook: StepHook | None = None) -> None:
        for _ in range(int(n)):
            mujoco.mj_step(self.model, self.data)
            if hook is not None:
                hook(self.model, self.data)

    def settle(self, seconds: float = 0.2, hook: StepHook | None = None) -> None:
        self.step(int(round(seconds / self.dt)), hook)

    def hold_until_converged(self, target_pos, target_mat=None, *, tol: float = 1.5e-3,
                             max_time: float = 0.6, hook: StepHook | None = None) -> float:
        """Hold the current setpoint until the TCP actually gets there.

        The arm is driven by position servos with finite stiffness, so the TCP
        trails its setpoint. Measured on the descent phase the lag reached ~15 mm,
        which is enough for the fingertip pads to sit above a 35 mm-tall object
        when the gripper is told to close: the fingers then catch the top edge and
        squirt the object sideways instead of gripping it. Every failed grasp of a
        flat object traced back to this.

        Returns the final position error so callers can record it.
        """
        target_pos = np.asarray(target_pos, dtype=np.float64)
        steps = max(1, int(round(max_time / self.dt)))
        err = np.inf
        for _ in range(steps):
            pos, _ = self.arm.tcp_pose(self.data)
            err = float(np.linalg.norm(pos - target_pos))
            if err < tol:
                break
            self.step(1, hook)
        return err

    def solve_ik(self, pos, mat, q_init=None, **kw) -> IKResult:
        if q_init is None:
            q_init = self.arm.get_arm_qpos(self.data)
        return solve_ik(self.model, self.arm, pos, mat, q_init, scratch=self._scratch, **kw)

    # -- moves -------------------------------------------------------------- #
    def move_to_joint(self, q_target: np.ndarray, duration: float = 1.2,
                      hook: StepHook | None = None) -> None:
        """Minimum-jerk interpolation from the current commanded pose to ``q_target``."""
        q0 = self.arm.get_arm_qpos(self.data)
        q1 = np.asarray(q_target, dtype=np.float64)
        n = max(1, int(round(duration / self.dt)))
        for i in range(1, n + 1):
            s = minimum_jerk(i / n)
            self.arm.set_arm_ctrl(self.data, q0 + s * (q1 - q0))
            self.step(1, hook)

    def move_to_pose(self, pos, mat, duration: float = 1.0, cartesian: bool = False,
                     waypoints: int = 12, hook: StepHook | None = None,
                     converge: bool = True, converge_tol: float = 1.5e-3,
                     converge_time: float = 0.6) -> IKResult:
        """Move the TCP to ``(pos, mat)``.

        ``cartesian=True`` interpolates the *TCP position* along a straight line
        and re-solves IK at each waypoint. That matters for the descent and lift
        phases, where a joint-space shortcut would swing the fingers sideways
        into the object.
        """
        if not cartesian:
            res = self.solve_ik(pos, mat)
            if not res.success:
                self.ik_failures += 1
            self.move_to_joint(res.qpos, duration, hook)
            if converge:
                self.last_convergence_error = self.hold_until_converged(
                    pos, mat, tol=converge_tol, max_time=converge_time, hook=hook)
            return res

        p0, _ = self.arm.tcp_pose(self.data)
        p1 = np.asarray(pos, dtype=np.float64)
        q_seed = self.arm.get_arm_qpos(self.data)
        last = IKResult(q_seed, True, 0.0, 0.0, 0)
        seg = max(1, int(waypoints))
        for k in range(1, seg + 1):
            s = minimum_jerk(k / seg)
            target = p0 + s * (p1 - p0)
            last = self.solve_ik(target, mat, q_init=q_seed)
            if not last.success:
                self.ik_failures += 1
            q_seed = last.qpos
            self._ramp_to(last.qpos, duration / seg, hook)
        if converge:
            self.last_convergence_error = self.hold_until_converged(
                p1, mat, tol=converge_tol, max_time=converge_time, hook=hook)
        return last

    def _ramp_to(self, q_target: np.ndarray, duration: float, hook: StepHook | None) -> None:
        q0 = np.array(self.data.ctrl[self.arm.arm_act], dtype=np.float64)
        n = max(1, int(round(duration / self.dt)))
        for i in range(1, n + 1):
            s = i / n  # segments are already globally min-jerk shaped
            self.arm.set_arm_ctrl(self.data, q0 + s * (q_target - q0))
            self.step(1, hook)

    # -- gripper ------------------------------------------------------------ #
    def set_gripper(self, width_m: float, duration: float = 0.4, hook: StepHook | None = None) -> None:
        self.arm.set_gripper_ctrl(self.data, width_m)
        self.step(max(1, int(round(duration / self.dt))), hook)

    def close_gripper(self, duration: float = 0.7, hook: StepHook | None = None) -> None:
        """Command a full close; the object stops the fingers and the servo squeezes."""
        self.set_gripper(0.0, duration, hook)

    def open_gripper(self, duration: float = 0.4, hook: StepHook | None = None) -> None:
        self.set_gripper(GRIPPER_MAX_WIDTH, duration, hook)

    def reset_to(self, q: np.ndarray, gripper_width: float = GRIPPER_MAX_WIDTH) -> None:
        """Teleport the arm (used at episode reset, never mid-episode)."""
        self.arm.set_arm_qpos(self.data, q)
        self.arm.set_arm_ctrl(self.data, q)
        self.arm.set_gripper_ctrl(self.data, gripper_width)
        self.arm.set_gripper_qpos(self.data, gripper_width)
        mujoco.mj_forward(self.model, self.data)

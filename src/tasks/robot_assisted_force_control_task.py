"""Minimal force-control task for the Starlink-mounted manipulator.

This task is intentionally simple:
- spawn the space manipulator
- keep the gripper untouched
- apply piecewise-constant joint torques/forces to the arm motor DOFs
- verify motion by tracking joint displacement
"""

import sys
import time
from pathlib import Path
from typing import Optional

import torch

current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from controllers.backend import EmptyBackend
from robots.satellite_manipulator import SatelliteManipulator
from tasks.task import Task, TaskStatus


class RobotAssistedForceControlTask(Task):
    def __init__(self):
        super().__init__(task_name="robot_assisted_force_control")
        self.starlink_manipulator: Optional[SatelliteManipulator] = None

        self.max_steps = 540
        self.settle_steps = 60
        self.phase_steps = 120
        self.motion_threshold = 0.18

        self.initial_joint_qpos = None
        self.force_baseline_qpos = None
        self.latest_joint_qpos = None
        self.active_phase = "not_started"

    def initialize(self) -> bool:
        try:
            if self.starlink_manipulator is None:
                self.starlink_manipulator = SatelliteManipulator(
                    name="franka_merge",
                    sensors=[],
                    backends=[EmptyBackend()],
                )

            self.gsim.start()
            self.starlink_manipulator.initialize()

            self.initial_joint_qpos = self._get_motor_qpos()
            self.force_baseline_qpos = None
            self.latest_joint_qpos = self.initial_joint_qpos.detach().clone()
            self.active_phase = "settle"
            self.success = False
            self.status = TaskStatus.RUNNING

            self.logger.info(
                "Force-control task initialized. motors_dof=%s initial_qpos=%s",
                self.starlink_manipulator.motors_dof,
                [round(x, 5) for x in self.initial_joint_qpos.detach().cpu().tolist()],
            )
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.status = TaskStatus.FAILED
            return False

    def _get_motor_qpos(self) -> torch.Tensor:
        qpos = self.starlink_manipulator.robot.get_qpos(
            qs_idx_local=self.starlink_manipulator.motors_qs
        )
        qpos = torch.as_tensor(qpos, dtype=self.datatype, device=self.device)
        if qpos.dim() == 2:
            qpos = qpos[0]
        return qpos.reshape(-1)

    def _get_force_command(self) -> torch.Tensor:
        zero = torch.zeros(6, dtype=self.datatype, device=self.device)

        if self.step_count < self.settle_steps:
            self.active_phase = "settle"
            return zero

        phase_idx = (self.step_count - self.settle_steps) // self.phase_steps
        phase_id = phase_idx % 2

        if phase_id == 0:
            self.active_phase = "all_joints_forward"
            return torch.tensor([32.0, -28.0, 26.0, -22.0, 12.0, -12.0], dtype=self.datatype, device=self.device)

        self.active_phase = "all_joints_reverse"
        return torch.tensor([-32.0, 28.0, -26.0, 22.0, -12.0, 12.0], dtype=self.datatype, device=self.device)

    def step(self, dt: float = None) -> bool:
        if self.status != TaskStatus.RUNNING:
            return False

        try:
            force_cmd = self._get_force_command()
            self.starlink_manipulator.robot.control_dofs_force(
                force=force_cmd,
                dofs_idx_local=self.starlink_manipulator.motors_dof,
            )

            self.starlink_manipulator.step()
            self.gsim.step()
            self.step_count += 1

            self.latest_joint_qpos = self._get_motor_qpos()
            self.total_reward += self.reward().item()

            if self.step_count == self.settle_steps and self.force_baseline_qpos is None:
                self.force_baseline_qpos = self.latest_joint_qpos.detach().clone()
                self.logger.info(
                    "Settle finished. Force-control baseline qpos=%s",
                    [round(x, 5) for x in self.force_baseline_qpos.detach().cpu().tolist()],
                )

            if self.step_count % 30 == 0:
                baseline = self.force_baseline_qpos if self.force_baseline_qpos is not None else self.initial_joint_qpos
                delta = self.latest_joint_qpos - baseline
                self.logger.info(
                    "step=%d phase=%s force=%s qpos=%s delta=%s",
                    self.step_count,
                    self.active_phase,
                    [round(x, 4) for x in force_cmd.detach().cpu().tolist()],
                    [round(x, 4) for x in self.latest_joint_qpos.detach().cpu().tolist()],
                    [round(x, 4) for x in delta.detach().cpu().tolist()],
                )

            if self.check_termination():
                self.status = TaskStatus.SUCCEED if self.success else TaskStatus.COMPLETED
                return False

            return True
        except Exception as e:
            self.logger.error(f"Step error: {e}")
            self.status = TaskStatus.FAILED
            return False

    def check_termination(self) -> bool:
        if self.force_baseline_qpos is not None and self.latest_joint_qpos is not None:
            max_motion = torch.max(torch.abs(self.latest_joint_qpos - self.force_baseline_qpos)).item()
            if max_motion >= self.motion_threshold:
                self.success = True
                self.logger.info(
                    "Force-control motion detected after settle. max_joint_displacement=%.5f rad",
                    max_motion,
                )
                return True

        if self.step_count >= self.max_steps:
            self.logger.info("Force-control test reached max_steps without enough motion.")
            return True

        return False

    def reward(self) -> torch.Tensor:
        if self.latest_joint_qpos is None:
            return torch.tensor(0.0, dtype=self.datatype, device=self.device)

        baseline = self.force_baseline_qpos if self.force_baseline_qpos is not None else self.initial_joint_qpos
        if baseline is None:
            return torch.tensor(0.0, dtype=self.datatype, device=self.device)

        displacement = torch.abs(self.latest_joint_qpos - baseline)
        return torch.max(displacement)

    def stop(self) -> bool:
        if self.status in {TaskStatus.NOT_STARTED, TaskStatus.CANCELLED}:
            return False

        try:
            if self.starlink_manipulator is not None:
                zero = torch.zeros(6, dtype=self.datatype, device=self.device)
                self.starlink_manipulator.robot.control_dofs_force(
                    force=zero,
                    dofs_idx_local=self.starlink_manipulator.motors_dof,
                )
                self.starlink_manipulator.stop()

            self.gsim.stop()
            self.status = TaskStatus.SUCCEED if self.success else TaskStatus.CANCELLED
            return True
        except Exception as e:
            self.logger.error(f"Stop error: {e}")
            self.status = TaskStatus.FAILED
            return False

    def reset(self) -> bool:
        self.logger.warning("Reset is not implemented for the minimal force-control task.")
        return False


def main():
    print("=== RobotAssistedForceControlTask Test ===")
    task = RobotAssistedForceControlTask()

    if not task.initialize():
        print("Task initialization failed")
        return

    print("Task initialized successfully")

    try:
        while task.status == TaskStatus.RUNNING:
            if not task.step():
                break

            if task.step_count % 30 == 0:
                reward = task.reward().item()
                print(
                    f"step={task.step_count} phase={task.active_phase} "
                    f"reward={reward:.4f} success={task.success}"
                )
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        print(f"Final status: {task.status}")
        print(f"Steps: {task.step_count}")
        print(f"Success: {task.success}")
        task.stop()


if __name__ == "__main__":
    main()

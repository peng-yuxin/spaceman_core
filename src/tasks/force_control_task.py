"""Minimal gripper force-control task.

This task only verifies the most basic gripper force-control sequence:
- initialize the Starlink manipulator
- wait a few simulation steps for settling
- close the gripper
- reopen the gripper
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


class ForceControlTask(Task):
    def __init__(self):
        super().__init__(task_name="force_control")
        self.starlink_manipulator: Optional[SatelliteManipulator] = None

        self.max_steps = 120
        self.settle_steps = 20
        self.close_hold_steps = 30
        self.open_hold_steps = 30

        self.phase = "not_started"
        self.last_gripper_command = None
        self.gripper_qpos = None

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

            self.phase = "settle"
            self.last_gripper_command = None
            self.gripper_qpos = self._get_gripper_qpos()
            self.success = False
            self.status = TaskStatus.RUNNING

            self.logger.info(
                "ForceControlTask initialized. fingers_dof=%s initial_gripper_qpos=%s",
                self.starlink_manipulator.fingers_dof,
                [round(x, 5) for x in self.gripper_qpos.detach().cpu().tolist()],
            )
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.status = TaskStatus.FAILED
            return False

    def _get_gripper_qpos(self) -> torch.Tensor:
        qpos = self.starlink_manipulator.robot.get_qpos(
            qs_idx_local=self.starlink_manipulator.fingers_qs
        )
        qpos = torch.as_tensor(qpos, dtype=self.datatype, device=self.device)
        if qpos.dim() == 2:
            qpos = qpos[0]
        return qpos.reshape(-1)

    def _get_gripper_command(self) -> float:
        if self.step_count < self.settle_steps:
            self.phase = "settle"
            return 1.0

        if self.step_count < self.settle_steps + self.close_hold_steps:
            self.phase = "close"
            return 0.0

        if self.step_count < self.settle_steps + self.close_hold_steps + self.open_hold_steps:
            self.phase = "open"
            return 1.0

        self.phase = "done"
        return 1.0

    def step(self, dt: float = None) -> bool:
        if self.status != TaskStatus.RUNNING:
            return False

        try:
            previous_phase = self.phase
            gripper_command = self._get_gripper_command()
            if self.phase == "close" and previous_phase != "close":
                self.logger.info("Starting gripper close sequence.")
            if self.phase == "open" and previous_phase != "open":
                self.logger.info("Starting gripper open sequence.")
            self.starlink_manipulator.control_gripper(gripper_command)
            self.starlink_manipulator.step()
            self.gsim.step()

            self.step_count += 1
            self.last_gripper_command = gripper_command
            self.gripper_qpos = self._get_gripper_qpos()
            self.total_reward += self.reward().item()

            if self.step_count % 10 == 0 or self.phase == "done":
                self.logger.info(
                    "step=%d phase=%s gripper_command=%.1f gripper_qpos=%s",
                    self.step_count,
                    self.phase,
                    gripper_command,
                    [round(x, 5) for x in self.gripper_qpos.detach().cpu().tolist()],
                )

            if self.phase == "done":
                self.success = True
                self.status = TaskStatus.SUCCEED
                return False

            if self.step_count >= self.max_steps:
                self.status = TaskStatus.COMPLETED
                return False

            return True
        except Exception as e:
            self.logger.error(f"Step error: {e}")
            self.status = TaskStatus.FAILED
            return False

    def reward(self) -> torch.Tensor:
        if self.gripper_qpos is None:
            return torch.tensor(0.0, dtype=self.datatype, device=self.device)
        return torch.sum(torch.abs(self.gripper_qpos))

    def reset(self) -> bool:
        self.logger.warning("Reset is not implemented for ForceControlTask.")
        return False

    def stop(self) -> bool:
        if self.status in {TaskStatus.NOT_STARTED, TaskStatus.CANCELLED}:
            return False

        try:
            if self.starlink_manipulator is not None:
                self.starlink_manipulator.control_gripper(1.0)
                self.starlink_manipulator.stop()

            self.gsim.stop()
            if self.status not in {TaskStatus.SUCCEED, TaskStatus.FAILED}:
                self.status = TaskStatus.CANCELLED
            return True
        except Exception as e:
            self.logger.error(f"Stop error: {e}")
            self.status = TaskStatus.FAILED
            return False


def main():
    print("=== ForceControlTask Test ===")
    task = ForceControlTask()

    if not task.initialize():
        print("Task initialization failed")
        return

    print("Task initialized successfully")

    try:
        while task.status == TaskStatus.RUNNING:
            if not task.step():
                break
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

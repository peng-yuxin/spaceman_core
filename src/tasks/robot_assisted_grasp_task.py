"""Grasp-and-transport test for the Starlink-mounted manipulator.

This task keeps the setup intentionally small:
- spawn the combined space manipulator asset
- spawn one movable target object (`satellite_part`)
- place the target directly at the gripper center
- execute a fixed settle -> close -> lift -> transport sequence
- declare success if the object stays grasped during a larger follow-up move
"""

import sys
import time
from pathlib import Path
from typing import Optional

import genesis as gs
import torch

current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from envs.genesis_env import GenesisSim
from robots.robot import Robot
from robots.satellite_manipulator import SatelliteManipulator
from tasks.task import Task, TaskStatus
from utils.utils import as_rotation_matrix


class _TaskEntityWrapper:
    def __init__(self, robot, base_link_name: str):
        self.robot = robot
        self.params = {"base": base_link_name}

    def initialize(self):
        return

    def step(self):
        return

    def stop(self):
        return


class RobotAssistedGraspTask(Task):
    def __init__(self):
        super().__init__(task_name="robot_assisted_grasp")
        self.starlink_manipulator: Optional[SatelliteManipulator] = None
        self.target_object: Optional[Robot] = None

        self.target_link_name = "base_link"
        self.target_initial_pos = None
        self.goal_quat = None
        self.grasp_center_pos = None
        self.ee_to_grasp_center_offset_local = None
        self.phase = "not_started"
        self.last_gripper_command = None
        self.debug_pose_logging = True
        self.debug_max_logged_steps = 3
        self.debug_escape_threshold = 0.005
        self.debug_escape_reported = False

        self.max_steps = 240
        self.lift_success_height = 0.04
        self.settle_step = 30
        self.close_step = 50
        self.lift_step = 100
        self.transport_step = 120
        self.jolt_step = 145
        self.transport_hold_step = 185
        self.transport_offset_stage1 = torch.tensor([0.22, -0.18, 0.1], dtype=self.datatype, device=self.device)
        self.transport_offset_stage2 = torch.tensor([0.32, 0.12, 0.3], dtype=self.datatype, device=self.device)
        self.transport_success_threshold = 0.05
        self.grasp_finger_link_names = (
            "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Finger1_2_Link",
            "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Finger2_1_Link",
        )
        # For the Genesis cube, align the object center with the finger midpoint.
        self.object_center_offset_local = torch.tensor([0.0, -0.15, 0.0], dtype=self.datatype, device=self.device)
        self.object_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=self.datatype, device=self.device)
        # Shrink cube volume to 1/4 of the current cube:
        # edge_new = edge_old * (1/4)^(1/3) ~= 0.1 * 0.62996
        self.cube_size = (0.03, 0.03, 0.03)
        # Previous custom satellite part grasp tuning, kept for reference:
        # self.object_center_offset_local = torch.tensor([0,-0.05,0.0], dtype=self.datatype, device=self.device)
        # self.object_quat = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068], dtype=self.datatype, device=self.device)

    def _create_target_cube(self):
        cube = self._scene.add_entity(
            morph=gs.morphs.Box(
                pos=(-1.1, 0.3, 1.0),
                size=self.cube_size,
                fixed=False,
            ),
            material=gs.materials.Rigid(
                gravity_compensation=1.0,
            ),
        )
        return _TaskEntityWrapper(robot=cube, base_link_name="box_baselink")

    def _get_gripper_debug_state(self):
        finger_link_a = self.starlink_manipulator.robot.get_link(self.grasp_finger_link_names[0])
        finger_link_b = self.starlink_manipulator.robot.get_link(self.grasp_finger_link_names[1])

        finger_pos_a = finger_link_a.get_pos().detach().cpu()
        finger_pos_b = finger_link_b.get_pos().detach().cpu()
        finger_delta = finger_pos_a - finger_pos_b
        finger_distance = torch.norm(finger_delta).item()

        gripper_qpos = self.starlink_manipulator.robot.get_qpos(
            qs_idx_local=self.starlink_manipulator.fingers_qs
        )
        gripper_qpos = torch.as_tensor(gripper_qpos, dtype=self.datatype, device=self.device).detach().cpu()

        return {
            "finger_a": finger_pos_a,
            "finger_b": finger_pos_b,
            "finger_delta": finger_delta,
            "finger_distance": finger_distance,
            "gripper_qpos": gripper_qpos,
        }

    def _log_debug_pose_snapshot(self, tag: str):
        if not self.debug_pose_logging:
            return

        ee_link = self.starlink_manipulator.robot.get_link(self.starlink_manipulator.params["end_effector"])
        object_link = self.target_object.robot.get_link(self.target_link_name)
        gripper_state = self._get_gripper_debug_state()

        ee_pos = ee_link.get_pos().detach().cpu()
        ee_quat = ee_link.get_quat().detach().cpu()
        object_pos = object_link.get_pos().detach().cpu()
        object_quat = object_link.get_quat().detach().cpu()
        grasp_center = self.grasp_center_pos.detach().cpu() if self.grasp_center_pos is not None else None
        obj_minus_center = object_pos - grasp_center if grasp_center is not None else None
        obj_minus_ee = object_pos - ee_pos

        self.logger.info(
            "[DEBUG_POSE] %s | ee_pos=%s ee_quat=%s | grasp_center=%s | "
            "finger_a=%s finger_b=%s finger_delta=%s finger_distance=%.6f | "
            "gripper_qpos=%s | obj_pos=%s obj_quat=%s | obj-grasp_center=%s | obj-ee=%s",
            tag,
            [round(x, 5) for x in ee_pos.tolist()],
            [round(x, 5) for x in ee_quat.tolist()],
            None if grasp_center is None else [round(x, 5) for x in grasp_center.tolist()],
            [round(x, 5) for x in gripper_state["finger_a"].tolist()],
            [round(x, 5) for x in gripper_state["finger_b"].tolist()],
            [round(x, 5) for x in gripper_state["finger_delta"].tolist()],
            gripper_state["finger_distance"],
            [round(x, 5) for x in gripper_state["gripper_qpos"].reshape(-1).tolist()],
            [round(x, 5) for x in object_pos.tolist()],
            [round(x, 5) for x in object_quat.tolist()],
            None if obj_minus_center is None else [round(x, 5) for x in obj_minus_center.tolist()],
            [round(x, 5) for x in obj_minus_ee.tolist()],
        )

    def _check_object_escape(self):
        if not self.debug_pose_logging or self.debug_escape_reported or self.grasp_center_pos is None:
            return

        object_link = self.target_object.robot.get_link(self.target_link_name)
        object_pos = object_link.get_pos().detach()
        delta = object_pos - self.grasp_center_pos
        distance = torch.norm(delta).item()

        if distance > self.debug_escape_threshold:
            self.debug_escape_reported = True
            self.logger.warning(
                "[DEBUG_ESCAPE] object left grasp center threshold at step=%d phase=%s "
                "distance=%.6f threshold=%.6f delta=%s obj_pos=%s grasp_center=%s",
                self.step_count,
                self.phase,
                distance,
                self.debug_escape_threshold,
                [round(x, 5) for x in delta.detach().cpu().tolist()],
                [round(x, 5) for x in object_pos.detach().cpu().tolist()],
                [round(x, 5) for x in self.grasp_center_pos.detach().cpu().tolist()],
            )

    def _compute_grasp_center_from_fingers(self):
        finger_link_a = self.starlink_manipulator.robot.get_link(self.grasp_finger_link_names[0])
        finger_link_b = self.starlink_manipulator.robot.get_link(self.grasp_finger_link_names[1])
        ee_link = self.starlink_manipulator.robot.get_link(self.starlink_manipulator.params["end_effector"])
        finger_pos_a = finger_link_a.get_pos().detach().clone()
        finger_pos_b = finger_link_b.get_pos().detach().clone()
        ee_pos = ee_link.get_pos().detach().clone()
        finger_midpoint = 0.5 * (finger_pos_a + finger_pos_b)

        ee_rot = as_rotation_matrix(self.goal_quat, order="wxyz")
        center_offset_world = torch.matmul(ee_rot, self.object_center_offset_local)
        grasp_center = finger_midpoint + center_offset_world

        self.logger.info(
            "[DEBUG_GRASP_CENTER] finger_links=%s | finger_a=%s | finger_b=%s | ee_pos=%s | "
            "midpoint=%s | midpoint-ee=%s | local_center_offset=%s | center_offset_world=%s | grasp_center=%s",
            list(self.grasp_finger_link_names),
            [round(x, 5) for x in finger_pos_a.detach().cpu().tolist()],
            [round(x, 5) for x in finger_pos_b.detach().cpu().tolist()],
            [round(x, 5) for x in ee_pos.detach().cpu().tolist()],
            [round(x, 5) for x in finger_midpoint.detach().cpu().tolist()],
            [round(x, 5) for x in (finger_midpoint - ee_pos).detach().cpu().tolist()],
            [round(x, 5) for x in self.object_center_offset_local.detach().cpu().tolist()],
            [round(x, 5) for x in center_offset_world.detach().cpu().tolist()],
            [round(x, 5) for x in grasp_center.detach().cpu().tolist()],
        )
        return grasp_center

    def _compute_ee_target_from_grasp_center(self, grasp_center_target: torch.Tensor):
        if self.ee_to_grasp_center_offset_local is None:
            raise ValueError("ee_to_grasp_center_offset_local is not initialized")

        ee_rot = as_rotation_matrix(self.goal_quat, order="wxyz")
        ee_to_grasp_center_offset_world = torch.matmul(
            ee_rot, self.ee_to_grasp_center_offset_local
        )
        ee_target = grasp_center_target - ee_to_grasp_center_offset_world

        self.logger.debug(
            "Computed ee target from grasp center. grasp_center=%s offset_local=%s offset_world=%s ee_target=%s",
            [round(x, 5) for x in grasp_center_target.detach().cpu().tolist()],
            [round(x, 5) for x in self.ee_to_grasp_center_offset_local.detach().cpu().tolist()],
            [round(x, 5) for x in ee_to_grasp_center_offset_world.detach().cpu().tolist()],
            [round(x, 5) for x in ee_target.detach().cpu().tolist()],
        )
        return ee_target

    def initialize(self) -> bool:
        try:
            if self.starlink_manipulator is None:
                self.starlink_manipulator = SatelliteManipulator(name="franka_merge", sensors=[], backends=[])
            if self.target_object is None:
                self.target_object = self._create_target_cube()
                # Previous config-based custom satellite part, kept for reference:
                # self.target_object = Robot(name="satellite_part")

            self.gsim.start()
            self.starlink_manipulator.initialize()
            self.target_object.initialize()
            self.target_link_name = self.target_object.params["base"]

            self._log_debug_pose_snapshot("after_robot_initialize_before_open")
            self.starlink_manipulator.update_state()
            ee_pos = self.starlink_manipulator.ee_global_position.detach().clone()
            self.goal_quat = self.starlink_manipulator.ee_global_quaternion.detach().clone()
            self.grasp_center_pos = self._compute_grasp_center_from_fingers()
            ee_rot = as_rotation_matrix(self.goal_quat, order="wxyz")
            self.ee_to_grasp_center_offset_local = torch.matmul(
                ee_rot.transpose(0, 1), self.grasp_center_pos - ee_pos
            )
            self.last_gripper_command = 1.0
            self.logger.info(
                "[DEBUG_INIT] ee_pos=%s finger_links=%s object_center_offset_local=%s "
                "ee_to_grasp_center_offset_local=%s object_quat=%s",
                [round(x, 5) for x in ee_pos.detach().cpu().tolist()],
                list(self.grasp_finger_link_names),
                [round(x, 5) for x in self.object_center_offset_local.detach().cpu().tolist()],
                [round(x, 5) for x in self.ee_to_grasp_center_offset_local.detach().cpu().tolist()],
                [round(x, 5) for x in self.object_quat.detach().cpu().tolist()],
            )

            # Place the cube directly at the pinch region center.
            spawn_pos = self.grasp_center_pos.detach().clone()
            self._log_debug_pose_snapshot("before_set_object_pose")
            self.target_object.robot.set_pos(spawn_pos, zero_velocity=True)
            self.target_object.robot.set_quat(self.object_quat, zero_velocity=True)
            self._log_debug_pose_snapshot("after_set_object_pose_before_step")
            self.gsim.step()
            self._log_debug_pose_snapshot("after_initialize_step")

            self.target_initial_pos = self.target_object.robot.get_link(self.target_link_name).get_pos().detach().clone()
            self.phase = "approach"
            self.status = TaskStatus.RUNNING
            self.success = False
            self.debug_escape_reported = False
            self.logger.info("RobotAssistedGraspTask initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.status = TaskStatus.FAILED
            return False

    def _compute_cartesian_target(self):
        if self.step_count < self.settle_step:
            self.phase = "settle"
            target_pos = self.grasp_center_pos.detach().clone()
            gripper_open = 1.0
        elif self.step_count < self.close_step:
            self.phase = "hold_center"
            target_pos = self.grasp_center_pos.detach().clone()
            gripper_open = 1.0
        elif self.step_count < self.lift_step:
            self.phase = "grasp"
            target_pos = self.grasp_center_pos.detach().clone()
            gripper_open = 0.0
        elif self.step_count < self.transport_step:
            self.phase = "lift"
            target_pos = self.grasp_center_pos + torch.tensor([0.0, 0.0, 0.12], dtype=self.datatype, device=self.device)
            gripper_open = 0.0
        elif self.step_count < self.jolt_step:
            self.phase = "jolt_transport_1"
            progress = (self.step_count - self.transport_step) / max(1, self.jolt_step - self.transport_step)
            target_pos = self.grasp_center_pos + progress * self.transport_offset_stage1
            gripper_open = 0.0
        elif self.step_count < self.transport_hold_step:
            self.phase = "jolt_transport_2"
            progress = (self.step_count - self.jolt_step) / max(1, self.transport_hold_step - self.jolt_step)
            stage_delta = self.transport_offset_stage2 - self.transport_offset_stage1
            target_pos = self.grasp_center_pos + self.transport_offset_stage1 + progress * stage_delta
            gripper_open = 0.0
        else:
            self.phase = "hold_transport"
            target_pos = self.grasp_center_pos + self.transport_offset_stage2
            gripper_open = 0.0

        return target_pos, gripper_open

    def step(self, dt: float = None) -> bool:
        if self.status != TaskStatus.RUNNING:
            return False

        try:
            target_grasp_center, gripper_open = self._compute_cartesian_target()
            target_pos = self._compute_ee_target_from_grasp_center(target_grasp_center)
            # Force-controlled grippers need a persistent command instead of a
            # single edge-triggered pulse, otherwise the fingers move briefly
            # and then get pulled back by joint PD / contact dynamics.
            self.starlink_manipulator.control_gripper(gripper_open)
            self.last_gripper_command = gripper_open
            self.starlink_manipulator.control_joints(target_pos, self.goal_quat)

            self.starlink_manipulator.step()
            self.target_object.step()
            self.gsim.step()
            self.step_count += 1
            self._check_object_escape()

            if self.step_count <= self.debug_max_logged_steps:
                self._log_debug_pose_snapshot(f"after_task_step_{self.step_count}")

            if self.check_termination():
                self.status = TaskStatus.SUCCEED if self.success else TaskStatus.COMPLETED
                return False

            return True
        except Exception as e:
            self.logger.error(f"Step error: {e}")
            self.status = TaskStatus.FAILED
            return False

    def check_termination(self) -> bool:
        object_pos = self.target_object.robot.get_link(self.target_link_name).get_pos()
        lift_height = (object_pos[2] - self.target_initial_pos[2]).item()
        final_transport_target = self.grasp_center_pos + self.transport_offset_stage2
        transport_error = torch.norm(object_pos - final_transport_target).item()

        if (
            self.step_count >= self.transport_hold_step
            and lift_height > self.lift_success_height
            and transport_error < self.transport_success_threshold
        ):
            self.success = True
            self.logger.info(
                "Grasp succeeded after transport. lift_height=%.4f m transport_error=%.4f m",
                lift_height,
                transport_error,
            )
            return True

        if self.step_count >= self.max_steps:
            self.logger.info(
                "Grasp test ended without success. Final lift height: %.4f m transport_error: %.4f m",
                lift_height,
                transport_error,
            )
            return True

        return False

    def reward(self) -> torch.Tensor:
        object_pos = self.target_object.robot.get_link(self.target_link_name).get_pos()
        lift_height = object_pos[2] - self.target_initial_pos[2]
        return torch.as_tensor(lift_height, dtype=self.datatype, device=self.device)

    def stop(self) -> bool:
        if self.status in {TaskStatus.NOT_STARTED, TaskStatus.CANCELLED}:
            return False

        try:
            if self.starlink_manipulator is not None:
                self.starlink_manipulator.stop()
            if self.target_object is not None:
                self.target_object.stop()
            self.gsim.stop()
            self.status = TaskStatus.CANCELLED if not self.success else TaskStatus.SUCCEED
            return True
        except Exception as e:
            self.logger.error(f"Stop error: {e}")
            self.status = TaskStatus.FAILED
            return False

    def reset(self) -> bool:
        self.logger.warning("Reset is not implemented for the minimal grasp task.")
        return False


def main():
    print("=== RobotAssistedGraspTask Test ===")
    task = RobotAssistedGraspTask()

    if not task.initialize():
        print("❌ Task initialization failed")
        return

    print("✅ Task initialized successfully")

    try:
        while task.status == TaskStatus.RUNNING:
            if not task.step():
                break

            if task.step_count % 20 == 0:
                reward = task.reward().item()
                print(f"step={task.step_count} phase={task.phase} reward={reward:.4f} success={task.success}")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    finally:
        print(f"Final status: {task.status}")
        print(f"Steps: {task.step_count}")
        print(f"Success: {task.success}")
        task.stop()


if __name__ == "__main__":
    main()

# 输入：TCP位姿、目标上抓取点位姿、基座卫星位姿（18维）
# 输出：机械臂关节角、卫星二维平面位置（8维）
# 奖励：底座先对位，再让TCP逼近抓取点，最后闭合夹爪完成抓取

import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from robots.robot import Robot
from robots.satellite_manipulator import SatelliteManipulator
from tasks.task import Task, TaskStatus
from utils.utils import as_rotation_matrix

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    gym = None
    spaces = None

try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None


class RLTask(Task):
    def __init__(self, target_asset_name: str = "starlink"):
        super().__init__(task_name="RL_task")

        self.starlink_manipulator: Optional[SatelliteManipulator] = None
        self.target_object: Optional[Robot] = None
        self.target_asset_name = target_asset_name

        self.target_link_name = ""
        self.target_grasp_offset_local = torch.tensor(
            [0.0, 0.0, 0.0], dtype=self.datatype, device=self.device
        )

        # TCP uses the gripper grasp center instead of the arm-hand connection link.
        self.tcp_offset_local = torch.tensor(
            [0.002085, -0.065496, 0.0004827],
            dtype=self.datatype,
            device=self.device,
        )

        self.max_steps = 300
        self.base_alignment_threshold = 0.18
        self.grasp_success_threshold = 0.06
        self.base_force_scale_xy = 60.0
        self.joint_action_scale = 0.5
        self.base_action_scale = 0.03
        self.gripper_close_value = 0.0
        self.gripper_open_value = 1.0
        self.time_penalty = 0.01
        self.base_planar_reference = None
        self.joint_reference = None
        self.initial_base_pos = None
        self.initial_base_quat = None
        self.initial_target_pos = None
        self.initial_target_quat = None
        self.initial_joint_positions = None

        # 输入的三个量
        self.tcp_pose = None
        self.target_grasp_pose = None
        self.base_satellite_pose = None

        # 输出的两个量
        self.arm_joint_positions = None
        self.base_planar_position = None

        self.observation = None
        self.action = None

        self.base_alignment_error = None
        self.tcp_to_grasp_error = None
        self.phase = "not_started"
        self.last_gripper_command = self.gripper_open_value
        self.contact_hold_steps = 12
        self.edge_contact_tolerance = 0.08
        self.contact_force_threshold = 1e-3
        self.edge_contact_hold_count = 0
        self.edge_contact_active = False
        self.approach_alignment_threshold = 0.85
        self.approach_alignment = None
        self.grasp_finger_link_names = (
            "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Finger1_2_Link",
            "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Finger2_1_Link",
        )

    def _get_tcp_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ee_link = self.starlink_manipulator.robot.get_link(
            self.starlink_manipulator.params["end_effector"]
        )
        ee_pos = ee_link.get_pos()
        ee_quat = ee_link.get_quat()
        ee_rot = as_rotation_matrix(ee_quat, order="wxyz")
        tcp_pos = ee_pos + torch.matmul(ee_rot, self.tcp_offset_local)
        return tcp_pos, ee_quat

    def _get_target_grasp_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        target_link = self.target_object.robot.get_link(self.target_link_name)
        target_pos = target_link.get_pos()
        target_quat = target_link.get_quat()
        target_rot = as_rotation_matrix(target_quat, order="wxyz")
        grasp_point_pos = target_pos + torch.matmul(
            target_rot, self.target_grasp_offset_local
        )
        return grasp_point_pos, target_quat

    def _get_base_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        base_link = self.starlink_manipulator.robot.get_link(
            self.starlink_manipulator.params["base"]
        )
        return base_link.get_pos(), base_link.get_quat()

    def _compute_approach_alignment(self, tcp_quat: torch.Tensor) -> torch.Tensor:
        tcp_rot = as_rotation_matrix(tcp_quat, order="wxyz")
        local_approach_axis = self.tcp_offset_local / torch.clamp(
            torch.norm(self.tcp_offset_local),
            min=torch.tensor(1e-6, dtype=self.datatype, device=self.device),
        )
        world_approach_axis = torch.matmul(tcp_rot, local_approach_axis)
        world_down_axis = torch.tensor(
            [0.0, 0.0, -1.0], dtype=self.datatype, device=self.device
        )
        return torch.dot(world_approach_axis, world_down_axis)

    def _get_contact_world_positions(self, finger_link_name: str):
        finger_link = self.starlink_manipulator.robot.get_link(finger_link_name)
        contacts = self.starlink_manipulator.robot.get_contacts(with_entity=self.target_object.robot)
        positions = []

        if contacts is None:
            return positions

        link_a = contacts.get("link_a")
        link_b = contacts.get("link_b")
        contact_pos = contacts.get("position")
        force_a = contacts.get("force_a")
        force_b = contacts.get("force_b")

        if link_a is None or link_b is None or contact_pos is None:
            return positions

        for idx in range(link_a.shape[0]):
            match_a = int(link_a[idx].item()) == finger_link.idx
            match_b = int(link_b[idx].item()) == finger_link.idx
            if not (match_a or match_b):
                continue

            force = force_a[idx] if match_a else force_b[idx]
            if torch.norm(force).item() < self.contact_force_threshold:
                continue

            positions.append(contact_pos[idx].detach().clone())

        return positions

    def _check_edge_grasp_contact(self) -> bool:
        grasp_point_pos, _ = self._get_target_grasp_pose()
        finger_a_contacts = self._get_contact_world_positions(self.grasp_finger_link_names[0])
        finger_b_contacts = self._get_contact_world_positions(self.grasp_finger_link_names[1])

        has_finger_a_edge_contact = any(
            torch.norm(contact_pos - grasp_point_pos).item() <= self.edge_contact_tolerance
            for contact_pos in finger_a_contacts
        )
        has_finger_b_edge_contact = any(
            torch.norm(contact_pos - grasp_point_pos).item() <= self.edge_contact_tolerance
            for contact_pos in finger_b_contacts
        )

        self.edge_contact_active = has_finger_a_edge_contact and has_finger_b_edge_contact
        if self.edge_contact_active:
            self.edge_contact_hold_count += 1
        else:
            self.edge_contact_hold_count = 0

        return self.edge_contact_active

    def _update_io_tensors(self):
        tcp_pos, tcp_quat = self._get_tcp_pose()
        grasp_point_pos, grasp_point_quat = self._get_target_grasp_pose()
        base_pos, base_quat = self._get_base_pose()
        joint_qpos = self.starlink_manipulator.robot.get_qpos(
            qs_idx_local=self.starlink_manipulator.motors_qs
        )

        self.tcp_pose = torch.cat([tcp_pos, tcp_quat]).to(
            dtype=self.datatype, device=self.device
        )
        self.target_grasp_pose = torch.cat([grasp_point_pos, grasp_point_quat]).to(
            dtype=self.datatype, device=self.device
        )
        self.base_satellite_pose = torch.cat([base_pos, base_quat]).to(
            dtype=self.datatype, device=self.device
        )
        self.arm_joint_positions = torch.as_tensor(
            joint_qpos, dtype=self.datatype, device=self.device
        ).reshape(-1)
        self.base_planar_position = base_pos[:2].to(
            dtype=self.datatype, device=self.device
        )

        self.observation = torch.cat(
            [self.tcp_pose, self.target_grasp_pose, self.base_satellite_pose]
        )
        self.action = torch.cat(
            [self.arm_joint_positions, self.base_planar_position]
        )

        self.base_alignment_error = torch.norm(
            grasp_point_pos[:2] - self.base_planar_position
        )
        self.tcp_to_grasp_error = torch.norm(grasp_point_pos - tcp_pos)
        self.approach_alignment = self._compute_approach_alignment(tcp_quat)

        if self.base_alignment_error <= self.base_alignment_threshold:
            self.phase = "arm_approach"
        else:
            self.phase = "base_alignment"

        if (
            self.tcp_to_grasp_error <= self.grasp_success_threshold
            and self.approach_alignment >= self.approach_alignment_threshold
        ):
            self.phase = "grasp"

    def initialize(self) -> bool:
        try:
            if self.starlink_manipulator is None:
                self.starlink_manipulator = SatelliteManipulator(
                    name="franka_merge",
                    sensors=[],
                    backends=[],
                )
            if self.target_object is None:
                self.target_object = Robot(name=self.target_asset_name)

            self.gsim.start()
            self.starlink_manipulator.initialize()
            self.target_object.initialize()

            self.target_link_name = self.target_object.params.get(
                "grasp_link", self.target_object.params["base"]
            )
            self.target_grasp_offset_local = torch.tensor(
                self.target_object.params.get("grasp_offset_local", [0.0, 0.0, 0.0]),
                dtype=self.datatype,
                device=self.device,
            )
            self.starlink_manipulator.update_state()
            self._update_io_tensors()

            self.initial_target_pos = self.target_object.robot.get_link(
                self.target_link_name
            ).get_pos().detach().clone()
            self.initial_target_quat = self.target_object.robot.get_link(
                self.target_link_name
            ).get_quat().detach().clone()
            self.initial_base_pos = self.base_satellite_pose[:3].detach().clone()
            self.initial_base_quat = self.base_satellite_pose[3:].detach().clone()
            self.initial_joint_positions = self.arm_joint_positions.detach().clone()

            self.base_planar_reference = self.base_planar_position.detach().clone()
            self.joint_reference = self.arm_joint_positions.detach().clone()

            self.step_count = 0
            self.total_reward = 0.0
            self.success = False
            self.last_gripper_command = self.gripper_open_value
            self.edge_contact_hold_count = 0
            self.edge_contact_active = False
            self.status = TaskStatus.RUNNING
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.status = TaskStatus.FAILED
            return False

    def _apply_base_action(self, planar_action: torch.Tensor):
        force_xy = planar_action * self.base_force_scale_xy
        force_cmd = torch.tensor(
            [force_xy[0], force_xy[1], 0.0],
            dtype=self.datatype,
            device=self.device,
        )
        self.starlink_manipulator.apply_force(
            force=force_cmd,
            link_name=self.starlink_manipulator.params["base"],
        )

    def _apply_arm_action(self, joint_action: torch.Tensor):
        target_joint = self.arm_joint_positions + joint_action * self.joint_action_scale
        self.starlink_manipulator.control_joint_pos(target_joint)

    def apply_rl_action(self, action: torch.Tensor):
        action = torch.as_tensor(action, dtype=self.datatype, device=self.device).reshape(-1)
        if action.shape[0] != 8:
            raise ValueError(f"RL action should have 8 dims, got {tuple(action.shape)}")

        joint_action = torch.clamp(action[:6], -1.0, 1.0)
        planar_action = torch.clamp(action[6:], -1.0, 1.0)

        self._apply_base_action(planar_action)
        self._apply_arm_action(joint_action)

        if self.phase == "grasp":
            self.last_gripper_command = self.gripper_close_value
        else:
            self.last_gripper_command = self.gripper_open_value
        self.starlink_manipulator.control_gripper(self.last_gripper_command)

    def step(self, dt: float = None, action: Optional[torch.Tensor] = None) -> bool:
        if self.status != TaskStatus.RUNNING:
            return False

        try:
            if action is not None:
                self.apply_rl_action(action)
            else:
                self.starlink_manipulator.control_gripper(self.last_gripper_command)

            self.starlink_manipulator.step()
            self.target_object.step()
            self.gsim.step()
            self._update_io_tensors()
            self._check_edge_grasp_contact()

            reward = self.reward()
            self.total_reward += reward.item()
            self.step_count += 1

            if self.is_success():
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
        if (
            self.base_alignment_error is None
            or self.tcp_to_grasp_error is None
            or self.approach_alignment is None
        ):
            return torch.tensor(0.0, dtype=self.datatype, device=self.device)

        reward = -0 * self.base_alignment_error - 1.5 * self.tcp_to_grasp_error

        if self.tcp_to_grasp_error <= self.grasp_success_threshold:
            reward = reward + 10.0
        if not self.is_success():
            reward = reward - self.time_penalty

        return reward.to(dtype=self.datatype, device=self.device)

    def is_success(self) -> bool:
        if (
            self.base_alignment_error is None
            or self.tcp_to_grasp_error is None
            or self.approach_alignment is None
        ):
            return False
        return self.tcp_to_grasp_error <= self.grasp_success_threshold

    def reset(self) -> bool:
        try:
            if self.starlink_manipulator is None or self.target_object is None:
                self.logger.warning("Reset skipped because task is not initialized.")
                return False

            self.gsim.reset()

            if self.initial_joint_positions is None:
                self.initial_joint_positions = self.joint_reference.detach().clone()
            if self.initial_target_pos is None or self.initial_target_quat is None:
                target_link = self.target_object.robot.get_link(self.target_link_name)
                self.initial_target_pos = target_link.get_pos().detach().clone()
                self.initial_target_quat = target_link.get_quat().detach().clone()

            target_pos_noise = torch.tensor(
                [
                    np.random.uniform(-0.03, 0.03),
                    np.random.uniform(-0.03, 0.03),
                    np.random.uniform(-0.02, 0.02),
                ],
                dtype=self.datatype,
                device=self.device,
            )
            target_yaw_noise = float(np.random.uniform(-0.15, 0.15))
            yaw_half = 0.5 * target_yaw_noise
            yaw_quat = torch.tensor(
                [math.cos(yaw_half), 0.0, 0.0, math.sin(yaw_half)],
                dtype=self.datatype,
                device=self.device,
            )

            self.target_object.robot.set_pos(
                self.initial_target_pos + target_pos_noise,
                zero_velocity=True,
            )
            self.target_object.robot.set_quat(
                self.initial_target_quat,
                zero_velocity=True,
            )
            self.target_object.robot.set_quat(
                yaw_quat,
                zero_velocity=True,
                relative=True,
            )

            all_dofs = self.starlink_manipulator.motors_dof + self.starlink_manipulator.fingers_dof
            initial_full_dofs = self.starlink_manipulator.config["initial_dofs"].detach().clone()
            initial_full_dofs[:6] = self.initial_joint_positions
            self.starlink_manipulator.robot.set_dofs_position(
                initial_full_dofs,
                dofs_idx_local=all_dofs,
                zero_velocity=True,
            )
            self.starlink_manipulator.robot.set_dofs_velocity(
                [0.0] * len(all_dofs),
                dofs_idx_local=all_dofs,
            )

            self.starlink_manipulator.update_state()
            if hasattr(self.starlink_manipulator, "pid"):
                self.starlink_manipulator.pid.reset()
                base_pose_setpoint = torch.cat(
                    [
                        self.starlink_manipulator.ee_state.link_parent_global_state.position.detach().clone(),
                        self.starlink_manipulator.ee_state.link_parent_global_state.orient.detach().clone(),
                    ]
                )
                self.starlink_manipulator.pid.update_setpoint(base_pose_setpoint)

            self.last_gripper_command = self.gripper_open_value
            self.starlink_manipulator.control_gripper(self.gripper_open_value)
            self.gsim.step()
            self.starlink_manipulator.update_state()
            self._update_io_tensors()

            self.step_count = 0
            self.total_reward = 0.0
            self.success = False
            self.edge_contact_hold_count = 0
            self.edge_contact_active = False
            self.status = TaskStatus.RUNNING
            return True
        except Exception as e:
            self.logger.error(f"Reset failed: {e}")
            self.status = TaskStatus.FAILED
            return False

    def stop(self) -> bool:
        try:
            if self.target_object is not None:
                self.target_object.stop()
            if self.starlink_manipulator is not None:
                self.starlink_manipulator.stop()
            self.gsim.stop()

            if self.status not in {TaskStatus.SUCCEED, TaskStatus.FAILED}:
                self.status = TaskStatus.CANCELLED
            return True
        except Exception as e:
            self.logger.error(f"Stop error: {e}")
            self.status = TaskStatus.FAILED
            return False


if gym is not None and spaces is not None:
    class SatelliteSideGraspEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, target_asset_name: str = "starlink"):
            super().__init__()
            self.task = RLTask(target_asset_name=target_asset_name)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(8,), dtype=np.float32
            )

        def _get_obs(self):
            return self.task.observation.detach().cpu().numpy().astype(np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if self.task.status == TaskStatus.NOT_STARTED:
                if not self.task.initialize():
                    raise RuntimeError("RLTask initialize failed during env reset")
            else:
                if not self.task.reset():
                    raise RuntimeError("RLTask reset failed during env reset")
            return self._get_obs(), {}

        def step(self, action):
            keep_running = self.task.step(action=action)
            obs = self._get_obs()
            reward = float(self.task.reward().item())
            terminated = self.task.status in {TaskStatus.SUCCEED, TaskStatus.FAILED}
            truncated = (not keep_running) and self.task.status == TaskStatus.COMPLETED
            info = {
                "phase": self.task.phase,
                "base_alignment_error": float(self.task.base_alignment_error.item()),
                "tcp_to_grasp_error": float(self.task.tcp_to_grasp_error.item()),
                "success": self.task.success,
            }
            return obs, reward, terminated, truncated, info

        def close(self):
            self.task.stop()


def train_ppo(total_timesteps: int = 100_000, model_path: str = "ppo_satellite_side_grasp"):
    if gym is None or spaces is None:
        raise ImportError("gymnasium is not installed. Please install gymnasium first.")
    if PPO is None:
        raise ImportError("stable_baselines3 is not installed. Please install stable-baselines3 first.")

    env = SatelliteSideGraspEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    env.close()
    return model


def describe_observation_layout():
    return {
        "tcp_pose": 7,
        "target_grasp_pose": 7,
        "base_satellite_pose": 7,
        "total": 21,
    }


def describe_action_layout():
    return {
        "arm_joint_angles": 6,
        "base_planar_motion": 2,
        "total": 8,
    }

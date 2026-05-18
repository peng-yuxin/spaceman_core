"""Parallel-friendly wrapper for the satellite side grasp RL task.

This module keeps the task definition identical to ``RL_task.py`` and only
adds a vectorized environment/training interface for parallel rollout.
"""

import os
import sys
from pathlib import Path

current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from tasks.RL_task import (  # noqa: E402
    PPO,
    RLTask,
    TaskStatus,
    describe_action_layout,
    describe_observation_layout,
    gym,
    resolve_training_device,
    spaces,
)

try:
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
except Exception:
    DummyVecEnv = None
    SubprocVecEnv = None
    VecMonitor = None


class ParallelRLTask(RLTask):
    """Task definition identical to ``RLTask``.

    The current Genesis simulation is a singleton scene, so true in-process
    batched simulation is not available here. Parallelism is therefore
    implemented at the vectorized-environment level.
    """

    def __init__(self, target_asset_name: str = "starlink"):
        super().__init__(target_asset_name=target_asset_name)
        self.task_name = "RL_task_parallel"


if gym is not None and spaces is not None:
    class ParallelSatelliteSideGraspEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, target_asset_name: str = "starlink"):
            super().__init__()
            self.task = ParallelRLTask(target_asset_name=target_asset_name)
            self.observation_space = spaces.Box(
                low=-float("inf"), high=float("inf"), shape=(21,), dtype="float32"
            )
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(8,), dtype="float32"
            )

        def _get_obs(self):
            return self.task.observation.detach().cpu().numpy().astype("float32")

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if self.task.status == TaskStatus.NOT_STARTED:
                if not self.task.initialize():
                    raise RuntimeError(
                        "ParallelRLTask initialize failed during env reset"
                    )
            else:
                if not self.task.reset():
                    raise RuntimeError("ParallelRLTask reset failed during env reset")
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


def build_parallel_env(target_asset: str = "starlink"):
    if gym is None or spaces is None:
        raise ImportError("gymnasium is not installed. Please install gymnasium first.")
    return ParallelSatelliteSideGraspEnv(target_asset_name=target_asset)


def make_parallel_vec_env(
    num_envs: int = 4,
    target_asset: str = "starlink",
    start_method: str = "spawn",
):
    if gym is None or spaces is None:
        raise ImportError("gymnasium is not installed. Please install gymnasium first.")
    if DummyVecEnv is None or VecMonitor is None:
        raise ImportError(
            "stable_baselines3 vectorized env utilities are not installed."
        )

    num_envs = max(1, int(num_envs))

    def _make_env(rank: int):
        def _init():
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            env = build_parallel_env(target_asset=target_asset)
            env.reset(seed=rank)
            return env

        return _init

    env_fns = [_make_env(rank) for rank in range(num_envs)]

    if num_envs == 1 or SubprocVecEnv is None:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns, start_method=start_method)

    return VecMonitor(vec_env)


def train_parallel_ppo(
    total_timesteps: int = 100_000,
    model_path: str = "ppo_satellite_side_grasp_parallel",
    num_envs: int = 4,
    learning_rate: float = 3e-4,
    n_steps: int = 1024,
    batch_size: int = 256,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    seed: int = 42,
    device: str = "auto",
    target_asset: str = "starlink",
):
    if gym is None or spaces is None:
        raise ImportError("gymnasium is not installed. Please install gymnasium first.")
    if PPO is None:
        raise ImportError(
            "stable_baselines3 is not installed. Please install stable-baselines3 first."
        )

    training_device = resolve_training_device(device)

    env = make_parallel_vec_env(
        num_envs=num_envs,
        target_asset=target_asset,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
        seed=seed,
        device=training_device,
    )
    print(f"Using parallel PPO training device: {training_device}")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(model_path)
    env.close()
    return model


__all__ = [
    "ParallelRLTask",
    "ParallelSatelliteSideGraspEnv",
    "build_parallel_env",
    "make_parallel_vec_env",
    "train_parallel_ppo",
    "describe_observation_layout",
    "describe_action_layout",
]

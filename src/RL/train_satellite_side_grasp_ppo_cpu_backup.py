import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

current_file_path = Path(__file__).resolve().parent
project_src_path = current_file_path.parent
project_root_path = project_src_path.parent
sys.path.append(str(project_src_path))

from tasks.RL_task import PPO, SatelliteSideGraspEnv, gym, spaces

try:
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
except Exception:
    CheckpointCallback = None
    DummyVecEnv = None
    VecMonitor = None


def build_env(target_asset: str):
    return SatelliteSideGraspEnv(target_asset_name=target_asset)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO for the satellite side-edge grasp task."
    )
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--save-dir", type=str, default="outputs/rl/ppo_satellite_side_grasp")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-freq", type=int, default=20_000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--target-asset", type=str, default="starlink")
    return parser.parse_args()


def ensure_dependencies():
    missing = []
    if gym is None or spaces is None:
        missing.append("gymnasium")
    if PPO is None:
        missing.append("stable-baselines3")
    if CheckpointCallback is None or DummyVecEnv is None or VecMonitor is None:
        missing.append("stable-baselines3 common utilities")

    if missing:
        deps = ", ".join(sorted(set(missing)))
        raise ImportError(
            f"Missing training dependencies: {deps}. "
            "Please install them in the Spaceman_env environment before training."
        )


def make_run_dir(base_dir: Path, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = run_name if run_name else f"run_{timestamp}"
    run_dir = base_dir / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_config(run_dir: Path, args):
    config = {
        "total_timesteps": args.total_timesteps,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "checkpoint_freq": args.checkpoint_freq,
        "target_asset": args.target_asset,
    }
    with open(run_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def train():
    args = parse_args()
    ensure_dependencies()

    base_dir = project_root_path / args.save_dir
    run_dir = make_run_dir(base_dir, args.run_name)
    save_run_config(run_dir, args)

    env = DummyVecEnv([lambda: build_env(args.target_asset)])
    env = VecMonitor(env)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_satellite_side_grasp",
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=str(run_dir / "tensorboard"),
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    model.save(str(run_dir / "final_model"))
    env.close()

    print(f"Training completed. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    train()

"""
Training script for hide and seek multi-agent RL with Ray RLlib.

Usage:
    python scripts/train.py --num-hiders 2 --num-seekers 2 --num-workers 4
"""

import argparse
import os
from pathlib import Path

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
import supersuit as ss

from environments.hide_and_seek import env_creator
from callbacks.video_recorder import VideoRecorderCallback


def make_env_creator(env_config):
    """
    Create and wrap the environment for RLlib.

    Applies preprocessing wrappers and converts to RLlib-compatible format.
    """
    # Create base environment
    env = env_creator(env_config)

    # Apply SuperSuit wrappers
    # Normalize observations to [0, 1]
    env = ss.dtype_v0(env, dtype="float32")

    # Flatten observation if needed (optional)
    # env = ss.flatten_v0(env)

    # Wrap for RLlib
    env = ParallelPettingZooEnv(env)

    return env


def register_custom_env():
    """Register the custom environment with Ray."""
    register_env("hide_and_seek_v0", make_env_creator)


def train(args):
    """
    Main training function.

    Sets up Ray RLlib PPO algorithm and runs training.
    """
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=True,
            num_cpus=args.num_cpus,
            num_gpus=args.num_gpus,
        )

    # Register environment
    register_custom_env()

    # Environment configuration
    env_config = {
        "grid_size": args.grid_size,
        "num_hiders": args.num_hiders,
        "num_seekers": args.num_seekers,
        "fov_size": args.fov_size,
        "hiding_steps": args.hiding_steps,
        "seeking_steps": args.seeking_steps,
        "wall_density": args.wall_density,
        "catch_radius": args.catch_radius,
        "render_mode": "rgb_array" if args.record_video else None,
    }

    # Get list of all agents for policy mapping
    total_agents = args.num_hiders + args.num_seekers

    # Policy mapping: separate policies for hiders and seekers
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        """Map agent IDs to policy IDs."""
        if agent_id.startswith("hider"):
            return "hider_policy"
        else:
            return "seeker_policy"

    # Create PPO configuration
    config = (
        PPOConfig()
        .environment(
            env="hide_and_seek_v0",
            env_config=env_config,
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=args.num_workers,
            num_envs_per_worker=1,
            rollout_fragment_length=200,
        )
        .training(
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=10,
            lr=args.learning_rate,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            model={
                # CNN configuration for vision-based observations
                "conv_filters": [
                    [16, [3, 3], 1],   # 16 filters, 3x3 kernel, stride 1
                    [32, [3, 3], 1],   # 32 filters, 3x3 kernel, stride 1
                ],
                "conv_activation": "relu",
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "vf_share_layers": True,
            },
        )
        .multi_agent(
            policies={
                "hider_policy": (
                    None,  # Use default policy class
                    None,  # Observation space (auto-detected)
                    None,  # Action space (auto-detected)
                    {},    # Policy config
                ),
                "seeker_policy": (
                    None,
                    None,
                    None,
                    {},
                ),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["hider_policy", "seeker_policy"],
        )
        .resources(
            num_gpus=1 if args.num_gpus > 0 else 0,
            num_cpus_per_worker=1,
        )
        .debugging(
            log_level="INFO",
        )
        .callbacks(VideoRecorderCallback if args.record_video else None)
    )

    # Add evaluation workers for periodic evaluation
    if args.eval_interval > 0:
        config = config.evaluation(
            evaluation_interval=args.eval_interval,
            evaluation_num_workers=1,
            evaluation_duration=10,
            evaluation_config={
                "render_env": args.record_video,
            },
        )

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stop conditions
    stop_config = {
        "timesteps_total": args.total_timesteps,
    }

    print("=" * 60)
    print("Starting Hide and Seek Training")
    print("=" * 60)
    print(f"Environment: {args.num_hiders} hiders vs {args.num_seekers} seekers")
    print(f"Grid size: {args.grid_size}x{args.grid_size}")
    print(f"Field of view: {args.fov_size}x{args.fov_size}")
    print(f"Workers: {args.num_workers}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Run training with Tune
    results = tune.run(
        "PPO",
        name="hide_and_seek_experiment",
        config=config.to_dict(),
        stop=stop_config,
        checkpoint_freq=10,
        checkpoint_at_end=True,
        local_dir=str(output_dir),
        verbose=1,
        restore=args.restore_checkpoint,
    )

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

    # Get best checkpoint
    best_checkpoint = results.get_best_checkpoint(
        trial=results.get_best_trial("episode_reward_mean", "max"),
        metric="episode_reward_mean",
        mode="max",
    )

    if best_checkpoint:
        print(f"Best checkpoint: {best_checkpoint}")

    return results


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(
        description="Train hide and seek agents with Ray RLlib"
    )

    # Environment arguments
    env_group = parser.add_argument_group("Environment")
    env_group.add_argument("--grid-size", type=int, default=15,
                          help="Size of the grid world")
    env_group.add_argument("--num-hiders", type=int, default=2,
                          help="Number of hider agents")
    env_group.add_argument("--num-seekers", type=int, default=2,
                          help="Number of seeker agents")
    env_group.add_argument("--fov-size", type=int, default=5,
                          help="Field of view size (must be odd)")
    env_group.add_argument("--hiding-steps", type=int, default=100,
                          help="Number of steps in hiding phase")
    env_group.add_argument("--seeking-steps", type=int, default=400,
                          help="Number of steps in seeking phase")
    env_group.add_argument("--wall-density", type=float, default=0.1,
                          help="Density of walls (0-1)")
    env_group.add_argument("--catch-radius", type=float, default=1.0,
                          help="Distance for catching hiders")

    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--num-workers", type=int, default=4,
                            help="Number of rollout workers")
    train_group.add_argument("--num-cpus", type=int, default=None,
                            help="Number of CPUs for Ray (default: auto)")
    train_group.add_argument("--num-gpus", type=int, default=0,
                            help="Number of GPUs")
    train_group.add_argument("--total-timesteps", type=int, default=1_000_000,
                            help="Total training timesteps")
    train_group.add_argument("--learning-rate", type=float, default=3e-4,
                            help="Learning rate")
    train_group.add_argument("--eval-interval", type=int, default=10,
                            help="Evaluation interval (0 to disable)")

    # Logging arguments
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--output-dir", type=str, default="./outputs",
                          help="Output directory for checkpoints and logs")
    log_group.add_argument("--record-video", action="store_true",
                          help="Record videos during training")
    log_group.add_argument("--restore-checkpoint", type=str, default=None,
                          help="Path to checkpoint to restore from")

    args = parser.parse_args()

    # Validate arguments
    if args.fov_size % 2 == 0:
        parser.error("fov-size must be an odd number")

    # Run training
    try:
        results = train(args)
        print("\nTraining finished successfully!")
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        raise
    finally:
        # Cleanup Ray
        ray.shutdown()


if __name__ == "__main__":
    main()

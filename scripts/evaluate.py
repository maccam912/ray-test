"""
Evaluation script for trained hide and seek agents.

Loads a trained checkpoint and runs episodes to evaluate performance.

Usage:
    python scripts/evaluate.py --checkpoint path/to/checkpoint --num-episodes 10 --render
"""

import argparse
import time
from pathlib import Path
import numpy as np

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
import supersuit as ss

from environments.hide_and_seek import env_creator


def make_env_creator(env_config):
    """Create and wrap the environment for RLlib."""
    env = env_creator(env_config)
    env = ss.dtype_v0(env, dtype="float32")
    env = ParallelPettingZooEnv(env)
    return env


def register_custom_env():
    """Register the custom environment with Ray."""
    register_env("hide_and_seek_v0", make_env_creator)


def evaluate(args):
    """
    Load trained model and run evaluation episodes.

    Args:
        args: Command line arguments
    """
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    # Register environment
    register_custom_env()

    # Load the trained algorithm
    print(f"Loading checkpoint from: {args.checkpoint}")
    try:
        algo = PPO.from_checkpoint(args.checkpoint)
        print("✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Create environment for evaluation
    env_config = {
        "grid_size": args.grid_size,
        "num_hiders": args.num_hiders,
        "num_seekers": args.num_seekers,
        "fov_size": args.fov_size,
        "hiding_steps": args.hiding_steps,
        "seeking_steps": args.seeking_steps,
        "wall_density": args.wall_density,
        "catch_radius": args.catch_radius,
        "render_mode": "rgb_array" if args.save_video else ("human" if args.render else None),
    }

    env = make_env_creator(env_config)

    # Statistics tracking
    episode_rewards = []
    episode_lengths = []
    hiders_caught_stats = []
    phase_stats = {"hiding": [], "seeking": []}

    print(f"\nRunning {args.num_episodes} evaluation episodes...")
    print("=" * 60)

    for episode in range(args.num_episodes):
        obs, info = env.reset()
        episode_reward = {agent: 0.0 for agent in obs.keys()}
        episode_length = 0
        frames = []
        done = {agent: False for agent in obs.keys()}

        print(f"\nEpisode {episode + 1}/{args.num_episodes}")

        while not all(done.values()):
            # Get actions from policies
            actions = {}
            for agent_id, agent_obs in obs.items():
                if not done.get(agent_id, False):
                    # Determine which policy to use
                    if agent_id.startswith("hider"):
                        policy_id = "hider_policy"
                    else:
                        policy_id = "seeker_policy"

                    # Compute action
                    action = algo.compute_single_action(
                        agent_obs,
                        policy_id=policy_id,
                        explore=False,  # Use deterministic policy
                    )
                    actions[agent_id] = action

            # Step environment
            obs, rewards, terminations, truncations, infos = env.step(actions)

            # Update done flags
            for agent_id in obs.keys():
                done[agent_id] = terminations.get(agent_id, False) or truncations.get(agent_id, False)

            # Accumulate rewards
            for agent_id, reward in rewards.items():
                episode_reward[agent_id] += reward

            episode_length += 1

            # Render if requested
            if args.render or args.save_video:
                frame = env.render()
                if args.save_video and frame is not None:
                    frames.append(frame)

                if args.render:
                    time.sleep(0.1)  # Slow down for human viewing

        # Collect statistics
        total_episode_reward = sum(episode_reward.values())
        episode_rewards.append(total_episode_reward)
        episode_lengths.append(episode_length)

        # Get final info
        final_info = list(infos.values())[0] if infos else {}
        num_caught = final_info.get("num_caught", 0)
        hiders_caught_stats.append(num_caught)

        print(f"  Total reward: {total_episode_reward:.2f}")
        print(f"  Episode length: {episode_length}")
        print(f"  Hiders caught: {num_caught}/{args.num_hiders}")
        print(f"  Hider rewards: {sum(r for a, r in episode_reward.items() if a.startswith('hider')):.2f}")
        print(f"  Seeker rewards: {sum(r for a, r in episode_reward.items() if a.startswith('seeker')):.2f}")

        # Save video if requested
        if args.save_video and len(frames) > 0:
            save_episode_video(frames, episode, args.output_dir)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Episodes: {args.num_episodes}")
    print(f"Average total reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Average hiders caught: {np.mean(hiders_caught_stats):.2f} ± {np.std(hiders_caught_stats):.2f}")
    print(f"Catch rate: {np.mean(hiders_caught_stats) / args.num_hiders * 100:.1f}%")
    print("=" * 60)

    # Cleanup
    env.close()
    algo.stop()
    ray.shutdown()


def save_episode_video(frames, episode_num, output_dir):
    """Save episode frames as video."""
    try:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        output_path = Path(output_dir) / "videos" / f"eval_episode_{episode_num}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        clip = ImageSequenceClip(list(frames), fps=10)
        clip.write_videofile(
            str(output_path),
            codec='libx264',
            audio=False,
            verbose=False,
            logger=None
        )

        print(f"  ✓ Saved video: {output_path}")

    except ImportError:
        print("  Warning: moviepy not installed, cannot save video")
    except Exception as e:
        print(f"  Error saving video: {e}")


def main():
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained hide and seek agents"
    )

    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained checkpoint directory")

    # Evaluation arguments
    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument("--num-episodes", type=int, default=10,
                           help="Number of episodes to evaluate")
    eval_group.add_argument("--render", action="store_true",
                           help="Render episodes in real-time")
    eval_group.add_argument("--save-video", action="store_true",
                           help="Save episodes as videos")
    eval_group.add_argument("--output-dir", type=str, default="./outputs",
                           help="Output directory for videos")

    # Environment arguments (should match training config)
    env_group = parser.add_argument_group("Environment")
    env_group.add_argument("--grid-size", type=int, default=15,
                          help="Size of the grid world")
    env_group.add_argument("--num-hiders", type=int, default=2,
                          help="Number of hider agents")
    env_group.add_argument("--num-seekers", type=int, default=2,
                          help="Number of seeker agents")
    env_group.add_argument("--fov-size", type=int, default=5,
                          help="Field of view size")
    env_group.add_argument("--hiding-steps", type=int, default=100,
                          help="Number of steps in hiding phase")
    env_group.add_argument("--seeking-steps", type=int, default=400,
                          help="Number of steps in seeking phase")
    env_group.add_argument("--wall-density", type=float, default=0.1,
                          help="Density of walls (0-1)")
    env_group.add_argument("--catch-radius", type=float, default=1.0,
                          help="Distance for catching hiders")

    args = parser.parse_args()

    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    # Run evaluation
    try:
        evaluate(args)
        print("\nEvaluation completed successfully!")
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\n\nEvaluation failed with error: {e}")
        raise


if __name__ == "__main__":
    main()

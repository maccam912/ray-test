"""
Quick start script to verify installation and run a short demo.

This script:
1. Verifies all dependencies are installed
2. Creates a simple environment instance
3. Runs a few random episodes
4. Shows basic statistics

Usage:
    python scripts/quick_start.py
"""

import sys


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("Checking dependencies...")
    print("-" * 60)

    dependencies = [
        ("ray", "Ray"),
        ("pettingzoo", "PettingZoo"),
        ("gymnasium", "Gymnasium"),
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pygame", "Pygame"),
        ("supersuit", "SuperSuit"),
    ]

    all_installed = True
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print(f"✓ {display_name}")
        except ImportError:
            print(f"✗ {display_name} - NOT INSTALLED")
            all_installed = False

    print("-" * 60)

    if not all_installed:
        print("\nSome dependencies are missing. Install them with:")
        print("  uv sync")
        print("  # or")
        print("  pip install -e .")
        sys.exit(1)

    print("✓ All dependencies installed!\n")


def run_demo():
    """Run a quick demo of the environment."""
    import numpy as np
    from environments.hide_and_seek import HideAndSeekEnv

    print("Creating Hide and Seek environment...")
    print("-" * 60)

    # Create environment with smaller grid for quick demo
    env = HideAndSeekEnv(
        grid_size=10,
        num_hiders=2,
        num_seekers=1,
        fov_size=5,
        hiding_steps=20,
        seeking_steps=80,
        wall_density=0.1,
        render_mode=None,  # No rendering for quick demo
    )

    print(f"Grid size: {env.grid_size}x{env.grid_size}")
    print(f"Agents: {env.num_hiders} hiders vs {env.num_seekers} seekers")
    print(f"Total episode length: {env.max_steps} steps")
    print(f"Observation space: {env.observation_space('hider_0')}")
    print(f"Action space: {env.action_space('hider_0')}")
    print("-" * 60)

    # Run a few random episodes
    num_episodes = 3
    print(f"\nRunning {num_episodes} random episodes...\n")

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = {agent: False for agent in env.agents}
        episode_rewards = {agent: 0.0 for agent in env.agents}
        steps = 0

        while not all(done.values()) and steps < env.max_steps:
            # Random actions
            actions = {
                agent: env.action_space(agent).sample()
                for agent in obs.keys()
                if not done.get(agent, False)
            }

            obs, rewards, terminations, truncations, infos = env.step(actions)

            # Update done flags
            for agent in env.agents:
                done[agent] = terminations.get(agent, False) or truncations.get(agent, False)

            # Accumulate rewards
            for agent, reward in rewards.items():
                episode_rewards[agent] += reward

            steps += 1

        # Print episode summary
        total_reward = sum(episode_rewards.values())
        num_caught = len(env.caught_hiders)

        print(f"Episode {episode + 1}:")
        print(f"  Steps: {steps}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Hiders caught: {num_caught}/{env.num_hiders}")
        print(f"  Hider total reward: {sum(r for a, r in episode_rewards.items() if a.startswith('hider')):.2f}")
        print(f"  Seeker total reward: {sum(r for a, r in episode_rewards.items() if a.startswith('seeker')):.2f}")

    env.close()

    print("\n" + "=" * 60)
    print("✓ Demo completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Train agents:")
    print("     python scripts/train.py --num-workers 2 --total-timesteps 100000")
    print("\n  2. Evaluate trained model:")
    print("     python scripts/evaluate.py --checkpoint <path> --render")
    print("\n  3. Deploy to Kubernetes:")
    print("     ./kubernetes/submit-job.sh")
    print("\nSee README.md for full documentation.")


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("Hide and Seek Multi-Agent RL - Quick Start")
    print("=" * 60)
    print()

    try:
        check_dependencies()
        run_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

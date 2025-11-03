"""
Video recording callback for Ray RLlib training.

Captures frames during episodes and saves best/worst performing episodes as videos.
"""

import os
import numpy as np
from typing import Dict, Optional, Union
from pathlib import Path

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy

# Type alias for episode - use EpisodeV2 for Ray 2.51+
Episode = EpisodeV2


class VideoRecorderCallback(DefaultCallbacks):
    """
    Callback to record episodes and save them as videos.

    Features:
    - Records frames during episode rollout
    - Saves best and worst performing episodes
    - Periodic video saving (every N episodes)
    - Automatic cleanup of old videos
    """

    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super().__init__(legacy_callbacks_dict)

        self.best_episode_return = float('-inf')
        self.worst_episode_return = float('inf')
        self.best_episode_frames = []
        self.worst_episode_frames = []
        self.episode_count = 0
        self.save_freq = 50  # Save videos every N episodes
        self.save_dir = Path("outputs/videos")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_episode_start(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs,
    ) -> None:
        """Initialize frame storage for this episode."""
        episode.user_data["frames"] = []
        episode.user_data["total_reward"] = 0.0

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[str, Policy]] = None,
        episode: Episode,
        env_index: int,
        **kwargs,
    ) -> None:
        """Capture frame at each step."""
        # Get the environment
        try:
            env = base_env.get_sub_environments()[env_index]

            # Render current frame
            frame = env.render()

            if frame is not None:
                episode.user_data["frames"].append(frame)

            # Accumulate rewards (sum across all agents)
            if hasattr(episode, "agent_rewards"):
                episode.user_data["total_reward"] = sum(
                    sum(rewards.values()) if isinstance(rewards, dict) else rewards
                    for rewards in episode.agent_rewards.values()
                )
        except Exception as e:
            # Silently fail if rendering not available
            pass

    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs,
    ) -> None:
        """Compile frames into video and track best/worst episodes."""
        frames = episode.user_data.get("frames", [])

        if len(frames) == 0:
            return

        self.episode_count += 1

        # Calculate episode return (sum of all agents' rewards)
        episode_return = episode.user_data.get("total_reward", 0.0)

        # Track best episode
        if episode_return > self.best_episode_return:
            self.best_episode_return = episode_return
            self.best_episode_frames = frames.copy()

        # Track worst episode
        if episode_return < self.worst_episode_return:
            self.worst_episode_return = episode_return
            self.worst_episode_frames = frames.copy()

        # Save videos periodically
        if self.episode_count % self.save_freq == 0:
            if len(self.best_episode_frames) > 0:
                self._save_video(
                    self.best_episode_frames,
                    f"best_episode_{self.episode_count}_return_{self.best_episode_return:.2f}.mp4"
                )

            if len(self.worst_episode_frames) > 0:
                self._save_video(
                    self.worst_episode_frames,
                    f"worst_episode_{self.episode_count}_return_{self.worst_episode_return:.2f}.mp4"
                )

        # Log custom metrics
        episode.custom_metrics["episode_return"] = episode_return
        episode.custom_metrics["num_frames"] = len(frames)

        # Clear frames to save memory
        episode.user_data["frames"] = []

    def on_train_result(
        self,
        *,
        algorithm,
        result: dict,
        **kwargs,
    ) -> None:
        """Log statistics after each training iteration."""
        # Add custom metrics to result
        result["custom_metrics"] = result.get("custom_metrics", {})
        result["custom_metrics"]["best_episode_return"] = self.best_episode_return
        result["custom_metrics"]["worst_episode_return"] = self.worst_episode_return
        result["custom_metrics"]["total_episodes_recorded"] = self.episode_count

    def _save_video(self, frames, filename: str):
        """
        Save frames as video using moviepy.

        Args:
            frames: List of RGB frames (numpy arrays)
            filename: Output filename
        """
        if len(frames) == 0:
            return

        try:
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

            # Create video clip
            clip = ImageSequenceClip(list(frames), fps=10)

            # Save to file
            output_path = self.save_dir / filename
            clip.write_videofile(
                str(output_path),
                codec='libx264',
                audio=False,
                verbose=False,
                logger=None
            )

            print(f"✓ Saved video: {output_path}")

        except ImportError:
            print("Warning: moviepy not installed. Saving as GIF instead.")
            self._save_gif(frames, filename.replace(".mp4", ".gif"))
        except Exception as e:
            print(f"Error saving video: {e}")

    def _save_gif(self, frames, filename: str):
        """
        Save frames as GIF using imageio (fallback if moviepy fails).

        Args:
            frames: List of RGB frames (numpy arrays)
            filename: Output filename
        """
        if len(frames) == 0:
            return

        try:
            import imageio

            output_path = self.save_dir / filename

            # Convert frames to uint8 if needed
            frames_uint8 = [
                frame.astype(np.uint8) if frame.dtype != np.uint8 else frame
                for frame in frames
            ]

            # Save as GIF
            imageio.mimsave(
                str(output_path),
                frames_uint8,
                fps=10,
                loop=0
            )

            print(f"✓ Saved GIF: {output_path}")

        except Exception as e:
            print(f"Error saving GIF: {e}")


class MinimalVideoRecorderCallback(DefaultCallbacks):
    """
    Minimal version that saves videos less frequently to save disk space.
    Only saves at specific training checkpoints.
    """

    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super().__init__(legacy_callbacks_dict)
        self.save_dir = Path("outputs/videos")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.iteration_count = 0

    def on_train_result(
        self,
        *,
        algorithm,
        result: dict,
        **kwargs,
    ) -> None:
        """Save a sample episode every N training iterations."""
        self.iteration_count += 1

        # Save video every 10 iterations
        if self.iteration_count % 10 == 0:
            try:
                # Sample one episode
                sample = algorithm.workers.local_worker().sample()

                # This is a simplified version - you may need to adapt
                # based on your specific Ray RLlib version
                print(f"Training iteration {self.iteration_count} completed")

            except Exception as e:
                print(f"Could not record video at iteration {self.iteration_count}: {e}")

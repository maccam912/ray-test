"""
Hide and Seek Multi-Agent Environment

Classic mechanics:
1. Hiding phase: Hiders can move, seekers are frozen
2. Seeking phase: All agents can move, seekers try to catch hiders
"""

import functools
import numpy as np
from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv
from typing import Dict, Optional, Tuple


class HideAndSeekEnv(ParallelEnv):
    """
    Multi-agent hide and seek environment with partial observability.

    Features:
    - Two-phase gameplay: hiding phase → seeking phase
    - Limited vision cones (5x5 FOV per agent)
    - Randomly generated walls and obstacles
    - Vision-based observations (multi-channel images)
    """

    metadata = {
        "name": "hide_and_seek_v0",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        grid_size: int = 15,
        num_hiders: int = 2,
        num_seekers: int = 2,
        fov_size: int = 5,
        hiding_steps: int = 100,
        seeking_steps: int = 400,
        wall_density: float = 0.1,
        catch_radius: float = 1.0,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            grid_size: Size of the square grid
            num_hiders: Number of hider agents
            num_seekers: Number of seeker agents
            fov_size: Field of view size (must be odd)
            hiding_steps: Steps in hiding phase (seekers frozen)
            seeking_steps: Steps in seeking phase (all agents move)
            wall_density: Probability of wall in each cell (0-1)
            catch_radius: Distance within which seeker catches hider
            render_mode: Rendering mode ("human" or "rgb_array")
            seed: Random seed for reproducibility
        """
        super().__init__()

        self.grid_size = grid_size
        self.num_hiders = num_hiders
        self.num_seekers = num_seekers
        self.fov_size = fov_size
        self.hiding_steps = hiding_steps
        self.seeking_steps = seeking_steps
        self.max_steps = hiding_steps + seeking_steps
        self.wall_density = wall_density
        self.catch_radius = catch_radius
        self.render_mode = render_mode
        self._seed = seed

        # Define agent names
        self.possible_agents = (
            [f"hider_{i}" for i in range(num_hiders)] +
            [f"seeker_{i}" for i in range(num_seekers)]
        )

        # State variables (initialized in reset)
        self.agents = []
        self.agent_positions: Dict[str, np.ndarray] = {}
        self.agent_types: Dict[str, str] = {}
        self.walls: np.ndarray = None
        self.timestep = 0
        self.caught_hiders = set()

        # Rendering
        self._screen = None
        self._clock = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """
        Vision-based observation: 4D tensor (channels, height, width)
        Channels:
          0: Current agent position
          1: Same team agents
          2: Opposite team agents
          3: Walls
        """
        return Box(
            low=0,
            high=1,
            shape=(4, self.fov_size, self.fov_size),
            dtype=np.float32
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """5 discrete actions: stay, up, down, left, right"""
        return Discrete(5)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state."""
        if seed is not None:
            self._seed = seed
        if self._seed is not None:
            np.random.seed(self._seed)

        # Reset agent list and types
        self.agents = self.possible_agents.copy()
        self.agent_types = {}
        for agent in self.agents:
            if agent.startswith("hider"):
                self.agent_types[agent] = "hider"
            else:
                self.agent_types[agent] = "seeker"

        self.timestep = 0
        self.caught_hiders = set()

        # Generate walls (random obstacles)
        self.walls = np.random.random((self.grid_size, self.grid_size)) < self.wall_density

        # Ensure borders are not walls
        self.walls[0, :] = False
        self.walls[-1, :] = False
        self.walls[:, 0] = False
        self.walls[:, -1] = False

        # Initialize agent positions (avoiding walls and overlaps)
        self.agent_positions = {}
        occupied = set()

        for agent in self.agents:
            while True:
                pos = np.random.randint(1, self.grid_size - 1, size=2)
                pos_tuple = tuple(pos)
                if not self.walls[pos[0], pos[1]] and pos_tuple not in occupied:
                    self.agent_positions[agent] = pos
                    occupied.add(pos_tuple)
                    break

        # Generate observations
        observations = {
            agent: self._get_observation(agent)
            for agent in self.agents
        }

        infos = {agent: {"phase": "hiding"} for agent in self.agents}

        return observations, infos

    def step(self, actions: Dict[str, int]):
        """Execute one step with all agents acting simultaneously."""
        if not self.agents:
            return {}, {}, {}, {}, {}

        current_phase = self._get_current_phase()

        # Update positions based on actions
        for agent, action in actions.items():
            if agent not in self.agents:
                continue

            # During hiding phase, only hiders can move
            if current_phase == "hiding" and self.agent_types[agent] == "seeker":
                continue

            # Skip caught hiders
            if agent in self.caught_hiders:
                continue

            self._update_position(agent, action)

        # Check for catches (only during seeking phase)
        if current_phase == "seeking":
            self._check_catches()

        # Calculate rewards
        rewards = {agent: self._compute_reward(agent, current_phase) for agent in self.agents}

        # Check termination conditions
        self.timestep += 1
        terminations = {agent: agent in self.caught_hiders for agent in self.agents}
        truncations = {agent: self.timestep >= self.max_steps for agent in self.agents}

        # Generate new observations
        observations = {
            agent: self._get_observation(agent)
            for agent in self.agents
        }

        # Update infos
        infos = {
            agent: {
                "phase": current_phase,
                "caught": agent in self.caught_hiders,
                "num_caught": len(self.caught_hiders),
            }
            for agent in self.agents
        }

        # Note: We keep all agents in self.agents for observation purposes
        # even when caught/terminated, but they won't receive further rewards

        return observations, rewards, terminations, truncations, infos

    def _get_current_phase(self) -> str:
        """Determine current game phase based on timestep."""
        if self.timestep < self.hiding_steps:
            return "hiding"
        else:
            return "seeking"

    def _get_observation(self, agent: str) -> np.ndarray:
        """
        Generate vision-based observation centered on agent with limited FOV.
        Returns a 4-channel image representing different entity types.
        """
        obs = np.zeros((4, self.fov_size, self.fov_size), dtype=np.float32)

        if agent not in self.agent_positions:
            return obs

        pos = self.agent_positions[agent]
        half_fov = self.fov_size // 2
        agent_type = self.agent_types[agent]

        # Iterate through field of view
        for dy in range(-half_fov, half_fov + 1):
            for dx in range(-half_fov, half_fov + 1):
                world_x = pos[0] + dx
                world_y = pos[1] + dy

                obs_x = dx + half_fov
                obs_y = dy + half_fov

                # Check bounds
                if 0 <= world_x < self.grid_size and 0 <= world_y < self.grid_size:
                    # Channel 0: Current agent
                    if world_x == pos[0] and world_y == pos[1]:
                        obs[0, obs_y, obs_x] = 1.0

                    # Channel 1: Same team agents
                    # Channel 2: Opposite team agents
                    for other_agent in self.agents:
                        if other_agent == agent or other_agent not in self.agent_positions:
                            continue

                        other_pos = self.agent_positions[other_agent]
                        if other_pos[0] == world_x and other_pos[1] == world_y:
                            if self.agent_types[other_agent] == agent_type:
                                obs[1, obs_y, obs_x] = 1.0  # Same team
                            else:
                                obs[2, obs_y, obs_x] = 1.0  # Opposite team

                    # Channel 3: Walls
                    if self.walls[world_x, world_y]:
                        obs[3, obs_y, obs_x] = 1.0
                else:
                    # Out of bounds = wall
                    obs[3, obs_y, obs_x] = 1.0

        return obs

    def _update_position(self, agent: str, action: int):
        """
        Update agent position based on action.
        Actions: 0=stay, 1=up, 2=down, 3=left, 4=right
        """
        if agent not in self.agent_positions:
            return

        new_pos = self.agent_positions[agent].copy()

        if action == 1:  # up
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 2:  # down
            new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)
        elif action == 3:  # left
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 4:  # right
            new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1)
        # action == 0 means stay in place

        # Check if new position is valid (not a wall)
        if not self.walls[new_pos[0], new_pos[1]]:
            self.agent_positions[agent] = new_pos

    def _check_catches(self):
        """Check if any seeker has caught any hider."""
        for seeker in self.agents:
            if self.agent_types[seeker] != "seeker":
                continue

            seeker_pos = self.agent_positions[seeker]

            for hider in self.agents:
                if self.agent_types[hider] != "hider" or hider in self.caught_hiders:
                    continue

                hider_pos = self.agent_positions[hider]
                distance = np.linalg.norm(seeker_pos - hider_pos)

                if distance <= self.catch_radius:
                    self.caught_hiders.add(hider)

    def _compute_reward(self, agent: str, phase: str) -> float:
        """
        Compute reward for agent based on game state.

        Hiders:
        - Small positive reward for each step they remain hidden during seeking phase
        - Large negative reward when caught
        - No reward during hiding phase

        Seekers:
        - Large positive reward for catching a hider
        - Small negative reward per step (encourages efficiency)
        """
        reward = 0.0
        agent_type = self.agent_types[agent]

        if agent_type == "hider":
            if agent in self.caught_hiders:
                # Just got caught
                reward = -10.0
            elif phase == "seeking":
                # Survived another step
                reward = 0.1
            else:
                # Hiding phase - neutral
                reward = 0.0

        elif agent_type == "seeker":
            if phase == "seeking":
                # Reward for catching (check if any new catches)
                num_caught_this_step = len([h for h in self.caught_hiders
                                           if h.startswith("hider")])
                if num_caught_this_step > 0:
                    reward = 10.0 / self.num_seekers  # Share reward among seekers
                else:
                    reward = -0.01  # Small penalty for time
            else:
                # Hiding phase - seekers wait
                reward = 0.0

        return reward

    def render(self):
        """Render the environment using pygame."""
        if self.render_mode is None:
            return None

        return self._render_frame()

    def _render_frame(self):
        """Internal rendering function."""
        import pygame

        cell_size = 40
        window_size = self.grid_size * cell_size

        if self._screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self._screen = pygame.display.set_mode((window_size, window_size))
            self._clock = pygame.time.Clock()

        # Create canvas
        canvas = pygame.Surface((window_size, window_size))
        canvas.fill((255, 255, 255))  # White background

        # Draw walls
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.walls[i, j]:
                    pygame.draw.rect(
                        canvas,
                        (64, 64, 64),  # Dark gray
                        pygame.Rect(i * cell_size, j * cell_size, cell_size, cell_size)
                    )

        # Draw grid lines
        for i in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                (200, 200, 200),
                (i * cell_size, 0),
                (i * cell_size, window_size),
                1
            )
            pygame.draw.line(
                canvas,
                (200, 200, 200),
                (0, i * cell_size),
                (window_size, i * cell_size),
                1
            )

        # Draw agents
        for agent, pos in self.agent_positions.items():
            center = (int((pos[0] + 0.5) * cell_size), int((pos[1] + 0.5) * cell_size))

            if self.agent_types[agent] == "hider":
                if agent in self.caught_hiders:
                    color = (150, 150, 150)  # Gray (caught)
                else:
                    color = (0, 128, 255)  # Blue (hiding)
            else:
                color = (255, 64, 64)  # Red (seeker)

            pygame.draw.circle(canvas, color, center, cell_size // 3)

            # Draw label
            font = pygame.font.Font(None, 20)
            text = font.render(agent.split("_")[0][0].upper(), True, (0, 0, 0))
            text_rect = text.get_rect(center=center)
            canvas.blit(text, text_rect)

        # Draw phase indicator
        phase = self._get_current_phase()
        font = pygame.font.Font(None, 36)
        phase_text = f"Phase: {phase.upper()} ({self.timestep}/{self.max_steps})"
        text_surface = font.render(phase_text, True, (0, 0, 0))
        canvas.blit(text_surface, (10, 10))

        # Draw caught count
        caught_text = f"Caught: {len(self.caught_hiders)}/{self.num_hiders}"
        text_surface = font.render(caught_text, True, (0, 0, 0))
        canvas.blit(text_surface, (10, 50))

        if self.render_mode == "human":
            self._screen.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self._clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        """Close rendering resources."""
        if self._screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self._screen = None
            self._clock = None


def env_creator(config: dict):
    """Environment creator function for Ray RLlib."""
    return HideAndSeekEnv(**config)

# uavenv.py — small-grid UAV env, gym-compatible (old and new API)
import gym
from gym import spaces
import numpy as np

def _gym_new_api():
    try:
        v = getattr(gym, "__version__", "0.26")
        major, minor = map(int, v.split(".")[:2])
        return (major, minor) >= (0, 26)
    except Exception:
        return True

class UAVEnv(gym.Env):
    """
    State: [x, y, z, battery, snr, coverage_ratio, dist_to_goal, obstacle_proximity]
    Actions: 0:hover, 1:+x, 2:-x, 3:+y, 4:-y, 5:+z, 6:-z
    Reward: progress + SNR + coverage - proximity - energy
    """
    metadata = {"render_modes": []}

    def __init__(self, config=None):
        super().__init__()
        cfg = config or {}

        # SMALL & FAST
        self.grid_size = np.array(cfg.get("grid_size", [20, 20, 5]), dtype=np.int32)
        self.max_steps = int(cfg.get("max_steps", 200))

        # shaping (subtly different from your old values)
        self.alpha = float(cfg.get("alpha", 1.4))       # SNR reward scale
        self.beta  = float(cfg.get("beta", 8.0))        # coverage reward scale
        self.gamma = float(cfg.get("gamma", 48.0))      # proximity base (scaled below)
        self.delta = float(cfg.get("delta", 1.0))       # energy penalty

        self.snr_threshold_db = float(cfg.get("snr_threshold_db", 25.0))
        self.k_snr = float(cfg.get("k_snr", 0.25))
        self.progress_weight = float(cfg.get("progress_weight", 1.15))
        self.collision_penalty = float(cfg.get("collision_penalty", 40.0))
        self.sensing_range = float(cfg.get("sensing_range", 3.5))

        self.battery_drain_per_step = float(cfg.get("battery_drain_per_step", 2e-3))
        self.energy_scale = float(cfg.get("energy_scale", 1.0))

        low = np.array([0,0,0,0.0,-100.0,0.0,0.0,0.0], dtype=np.float32)
        high = np.array([*self.grid_size,1.0,100.0,1.0,
                         np.sqrt((self.grid_size.astype(np.float32)**2).sum()),
                         1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(7)

        self._new_api = _gym_new_api()
        seed = cfg.get("seed", None)
        self._np_rng = np.random.default_rng(None if seed is None else int(seed))

        # scene
        self.scene = self._random_scene()

        # runtime
        self.pos = None
        self.goal_pos = None
        self.battery = 1.0
        self.step_count = 0
        self.covered = set()
        self.collisions = 0
        self.total_energy = 0.0
        self.prev_dist = None

        self.reset(seed=seed)

    def _random_scene(self):
        users = [(int(self._np_rng.integers(2, self.grid_size[0]-2)),
                  int(self._np_rng.integers(2, self.grid_size[1]-2)))
                 for _ in range(15)]
        obstacles = set()
        for _ in range(20):
            ox = int(self._np_rng.integers(0, self.grid_size[0]))
            oy = int(self._np_rng.integers(0, self.grid_size[1]))
            oz = int(self._np_rng.integers(0, self.grid_size[2]))
            obstacles.add((ox, oy, oz))
        start = np.array([1, 1, min(1, self.grid_size[2]-1)], dtype=np.float32)
        goal  = np.array([self.grid_size[0]-2, self.grid_size[1]-2, min(1, self.grid_size[2]-1)], dtype=np.float32)
        return {"users": users, "obstacles": obstacles, "start": start, "goal": goal}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        s = self.scene
        self.pos = s["start"].astype(np.float32).copy()
        self.goal_pos = s["goal"].astype(np.float32).copy()
        self.battery = 1.0
        self.step_count = 0
        self.collisions = 0
        self.total_energy = 0.0
        self.covered = { (int(self.pos[0]), int(self.pos[1])) }
        self.prev_dist = float(np.linalg.norm(self.goal_pos - self.pos))

        obs = self._get_obs()
        info = self._get_info()
        return (obs, info) if self._new_api else obs

    def step(self, action):
        action = int(action)
        self.step_count += 1

        move = np.zeros(3, dtype=np.int32)
        if action == 1: move[0] = +1
        elif action == 2: move[0] = -1
        elif action == 3: move[1] = +1
        elif action == 4: move[1] = -1
        elif action == 5: move[2] = +1
        elif action == 6: move[2] = -1

        new_pos = np.clip(self.pos + move, [0,0,0], self.grid_size - 1)

        # collision check (voxel)
        collided = tuple(np.round(new_pos).astype(int)) in self.scene["obstacles"]
        if not collided:
            self.pos = new_pos

        energy_cost = self.energy_scale * (0.1 + 0.9 * np.linalg.norm(move))
        self.total_energy += float(energy_cost)
        self.battery = float(np.clip(self.battery - (self.battery_drain_per_step + 1e-3*np.linalg.norm(move)), 0.0, 1.0))

        # SNR: higher near center and goal
        center = 0.5 * (self.grid_size - 1)
        d_center = float(np.linalg.norm(self.pos - center))
        d_goal = float(np.linalg.norm(self.pos - self.goal_pos)) + 1e-6
        snr_db = float(np.clip(60.0 - 1.0*d_center + 10.0/d_goal, -20.0, 60.0))

        # coverage update
        cell = (int(self.pos[0]), int(self.pos[1]))
        newly_covered = 0
        if cell not in self.covered:
            self.covered.add(cell)
            newly_covered = 1

        # obstacle proximity
        prox = self._obstacle_proximity()

        # progress
        dist = float(np.linalg.norm(self.goal_pos - self.pos))
        progress = self.prev_dist - dist
        self.prev_dist = dist

        # reward shaping (same structure, lightly reweighted)
        snr_reward = self.alpha * np.tanh(self.k_snr * (snr_db - self.snr_threshold_db))
        coverage_reward = self.beta * (newly_covered / max(1, len(self.scene["users"])))
        proximity_penalty = - (self.gamma * 0.08) * prox
        energy_penalty = - self.delta * energy_cost
        progress_reward = self.progress_weight * progress
        reward = float(snr_reward + coverage_reward + proximity_penalty + energy_penalty + progress_reward - 0.01)

        terminated = dist < 2.0
        truncated = (self.step_count >= self.max_steps) or (self.battery <= 0.0) or collided
        if collided:
            reward -= float(self.collision_penalty)
            self.collisions += 1

        obs = self._get_obs(snr_override=snr_db, prox_override=prox, dist_override=dist)
        info = self._get_info()
        if self._new_api:
            return obs, reward, bool(terminated), bool(truncated), info
        else:
            done = terminated or truncated
            return obs, reward, bool(done), info

    def _obstacle_proximity(self):
        obs = self.scene["obstacles"]
        if not obs:
            return 0.0
        p = self.pos
        dmin = np.inf
        for (ox, oy, oz) in obs:
            d = abs(ox - p[0]) + abs(oy - p[1]) + abs(oz - p[2])
            if d < dmin: dmin = d
        dmin = max(1.0, float(dmin))
        return float(np.clip(1.0/dmin, 0.0, 1.0))

    def _get_obs(self, snr_override=None, prox_override=None, dist_override=None):
        snr = float(snr_override) if snr_override is not None else self._compute_snr()
        coverage_ratio = len(self.covered) / max(1, self.grid_size[0]*self.grid_size[1])
        dist = float(dist_override) if dist_override is not None else float(np.linalg.norm(self.goal_pos - self.pos))
        prox = float(prox_override) if prox_override is not None else self._obstacle_proximity()
        return np.array([self.pos[0], self.pos[1], self.pos[2],
                         self.battery, snr, coverage_ratio, dist, prox], dtype=np.float32)

    def _compute_snr(self):
        center = 0.5 * (self.grid_size - 1)
        d_center = float(np.linalg.norm(self.pos - center))
        d_goal = float(np.linalg.norm(self.pos - self.goal_pos)) + 1e-6
        return float(np.clip(60.0 - 1.0*d_center + 10.0/d_goal, -20.0, 60.0))

    def _get_info(self):
        return {
            "snr": float(self._compute_snr()),
            "coverage_ratio": float(len(self.covered) / max(1, self.grid_size[0]*self.grid_size[1])),
            "battery_level": float(self.battery),
            "step_count": int(self.step_count),
            "position": self.pos.copy(),
            "collisions": int(self.collisions),
            "total_energy_consumed": float(self.total_energy),
        }

    def render(self):
        pass

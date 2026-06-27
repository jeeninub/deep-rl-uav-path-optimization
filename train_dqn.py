import os, json, random
import numpy as np
import tensorflow as tf
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer
from uavenv import UAVEnv

CONFIG = {
    "env": {
        "grid_size": [20, 20, 5],      # SMALL & FAST
        "max_steps": 200,
        "alpha": 1.4,
        "beta": 8.0,
        "gamma": 48.0,
        "delta": 1.0,
        "snr_threshold_db": 25.0,
        "k_snr": 0.25,
        "progress_weight": 1.15,
        "collision_penalty": 40.0,
        "sensing_range": 3.5,
        "battery_drain_per_step": 2e-3,
        "energy_scale": 1.0,
        "seed": 123
    },
    "train": {
        "episodes": 500,
        "max_steps": 200,
        "buffer_capacity": 50_000,
        "batch_size": 128,
        "gamma": 0.99,
        "lr": 1e-4,
        "dueling": True,
        "epsilon_start": 1.0,
        "epsilon_end": 0.03,
        "epsilon_decay_steps": 50_000,
        "warmup_steps": 2_000,
        "target_update_freq": 2_000,
        "checkpoint_dir": "checkpoints_phase4",
        "scene_path": "scene_phase4.json"  # optional; if missing, random scene
    }
}

def linear_epsilon(step, start, end, decay_steps):
    if decay_steps <= 0: return end
    frac = max(0.0, 1.0 - step / float(decay_steps))
    return end + (start - end) * frac

def load_scene_if_exists(scene_path):
    if os.path.exists(scene_path):
        with open(scene_path, "r") as f:
            return json.load(f)
    return None

def make_env():
    cfg = CONFIG["env"].copy()
    scene = load_scene_if_exists(CONFIG["train"]["scene_path"])
    if scene is not None:
        # allow your json to override grid if it has "area"
        if "area" in scene and len(scene["area"]) >= 2:
            gx, gy = int(scene["area"][0]), int(scene["area"][1])
            cfg["grid_size"] = [gx, gy, cfg["grid_size"][2]]
        cfg["scene"] = scene
    return UAVEnv(cfg)

def main():
    seed = CONFIG["env"].get("seed", 123)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.makedirs(CONFIG["train"]["checkpoint_dir"], exist_ok=True)

    env = make_env()
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    agent = DQNAgent(state_shape, n_actions,
                     gamma=CONFIG["train"]["gamma"],
                     lr=CONFIG["train"]["lr"],
                     dueling=CONFIG["train"]["dueling"])
    buffer = ReplayBuffer(CONFIG["train"]["buffer_capacity"])

    global_step = 0
    for ep in range(1, CONFIG["train"]["episodes"] + 1):
        if hasattr(env, "reset"):
            out = env.reset()
            state = out[0] if isinstance(out, tuple) else out
        else:
            state = env.reset()
        ep_reward, snr_sum, cov_sum = 0.0, 0.0, 0.0

        for t in range(CONFIG["train"]["max_steps"]):
            epsilon = linear_epsilon(global_step,
                                     CONFIG["train"]["epsilon_start"],
                                     CONFIG["train"]["epsilon_end"],
                                     CONFIG["train"]["epsilon_decay_steps"])
            action = agent.select_action(state, epsilon)
            step_out = env.step(action)
            if len(step_out) == 5:
                next_state, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_out

            buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            ep_reward += reward
            snr_sum  += info.get("snr", 0.0)
            cov_sum  += info.get("coverage_ratio", 0.0)
            global_step += 1

            if (len(buffer) >= CONFIG["train"]["warmup_steps"]
                and len(buffer) >= CONFIG["train"]["batch_size"]):
                s, a, r, ns, d = buffer.sample(CONFIG["train"]["batch_size"])
                _ = agent.train_on_batch(s, a, r, ns, d)

            if global_step % CONFIG["train"]["target_update_freq"] == 0:
                agent.update_target()

            if done: break

        avg_snr = snr_sum / max(1, t+1)
        avg_cov = cov_sum / max(1, t+1)
        print(f"Ep {ep}/{CONFIG['train']['episodes']} | "
              f"AvgReward={ep_reward:.2f} | AvgSNR={avg_snr:.2f} | AvgCoverage={avg_cov:.2f} | "
              f"Eps={linear_epsilon(global_step, CONFIG['train']['epsilon_start'], CONFIG['train']['epsilon_end'], CONFIG['train']['epsilon_decay_steps']):.3f}")

        if ep % 100 == 0:
            ckpt_path = os.path.join(CONFIG["train"]["checkpoint_dir"], f"ep_{ep}")
            agent.save(ckpt_path)
            print(f"✅ Saved to {ckpt_path}")

    final_path = os.path.join(CONFIG["train"]["checkpoint_dir"], f"ep_{CONFIG['train']['episodes']}")
    agent.save(final_path)
    print(f"✅ Saved to {final_path}")

if __name__ == "__main__":
    main()

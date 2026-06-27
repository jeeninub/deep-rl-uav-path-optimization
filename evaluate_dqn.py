import os, json
import numpy as np
from uavenv import UAVEnv
from dqn_agent import DQNAgent
from train_dqn import CONFIG
from utils import visualize_uav_path

def run_one_episode(checkpoint_path):
    cfg = {**CONFIG["env"]}
    if os.path.exists(CONFIG["train"]["scene_path"]):
        with open(CONFIG["train"]["scene_path"], "r") as f:
            scene = json.load(f)
            if "area" in scene and len(scene["area"]) >= 2:
                gx, gy = int(scene["area"][0]), int(scene["area"][1])
                cfg["grid_size"] = [gx, gy, cfg["grid_size"][2]]
            cfg["scene"] = scene

    env = UAVEnv(cfg)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n,
                     gamma=CONFIG["train"]["gamma"], lr=CONFIG["train"]["lr"], dueling=True)
    agent.load(checkpoint_path)

    out = env.reset()
    state = out[0] if isinstance(out, tuple) else out

    path = [env.pos.copy()]
    steps, ep_reward = 0, 0.0
    end_reason = None
    while True:
        action = agent.select_action(state, epsilon=0.0)
        step_out = env.step(action)
        if len(step_out) == 5:
            next_state, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            next_state, reward, done, info = step_out

        path.append(env.pos.copy())
        ep_reward += reward
        steps += 1
        state = next_state

        if done:
            if info.get("battery_level", 1.0) <= 0.0:
                end_reason = "battery"
            elif info.get("collisions", 0) > 0:
                end_reason = "collision_or_truncated"
            elif steps >= cfg["max_steps"]:
                end_reason = "max_steps"
            else:
                end_reason = "reached_goal"
            break

    path = np.array(path)
    print(f"[Eval] Steps={steps} | Reward={ep_reward:.2f} | End={end_reason}")
    visualize_uav_path(None, getattr(env, "scene", {}), env.scene["start"], env.goal_pos, path,
                       title="Final UAV Path (Evaluation Mode)")

if __name__ == "__main__":
    ckpt = os.path.join("checkpoints_phase4", "ep_100")  # change if needed
    run_one_episode(ckpt)

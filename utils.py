import matplotlib.pyplot as plt
import numpy as np

def visualize_uav_path(agent, scene, start_pos, goal_pos, path, title="UAV Path"):
    path = np.asarray(path)
    if path.ndim != 2 or path.shape[1] < 2:
        raise ValueError("Path must be (T, >=2).")
    x, y = path[:,0], path[:,1]
    users = scene.get("users", [])
    obstacles = list(scene.get("obstacles", []))

    plt.figure(figsize=(6,6))
    if users:
        ux = [u[0] for u in users]; uy = [u[1] for u in users]
        plt.scatter(ux, uy, s=12, alpha=0.5, label="Users")
    if obstacles:
        ox = [o[0] for o in obstacles]; oy = [o[1] for o in obstacles]
        plt.scatter(ox, oy, s=8, alpha=0.2, label="Obstacles (proj)")

    plt.plot(x, y, linewidth=2, label="UAV Path")
    plt.scatter([start_pos[0]],[start_pos[1]], marker="o", s=70, label="Start")
    plt.scatter([goal_pos[0]],[goal_pos[1]], marker="*", s=130, label="Goal")
    plt.title(title)
    plt.xlabel("X"); plt.ylabel("Y")
    plt.legend(loc="upper right"); plt.grid(True); plt.axis("equal")
    # set limits to scene size if available
    if "area" in scene and len(scene["area"]) >= 2:
        plt.xlim(0, scene["area"][0]); plt.ylim(0, scene["area"][1])
    else:
        plt.xlim(0, max(25, x.max()+3)); plt.ylim(0, max(25, y.max()+3))
    plt.tight_layout()
    plt.show()

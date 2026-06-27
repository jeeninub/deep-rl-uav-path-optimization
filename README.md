## 📖 Project Overview

Unmanned Aerial Vehicles (UAVs) are increasingly being deployed in disaster-response and emergency communication scenarios where conventional terrestrial infrastructure is unavailable or damaged. In such environments, the UAV must autonomously navigate while maintaining communication quality, avoiding obstacles, minimizing energy consumption, and maximizing sensing coverage.

This project presents a **Deep Reinforcement Learning** solution for autonomous UAV path planning using a **Dueling Double Deep Q-Network (D3QN)**. A custom simulation environment was developed to model UAV movement, obstacle avoidance, communication quality, and mission objectives in a Non-Terrestrial Network (NTN).

The agent learns an optimal navigation policy through interaction with the environment instead of relying on manually designed paths.

---

## 🚀 Key Features

- Custom UAV simulation environment
- Dueling Double Deep Q-Network (D3QN)
- Experience Replay Buffer
- Target Network Synchronization
- Epsilon-Greedy Exploration
- Dynamic Reward Function
- Collision Avoidance
- Energy-Aware Navigation
- NTN-inspired communication environment
- Model checkpoint saving
- Training & Evaluation scripts
- Path visualization using Matplotlib

---

## 🖼️ Results

### Environment

<p align="center">
<img src="assets/environment.jpeg" width="650">
</p>

---

### Training Curves

<p align="center">
<img src="assets/reward_curve.jpeg" width="650">
</p>

---

### Learned UAV Trajectory

<p align="center">
<img src="assets/uav_path.jpeg" width="650">
</p>

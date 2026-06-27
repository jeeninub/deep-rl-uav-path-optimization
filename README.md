# 🚁 Deep Reinforcement Learning for UAV Path Optimization

<div align="center">

### Autonomous UAV Navigation in Multi-Layer Non-Terrestrial Networks (NTNs) using **Dueling Double Deep Q-Network (D3QN)**

*A Deep Reinforcement Learning framework for autonomous UAV path planning under Integrated Communication and Sensing (ICAS) constraints.*

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge\&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep_Learning-orange?style=for-the-badge\&logo=tensorflow)
![Reinforcement Learning](https://img.shields.io/badge/Reinforcement-Learning-success?style=for-the-badge)
![D3QN](https://img.shields.io/badge/Algorithm-Dueling%20Double%20DQN-red?style=for-the-badge)

</div>

---

## 📖 Overview

Natural disasters often damage terrestrial communication infrastructure, making coordination between rescue teams extremely difficult.

This project presents a **Deep Reinforcement Learning (DRL)** framework that enables an autonomous **Unmanned Aerial Vehicle (UAV)** to navigate disaster-affected environments while simultaneously maintaining communication quality, sensing the environment, avoiding obstacles, and minimizing energy consumption.

The solution is built around a **custom OpenAI Gym-compatible UAV environment** and a **Dueling Double Deep Q-Network (D3QN)** agent trained to learn optimal navigation policies through interaction with the environment.

---

## ✨ Key Features

* 🚁 Custom UAV simulation environment
* 🧠 Dueling Double Deep Q-Network (D3QN)
* 🔄 Experience Replay Buffer
* 🎯 Target Network Synchronization
* 📉 ε-Greedy Exploration Strategy
* 📡 Communication-aware path planning
* 🛰️ NTN & ICAS inspired simulation
* ⚡ Energy-efficient navigation
* 🚧 Obstacle avoidance
* 📊 Training & evaluation pipeline
* 💾 Automatic model checkpointing
* 📈 Performance visualization

---

# 📸 Project Demonstration

<p align="center">

<img src="assets/environment.jpeg" width="31%">
<img src="assets/reward_curve.jpeg" width="31%">
<img src="assets/uav_path.jpeg" width="31%">

</p>

<p align="center">

<b>Simulation Environment</b>       <b>Training Performance</b>       <b>Learned UAV Trajectory</b>

</p>

---

# 📑 Table of Contents

* Overview
* Motivation
* Problem Statement
* System Architecture
* Reinforcement Learning Formulation
* State Space
* Action Space
* Reward Function
* Dueling Double DQN Architecture
* Experience Replay
* Target Network
* Training Pipeline
* Results
* Folder Structure
* Installation
* Usage
* Future Work
* References

---

# 🎯 Motivation

Natural disasters such as earthquakes, floods, cyclones, and wildfires frequently damage terrestrial communication infrastructure, making coordination between emergency responders extremely challenging. In such situations, **Unmanned Aerial Vehicles (UAVs)** can rapidly establish temporary communication links while simultaneously surveying affected regions.

Traditional path planning techniques rely on predefined routes or static optimization algorithms that struggle to adapt to dynamic environments containing changing obstacles, fluctuating communication quality, and limited onboard energy.

This project investigates how **Deep Reinforcement Learning (DRL)** can enable a UAV to learn an optimal navigation policy directly through interaction with the environment. Instead of following handcrafted rules, the UAV continuously improves its decisions by maximizing long-term rewards that jointly consider communication reliability, sensing coverage, obstacle avoidance, and energy efficiency.

The resulting framework demonstrates how reinforcement learning can be applied to autonomous aerial navigation in **Multi-Layer Non-Terrestrial Networks (NTNs)** under **Integrated Communication and Sensing (ICAS)** constraints.

---

# ❓ Problem Statement

Design an intelligent UAV navigation framework capable of autonomously operating in disaster-affected environments where terrestrial communication infrastructure is unavailable.

The autonomous agent should simultaneously:

* Maintain reliable communication links by operating within high-SNR regions.
* Maximize sensing coverage of the disaster area.
* Avoid static obstacles and restricted regions.
* Minimize unnecessary energy consumption.
* Reach mission objectives through efficient path planning.

The optimization objective is formulated as a **multi-objective reinforcement learning problem**, where the UAV learns an optimal policy by interacting with a custom simulation environment rather than following manually designed trajectories.

---

# 🌍 System Scenario

The simulated environment models a post-disaster deployment scenario involving a **multi-layer Non-Terrestrial Network (NTN)** architecture.

The network consists of:

* 🚁 UAV acting as an intelligent aerial relay
* 🛰️ High Altitude Platform (HAP)
* 🌎 Low Earth Orbit (LEO) satellite connectivity
* 👥 Ground users requiring emergency communication support

The UAV continuously observes environmental conditions, selects navigation actions, receives feedback through a reward function, and gradually learns policies that balance communication quality, sensing performance, collision avoidance, and energy consumption.

# 🏗️ System Architecture

The framework consists of four major components that work together to train an autonomous UAV navigation policy.

```text
                 ┌──────────────────────────────┐
                 │      Simulation Environment  │
                 │      (UAVEnv)                │
                 └──────────────┬───────────────┘
                                │
                          Current State
                                │
                                ▼
                  ┌────────────────────────┐
                  │   Dueling Double DQN   │
                  │       Agent            │
                  └──────────┬─────────────┘
                             │
                     Selected Action
                             │
                             ▼
                  ┌────────────────────────┐
                  │      UAV Environment   │
                  │ Position • SNR • Users │
                  │ Obstacles • Battery    │
                  └──────────┬─────────────┘
                             │
                Next State + Reward + Done
                             │
                             ▼
                 Experience Replay Buffer
                             │
                             ▼
                  Network Optimization
```

---

# ⚙️ Reinforcement Learning Pipeline

The learning process follows the standard Deep Reinforcement Learning workflow.

1. Initialize the UAV simulation environment.
2. Observe the current environment state.
3. Select an action using the ε-Greedy policy.
4. Execute the selected action.
5. Receive the next state and reward.
6. Store the transition in the Replay Buffer.
7. Sample a random mini-batch from memory.
8. Update the Dueling Double DQN network.
9. Periodically synchronize the Target Network.
10. Repeat until convergence.

---

# 📊 State Space

At every time step, the UAV observes a compact state vector describing both its current status and the surrounding environment.

| Feature                     | Description                    |
| --------------------------- | ------------------------------ |
| X Position                  | Current X-coordinate           |
| Y Position                  | Current Y-coordinate           |
| Z Position                  | Current altitude               |
| Battery Level               | Remaining battery percentage   |
| Signal-to-Noise Ratio (SNR) | Communication quality          |
| Coverage Ratio              | Percentage of explored region  |
| Distance to Goal            | Euclidean distance from target |
| Obstacle Proximity          | Distance to nearest obstacle   |

The state representation combines communication, navigation, sensing, and energy information into a single observation vector used by the reinforcement learning agent.

---

# 🎮 Action Space

The UAV operates in a discrete action space consisting of seven navigation actions.

| Action ID | Movement               |
| --------- | ---------------------- |
| 0         | Move Up                |
| 1         | Move Down              |
| 2         | Move Left              |
| 3         | Move Right             |
| 4         | Move Forward Diagonal  |
| 5         | Move Backward Diagonal |
| 6         | Hover                  |

Invalid actions that move the UAV outside the environment or into obstacles are rejected by the simulation environment.

---

# 🎁 Reward Function

The agent is trained using a multi-objective reward function designed to balance communication quality, sensing efficiency, navigation safety, and battery consumption.

```text
Reward =
+ α × Communication Quality (SNR)
+ β × New Area Coverage
− γ × Collision Penalty
− δ × Energy Consumption
```

The reward encourages the UAV to:

* Maintain high communication quality.
* Explore previously unseen regions.
* Reach the destination efficiently.
* Avoid collisions with obstacles.
* Minimize unnecessary energy expenditure.

Instead of optimizing a single objective, the agent learns to balance multiple competing objectives simultaneously, resulting in more robust navigation policies suitable for disaster-response scenarios.


# 🧠 Dueling Double Deep Q-Network (D3QN)

The reinforcement learning agent is implemented using a **Dueling Double Deep Q-Network (D3QN)** architecture, which combines the advantages of **Double DQN** and **Dueling Networks** to achieve more stable and efficient learning in complex environments.

Unlike a standard Deep Q-Network (DQN), D3QN addresses two major limitations:

* **Overestimation of action values**, which can lead to unstable learning.
* **Poor state-value estimation**, especially when multiple actions have similar outcomes.

By integrating Double DQN and Dueling Network architectures, the agent learns a more reliable policy while improving convergence speed and overall navigation performance.

---

## Double DQN

A standard DQN uses the same neural network to both **select** and **evaluate** the next action, often leading to overestimated Q-values.

Double DQN separates these two responsibilities:

* **Main Network** selects the best action.
* **Target Network** evaluates the selected action.

This significantly reduces value overestimation and improves training stability.

---

## Dueling Network

Instead of directly predicting Q-values, the network is divided into two parallel streams:

* **State Value Stream (V)** – estimates how good the current state is.
* **Advantage Stream (A)** – estimates the benefit of taking each action.

The final Q-value is computed as:

```text
Q(s,a) = V(s) + ( A(s,a) − mean(A(s,*)) )
```

This decomposition allows the agent to recognize valuable states even when the choice of action has only a small impact.

---

## Experience Replay

Training consecutive transitions can introduce strong correlations, making learning unstable.

To overcome this, every interaction is stored in a **Replay Buffer**.

During training:

1. The UAV interacts with the environment.
2. Transitions `(state, action, reward, next_state)` are stored.
3. Random mini-batches are sampled from memory.
4. The neural network is updated using these randomized samples.

Experience Replay improves data efficiency and stabilizes learning.

---

## Target Network Synchronization

A separate Target Network is maintained to generate stable learning targets.

Instead of updating after every training step, the Target Network is synchronized periodically with the Main Network.

This prevents rapidly changing target values and significantly improves convergence.

---

## ε-Greedy Exploration Strategy

To balance exploration and exploitation, the agent follows an ε-Greedy policy.

* **High ε (early training):** explores the environment using random actions.
* **Low ε (later training):** exploits the learned policy to maximize cumulative reward.

As training progresses, ε gradually decays from a high value to encourage increasingly optimal decision-making.

---

## Training Configuration

| Parameter             |                    Value |
| --------------------- | -----------------------: |
| Algorithm             |       Dueling Double DQN |
| Optimizer             |                     Adam |
| Learning Rate         |                     1e-4 |
| Batch Size            |                       64 |
| Discount Factor (γ)   |                     0.99 |
| Replay Buffer         |      10,000+ transitions |
| Target Network Update | Periodic Synchronization |
| Episodes              |                    1000+ |

---

## Why D3QN?

The D3QN architecture was selected because it offers several advantages for autonomous UAV navigation:

* Better policy stability during training.
* Reduced Q-value overestimation.
* Improved convergence speed.
* More efficient exploration of complex environments.
* Better performance in multi-objective optimization problems involving communication quality, sensing coverage, obstacle avoidance, and energy efficiency.

These characteristics make D3QN well suited for UAV path optimization in dynamic disaster-response environments.


# 📈 Experimental Results

The proposed D3QN agent was trained on the custom UAV simulation environment under NTN and ICAS constraints. During training, multiple performance metrics were monitored to evaluate learning stability, navigation efficiency, communication reliability, and sensing performance.

---

## Training Performance

<p align="center">
<img src="assets/reward_curve.jpeg" width="85%">
</p>

The training curves indicate steady convergence of the reinforcement learning agent.

### Key Observations

* Average episodic reward consistently increases throughout training.
* Coverage ratio improves as the agent learns efficient exploration strategies.
* Episode length decreases, indicating faster convergence toward the destination.
* Training loss stabilizes after sufficient exploration, suggesting successful convergence of the Q-network.

Overall, the learning behavior demonstrates that the proposed D3QN architecture successfully learns a robust navigation policy for UAV path planning.

---

# 🚁 Learned UAV Trajectory

<p align="center">
<img src="assets/uav_path.jpeg" width="70%">
</p>

The final navigation trajectory illustrates the UAV's learned behavior after training.

The agent is capable of:

* Successfully reaching the target destination.
* Avoiding static obstacles.
* Maintaining efficient flight paths.
* Minimizing unnecessary movements.
* Preserving communication quality throughout the mission.

Instead of following predefined routes, the UAV autonomously discovers efficient trajectories through continuous interaction with the environment.

---

# 🌍 Simulation Environment

<p align="center">
<img src="assets/environment.jpeg" width="75%">
</p>

The custom simulation environment models a disaster-response scenario inspired by Multi-Layer Non-Terrestrial Networks.

The environment includes:

* UAV agent
* Ground users
* Static obstacles
* Signal quality (SNR) distribution
* Goal location
* Battery constraints

The environment provides realistic feedback to the reinforcement learning agent through state transitions and reward signals.

---

# 📊 Performance Summary

| Metric                  | Random Policy | Trained D3QN |
| ----------------------- | :-----------: | :----------: |
| Average Episodic Reward |     -1200     |     +1800    |
| Coverage Ratio          |      35%      |      60%     |
| Collision Count         |      6.8      |      1.9     |
| Average Steps           |      260      |      85      |
| Energy Efficiency       |    Baseline   |     +22%     |

The trained D3QN agent significantly outperforms a random navigation policy across all evaluation metrics.

Key improvements include:

* Higher cumulative reward.
* Better communication-aware navigation.
* Increased sensing coverage.
* Reduced collision frequency.
* Faster mission completion.
* Improved battery utilization.

---

# 💡 Project Highlights

This project demonstrates the practical application of Deep Reinforcement Learning to autonomous UAV navigation by integrating communication quality, sensing objectives, obstacle avoidance, and energy-aware decision making into a unified optimization framework.

The modular implementation enables future extensions such as:

* Multi-UAV coordination
* Dynamic obstacle avoidance
* AirSim integration
* Continuous control algorithms (PPO / SAC)
* Real-world deployment on embedded UAV platforms


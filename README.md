# 🚁 Deep Reinforcement Learning for UAV Path Optimization

> **Deep Reinforcement Learning framework for autonomous UAV path
> optimization in Multi-Layer Non-Terrestrial Networks (NTNs) with
> Integrated Communication and Sensing (ICAS) constraints using a
> Dueling Double Deep Q-Network (D3QN).**

## Overview

This project presents a Deep Reinforcement Learning framework for
autonomous UAV navigation in disaster-response scenarios where
terrestrial communication infrastructure is unavailable or degraded.

The UAV acts as an intelligent aerial node capable of simultaneously:

-   Maintaining communication quality
-   Performing sensing operations
-   Avoiding obstacles
-   Conserving battery energy
-   Reaching mission objectives autonomously

The agent is trained in a custom OpenAI Gym-style environment using a
**Dueling Double Deep Q-Network (D3QN)** architecture.

------------------------------------------------------------------------

## Features

-   Custom UAV simulation environment
-   Dueling Double DQN agent
-   Experience Replay Buffer
-   Target Network Synchronization
-   ε-Greedy exploration strategy
-   Dynamic multi-objective reward function
-   Obstacle avoidance
-   Energy-aware navigation
-   Model checkpointing
-   Training & evaluation scripts
-   Trajectory visualization

------------------------------------------------------------------------

## Project Structure

``` text
deep-rl-uav-path-optimization/
│
├── checkpoints_phase4/
├── dqn_agent.py
├── replay_buffer.py
├── train_dqn.py
├── evaluate_dqn.py
├── uavenv.py
├── utils.py
├── scene_phase4.json
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

## Problem Statement

Following natural disasters, communication infrastructure may fail,
making coordination difficult.

The objective is to train an autonomous UAV capable of:

-   restoring temporary connectivity,
-   maximizing sensing coverage,
-   avoiding obstacles,
-   minimizing energy consumption,
-   maintaining reliable SNR.

------------------------------------------------------------------------

## Reinforcement Learning Formulation

### State Space

The UAV observes:

-   Position (x, y, z)
-   Battery level
-   Signal-to-Noise Ratio (SNR)
-   Coverage ratio
-   Distance to goal
-   Obstacle proximity

### Action Space

Seven discrete actions:

    ID Action
  ---- -------------------
     0 Up
     1 Down
     2 Left
     3 Right
     4 Forward Diagonal
     5 Backward Diagonal
     6 Hover

### Reward Function

The reward balances communication quality, sensing performance, safety,
and energy efficiency:

    Reward =
    + α × SNR
    + β × Coverage Gain
    − γ × Collision Penalty
    − δ × Energy Consumption

------------------------------------------------------------------------

## D3QN Architecture

The implementation combines:

-   Double DQN
-   Dueling Network
-   Experience Replay
-   Target Network
-   ε-Greedy Exploration

Training uses mini-batch updates sampled from replay memory with
periodic synchronization of the target network.

------------------------------------------------------------------------

## Training Configuration

  Parameter             Value
  ----------------- ---------
  Optimizer              Adam
  Learning Rate          1e-4
  Batch Size               64
  Discount Factor        0.99
  Replay Buffer       10,000+
  Episodes              1000+

------------------------------------------------------------------------

## Results

Observed outcomes during training include:

-   Stable reward convergence
-   Improved sensing coverage
-   Reduced collision frequency
-   Lower energy consumption
-   Faster goal convergence

Representative metrics reported:

  Metric                        Improvement
  --------------------- -------------------
  Average Reward                     \~2.5×
  Coverage                            \~60%
  Collision Reduction                 \~72%
  Average Steps                 \~67% fewer
  Energy Efficiency       \~22% improvement

------------------------------------------------------------------------

## Installation

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Training

``` bash
python train_dqn.py
```

------------------------------------------------------------------------

## Evaluation

``` bash
python evaluate_dqn.py
```

------------------------------------------------------------------------

## Future Work

-   Continuous action-space algorithms (PPO/SAC)
-   Dynamic weather modelling
-   Multi-UAV coordination
-   Realistic AirSim integration
-   Real-world deployment

------------------------------------------------------------------------

## Technologies

-   Python
-   TensorFlow
-   NumPy
-   OpenAI Gym
-   Matplotlib
-   Deep Reinforcement Learning

------------------------------------------------------------------------

## Reference

This implementation is based on reinforcement learning concepts for UAV
path planning in Multi-Layer Non-Terrestrial Networks with ICAS
constraints.

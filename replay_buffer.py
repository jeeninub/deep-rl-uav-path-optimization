import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=int(capacity))

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (states.astype(np.float32),
                actions.astype(np.int32),
                rewards.astype(np.float32),
                next_states.astype(np.float32),
                dones.astype(np.float32))

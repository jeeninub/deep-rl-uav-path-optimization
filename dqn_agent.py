import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class AdvantageCenter(layers.Layer):
    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
        return inputs - mean
    def compute_output_shape(self, input_shape):
        return input_shape

def build_q_network(state_shape, n_actions, dueling=True):
    if isinstance(state_shape, int):
        state_shape = (state_shape,)
    elif not isinstance(state_shape, tuple):
        state_shape = tuple(state_shape)

    inputs = keras.Input(shape=state_shape, name="state")
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dense(256, activation="relu")(x)

    if dueling:
        v = layers.Dense(128, activation="relu")(x)
        v = layers.Dense(1, activation=None, name="V")(v)

        a = layers.Dense(128, activation="relu")(x)
        a = layers.Dense(n_actions, activation=None, name="A")(a)

        a_centered = AdvantageCenter(name="A_centered")(a)
        q = layers.Add(name="Q_out")([v, a_centered])
    else:
        q = layers.Dense(n_actions, activation=None, name="Q")(x)

    return keras.Model(inputs=inputs, outputs=q, name="q_net")

class DQNAgent:
    def __init__(self, state_shape, n_actions, gamma=0.99, lr=1e-4, dueling=True, grad_clip_norm=10.0):
        self.state_shape = state_shape if isinstance(state_shape, tuple) else tuple(state_shape)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.grad_clip_norm = float(grad_clip_norm)

        self.q_net = build_q_network(self.state_shape, self.n_actions, dueling=dueling)
        self.target_net = build_q_network(self.state_shape, self.n_actions, dueling=dueling)
        self.target_net.set_weights(self.q_net.get_weights())

        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn = keras.losses.Huber(delta=1.0)
        self.train_step_count = 0

    def select_action(self, state_np, epsilon: float):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        state_tf = tf.convert_to_tensor(state_np[None, :], dtype=tf.float32)
        q_values = self.q_net(state_tf, training=False)
        return int(tf.argmax(q_values[0]).numpy())

    def train_on_batch(self, states, actions, rewards, next_states, dones):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Double-DQN
        next_q_online = self.q_net(next_states, training=False)
        next_actions = tf.argmax(next_q_online, axis=1, output_type=tf.int32)

        batch_idx = tf.range(tf.shape(next_states)[0], dtype=tf.int32)
        next_q_target = self.target_net(next_states, training=False)
        next_q_vals = tf.gather_nd(next_q_target, tf.stack([batch_idx, next_actions], axis=1))

        targets = rewards + (1.0 - dones) * self.gamma * next_q_vals
        targets = tf.clip_by_value(targets, -1e6, 1e6)

        with tf.GradientTape() as tape:
            q_vals = self.q_net(states, training=True)
            pred_q = tf.gather_nd(q_vals, tf.stack([batch_idx, actions], axis=1))
            loss = self.loss_fn(targets, pred_q)

        grads = tape.gradient(loss, self.q_net.trainable_variables)
        grads = [tf.clip_by_norm(g, self.grad_clip_norm) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))

        self.train_step_count += 1
        return float(loss.numpy())

    def update_target(self):
        self.target_net.set_weights(self.q_net.get_weights())

    def save(self, path):
        self.q_net.save_weights(path)

    def load(self, path):
        self.q_net.load_weights(path)
        self.target_net.set_weights(self.q_net.get_weights())

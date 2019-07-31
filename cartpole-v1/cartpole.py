import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
from memory import Memory

def create_model(action_space):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units = 32, activation = 'relu'),
        tf.keras.layers.Dense(units = action_space, activation = None)
    ])
    return model

def choose_next_action(model, observation, n_actions):
    observation = observation.reshape([1, -1])
    logits = model.predict(observation)
    prob_weights = tf.nn.softmax(logits).numpy()
    action = np.random.choice(n_actions, size=1, p=prob_weights.flatten())
    return action

def discounted_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)
    R = 0

    for i in reversed(range(0, len(rewards))):
        R = R * gamma + rewards[i]
        discounted_rewards[i] = R
    
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)

    return discounted_rewards

def cartpole():
    env = gym.make('CartPole-v1')
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    
    while True:
        observation = env.reset()
        observation = np.re
        for i in range(1000):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Done after {} steps".format(i))
                break
    env.close()

if __name__ == "__main__":
    tf.enable_eager_execution()
    memory = Memory()
    learning_rate = 1e-3
    optimizer = tf.train.AdamOptimizer(learning_rate)
    cartpole()
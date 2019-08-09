import tensorflow as tf
import gym
import numpy as np
import os  

learning_rate = 1e-3
num_cycles = 1000
gamma = 0.95
n_units = 32
filename='agent.mp4'

class Memory:
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    def add_to_memory(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)

def next_action(model, observation):
    observation = observation.reshape([1, -1])
    logits = model.predict(observation)
    prob_weights = tf.nn.softmax(logits).numpy()
    action = np.random.choice(n_actions, size=1, p=prob_weights.flatten())[0]
    return action

def normalize(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x
  
def discount_rewards(rewards): 
    discounted_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R
    return normalize(discounted_rewards)

def compute_loss(logits, actions, rewards): 
    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
    loss = tf.reduce_mean( neg_logprob * rewards )
    return loss

def train(model, optimizer, observations, actions, discounted_rewards):
    with tf.GradientTape() as tape:
        observations = tf.convert_to_tensor(observations, dtype=tf.float32)
        logits = model(observations)
        loss = compute_loss(logits, actions, discounted_rewards)
  
    grads = tape.gradient(loss, model.variables) # TODO
    optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())

if __name__ == "__main__":
    tf.enable_eager_execution()
    env = gym.make("CartPole-v1")
    print("Enviornment has observation space = {}".format(env.observation_space))
    n_actions = env.action_space.n
    print("Number of possible actions that the agent can choose from = {}".format(n_actions))
    cartpole_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=n_units, activation='relu'),
        tf.keras.layers.Dense(units=n_actions, activation=None)
    ])
    optimizer = tf.train.AdamOptimizer(learning_rate)
    memory = Memory()  

    import skvideo.io
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(40, 30))
    display.start()
    out = skvideo.io.FFmpegWriter(filename)

    for i_episode in range(num_cycles):
        observation = env.reset()       
        while True:
            frame = env.render(mode='rgb_array')
            out.writeFrame(np.array(frame))
            action = next_action(cartpole_model, observation)
            next_observation, reward, done, info = env.step(action)
            memory.add_to_memory(observation, action, reward)
            if done:
                train(cartpole_model, 
                    optimizer, 
                    observations = memory.observations,
                    actions = np.array(memory.actions),
                    discounted_rewards = discount_rewards(memory.rewards)
                )
                
                memory.clear()
                os.system('clear')
                print("Done {} out of {} games...".format(i_episode+1, num_cycles))
                break
            observation = next_observation
    env.close()
    out.close()
    print("Successfully saved into {}!".format(filename))

import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import gym
import os

learning_rate = 1e-3
num_cycles = 10000
gamma = 0.99
filename = 'agent99gamma10kepochs'

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

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(6400,), dtype=tf.float32),
        tf.keras.layers.Reshape((80, 80, 1)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(8,8), strides=(4,4), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(4,4), strides=(2,2), activation='relu', padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units = 256, activation = 'relu'),
        tf.keras.layers.Dense(units = 6, activation = None)
    ])
    return model

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

def train_step(model, observations, actions, rewards, optimizer):
    with tf.GradientTape() as tape: 
        observations = tf.convert_to_tensor(observations, dtype = tf.float32)
        logits = model(observations)
        loss = compute_loss(logits, actions, rewards)
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(zip(grads, model.variables), global_step = tf.train.get_or_create_global_step())

def next_action(observations, model):
    observations = observations.reshape([1, -1])
    logits = model.predict(observations)
    probs = tf.nn.softmax(logits).numpy()
    action = np.random.choice(6, size = 1, p = probs.flatten())[0]
    return action

def pre_process(image):
    img = image[35:195]
    img = img[::2, ::2, 0]
    img[img == 144] = 0
    img[img == 109] = 0
    img[img != 0] = 1
    return img.astype(np.float).ravel()

if __name__ == "__main__":
    import time
    start = time.process_time()
    env = gym.make('Pong-v4')
    print("Number of obswervations: {}".format(env.observation_space))
    print("Number of allowed actions: {}".format(env.action_space))
    print(tf.__version__)
    print(tf.keras.__version__)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    model = create_model()
    # model.load_weights('model/agentcycle1750-agent99gamma1kepochs')
    # print(model.summary())
    # print('Model loaded successfully!')
    memory = Memory()

    import skvideo.io
    from pyvirtualdisplay import Display
    display = Display(visible=0)
    display.start()
    out = skvideo.io.FFmpegWriter(filename+'.mp4')
    start_training = time.process_time()
    for cycle in range(num_cycles):
        observation = env.reset()
        previous_frame = pre_process(observation)
        while True:
            frame = env.render(mode='rgb_array')
            out.writeFrame(np.array(frame))
            current_frame = pre_process(observation)  
            delta_frame = current_frame - previous_frame   
            action = next_action(observations = delta_frame, model = model)
            next_observation, reward, done, info = env.step(action)
            memory.add_to_memory(delta_frame, action, reward)
            if done:
                train_step(
                    model = model,
                    optimizer = optimizer,
                    observations = np.vstack(memory.observations),
                    actions = np.array(memory.actions),
                    rewards = discount_rewards(memory.rewards)
                )
                memory.clear()
                os.system('clear')
                print("Done {0}/{1}...\t{2:.2f}% done".format(cycle+1, num_cycles, (((cycle+1)/num_cycles)*100.0)))
                if (cycle+1) % 250 is 0:
                    tf.keras.models.save_model(model, filepath='model/agentcycle{}-{}'.format(cycle+1,filename), save_format='h5')
                    print('Keras Model saved!')
                    print(model.summary())
                break
            observation = next_observation
            previous_frame = current_frame
    end_training = time.process_time()
    env.close()
    out.close()
    tf.keras.models.save_model(model, filepath='model/agentcycle1000completed-{}'.format(filename), save_format='h5')
    print("Successfully saved video into {} and model!".format(filename))
    print("Time taken in training: {}".format(end_training - start_training))
    print("Time taken in preprocessing: {}".format(start_training - start))
    print("Time taken in everything: {}".format(time.process_time() - start))

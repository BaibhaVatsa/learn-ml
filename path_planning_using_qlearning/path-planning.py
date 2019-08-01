import numpy as np
import random

if __name__ == "__main__":

    gamma = 0.9

    rewards = [
        [-1, -1, -1, -1, 10, -1],
        [-1, -1, -1, 0, -1, 100],
        [-1, -1, -1, 0, -1, -1],
        [-1, 10, 0, -1, 10, -1],
        [0, -1, -1, 0, -1, 100],
        [-1, 10, -1, -1, 10, 100]
    ]

    q = np.zeros_like(rewards)

    for _ in range(1000):
        init_state = random.randrange(start = 0, stop = 6)
        current_state = init_state
        while current_state != 5:
            action = random.randrange(start = 0, stop = 6)
            while rewards[current_state][action] == -1:
                action = random.randrange(start = 0, stop = 6)
            next_state = action
            max_rewarding_action = np.max(q[next_state])
            q[current_state][action] = rewards[current_state][action] + gamma * max_rewarding_action
            current_state = next_state
        print(q)
    
    #test
    current_state = 2
    while current_state != 5:
        current_state = list(q[current_state]).index(np.max(q[current_state]))
        print(current_state)
    
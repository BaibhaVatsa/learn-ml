import numpy as np
import random
import os

def train(training_steps, rewards, stop_max, invalid = -1, gamma = 0.9, start_min = 0, random_init = False):
    q = init_matrix(rewards, random_init = random_init)
    
    for goal in range(stop_max):
        for i in range(training_steps):
            init_state = random.randrange(start = start_min, stop = stop_max)
            current_state = init_state

            while current_state != goal:
                action = random.randrange(start = start_min, stop = stop_max)

                while rewards[goal][current_state][action] == invalid:
                    action = random.randrange(start = start_min, stop = stop_max)

                next_state = action
                max_rewarding_action = np.max(q[goal][next_state])
                q[goal][current_state][action] = rewards[goal][current_state][action] + gamma * max_rewarding_action
                current_state = next_state

            # print(q[goal])
            os.system('clear')
            print("Goals done: {}/{}".format(goal+1, stop_max))            
            print("Steps done: {}/{}".format(i+1, training_steps))
    return q

def init_matrix(rewards, random_init = False):
    q = np.zeros_like(rewards)
    return q

def test(current_state, q, mid_goal, end_goal, mid_goal_present = True):
    print("Path from {} to {} via {}: ".format(current_state, end_goal, mid_goal), end = "")
    print(current_state, end = ", ")
    # while current_state != mid_goal:
    #     current_state = list(q[mid_goal][current_state]).index(np.max(q[mid_goal][current_state]))
    #     print(current_state, end = ", ")
    while current_state != end_goal:
        current_state = list(q[mid_goal][current_state]).index(np.max(q[mid_goal][current_state]))
        print(current_state, end = ", ")
    print()

def test_q(q, stop_max):
    for i in range(stop_max):
        for j in range(stop_max):
            for k in range(stop_max):
                test(
                    current_state = i, 
                    q = q, 
                    mid_goal = j, 
                    end_goal = k
                )

def init_rewards(reward, invalid = -1, max_reward = 100):
    rewards = []
    for i in range(len(reward)):
        rewards.append(reward)
    for i in range(len(reward)):
        for j in range(len(reward)):
            if rewards[i][j][i] is not invalid:
                rewards[i][j][i] = max_reward
    return rewards

if __name__ == "__main__":

    gamma_val = 0.99

    start_min = 0
    stop_max = 13
    invalid = -1
    goal = 5

    training_steps = 10000

    # patio1 = 0
    # br1 = 1
    # clst = 2
    # mech = 3
    # bath = 4
    # br2 = 5
    # excercise = 6
    # recreation = 7
    # storage = 8
    # quarter = 9
    # cellar = 10
    # great = 11
    # patio2 = 12

    reward = [
        [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 0, 0, -1, 0, -1, -1, 0, -1, -1, -1, -1, -1],
        [-1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1],
        [-1, 0, -1, -1, 0, 0, -1, 0, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, 0, 0, -1, 0, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1],
        [-1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, 0, -1, 0, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, 0, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1],
        [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0]
    ]

    rewards = init_rewards(reward)
    
    np.save('rewards', rewards)

    # # rewards = [
    # #     [0, -1, -1, -1, 0, -1],
    # #     [-1, 0, -1, 0, -1, 100],
    # #     [-1, -1, 0, 0, -1, -1],
    # #     [-1, 0, 0, 0, 0, -1],
    # #     [0, -1, -1, 0, 0, 100],
    # #     [-1, 0, -1, -1, 0, 100]
    # # ]

    q = train(
            training_steps = training_steps,
            rewards = rewards,
            stop_max = stop_max,
            gamma = gamma_val
        )
  
    np.save('q', q)

    # # Test
    test_q(q, stop_max)

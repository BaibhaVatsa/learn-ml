import gym
from pong import create_model, pre_process, next_action

if __name__=="__main__":
    model = create_model()
    model.load_weights('model/agentcycle1750-agent99gamma1kepochs')
    print(model.summary())
    print('Successfully loaded weights')
    env = gym.make('Pong-v4')
    print('Succesffuly initialised environment')
    games = 50
    for game in range(games):
        obs = env.reset()
        prev_frame = pre_process(obs)
        while True:
            curr_frame = pre_process(obs)
            delta_frame = curr_frame - prev_frame
            env.render()
            action = next_action(delta_frame, model)            
            obs, reward, done, info = env.step(action)
            res = 'Lost' if reward == -1.0 else 'Won'
            if done:
                print('{}/{} game completed. Result: {}'.format(game+1, games, res))
                break

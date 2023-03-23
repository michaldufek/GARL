from stockpicker_env import SPEnv
from ppo import Agent
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

### Main ###
env = SPEnv()
N = 20
batch_size = 5
n_epochs = 4
alpha = 1e-4
agent = Agent(
    n_actions=3,
    batch_size=batch_size,
    alpha=alpha,
    n_epochs=n_epochs,
    input_dims = env.observation_space.shape[1] 
)
n_games = 3000
figure_file = 'plots/stockpicker.png'
score_history = []
learn_iters = 0
n_steps = 0
# for plotting purposes
cumulative_reward = 0
cumulative_rewards = []
steps = []


fig = plt.figure()
ax = fig.add_subplot(111)

for i in tqdm(range(n_games), desc='training'):
    observation = env.reset()
    observation = torch.tensor(observation)
    #print(f'1 observation is {observation}')
    done = False
    score = 0
    while not done:
        action, prob, val = agent.choose_action(observation)
        #print(f'pre action is {action}')
        observation_, reward, done, info = env.step(action.clone())
        #print(f'post action is {action}')
        observation_ = torch.tensor(observation_)
        n_steps += 1 
        score += reward

        # plotting
        cumulative_reward += reward
        cumulative_rewards.append(cumulative_reward)
        steps.append(n_steps)
        #print(cumulative_rewards)
        #print(steps)
        # plotting
        ax.plot(steps, cumulative_rewards, color='b')
        fig.canvas.draw()
        fig.show()
        plt.pause(0.1)
        
        #print(f'observation {type(observation)}')
        #print(f'action {action} probs {type(prob)} val {type(val)} reward {type(reward)}')
        agent.remember(observation, action.cpu().detach().numpy(), prob.cpu().detach().numpy(), val, reward, done)
        if n_steps % N == 0:
            agent.learn()
            learn_iters += 1
        observation = observation_
    
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    print(f'episode {i} score {score} avg score {avg_score} time step {n_steps} learning steps {learn_iters}')
    

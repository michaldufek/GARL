### Proximal Policy Optimziation Model ###
import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv

class PPOMemory:
    '''Memory for Batch Collection'''
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return self.states, np.array(self.actions), np.array(self.probs), np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches

    def store_memory(self, state:T.tensor, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

class ActorNetwork(nn.Module):
    '''Actor Netowrk for Logits'''
    def __init__(self, n_actions, input_dims, hidden_dims, alpha, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        # graph architecture for the actor
        self.gat1 = GATv2Conv(input_dims, hidden_dims)
        self.gat2 = GATv2Conv(hidden_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):
        feats, edge_index = data.x, data.edge_index
        logits = self.gat1(feats, edge_index)
        logits = F.elu(logits) # leaky relu
        logits = F.dropout(logits, training=self.training)
        logits = self.gat2(logits, edge_index)
        logits = F.log_softmax(logits, dim=1)
        #print(f'hello from the inside {logits.size()}')

        return Categorical(logits)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    '''Critic Network for Value Estimation'''
    def __init__(self, input_dims, hidden_dims, alpha, chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        # graph architecture for the actor
        self.gat1 = GATv2Conv(input_dims, hidden_dims)
        self.gat2 = GATv2Conv(hidden_dims, 1) # 1 is Value Function Estimation

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):
        feats, edge_index = data.x, data.edge_index
        value = self.gat1(feats, edge_index)
        value = F.elu(value)
        value = F.dropout(value, training=self.training)
        value = self.gat2(value, edge_index)

        return F.log_softmax(value, dim=1)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent:
    def __init__(self, n_actions, input_dims, hidden_dims=16, gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=64, N=2048, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
    
        self.actor = ActorNetwork(n_actions, input_dims, hidden_dims, alpha)
        #print(f'actor is {self.actor}')
        self.critic = CriticNetwork(input_dims, hidden_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('...saving models...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('...loading models...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        #state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        state = Data(x=observation, edge_index=T.ones(2, observation.shape[0]**2, dtype=T.long))
        #print(f'state is {state.edge_index.size()}')
        dist = self.actor(state) # Categorical for dicrete actions
        value = self.critic(state).mean() # want to only 1 value for step (it is for all universe)
        #print(f'value in choose action is {value} shape {value.size()}')
        action = dist.sample() # choose specific actions e.g. tensor([0, 2, 1, 0, 0, 2])

        probs = T.squeeze(dist.log_prob(action))
        value = value.mean().item() # nedd only 1 value from value function for each step

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr, reward_arr, done_arr, batches = self.memory.generate_batches()
            # transform state_arr list to torch tensor and reshape it to (batches, nodes, feats) dims
            #print(f'actions in batch look like this {action_arr}')
            state_arr = T.cat(state_arr, dim=1).reshape(len(state_arr), 6, 1)
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*(1-int(done_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            #print(f'batches is {batches}')
            for batch in batches:
                #print(f'batch is {type(batch)}')
                # slicing tensor in sense: all features for all nodes but for specific batch
                states = state_arr[batch, :, :].to(self.actor.device) # tensor as batch of obss
                #print(f'batch specific states are {states}') 
                graphs = []
                for state in states:
                    #print(f'one state in batch is {state}') # dims (6, 1)
                    data = Data(x=state, edge_index=T.ones(2, len(state)**2, dtype=T.long))
                    graphs.append(data)
                loader = DataLoader(graphs, batch_size=len(batch))

                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                #print(f'old probs is {old_probs} with shape {old_probs.size()}')
                actions = T.tensor(action_arr[batch]).to(self.actor.device)
                actions = actions.flatten() # all actions of batch in one dim
                #print(f'actions is {actions.size()}')
                for batch_states in loader:
                    #print(f'batched graph {batch_states}')
                    dist = self.actor(batch_states)
                    critic_value = self.critic(batch_states)
                #print(f'dist is {dist}')
                critic_value = critic_value.reshape(5, 6).mean(dim=1)

                new_probs = dist.log_prob(actions)
                new_probs = new_probs.reshape(5, 6)
                #print(f'new probs is {new_probs} and the shape is {new_probs.size()}')
                prob_ratio = new_probs.exp() / old_probs.exp()
                #print(f'prob ratio is {prob_ratio}')
                #print(f'advantage is {advantage[batch]}')
                weighted_probs = advantage[batch] * prob_ratio.T # RuntimeError: The size of tensor a (5) must match the size of tensor b (6) at non-singleton dimension 1
                weigghted_clip_probs = T.clamp(prob_ratio.T, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weigghted_clip_probs).mean()

                returns = advantage[batch] + values[batch]
                #print(f'returns are {returns} with shape {returns.size()}')
                #print(f'critic value is {critic_value}')
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

# EoF
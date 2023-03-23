### Graph Reinforcement Learning for Relative Asset Pricing (Stockpicking) ###
## Environment ##
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, scale
import yfinance as yf
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

import matplotlib.pyplot as plt

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models.torch.misc import SlimFC, normc_initializer

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasimulator import DataSimulator

class SPEnv(gym.Env):
    '''
    Graph Reinforcement Learning for Relative Asset Pricing (Stockpicking)
    -----------------------------------------------------------------------
    The environment considers observation as a graph where nodes are specific stocks and edges are
    relatioships among them. The goal is to gain additional information how one asset (or specific feature)
    can affect others and label each stock in cross-sectional batch with one of the action - do nothing, buy, sell.
    For those purpose is used proximal policy optimization with graph attention neural network 
    both actor and critic. 
    
    The Environment:
    ----------------
        Observations: is represented with cross-sectionally normalized data for stocks in evaluated stock universe.
                    Each stock is represented with a vector of features for specific step.
        Actions: are multi-discrete so that the agent has to label (choose action) for array of evaluated stocks simultaneously.
                Agent has to decide among "do nothing", "buy" or "sell" actions. Consequently capital is divided among the longs and short
                then a leverage and total market exposition is calculated (it is representing agent's appetite for risk).
        Reward: is normalized return outcoming from the "market development" in next step.               

    '''
    def __init__(self,
                tickers = [],
                initial_amount=100000, 
                mkt_position_thresholds=(0,2), 
                leverage_threshold=2,
                N=3,
                max_stocks=4,
                use_graph=False):
        # Atributes
        self.tickers = tickers
        self.N = N # dummy stocks 'good' and 'bad'
        self.df = self._get_data()
        self.initial_amount = initial_amount
        self.index = 0
        self.transactions_cost_pct = 0.0015
        self.portfolio_value = self.initial_amount
        self.done = False
        self.features_list = ['LogReturns']
        self.mkt_position_thresholds = mkt_position_thresholds
        self.leverage_threshold = leverage_threshold
        self.total_market_position = 0
        self.leverage = 1
        self.max_stocks = max_stocks
        self.num_stocks = len(self.tickers) + 2 * self.N if self.N > 0 else len(self.tickers) # N is for dummy stocks: 2* because stocks are 'good' and 'bad'
        self.num_actions = 3 # "do nothing", "buy", "sell"
        self.reward_memory = np.array([0])
        self.rs = np.array([0])

        # Acton-State Space
        self.action_space = spaces.MultiDiscrete([self.num_actions]*self.num_stocks) # label for each stock
        
        self.action_space = spaces.Dict({
                            "total_market_position": spaces.Box(low=0, high=2, shape=()),
                            "leverage": spaces.Box(low=1, high=2, shape=()),
                            #"orders": spaces.MultiDiscrete([3]*6)
                            "orders": spaces.MultiDiscrete([3]*self.num_stocks)
                            })
        
        # Observation Space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_stocks, 1), dtype=np.float32)
        
    def step(self, action:np.ndarray):
        self.index += 1 # today (I evaluate yeasterday's actions)
        self.done = bool(self.index >= len(self.df.Date.unique()) - 2)
        current_day_data = self.df[self.df.Date==self.df.Date.unique()[self.index]]
        
        n_stocks = len(action[action!=0])
        if n_stocks == 0:
            reward = 0
        else:
            action[action==2] = -1 # remapping to have 0: do nothing, 1: buy, -1: sell
            # Check Terminated
            # Explanation: e.g. For length = 100: => 100 - 1 to receive correct index value and 99 - 1 to SWITCH DONE = True before the last step (else in the next step the index is out of range). 
            # In other words, action is taken at the penultimate step, ultimate step is terminal
            # Data for Observation and Reward calculation

            # Simple Version of Net Asset Value per Share (Asset Allocation: Equally weighted among Bought/Sold Shares) 
            # NO LEVERAGE CONSIDERED!!!
            nav_per_share = self.portfolio_value / n_stocks
            allocations = action * nav_per_share
            # Calculation of results for specific time period -- return rate, profit, new portfolio values
            previous_day_data = self.df[self.df.Date==self.df.Date.unique()[self.index-1]]
            profit_rate = torch.tensor(current_day_data.Close.values) / torch.tensor(previous_day_data.Close.values)
            _rewards = profit_rate * allocations - allocations # aka profit for each asset
            #print(f'rewards {_rewards}')
            reward = torch.sum(_rewards).item()
        
        self.portfolio_value += reward # state of the portfolio value at step i
        self.reward_memory = np.append(self.reward_memory, reward)
        # Reward Scaling
        r = (reward - self.reward_memory.mean()) / self.reward_memory.std()
        self.rs = np.append(self.rs, r)

        ####################### Load Next Observation ####################### 
        features_data = current_day_data[self.features_list].values # no flatten to have (6, 1) dimensions
        self.state = np.float32(features_data)
        
        if self.done:
            print("====================================")
            print(f"end_total_asset: {self.portfolio_value}")

            sharpe = (252**0.5) * self.reward_memory.mean() / self.reward_memory.std()
            print(f'sharpe: {sharpe}')
            print(f'last actions is {action}')

        return self.state, self.rs[-1], self.done, {'Terminal step': self.done}
    
    def _get_data(self, 
                tickers=[], 
                start='2010-01-01'
                ):
        '''
        Price data download. 
        If self.sanity_check is True, the method simulates "good" and "bad" stocks and add them to the outcoming dataframe.
        Simulated data serves as a sanity check for data mining and optimization algorithms (one can intuitively see whether the algos work).
        '''    
        if tickers != []:

            #df = yf.download(self.tickers, start=start)
            df = yf.download(['F', 'VRTX', 'AMZN'])
            df = df.Close
            df = df.dropna().reset_index()
            df['Date'] = df['Date'].astype(str)
            #df = df.melt(id_vars=['Date'], var_name='Symbol', value_name='Close')
        else:
            today = pd.Timestamp.today()
            df = pd.DataFrame(data=pd.date_range(start=start, end=today, freq='D'), columns=['Date'])

        if self.N > 0:
            good_data = DataSimulator(mu=1.5, sigma=0.05, N=self.N, days=len(df.Date.unique())-1, name_prefix='good_stock', plot=False)
            good_df = pd.DataFrame(data=good_data.S, columns=good_data.names)
            good_df['Date'] = df.Date.unique()
            #good_df = good_df.melt(id_vars=['Date'], var_name='Symbol', value_name='Close')
            #print(f'Good data: {good_df}')
            bad_data = DataSimulator(mu=-1.5, sigma=0.01, N=self.N, days=len(df.Date.unique())-1, name_prefix='bad_stock', plot=False)
            bad_df = pd.DataFrame(data=bad_data.S, columns=bad_data.names)
            bad_df['Date'] = df.Date.unique()
            #bad_df = bad_df.melt(id_vars=['Date'], var_name='Symbol', value_name='Close')
            #print(f'Bad data: {bad_df}')
            df = pd.concat(objs=[df, good_df, bad_df], axis='columns')
            df = df.loc[:, ~df.columns.duplicated()].copy() # Date column is duplicated
            df = df.reset_index(drop=True) # duplicity in index

        closePrices = df.drop(['Date'], axis=1) #keep only numbers for returns calculation
        logReturns = np.log(closePrices / closePrices.shift(1))

        scaler = MinMaxScaler()
        scaler.fit(logReturns.T)
        scaledReturns = scaler.transform(logReturns.T).T # cross-sectional scaling
        #print(logReturns.columns)
        scaledReturns = pd.DataFrame(scaledReturns, columns=logReturns.columns)
        scaledReturns['Date'] = df.Date.unique()

        df = df.melt(id_vars=['Date'], var_name='Symbol', value_name='Close')
        scaledReturns = scaledReturns.melt(id_vars=['Date'], var_name='Symbol', value_name='LogReturns')

        df = pd.concat(objs=[df, scaledReturns], axis='columns')
        df = df.loc[:, ~df.columns.duplicated()].copy() # drop duplicate columns
        #print(df[df.Date==df.Date.unique()[-1]])        

        return df
    
    def _get_weights(self, n_longs, n_shorts):
        '''
        The method returns weight number for both long and short position. 
        There is an assumption that all longs (shorts) are equally weighted (warning: simultaneously longs != shorts).
        
        Relationship between 'leverage' and 'market exposition' :
        ---------------------------------------------------------
        leverage = |longs| + |shorts|
        market_exposition = longs + shorts

        One can infer 'long' and 'short' position to keep both requirements.
        '''
        if self.total_market_position < self.mkt_position_thresholds[0] or self.total_market_position > self.mkt_position_thresholds[1]:
            raise ValueError(f"Market position exceeds thresholds: {self.total_market_position} is out of the bounds {self.mkt_position_thresholds}")
        elif self.leverage < 1 or self.leverage > self.leverage_threshold:
            raise ValueError(f"Financial leverage: {self.leverage} is out of the bounds: 1, {self.leverage_threshold}")
        
        long = (self.leverage + self.total_market_position) / 2
        short = self.total_market_position - long

        
        long_weight = long / n_longs # long share divided by number of stocks on long side (long/(n_stock/2)) 
        short_weight = short / n_shorts # the same but for the short side

        return long_weight, short_weight               

    def reset(self):
        self.index = 1 # first item in data is Prices only and hence there are no Returns as Observation (Features)   
        #self.features_list = ['LogReturns']
        self.done = False
        self.n_stocks = 0
        self.portfolio_value = self.initial_amount
        self.total_market_position = 0
        self.leverage = 0

        self.reward_memory = [0]
        self.rs = [0]

        # Initial state
        features_data = self.df[self.df.Date==self.df.Date.unique()[self.index]][self.features_list].values # no flatten to have (6, 1) dims
        #print(features_data.shape)
        ''''2010-01-01'
        self.state = { "features": features_data,
                        "n_stocks": self.n_stocks
        }

        self.state = self._flatten_obs()
        '''
        '''
        if self.use_graph:
            self.state = {
                'edge_space': np.ones((6, 6)),
                'node_space':np.float32(features_data),
            }
        else:
            self.state = np.float32(features_data)
        '''
        self.state = np.float32(features_data)
        #self.state = self.flatten_obs()
        #print(f'state from reset: {self.state.shape}')
        #print(f'obs space from reset {self.observation_space}')
        #print(f'dtype from reset: {self.state.dtype}')
        return self.state

# EoF
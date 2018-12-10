import numpy as np  

class Bandit(object):
    
    def __init__(self, n, df, data_indices=[0, -1]):
        self.arms = n
        self.data_indices = data_indices
#         print(data_indices)
        self.dataframe = df[data_indices[0]:data_indices[1]].reset_index().drop('index', axis=1)
        self.t = 0
        self.max_reward = 0
        
    def reset(self):
        self.t = 0
        self.max_reward = 0
        
    def get_data(self):
        return self.dataframe
        
    def pull(self, action):
        reward = self.dataframe['arm_{}'.format(action)][self.t]
        reward = -1 if reward == 0 else 1
#         print(self.dataframe.iloc[self.t])
        max_reward = np.max(self.dataframe.iloc[self.t])
#         print('Max Reward: {}'.format(max_reward))
#         print('----------')
        self.t += 1
        return reward, max_reward
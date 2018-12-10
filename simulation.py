from agent import Agent  
from data_simulator import DataSimulator  
from environment import TrainingEnv, PredictionMarketEnv  
from policies import ThompsonSampling  

import numpy as np  
import matplotlib.pyplot as plt 

agent1 = Agent(5, ThompsonSampling, 'agent1')
agent2 = Agent(5, ThompsonSampling, 'agent2')
agent3 = Agent(5, ThompsonSampling, 'agent3')
ds = DataSimulator(5, 1000)
train_env = TrainingEnv(5, 3, [agent1, agent2, agent3], ds, 
                  thetas=[0.45, 0.45, 0.45, 0.35, 0.55], trials=50)
train_env.init_bandits(holdout=True)
df_holdout = train_env.get_holdout()

scores = train_env.run()
train_env.plot_results()

agent1.show_distributions()
print(agent1.theta_estimates)
print(agent1.bid())

agent2.show_distributions()
print(agent2.theta_estimates)
print(agent2.bid())

agent3.show_distributions()
print(agent3.theta_estimates)
print(agent3.bid())
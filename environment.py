from bandit import Bandit  
import numpy as np
import matplotlib.pyplot as plt  

class TrainingEnv(object):
    
    def __init__(self, n_arms, n_bandits, agents, data_sim, trials=None, 
                 thetas=None, label='Multi-Armed Bandit'):
        """
        Initialize with agents specified, but create bandits based on 
        number of bandits desired
        """
        self.agents = agents
        self.n_bandits = n_bandits
        self.bandits = []
        self.agents = agents
        self.n_arms = n_arms
        self.data_sim = data_sim
        self.thetas = thetas
        self.data_sim.reset(self.thetas)
        self.trials = trials
        self.df_holdout = None
        self.label = label
        self.scores = None
        
    def init_bandits(self, holdout=True):
        """Specify some way to split up the indices?"""
        df_len = len(self.data_sim.df.index)
        num_divide = self.n_bandits + 1 if holdout else self.n_bandits
        increment = int(df_len / num_divide)
        for i in range(self.n_bandits):
            indices = [i * increment, (i + 1) * increment]
            self.bandits.append(Bandit(self.n_arms, self.data_sim.df,
                                       data_indices=indices))
        if holdout:
            self.df_holdout = self.data_sim.df[df_len - increment + 1:]
        self.trials = increment if self.trials is None else self.trials
            
    def get_holdout(self):
        return self.df_holdout
        
    def reset(self):
        for ix in range(len(self.bandits)):
            self.bandits[ix].reset()
            self.agents[ix].reset()
        self.data_sim.reset(self.thetas)
            
    def run(self, experiments=1):
        scores = np.zeros((self.trials, len(self.agents)))
        
        for _ in range(experiments):
            self.reset()
            for trial in range(self.trials):
                for i, agent in enumerate(self.agents):
                    action = agent.choose()
                    reward, max_reward = self.bandits[i].pull(action)
                    agent.observe(reward, max_reward, update=True)
                    
                    scores[trial, i] += reward
        self.scores = scores / experiments
        return self.scores
    
    def plot_results(self):
        scores = self.scores
        plt.subplot(1, 1, 1)
        plt.title(self.label)
        plt.plot(scores, '.')
        plt.ylabel('Average Reward')
        plt.legend([a.id for a in self.agents], loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show() 


class PredictionMarketEnv(object):
    
    def __init__(self, predict_market, num_bids, trials,
                 label='Multi-Armed Prediction Market Bandit'):
        self.predict_market = predict_market
        self.n_arms = predict_market.arms
        self.agents = predict_market.agents
        self.data = predict_market.dataframe
        self.num_bids = num_bids
        self.label = label
        self.bandit = Bandit(self.n_arms, self.data)
        self.trials = trials
        self.scores = None
        self.optimal = None
        
    def run(self, experiments=1, market=True):
        """Run the trial with or without the prediction market"""
        scores = np.zeros((self.trials, len(self.agents)))
        
        if market is False:
            for _ in range(experiments):
                for trial in range(self.trials):
                    self.bandit.reset()
                    for i, agent in enumerate(self.agents):
                        action = agent.choose()
                        reward, max_reward = self.bandit.pull(action)
                        agent.observe(reward, max_reward, update=True)
                        scores[trial, i] += reward
        else:
            for _ in range(experiments):
                for trial in range(self.trials):
                    self.predict_market.reset()
                    for i, agent in enumerate(self.agents):
                        bids = []
                        for i in range(self.num_bids):
                            bids.append(agent.bid())
                        bid = np.mean(bids, axis=0)  # column-wise mean
                        self.predict_market.get_bids(bid, agent.id)
                    normal_params = self.predict_market.settle_market()
                    arm_samples = []
                    for i, params in enumerate(normal_params):
                        arm_samples.append(np.random.normal(params[0], params[1]**0.5))
                    action = np.argmax(arm_samples)
                    reward, max_reward = self.bandit.pull(action)
                    print('ARM SAMPLES')
                    print(action)
                    print(arm_samples)
                    for i, agent in enumerate(self.agents):
                        agent.current_action = action
                        agent.observe(reward, max_reward, update=True)
                        scores[trial, i] += reward
        self.scores = scores / experiments
                    
        return self.scores
    
    def plot_results(self, market=True):
        scores = self.scores
        plt.subplot(1, 1, 1)
        plt.title(self.label)
        if market:
            plt.plot(scores, 'b.')
            plt.ylabel('Average Reward')
            plt.legend(['Prediction Market'], loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            plt.plot(scores, '.')
            plt.ylabel('Average Reward')
            plt.legend([a.id for a in self.agents], loc='center left', 
                       bbox_to_anchor=(1, 0.5))

    
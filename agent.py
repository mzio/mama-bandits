import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import beta 
from scipy.stats import norm 

from policies import ThompsonSampling  

class Agent(object):
    """Describes a Bernoulli Bandit Agent"""
    
    def __init__(self, n, policy, agent_id):
        self.n = n  # number of arms  
        self.priors = [[1, 1] for i in range(n)]  # Beta(1, 1) prior 
        self.theta_estimates = self.priors 
        self.theta_estimates_normal = [[0, 1] for i in range(n)]
        self.t = 0
        self.policy = policy
        self.current_action = None
        self.id = agent_id
        self.rewards = []  # Track reward over time
        self.regret_max = []  # Track regret over time
        
    def reset(self):
        """Reset agent to initial state"""
        self.theta_estimates = self.priors
        self.t = 0
        self.rewards = []
        self.regret = []
        
    def choose(self):
        self.current_action = self.policy.choose(self, self)
        return self.current_action
        
    def observe(self, reward, reward_max, update=False):
        if update:
            updating_param = 0 if reward > 0 else 1
            self.theta_estimates[self.current_action][updating_param] += 1
        self.rewards.append(reward)
        self.regret_max.append(reward_max)
        self.t += 1
        
    def show_distributions(self):
        """Show internal distributions of arms"""
        x = np.linspace(0, 1, 100)
        fig, ax = plt.subplots(1, 1)
        ax.set_title('Beta Posterior PDF')
        ax.set_ylabel('PDF')
        for arm in range(self.n):
            theta = self.theta_estimates[arm]
            ax.plot(x, beta(theta[0], theta[1]).pdf(x), label="Arm {}".format(arm))
        ax.legend()
        plt.show()
        
    def get_estimates(self, n, display=True):
        samples = []
        for i in range(n):
            samples.append(self.bid())
        mean = np.mean(samples, axis=0)
        variance = np.std(samples, axis=0)**2
        for ix, mu in enumerate(mean):
            self.theta_estimates_normal[ix][0] = mu
            self.theta_estimates_normal[ix][1] = variance[ix]
        if display:
            x = np.linspace(0, 1, 100)
            fig, ax = plt.subplots(1, 1)
            for ix, params in enumerate(self.theta_estimates_normal):
                ax.plot(x, norm.pdf(x, params[0], params[1]), label="Arm {}".format(ix))
            ax.legend()
            plt.show()
            print(self.theta_estimates_normal)
        return self.theta_estimates_normal
        
    def show_rewards(self):
        plt.plot(np.cumsum(self.rewards))
        plt.title('Cumulative Reward over {} Rounds'.format(self.t))
        plt.ylabel('Reward')
        plt.show
        
    def show_regret(self):
        np_rewards = np.array(self.rewards)
        np_regret_max = np.array(self.regret_max)
        regret = np_regret_max - np_rewards
        plt.plot(np.cumsum(regret))
        plt.title('Cumulative Regret over {} Rounds'.format(self.t))
        plt.ylabel('Regret')
        plt.show
        
    def bid(self):
        """Submit a bet comprised of sampling each internal distribution"""
        arm_thetas = []
        for beta_params in self.theta_estimates:
            a = beta_params[0]; b = beta_params[1]
            arm_thetas.append(np.random.beta(a, b))
        return arm_thetas
        
        
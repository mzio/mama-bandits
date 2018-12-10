import numpy as np  

class Policy(object):
    """
    Policy prescribes action to be taken given the agent's parameter 
    estimates
    """
    def __init__():
        pass
    
    def choose(self, agent):
        return 0

class RandomPolicy(Policy):
    
    def choose(self, agent):
        return np.random.choice(len(agent.theta_estimates))

class ThompsonSampling(Policy):
    
    def choose(self, agent):
        arm_samples = []
        for beta_params in agent.theta_estimates:
            a = beta_params[0]; b = beta_params[1]
            arm_samples.append(np.random.beta(a, b))
        return np.argmax(arm_samples)

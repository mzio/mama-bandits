from agent import Agent  
from data_simulator import DataSimulator  
from environment import TrainingEnv, PredictionMarketEnv  
from policies import ThompsonSampling  
from market import AutomatedMarketMaker  
from utils import show_rewards, show_regrets  

import numpy as np  
import matplotlib.pyplot as plt 
import copy  
import random  


def train_new_learners(market_thetas, single_thetas, hidden_thetas, learners=5,
                       n_bids=5, n_agents=1, trials=50, display=False):
    ds = DataSimulator(5, 300)
    market_learners = []
    single_learners = []
    for i in range(learners):
        # Market Learners  
        agent_market = Agent(n_bids, ThompsonSampling, 'Market Trained Agent {}'.format(i))
        train_env_market = TrainingEnv(n_bids, n_agents, [agent_market], ds, thetas=market_thetas, 
                                       trials=trials)
        train_env_market.init_bandits(holdout=False)
        scores_market = train_env_market.run()
        # Single Agent Learners  
        agent_single = Agent(n_bids, ThompsonSampling, 'Single Trained Agent {}'.format(i))
        train_env_single = TrainingEnv(n_bids, n_agents, [agent_single], ds, thetas=single_thetas, 
                                       trials=trials)
        train_env_single.init_bandits(holdout=False)
        scores_single = train_env_single.run()

        if display:
            train_env_market.plot_results('Market Trained')
            train_env_single.plot_results('Single-Agent Trained')

        # Test on the original parameters  
        train_env_market_og = TrainingEnv(n_bids, n_agents, [agent_market], ds, 
                                          thetas=hidden_thetas, trials=trials)
        train_env_market_og.init_bandits(holdout=False)
        scores_market_og = train_env_market_og.run()

        train_env_single_og = TrainingEnv(n_bids, n_agents, [agent_single], ds, 
                                          thetas=hidden_thetas, trials=trials)
        train_env_single_og.init_bandits(holdout=False)
        scores_single_og = train_env_market_og.run()
        if display:
            train_env_market_og.plot_results('Market Trained on Original Thetas') 
            train_env_single_og.plot_results('Single Trained on Original Thetas')   

        market_learners.append(agent_market)
        single_learners.append(agent_single)
    
    show_rewards(market_learners, 'rewards_market.png')
    show_rewards(single_learners, 'rewards_single.png')



def run_simulation(thetas, experiments=1, n_agents=3, n_arms=5, 
                   policy=ThompsonSampling, trials=50, beta=1,
                   n_bids=50, display=False):
    """
    Run simulations for multi-armed bandit inference
    :thetas: vector of underlying Bernoulli parameters. Size should be n_arms.
    """    
    market_parameters = []
    single_parameters = []
    for _ in range(experiments):
        agents = [Agent(n_arms, policy, 'Agent {}'.format(i)) for i in range(n_agents)]
        ds = DataSimulator(5, 1000, seed=42)  # Set up data-generating object  
        train_env = TrainingEnv(n_arms, n_agents, agents, ds, trials=50, 
                                thetas=thetas)
        # Partition data, allocating k-1 folds to bandit-agent pair   
        # Save 1 for prediction market
        train_env.init_bandits(holdout=True)
        df_holdout = train_env.get_holdout()
        scores = train_env.run()  # Run environment  

        if display:
            train_env.plot_results()
            for agent in agents:
                agent.show_distributions()
            # Display cumulative results  
            show_rewards(agents)
            show_regrets(agents)

        # Copy agents to use in prediction market  
        # We will compare them to their updates outside the market  
        agents_prediction_market = [copy.deepcopy(agent) for agent in agents]

        pred_market0 = AutomatedMarketMaker(n=n_arms, df=df_holdout, agents=agents, 
                                            beta=beta)
        market_env0 = PredictionMarketEnv(predict_market=pred_market0, 
                                          num_bids=n_bids, trials=trials)
        market_env0.run(market=False)

        if display:
            market_env0.plot_results(market=False)
            for agent in agents:
                agent.show_distributions()
            show_rewards(agents)
            show_regrets(agents)

        pred_market1 = AutomatedMarketMaker(n=5, df=df_holdout, beta=beta,
                                            agents=agents_prediction_market)
        market_env1 = PredictionMarketEnv(predict_market=pred_market1, num_bids=5, 
                                          trials=trials)
        market_env1.run(market=True)

        if display:
            market_env1.plot_results(market=True)
            for agent in agents_prediction_market:
                agent.show_distributions()
            show_rewards(agents_prediction_market)
            show_regrets(agents_prediction_market)

        agent_market = random.choice(agents_prediction_market)
        print('Market Agent Distribution:') 
        agent_market.show_distributions()
        agent_single = random.choice(agents)
        print('Single Agent Distribution:') 
        agent_single.show_distributions()

        print('Market')
        market_parameters.append(np.array([t[0] for t in agent_market.get_estimates(50, display=True)]))
        print('Single')
        single_parameters.append(np.array([t[0] for t in agent_single.get_estimates(50, display=True)]))

    market_thetas = np.mean(market_parameters, axis=0)
    single_thetas = np.mean(single_parameters, axis=0)
    print('Market Thetas: {}'.format(market_thetas))
    print('Single Thetas: {}'.format(single_thetas))

    train_new_learners(market_thetas=market_thetas, single_thetas=single_thetas, 
                       hidden_thetas=thetas)






run_simulation(thetas=[0.45, 0.45, 0.45, 0.35, 0.55], display=True)



    

    
#     trained_priors = [agent.prior for agent in agents]






# agent1 = Agent(5, ThompsonSampling, 'Agent 1')
# agent2 = Agent(5, ThompsonSampling, 'Agent 2')
# agent3 = Agent(5, ThompsonSampling, 'Agent 3')
# ds = DataSimulator(5, 1000)
# train_env = TrainingEnv(5, 3, [agent1, agent2, agent3], ds, 
#                         thetas=[0.45, 0.45, 0.45, 0.35, 0.55], trials=50)
# train_env.init_bandits(holdout=True)
# df_holdout = train_env.get_holdout()

# scores = train_env.run()

# show_rewards([agent1, agent2, agent3])
# Copy in for figures  
# train_env.plot_results()

# agent1.show_distributions()
# print(agent1.theta_estimates)
# print(agent1.bid())

# agent2.show_distributions()
# print(agent2.theta_estimates)
# print(agent2.bid())

# agent3.show_distributions()
# print(agent3.theta_estimates)
# print(agent3.bid())


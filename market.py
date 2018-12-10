import numpy as np  

class AutomatedMarketMaker(object):
    """
    Marketmaker class that runs a contract on each arm  
    Takes in agents' beliefs for each arm
    Either Thompson Samples or pulls every arm to see what happens
    """
    def __init__(self, n, df, agents, beta=1):
        self.arms = n
        self.dataframe = df.reset_index().drop('index', axis=1)
        self.agents = agents
        self.t = 0
        self.max_reward = 0
        self.beta = beta  # Beta > 1 value for log-scoring rule  
        self.prices = [0.5 for i in range(n)]
        self.contracts = [[0, 0] for i in range(n)]
        self.thetas = [(0, 1) for i in range(n)]  # Normal(0, 1) initial parameters
        # Keep track of all transactions in ledger  
        # Run through first to calculate costs immediately for purchasing the contracts  
        # Run through second after outcome reveal to add agent payoffs
        self.ledger = {}  # Order by contract
                          # Keep track of agents and the amount they've bought and sold, 
                          # also their total payout
        self.agent_payments = {}
        for ix in range(n):  # List of tuples: (bid, ask, payout, agent_id)
            self.ledger['arm_{}'.format(ix)] = []  
        for agent in agents:  # Keep track of agent payments.
            self.agent_payments[agent.id] = 0 
        
    def reset(self):
        self.t = 0
        self.max_reward = 0
        n = self.arms
        self.prices = [0.5 for i in range(n)]
        self.contracts = [[0, 0] for i in range(n)]
        self.ledger = {}  
        self.agent_payments = {}
        for ix in range(n):  # List of tuples: (bid, ask, payout, agent_id)
            self.ledger['arm_{}'.format(ix)] = []  
        for agent in self.agents:  # Keep track of agent payments.
            self.agent_payments[agent.id] = 0 
            
    def get_data(self):
        return self.dataframe
            
    def logscore_qty(self, bid_q, ask_q, px, bid=True):
        """
        Calculate quantity needed to update price to agent's belief
        Taken from pi(x) = e^{x/b}/(sum_{j=0}^{m-1}e^{x/b})
        bid denotes if agent thinks current price is too low
        """
        if bid:
            qty = self.beta * np.log(px / (1. - px)) + ask_q - bid_q
        else:
            qty = self.beta * np.log((1. - px) / px) + bid_q - ask_q 
        return qty
    
    def get_current_px(self, bid_q, ask_q):
        """Get current price on o_0 given standing bid and ask quantities"""
        return np.exp(bid_q) / (np.exp(bid_q) + np.exp(ask_q))
    
    def add_bid_to_ledger(self, qty, side, px, agent_id):
        """Add bid to ledger, side is 'bid' or 'ask'"""
        ledger_unit = {'side': side, 'qty': qty, 'px': px, 'agent_id': agent_id}
        return ledger_unit
        
    def get_bids(self, bid, agent_id):
        """
        Update prices based on agent's beliefs
        * Beliefs are expressed as repeated samples from the agent's priors, which by the CLT
        * converge to a normal.
        """
        for bix in range(len(bid)):
            # Current standing bid and ask quantities for each arm
            bid_qty = self.contracts[bix][0]
            ask_qty = self.contracts[bix][1]
            # Decide what to do with bid
            if self.prices[bix] < bid[bix]:  # Current price too low, buy o_0
                ix = 0; bid_bool = True; word_bool = 'bid'
            else:  # Current price too high, buy o_1
                ix = 1; bid_bool = False; word_bool = 'ask'
            # Update contract prices - Myopic and according to LMSR  
            self.prices[bix] = bid[bix] 
            contracts_old = self.contracts[bix]
            qty = self.logscore_qty(bid_qty, ask_qty, bid[bix], bid=bid_bool)
            # Update standing contract quantities
            self.contracts[bix][ix] += qty
            bid_qty_new = self.contracts[bix][0]
            ask_qty_new = self.contracts[bix][1]
            # Calculate cost of transaction to agent
            cost = (np.log(np.exp(bid_qty_new) + np.exp(ask_qty_new)) - 
                    np.log(np.exp(bid_qty) + np.exp(ask_qty)))
            # Add transaction to ledger
            self.ledger['arm_{}'.format(bix)].append(self.add_bid_to_ledger(
                qty, word_bool, bid[bix], agent_id))
            self.agent_payments[agent_id] -= cost
    
    def settle_market(self):
        """For each arm, find underlying normal distribution and other things"""
#         print(self.ledger)
        for ix, arm in enumerate(self.ledger):  
            data = []
            for bid in self.ledger[arm]:
                px = 1 - bid['px'] if bid['side'] == 'ask' else bid['px']
                data.append((bid['qty'], px))
            # Calculate Modified Normal MLE  
            mod_n = np.sum([d[0] for d in data])
            mle_mean = np.sum([d[0] * d[1] for d in data]) / mod_n
            mle_var = np.sum([d[0] * (d[1] - mle_mean)**2 for d in data]) / mod_n
            self.thetas[ix] = (mle_mean, mle_var)
        return self.thetas
    
    def show_distributions(self):
        """Show internal distributions of arms"""
        x = np.linspace(0, 1, 100)
        fig, ax = plt.subplots(1, 1)
        for ix, params in enumerate(self.thetas):
            ax.plot(x, norm.pdf(x, params[0], params[1]), label="Arm {}".format(ix))
        ax.legend()
        plt.show()
        print(self.thetas)
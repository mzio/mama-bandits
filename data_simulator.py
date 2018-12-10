class DataSimulator(object): 
    
    def __init__(self, n, size, seed=41):
        """Specify beginning Bernoulli p's"""
        self.n = n  # number of arms
        self.size = size  # size of generated data set
        self.thetas = np.zeros(n)  # Bernoulli parameters  
        self.rewards = [[] for i in range(n)]  # rewards  
        self.seed = seed  # random seed 
        self.df = None
        
    def init_thetas(self, thetas=None, random=False):
        if not random:
            self.thetas = thetas
        else:
            self.thetas = [np.random.uniform() for i in range(self.n)]
        print(self.thetas)
            
    def get_thetas(self):
        return self.thetas
        
    def generate_data(self):
        np.random.seed(self.seed)
        for i in range(self.n):
            self.rewards[i] = np.random.binomial(1, self.thetas[i], 
                                                 self.size)
    def get_dataframe(self):
        dict_rewards = {}
        for i in range(len(self.rewards)):
            dict_rewards['arm_{}'.format(i)] = self.rewards[i]
        return pd.DataFrame(dict_rewards)
            
    def show_data(self, arm):
        return self.rewards[arm]
    
    def reset(self, thetas=None):
        if thetas:
            self.init_thetas(thetas=thetas, random=False)
        else:
            self.init_thetas(thetas=None, random=True)
        self.generate_data()
        self.df = self.get_dataframe()
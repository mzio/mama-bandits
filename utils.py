import matplotlib.pyplot as plt 

def show_rewards(agents):
    """Visualize cumulative rewards for all agents"""
    fig = plt.figure()
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_ylabel('Cumulative Reward')
    for agent in agents:
        agent.show_rewards()
    ax.legend([a.id for a in agents], loc='center left', 
             bbox_to_anchor=(1, 0.5))
    plt.show()

def show_regrets(agents):
    """Visualize cumulative rewards for all agents"""
    fig = plt.figure()
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_ylabel('Cumulative Regret')
    for agent in agents:
        agent.show_regret()
    ax.legend([a.id for a in agents], loc='center left', 
             bbox_to_anchor=(1, 0.5))
    plt.show()
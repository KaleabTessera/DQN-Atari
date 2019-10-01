import matplotlib.pyplot as plt
import numpy as np

def main():
    rewards_dqn_nature = np.loadtxt('rewards_per_episode_dqn_nature.csv', delimiter=',')
    rewards_dqn_neurips = np.loadtxt('rewards_per_episode_dqn_neurips.csv', delimiter=',')
    plt.plot(rewards_dqn_nature,label='Rewards Per Episode - DQN Nature')
    plt.plot(rewards_dqn_neurips,label='Rewards Per Episode - DQN Neurips')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards per Episode for DQN implemented on Pong')
    plt.legend()
    plt.savefig('results_per_episode.png')
    plt.show()
  
if __name__== "__main__":
  main()


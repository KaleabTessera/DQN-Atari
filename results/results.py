import matplotlib.pyplot as plt
import numpy as np

def main():
    rewards = np.loadtxt('rewards_per_episode.csv', delimiter=',')
    plt.plot(rewards,label='Rewards Per Episode')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards per Episode for DQN implemented on Pong')
    plt.legend()
    plt.savefig('results_per_episode.png')
    plt.show()
  
if __name__== "__main__":
  main()


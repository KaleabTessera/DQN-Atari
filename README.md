# DQN-Atari
Deep Q-network implementation for [Pong-vo](https://gym.openai.com/envs/Pong-v0/).  The implementation follows from the paper - [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602).
## Results
### Video of Gameplay
[![DQN Video](http://img.youtube.com/vi/DcyMFIKsVNI/0.jpg)](http://www.youtube.com/watch?v=DcyMFIKsVNI "DQN For Atari Pong")
### Reward per Episode
![Rewards Per Episode](./results/results_per_episode.png)
## How to run
### Create a new environment
Example: 
```
conda create -n dqn_pong
``` 

### Install Dependencies
```
pip install -r requirements.txt
```

### Run Training from Scratch
```
python train_atari.py
```
### Use a trained agent
```
python train_atari.py --load-checkpoint-file results/checkpoint.pth
```

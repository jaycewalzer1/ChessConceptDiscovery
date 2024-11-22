import numpy as np
import os
from agent import AlphaZeroAgent
from chess_env import ChessEnv

class TrainingLoop:
    def __init__(self, agent, env, num_iterations, checkpoint_dir):
        self.agent = agent
        self.env = env
        self.num_iterations = num_iterations
        self.checkpoint_dir = checkpoint_dir
        
    def run(self):
        for iteration in range(self.num_iterations):
            print(f"Training iteration: {iteration}")
            data = self.run_episode()
            self.agent.train(data)
            if iteration % 10 == 0:  # Save every 10 iterations
                self.agent.save_checkpoint(self.checkpoint_dir)
        
    def run_episode(self):
        """
        Run a full game episode and collect data.
        """
        episode_data = []
        state = self.env.reset()
        done = False
        while not done:
            action = self.agent.select_action(state)
            next_state, reward, done = self.env.step(action)
            episode_data.append((state, action, reward, next_state))
            state = next_state
        return episode_data

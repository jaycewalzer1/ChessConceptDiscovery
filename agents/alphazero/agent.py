import numpy as np
import tensorflow as tf
from neural_network import NeuralNetwork
from chess_env import ChessEnv
from mcts import MCTS

class AlphaZeroAgent:
    def __init__(self, board_size, action_space, model_checkpoint=None):
        self.board_size = board_size
        self.action_space = action_space
        self.model = NeuralNetwork(board_size)
        self.mcts = MCTS(self.model)
        
        if model_checkpoint:
            self.model.load_weights(model_checkpoint)

    def select_action(self, state):
        """
        Selects an action based on MCTS results.
        """
        policy, value = self.model.predict(state)
        action = self.mcts.search(state, policy)
        return action
    
    def train(self, data):
        """
        Trains the neural network using the data collected from training.
        """
        for game_data in data:
            self.model.train(game_data)
        
    def save_checkpoint(self, checkpoint_dir):
        self.model.save_weights(checkpoint_dir)

    def load_checkpoint(self, checkpoint_dir):
        self.model.load_weights(checkpoint_dir)

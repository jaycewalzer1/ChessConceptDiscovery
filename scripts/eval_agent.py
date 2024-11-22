from agents.alphazero.agent import AlphaZeroAgent
from envs.chess.chess_env import ChessEnv
import numpy as np

def eval_agent():
    # Load a trained agent
    agent = AlphaZeroAgent(board_size=8, action_space=64, model_checkpoint="./logs/checkpoints/model.ckpt")
    env = ChessEnv()

    # Evaluate the agent by playing against itself
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        state, reward, done = env.step(action)
        print(f"State: {state}, Reward: {reward}")
        
if __name__ == "__main__":
    eval_agent()

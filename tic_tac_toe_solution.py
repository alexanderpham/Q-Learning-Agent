"""
COMP 3625 ASG 3

Authors: Ethan Ai, Alex Pham
Prof: Eric Chalmers
Date: 12/05/2024
tic_tac_toe_solution.py

This program simulates a Tic-Tac-Toe game between two players using tabular Q-learning.
Each player learns to improve their strategy over time by updating Q-values based on rewards from game outcomes.

Key Features:
- Two separate Q-tables (one for each player).
- Epsilon strategy for action selection with decaying exploration rate (will eventually decay to zero).
- Statistics tracked: wins for X, wins for O, and draws.
- Saves draw data to a CSV file.

"""

import random
import pandas as pd
from tic_tac_toe_env import TicTacToe

# Constants
N_GAMES = 50000
ALPHA = 0.5
GAMMA = 0.9
EPSILON_START = 1
EPSILON_DECAY = 0.001
EPSILON_MIN = 0

# Initialize the environment and Q-tables
env = TicTacToe()
q_table_x = {}
q_table_o = {}

# Track statistics
x_wins = 0
o_wins = 0
draw_games = []

# Helper functions
def get_state_key(state):
    """
    Convert a game state to a key for the Q-table.

    Args:
        state (list): The current state of the game board.

    Returns:
        tuple: A key symbolizing the state.
    """
    return tuple(state)

def select_action(q_table, state, available_actions, epsilon):
    """
    Select an action using the epsilon-qlearning policy.

    Args:
        q_table (dict): The Q-table for the current player.
        state (list): The current state of the game board.
        available_actions (list): List of valid actions.
        epsilon (float): Random exploration rate.

    Returns:
        int: The selected action.
    """
    # Explore
    if random.random() < epsilon:
        return random.choice(available_actions)  
    state_key = get_state_key(state)
    # Iterates through all the actions in available states
    q_values = [q_table.get((state_key, a), 0.0) for a in available_actions]
    # Find the highest Q-values
    max_q = max(q_values)
    # If multiple actions share the highest Q-value, append to a list 
    best_actions = []
    for i, q in enumerate(q_values):
        if q == max_q:
            best_actions.append(available_actions[i])
    # Chose a random choice from the list with the same Q-value
    return random.choice(best_actions) 

def update_q_value(q_table, state, action, reward, next_state, available_actions):
    """
    Update the Q-value for a state-action pair using Q-learning formulas.

    Args:
        q_table (dict): The Q-table for the current player.
        state (list): The current state of the game board.
        action (int): The action chosen.
        reward (float): The reward from the chosen action.
        next_state (list): The next state of the game board.
        available_actions (list): List of valid actions in the next state.

    Returns:
        None
    """
    state_key = get_state_key(state)

    # Get Q-value of the currect state-action pair
    current_q = q_table.get((state_key, action), 0.0)
    # Get Q-value of the next state
    if next_state is not None:
        next_state_key = get_state_key(next_state)
        max_next_q = max([q_table.get((next_state_key, a), 0.0) for a in available_actions], default=0.0)
    else:
        max_next_q = 0.0
    # Update the Q-Table using the Q-learning Formula
    q_table[(state_key, action)] = current_q + ALPHA * (reward + GAMMA * max_next_q - current_q)

def handle_player_turn(q_table, state, last_state, last_action, reward, epsilon):
    """
    Handles a player's turn, including action selection and Q-value update.

    Args:
        q_table (dict): The Q-table for the current player.
        state (list): The current state of the game board.
        last_state (list): The previous state of the game board.
        last_action (int): The last action taken by the player.
        reward (float): The reward received.
        epsilon (float): Random exploration rate.

    Returns:
        tuple: The selected action and updated state.
    """
    available_actions = env.get_available_actions()
    # If this is not the first move, update Q-value for the last state-action pair
    if last_state is not None:
        update_q_value(q_table, last_state, last_action, reward, state, available_actions)
    # Select the next action using the epsilon-qlearning policy    
    action = select_action(q_table, state, available_actions, epsilon)
    return action, state

def track_outcome():
    """
    Track the outcome of the game by updating win/loss/draw stats.

    """
    global x_wins, o_wins, draw_games
    winner = env._game_won()
    # X wins
    if winner == 1:  
        x_wins += 1
    # O wins    
    elif winner == -1:  
        o_wins += 1
    # Draw
    else:  
        draw_games.append(True)

def decay_epsilon(game, decay_denom = 1000):
    """
    Decay epsilon over time.

    Args:
        game (int): Current game number.
        decay_denom (int): Number of games per decay

    Returns:
        float: The updated epsilon value.
    """
    
    decay_factor = EPSILON_DECAY ** (game / decay_denom)
    return max(EPSILON_MIN, EPSILON_START * decay_factor)

def analyze_results():
    """
    Analyze and print game statistics.

    """
    total_games = len(draw_games) + x_wins + o_wins
    draw_count = len(draw_games)
    draw_rate = (draw_count / total_games) * 100
    x_win_rate = (x_wins / total_games) * 100
    o_win_rate = (o_wins / total_games) * 100
    print(f"Total games played: {total_games}")
    print(f"X Wins: {x_wins} ({x_win_rate:.2f}%)")
    print(f"O Wins: {o_wins} ({o_win_rate:.2f}%)")
    print(f"Overall Draw Rate: {draw_rate:.2f}%")

# Main loop
epsilon = EPSILON_START
for game in range(N_GAMES):
    state, _ = env.reset()
    # Variable initialization
    terminated = False
    last_state_x, last_action_x, reward_x = None, None, 0
    last_state_o, last_action_o, reward_o = None, None, 0
    # While game is going 
    while not terminated:
        player_turn = env.get_player_turn()
        if player_turn == 1:  # X's turn
            action, last_state_x = handle_player_turn(q_table_x, state, last_state_x, last_action_x, reward_x, epsilon)
            last_action_x = action
        else:  # O's turn
            action, last_state_o = handle_player_turn(q_table_o, state, last_state_o, last_action_o, reward_o, epsilon)
            last_action_o = action

        # Execute action and update state
        next_state, reward, terminated, _, _ = env.step(action)
       
        # Game is over updates
        if terminated:
            winner = env._game_won()
            reward_x, reward_o = (reward, -reward) if player_turn == 1 else (-reward, reward)
            update_q_value(q_table_x, last_state_x, last_action_x, reward_x, None, [])
            update_q_value(q_table_o, last_state_o, last_action_o, reward_o, None, [])
            track_outcome()
        state = next_state
    
    if (game + 1) % 10000 == 0:
        print(f"Game {game + 1}: X Wins: {x_wins}, O Wins: {o_wins}, Draws: {len(draw_games)}")
    epsilon = decay_epsilon(game)

# Save results and analyze
pd.Series(name='draw_games', data=draw_games).to_csv('draw_games.csv')
analyze_results()

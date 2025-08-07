"""
Isolated worker module for Monte Carlo simulations to avoid pickling issues
"""
import random
import numpy as np
from typing import Tuple, Type, Dict, Any, Optional

# Import only what's needed, avoid logger
from base_strategy import BaseStrategy


def run_single_simulation(
    sim_id: int,
    strategy_class: Type[BaseStrategy],
    strategy_params: Dict[str, Any],
    max_rounds: int,
    stop_on_bankruptcy: bool,
    stop_on_target: bool,
    seed: Optional[int] = None
) -> dict:
    """
    Run a single simulation and return results as a dictionary.
    This function is designed to be pickle-safe for multiprocessing.
    """
    # Set seed for this simulation
    if seed is not None:
        random.seed(seed + sim_id)
        np.random.seed(seed + sim_id)
    
    try:
        # Create strategy instance
        strategy = strategy_class(**strategy_params)
    except Exception as e:
        # Return failure result
        return {
            'survived': False,
            'final_bankroll': 0.0,
            'rounds_played': 0,
            'max_bankroll': 0.0,
            'min_bankroll': 0.0,
            'target_reached': False,
            'bankruptcy_round': None,
            'target_round': None,
            'win_rate': 0.0,
            'error': str(e)
        }
    
    rounds = 0
    target_reached = False
    target_round = None
    bankruptcy_round = None
    
    try:
        while rounds < max_rounds:
            rounds += 1
            
            # Determine outcome
            won = random.random() < strategy.state.win_probability
            
            # Process round
            next_bet, new_bankroll = strategy.process_round(won)
            
            # Check for target reached
            if new_bankroll >= strategy.target_bankroll:
                target_reached = True
                target_round = rounds
                if stop_on_target:
                    break
            
            # Check for bankruptcy
            if new_bankroll <= 0 or next_bet > new_bankroll:
                bankruptcy_round = rounds
                if stop_on_bankruptcy:
                    break
        
        # Get final stats
        stats = strategy.get_stats()
        
        return {
            'survived': bankruptcy_round is None,
            'final_bankroll': strategy.state.current_bankroll,
            'rounds_played': rounds,
            'max_bankroll': stats["max_bankroll"],
            'min_bankroll': stats["min_bankroll"],
            'target_reached': target_reached,
            'bankruptcy_round': bankruptcy_round,
            'target_round': target_round,
            'win_rate': stats["win_rate"],
            'error': None
        }
        
    except Exception as e:
        # Return partial result on error
        return {
            'survived': False,
            'final_bankroll': getattr(strategy.state, 'current_bankroll', 0.0),
            'rounds_played': rounds,
            'max_bankroll': getattr(strategy.state, 'max_bankroll', 0.0),
            'min_bankroll': getattr(strategy.state, 'min_bankroll', 0.0),
            'target_reached': target_reached,
            'bankruptcy_round': bankruptcy_round,
            'target_round': target_round,
            'win_rate': 0.0,
            'error': str(e)
        }
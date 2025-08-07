"""
Safe worker module for multiprocessing that handles spawn method correctly.
This module provides the worker function that can be safely imported by spawned processes.
"""

import random
import os
import time
import warnings
import numpy as np
from typing import List, Dict, Any, Type

# Suppress warnings in worker processes
warnings.filterwarnings('ignore')


def run_simulation_batch_safe(
    batch_ids: List[int],
    strategy_class: Type,  # Don't import BaseStrategy to avoid logger initialization
    strategy_params: Dict[str, Any],
    max_rounds: int,
    stop_on_bankruptcy: bool,
    stop_on_target: bool,
    seed_offset: int = 0
) -> List[dict]:
    """
    Run a batch of simulations in a separate process with optimizations.
    This is a standalone function that can be safely pickled for multiprocessing.
    Returns results as dictionaries to avoid pickle issues.
    """
    batch_results = []
    
    # Set unique random seed for this batch
    base_seed = hash(f"{os.getpid()}_{time.time()}_{seed_offset}") % (2**32)
    
    for sim_id in batch_ids:
        # Use unique seed per simulation
        sim_seed = base_seed + sim_id
        random.seed(sim_seed)
        np.random.seed(sim_seed % (2**32))
        
        try:
            # Create strategy instance
            strategy = strategy_class(**strategy_params)
            
            rounds = 0
            target_reached = False
            target_round = None
            bankruptcy_round = None
            
            # Pre-generate random numbers for efficiency
            win_prob = strategy.state.win_probability
            cache_size = min(max_rounds, 1000)
            random_cache = np.random.random(cache_size)
            cache_pos = 0
            
            # Run simulation
            while rounds < max_rounds:
                rounds += 1
                
                # Use cached random number if available
                if cache_pos < len(random_cache):
                    rand_val = random_cache[cache_pos]
                    cache_pos += 1
                else:
                    rand_val = random.random()
                    
                won = rand_val < win_prob
                
                # Process round with error handling
                try:
                    next_bet, new_bankroll = strategy.process_round(won)
                except Exception:
                    break
                
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
            
            # Return as dictionary for multiprocessing compatibility
            result = {
                'survived': bankruptcy_round is None,
                'final_bankroll': strategy.state.current_bankroll,
                'rounds_played': rounds,
                'max_bankroll': stats["max_bankroll"],
                'min_bankroll': stats["min_bankroll"],
                'target_reached': target_reached,
                'bankruptcy_round': bankruptcy_round,
                'target_round': target_round,
                'win_rate': stats["win_rate"]
            }
            
            batch_results.append(result)
            
        except Exception as e:
            # Add failed result
            batch_results.append({
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
            })
    
    return batch_results
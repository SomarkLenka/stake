#!/usr/bin/env python3
"""
Isolated worker module for multiprocessing that avoids any imports that trigger logging.
This module is completely self-contained to work with spawn method.
"""

def run_simulation_batch_isolated(
    batch_ids,
    strategy_class,
    strategy_params,
    max_rounds,
    stop_on_bankruptcy,
    stop_on_target,
    seed_offset=0
):
    """
    Run a batch of simulations in a completely isolated manner.
    Imports are done inside the function to avoid module-level initialization issues.
    """
    # Do all imports inside the function to avoid initialization issues
    import random
    import os
    import time
    import warnings
    
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Import numpy only when needed
    try:
        import numpy as np
        has_numpy = True
    except ImportError:
        has_numpy = False
    
    batch_results = []
    
    # Set unique random seed for this batch
    base_seed = hash(f"{os.getpid()}_{time.time()}_{seed_offset}") % (2**32)
    
    for sim_id in batch_ids:
        # Use unique seed per simulation
        sim_seed = base_seed + sim_id
        random.seed(sim_seed)
        if has_numpy:
            np.random.seed(sim_seed % (2**32))
        
        try:
            # Create strategy instance
            strategy = strategy_class(**strategy_params)
            
            rounds = 0
            target_reached = False
            target_round = None
            bankruptcy_round = None
            
            # Pre-generate random numbers for efficiency if numpy available
            win_prob = strategy.state.win_probability
            random_cache = None
            cache_pos = 0
            
            if has_numpy:
                cache_size = min(max_rounds, 1000)
                random_cache = np.random.random(cache_size)
            
            # Run simulation
            while rounds < max_rounds:
                rounds += 1
                
                # Use cached random number if available
                if random_cache is not None and cache_pos < len(random_cache):
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
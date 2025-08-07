#!/usr/bin/env python3
"""Test script to verify progress logging works correctly"""

import sys


def main():
    """Test progress logging with large simulation count"""
    
    # Import inside main to avoid issues with multiprocessing spawn
    from monte_carlo_engine import MonteCarloEngine
    from simple_test_strategy import SimpleTestStrategy
    from logger_utils import init_logger, logLev
    
    logger = init_logger(name="test_progress", level=logLev.INFO, is_orc=True)
    
    logger.info("Testing progress logging with 100,000 simulations...")
    
    # Create engine
    engine = MonteCarloEngine(
        strategy_class=SimpleTestStrategy,
        strategy_params={
            'initial_bankroll': 10000,
            'target_bankroll': 20000,
            'win_probability': 0.49,
            'expected_return': 2.0,
            'base_bet_fraction': 0.02
        },
        max_rounds=1000
    )
    
    # Test with large simulation count to see progress
    logger.info("Starting large simulation run...")
    summary = engine.run_simulations(
        num_simulations=100000,
        parallel=True,
        use_multiprocessing=True
    )
    
    logger.info(f"✅ Completed!")
    logger.info(f"   Survival rate: {summary.survival_rate:.2%}")
    logger.info(f"   Target success: {summary.target_success_rate:.2%}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
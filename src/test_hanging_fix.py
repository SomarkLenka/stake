#!/usr/bin/env python3
"""Test script to verify hanging issue is fixed"""

import sys


def main():
    """Test that multiprocessing doesn't hang"""
    
    # Import inside main to avoid issues
    from monte_carlo_engine import MonteCarloEngine
    from simple_test_strategy import SimpleTestStrategy
    from logger_utils import init_logger, logLev
    
    logger = init_logger(name="test_hanging", level=logLev.INFO, is_orc=True)
    
    logger.info("Testing multiprocessing hanging fix...")
    
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
    
    # Test with multiprocessing 
    logger.info("Running 10,000 simulations...")
    try:
        summary = engine.run_simulations(
            num_simulations=10000,
            parallel=True,
            use_multiprocessing=True
        )
        
        logger.info(f"✅ Completed successfully!")
        logger.info(f"   Survival rate: {summary.survival_rate:.2%}")
        logger.info(f"   Target success: {summary.target_success_rate:.2%}")
        return 0
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
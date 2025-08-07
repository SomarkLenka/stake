#!/usr/bin/env python3
"""Test that threading is now the default and doesn't hang"""

import sys


def main():
    """Test with threading as default"""
    
    # Import inside main
    from monte_carlo_engine import MonteCarloEngine
    from simple_test_strategy import SimpleTestStrategy
    from logger_utils import init_logger, logLev
    
    logger = init_logger(name="test_threading", level=logLev.INFO, is_orc=True)
    
    logger.info("Testing with threading as default...")
    
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
    
    # Test WITHOUT specifying use_multiprocessing (should use threading by default)
    logger.info("Running 100,000 simulations with default settings...")
    summary = engine.run_simulations(
        num_simulations=100000,
        parallel=True
        # use_multiprocessing not specified, defaults to False
    )
    
    logger.info(f"✅ Completed successfully with threading!")
    logger.info(f"   Survival rate: {summary.survival_rate:.2%}")
    logger.info(f"   Target success: {summary.target_success_rate:.2%}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Test script to verify multiprocessing fix"""

import sys
from monte_carlo_engine import MonteCarloEngine
from simple_test_strategy import SimpleTestStrategy
from logger_utils import init_logger, logLev

logger = init_logger(name="test_mp_fix", level=logLev.INFO, is_orc=True)


def main():
    """Test multiprocessing with the fixed engine"""
    
    logger.info("Testing multiprocessing fix...")
    
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
    
    # Test with multiprocessing explicitly enabled
    logger.info("Running 5000 simulations with multiprocessing...")
    try:
        summary = engine.run_simulations(
            num_simulations=5000,
            parallel=True,
            use_multiprocessing=True
        )
        
        logger.info(f"✅ Multiprocessing completed successfully!")
        logger.info(f"   Survival rate: {summary.survival_rate:.2%}")
        logger.info(f"   Target success: {summary.target_success_rate:.2%}")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Multiprocessing failed: {e}")
        logger.info("Testing fallback to threading...")
        
        # Test threading fallback
        summary = engine.run_simulations(
            num_simulations=5000,
            parallel=True,
            use_multiprocessing=False
        )
        
        logger.info(f"✅ Threading fallback worked!")
        logger.info(f"   Survival rate: {summary.survival_rate:.2%}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
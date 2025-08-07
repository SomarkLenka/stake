#!/usr/bin/env python3
"""Test worker configuration for high-core systems"""

import sys
from multiprocessing import cpu_count


def main():
    """Show worker configuration"""
    
    from monte_carlo_engine import MonteCarloEngine
    from simple_test_strategy import SimpleTestStrategy
    from logger_utils import init_logger, logLev
    import time
    
    logger = init_logger(name="test_workers", level=logLev.INFO, is_orc=True)
    
    cores = cpu_count()
    logger.info(f"🖥️ System detected: {cores} CPU cores")
    
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
    
    # Test different worker configurations
    test_configs = [
        (10000, None, "Auto-detect"),
        (10000, 32, "32 workers"),
        (10000, 64, "64 workers"),
        (10000, 128, "128 workers"),
    ]
    
    for num_sims, workers, desc in test_configs:
        logger.info(f"\n📊 Testing: {desc}")
        start = time.time()
        
        summary = engine.run_simulations(
            num_simulations=num_sims,
            parallel=True,
            max_workers=workers
        )
        
        elapsed = time.time() - start
        rate = num_sims / elapsed if elapsed > 0 else 0
        
        logger.info(f"✅ {desc}: {elapsed:.2f}s ({rate:.0f} sims/sec)")
        logger.info(f"   Survival rate: {summary.survival_rate:.2%}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
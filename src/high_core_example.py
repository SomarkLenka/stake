#!/usr/bin/env python3
"""
Example configuration for high-core systems (128 cores, 256 threads)
"""

from monte_carlo_engine import MonteCarloEngine
from simple_test_strategy import SimpleTestStrategy
from logger_utils import init_logger, logLev
import time

logger = init_logger(name="high_core_example", level=logLev.INFO, is_orc=True)


def main():
    """Example for high-core system optimization"""
    
    logger.info("🖥️ High-Core System Configuration Example")
    logger.info("System: 128 cores, 256 threads")
    
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
    
    # Configuration options for high-core systems:
    
    # Option 1: Auto-detect (will use up to 256 workers for large simulations)
    logger.info("\n📊 Option 1: Auto-detect workers")
    start = time.time()
    summary = engine.run_simulations(
        num_simulations=1000000,
        parallel=True
        # max_workers=None means auto-detect
    )
    elapsed = time.time() - start
    logger.info(f"✅ Completed in {elapsed:.2f}s ({1000000/elapsed:.0f} sims/sec)")
    logger.info(f"   Survival rate: {summary.survival_rate:.2%}")
    
    # Option 2: Manually set worker count
    logger.info("\n📊 Option 2: Manual worker configuration")
    
    # For 100k simulations with 64 workers
    logger.info("Running with 64 workers...")
    start = time.time()
    summary = engine.run_simulations(
        num_simulations=100000,
        parallel=True,
        max_workers=64  # Explicitly set 64 workers
    )
    elapsed = time.time() - start
    logger.info(f"✅ Completed in {elapsed:.2f}s ({100000/elapsed:.0f} sims/sec)")
    
    # For 1M simulations with 128 workers
    logger.info("Running with 128 workers...")
    start = time.time()
    summary = engine.run_simulations(
        num_simulations=1000000,
        parallel=True,
        max_workers=128  # Use full core count
    )
    elapsed = time.time() - start
    logger.info(f"✅ Completed in {elapsed:.2f}s ({1000000/elapsed:.0f} sims/sec)")
    
    # Option 3: Maximum parallelism with 256 workers
    logger.info("\n📊 Option 3: Maximum parallelism (256 workers)")
    start = time.time()
    summary = engine.run_simulations(
        num_simulations=1000000,
        parallel=True,
        max_workers=256,  # Use all hardware threads
        batch_size=500    # Smaller batches for more parallelism
    )
    elapsed = time.time() - start
    logger.info(f"✅ Completed in {elapsed:.2f}s ({1000000/elapsed:.0f} sims/sec)")
    
    # Option 4: Custom batch sizing for optimal performance
    logger.info("\n📊 Option 4: Custom batch optimization")
    
    # Calculate optimal batch size: total_sims / (workers * 4)
    # This gives each worker about 4 batches to process
    optimal_workers = 128
    optimal_batch_size = 1000000 // (optimal_workers * 4)
    
    logger.info(f"Using {optimal_workers} workers with batch size {optimal_batch_size}")
    start = time.time()
    summary = engine.run_simulations(
        num_simulations=1000000,
        parallel=True,
        max_workers=optimal_workers,
        batch_size=optimal_batch_size
    )
    elapsed = time.time() - start
    logger.info(f"✅ Completed in {elapsed:.2f}s ({1000000/elapsed:.0f} sims/sec)")
    
    # Performance tips
    logger.info("\n💡 Performance Tips for High-Core Systems:")
    logger.info("1. Use threading (default) for better stability")
    logger.info("2. Set max_workers=128 or 256 for your system")
    logger.info("3. Adjust batch_size: smaller = more parallelism, larger = less overhead")
    logger.info("4. Optimal batch_size ≈ num_simulations / (workers * 4)")
    logger.info("5. Monitor CPU usage to find the sweet spot")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
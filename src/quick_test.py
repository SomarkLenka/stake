#!/usr/bin/env python3
"""Quick test of the optimization system"""

from advanced_optimizer import AdvancedOptimizer, OptimizationConfig
from probability_engine import ProbabilityEngine

# Test configuration with very small parameter space
config = OptimizationConfig(
    starting_bankroll=1000,
    target_bankroll=2000,
    win_probability=0.495,  # 3 decimal places as requested
    expected_return=2.0,
    max_rounds=10000,
    
    # Small parameter ranges for quick test
    base_bet_range=[10, 20],  # Just 2 values
    streak_threshold_range=[3],  # Just 1 value
    multiplier1_range=[2.0],  # Just 1 value
    multiplier2_range=[1.5],  # Just 1 value
    
    simulations_per_test=50,  # Very small for quick test
    optimization_goal="balanced",
    max_workers=2
)

print("\n" + "="*60)
print(" QUICK OPTIMIZATION TEST")
print("="*60)
print(f"\nTesting {len(config.base_bet_range)} bet amounts")
print(f"Total combinations: {len(config.base_bet_range)}")
print(f"Simulations per test: {config.simulations_per_test}")

# Create optimizer and run
optimizer = AdvancedOptimizer(config)
results = optimizer.optimize(top_n=2)

# Display results
if results:
    print("\nOptimization Results:")
    print("-"*40)
    for i, result in enumerate(results, 1):
        print(f"\n#{i} Configuration:")
        print(f"  Base Bet: ${result.parameters['base_bet']:.0f}")
        print(f"  Streak Threshold: {result.parameters['streak_threshold']}")
        print(f"  Multipliers: {result.parameters['multiplier1']}x / {result.parameters['multiplier2']}x")
        print(f"  Target Success: {result.target_success_rate:.1%}")
        print(f"  Survival Rate: {result.survival_rate:.1%}")
        print(f"  Score: {result.score:.3f}")

# Test the best configuration with more simulations
if results:
    print("\n" + "="*60)
    print(" TESTING BEST CONFIGURATION")
    print("="*60)
    
    best = results[0]
    engine = ProbabilityEngine()
    
    summary = engine.run_simulation(
        strategy_type="streak_multiplier",
        starting_bankroll=1000,
        base_bet=best.parameters['base_bet'],
        target_bankroll=2000,
        win_probability=0.495,
        expected_return=2.0,
        num_simulations=1000,  # More simulations for validation
        streak_threshold=best.parameters['streak_threshold'],
        multiplier1=best.parameters['multiplier1'],
        multiplier2=best.parameters['multiplier2']
    )
    
    print(f"\nValidation with 1000 simulations:")
    print(f"  Target Success: {summary.target_success_rate:.2%}")
    print(f"  Survival Rate: {summary.survival_rate:.2%}")
    print(f"  Avg Final Bankroll: ${summary.average_final_bankroll:.2f}")
    print(f"  75th Percentile: ${summary.percentile_75:.2f}")
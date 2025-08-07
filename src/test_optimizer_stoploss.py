#!/usr/bin/env python3
"""Test optimizer with stop-loss feature"""

from advanced_optimizer import AdvancedOptimizer, OptimizationConfig

# Test configuration with stop-loss parameter
config = OptimizationConfig(
    starting_bankroll=1000,
    target_bankroll=2000,
    win_probability=0.495,  # 3 decimal precision
    expected_return=2.0,
    max_rounds=10000,
    
    # Small parameter ranges for quick test
    base_bet_range=[10, 15],
    streak_threshold_range=[3, 4],
    multiplier1_range=[2.0],
    multiplier2_range=[1.5],
    max_loss_streak_range=[None, 6, 8],  # Test no stop-loss, 6, and 8
    
    simulations_per_test=100,  # Small for quick test
    optimization_goal="balanced",
    max_workers=2
)

print("\n" + "="*60)
print(" OPTIMIZER WITH STOP-LOSS TEST")
print("="*60)
print(f"\nTesting combinations with stop-loss feature:")
print(f"  Base bets: {config.base_bet_range}")
print(f"  Streak thresholds: {config.streak_threshold_range}")
print(f"  Stop-loss values: {config.max_loss_streak_range}")
print(f"  Total combinations: ~{len(config.base_bet_range) * len(config.streak_threshold_range) * len(config.max_loss_streak_range)}")

# Create optimizer and run
optimizer = AdvancedOptimizer(config)
results = optimizer.optimize(top_n=5)

# Display results
if results:
    print("\n" + "="*60)
    print(" TOP CONFIGURATIONS")
    print("="*60)
    
    for i, result in enumerate(results[:3], 1):
        print(f"\n#{i} Best Configuration:")
        print(f"  Base Bet: ${result.parameters['base_bet']:.0f}")
        print(f"  Streak Threshold: {result.parameters['streak_threshold']}")
        print(f"  Multipliers: {result.parameters['multiplier1']}x / {result.parameters['multiplier2']}x")
        
        if result.parameters.get('max_loss_streak'):
            print(f"  Stop-Loss: After {result.parameters['max_loss_streak']} losses")
        else:
            print(f"  Stop-Loss: Disabled")
            
        print(f"  ---")
        print(f"  Target Success: {result.target_success_rate:.1%}")
        print(f"  Survival Rate: {result.survival_rate:.1%}")
        print(f"  Score: {result.score:.3f}")
    
    # Analysis of stop-loss impact
    print("\n" + "="*60)
    print(" STOP-LOSS IMPACT ANALYSIS")
    print("="*60)
    
    # Group by stop-loss setting
    no_stoploss = [r for r in results if r.parameters.get('max_loss_streak') is None]
    with_stoploss = [r for r in results if r.parameters.get('max_loss_streak') is not None]
    
    if no_stoploss and with_stoploss:
        avg_no_stoploss = sum(r.score for r in no_stoploss) / len(no_stoploss)
        avg_with_stoploss = sum(r.score for r in with_stoploss) / len(with_stoploss)
        
        print(f"\nAverage Scores:")
        print(f"  Without Stop-Loss: {avg_no_stoploss:.3f}")
        print(f"  With Stop-Loss:    {avg_with_stoploss:.3f}")
        
        if avg_with_stoploss > avg_no_stoploss:
            improvement = (avg_with_stoploss - avg_no_stoploss) / avg_no_stoploss * 100
            print(f"\n✓ Stop-loss improves average performance by {improvement:.1f}%")
        else:
            reduction = (avg_no_stoploss - avg_with_stoploss) / avg_no_stoploss * 100
            print(f"\n✗ Stop-loss reduces average performance by {reduction:.1f}%")
    
    print("\nBest strategy uses:", end=" ")
    if results[0].parameters.get('max_loss_streak'):
        print(f"Stop-loss at {results[0].parameters['max_loss_streak']} losses")
    else:
        print("No stop-loss")
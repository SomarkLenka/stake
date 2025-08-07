#!/usr/bin/env python3
"""
Demo of the Advanced Optimizer - Finding optimal betting parameters
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from advanced_optimizer import AdvancedOptimizer, OptimizationConfig
import time

def main():
    print("\n" + "="*80)
    print(" ADVANCED PARAMETER OPTIMIZATION DEMO")
    print("="*80)
    print("\nThis will find the optimal combination of:")
    print("  • Base bet amount (% of bankroll)")
    print("  • Streak threshold (when to switch multipliers)")
    print("  • Multiplier 1 (aggressive - for short losing streaks)")
    print("  • Multiplier 2 (conservative - for long losing streaks)")
    
    # Configure optimization
    config = OptimizationConfig(
        # Fixed game parameters
        starting_bankroll=1000,
        target_bankroll=2000,  # 2x target
        win_probability=0.495,  # 49.5% win chance (3 decimal precision)
        expected_return=2.0,    # 2x payout on win
        max_rounds=10000,
        
        # Parameter ranges to test  
        base_bet_range=[10, 20, 30],  # Test 3 bet amounts for faster demo
        streak_threshold_range=[2, 3, 4],  # Switch multipliers after N losses
        multiplier1_range=[1.5, 2.0, 2.5],  # Aggressive multipliers
        multiplier2_range=[1.2, 1.5],  # Conservative multipliers
        
        # Optimization settings
        simulations_per_test=100,  # 100 simulations per parameter combo for faster demo
        optimization_goal="balanced",  # Balance survival and target success
        max_workers=4  # Parallel threads
    )
    
    # Create optimizer
    optimizer = AdvancedOptimizer(config)
    
    print(f"\nOptimization Configuration:")
    print(f"  • Testing {len(config.base_bet_range)} bet amounts: ${min(config.base_bet_range):.0f}-${max(config.base_bet_range):.0f}")
    print(f"  • Testing {len(config.streak_threshold_range)} streak thresholds: {min(config.streak_threshold_range)}-{max(config.streak_threshold_range)}")
    print(f"  • Testing {len(config.multiplier1_range)} aggressive multipliers: {min(config.multiplier1_range)}x-{max(config.multiplier1_range)}x")
    print(f"  • Testing {len(config.multiplier2_range)} conservative multipliers: {min(config.multiplier2_range)}x-{max(config.multiplier2_range)}x")
    
    total_combos = (len(config.base_bet_range) * 
                   len(config.streak_threshold_range) * 
                   len(config.multiplier1_range) * 
                   len(config.multiplier2_range))
    
    print(f"\nTotal combinations to test: ~{total_combos} (filtered for validity)")
    print(f"Simulations per test: {config.simulations_per_test}")
    print(f"Using {config.max_workers} parallel workers")
    
    # Run optimizations for different goals
    print("\n" + "="*80)
    print(" OPTIMIZATION ROUND 1: MAXIMIZE TARGET SUCCESS")
    print("="*80)
    
    target_results = optimizer.optimize_for_target(top_n=5)
    optimizer.print_results(target_results[:3], "Top 3 for TARGET SUCCESS")
    
    print("\n" + "="*80)
    print(" OPTIMIZATION ROUND 2: MAXIMIZE SURVIVAL")
    print("="*80)
    
    survival_results = optimizer.optimize_for_survival(top_n=5)
    optimizer.print_results(survival_results[:3], "Top 3 for SURVIVAL")
    
    print("\n" + "="*80)
    print(" OPTIMIZATION ROUND 3: BALANCED APPROACH")
    print("="*80)
    
    balanced_results = optimizer.optimize_balanced(top_n=5)
    optimizer.print_results(balanced_results[:3], "Top 3 for BALANCED")
    
    # Compare the best from each category
    print("\n" + "="*80)
    print(" COMPARISON OF BEST STRATEGIES")
    print("="*80)
    
    if target_results and survival_results and balanced_results:
        print("\nBest for TARGET SUCCESS:")
        best_target = target_results[0]
        print(f"  Base Bet: ${best_target.parameters['base_bet']:.2f}")
        print(f"  Threshold: {best_target.parameters['streak_threshold']}")
        print(f"  Multipliers: {best_target.parameters['multiplier1']:.1f}x / {best_target.parameters['multiplier2']:.1f}x")
        print(f"  → Target: {best_target.target_success_rate:.1%}, Survival: {best_target.survival_rate:.1%}")
        
        print("\nBest for SURVIVAL:")
        best_survival = survival_results[0]
        print(f"  Base Bet: ${best_survival.parameters['base_bet']:.2f}")
        print(f"  Threshold: {best_survival.parameters['streak_threshold']}")
        print(f"  Multipliers: {best_survival.parameters['multiplier1']:.1f}x / {best_survival.parameters['multiplier2']:.1f}x")
        print(f"  → Target: {best_survival.target_success_rate:.1%}, Survival: {best_survival.survival_rate:.1%}")
        
        print("\nBest BALANCED:")
        best_balanced = balanced_results[0]
        print(f"  Base Bet: ${best_balanced.parameters['base_bet']:.2f}")
        print(f"  Threshold: {best_balanced.parameters['streak_threshold']}")
        print(f"  Multipliers: {best_balanced.parameters['multiplier1']:.1f}x / {best_balanced.parameters['multiplier2']:.1f}x")
        print(f"  → Target: {best_balanced.target_success_rate:.1%}, Survival: {best_balanced.survival_rate:.1%}")
    
    # Save results
    optimizer.save_results(balanced_results, "optimal_parameters.json")
    print("\n✓ Results saved to optimal_parameters.json")
    
    # Show how to test with different win probabilities
    print("\n" + "="*80)
    print(" TESTING DIFFERENT WIN PROBABILITIES")
    print("="*80)
    
    probabilities = [0.485, 0.490, 0.495, 0.500, 0.505]  # 48.5% to 50.5%
    
    print("\nTesting optimal parameters across different win probabilities:")
    print("(Using best balanced configuration found above)")
    
    if balanced_results:
        best = balanced_results[0]
        from probability_engine import ProbabilityEngine
        engine = ProbabilityEngine()
        
        for prob in probabilities:
            summary = engine.run_simulation(
                strategy_type="streak_multiplier",
                starting_bankroll=1000,
                base_bet=best.parameters['base_bet'],
                target_bankroll=2000,
                win_probability=prob,
                expected_return=2.0,
                num_simulations=500,
                streak_threshold=best.parameters['streak_threshold'],
                multiplier1=best.parameters['multiplier1'],
                multiplier2=best.parameters['multiplier2']
            )
            
            print(f"\nP(win) = {prob:.3f} ({prob*100:.1f}%):")
            print(f"  Target Success: {summary.target_success_rate:.1%}")
            print(f"  Survival Rate:  {summary.survival_rate:.1%}")
            print(f"  Avg Bankroll:   ${summary.average_final_bankroll:.2f}")


if __name__ == "__main__":
    main()
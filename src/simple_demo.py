#!/usr/bin/env python3
"""
Simple demonstration of the probability engine without excessive logging
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from probability_engine import ProbabilityEngine

def main():
    print("\n" + "="*60)
    print("PROBABILITY ENGINE - MONTE CARLO SIMULATION")
    print("="*60)
    print("\nConfiguration:")
    print("- Starting Bankroll: $1,000")
    print("- Base Bet: $10")
    print("- Target: $2,000 (2x bankroll)")
    print("- Win Probability: 49%")
    print("- Payout: 2x")
    print("\nStrategy: Streak-based Multiplier")
    print("- On WIN: Reset to base bet ($10)")
    print("- On LOSS (streak < 3): Multiply bet by 2.0")
    print("- On LOSS (streak >= 3): Multiply bet by 1.5")
    print("\nRunning 1000 simulations...")
    print("-" * 60)
    
    engine = ProbabilityEngine()
    
    # Run simulation
    summary = engine.run_simulation(
        strategy_type="streak_multiplier",
        starting_bankroll=1000,
        base_bet=10,
        target_bankroll=2000,
        win_probability=0.49,
        expected_return=2.0,
        num_simulations=1000,
        max_rounds=10000,
        streak_threshold=3,
        multiplier1=2.0,
        multiplier2=1.5
    )
    
    # Display results
    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    print(f"Survival Rate:        {summary.survival_rate:.1%}")
    print(f"Target Success Rate:  {summary.target_success_rate:.1%}")
    print(f"Average Rounds:       {summary.average_rounds:.0f}")
    print(f"Average Final Bank:   ${summary.average_final_bankroll:.2f}")
    print(f"\nBankroll Distribution:")
    print(f"  25th percentile:    ${summary.percentile_25:.2f}")
    print(f"  50th (median):      ${summary.median_final_bankroll:.2f}")
    print(f"  75th percentile:    ${summary.percentile_75:.2f}")
    print(f"  95th percentile:    ${summary.percentile_95:.2f}")
    
    if summary.average_bankruptcy_round:
        print(f"\nAvg Bankruptcy Round: {summary.average_bankruptcy_round:.0f}")
    if summary.average_target_round:
        print(f"Avg Target Round:     {summary.average_target_round:.0f}")
    
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    if summary.survival_rate > 0.4:
        print("✓ Good survival rate - strategy is relatively safe")
    elif summary.survival_rate > 0.2:
        print("⚠ Moderate survival rate - strategy has significant risk")
    else:
        print("✗ Poor survival rate - strategy is very risky")
    
    if summary.target_success_rate > 0.3:
        print("✓ Good chance of reaching target")
    elif summary.target_success_rate > 0.15:
        print("⚠ Moderate chance of reaching target")
    else:
        print("✗ Low chance of reaching target")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
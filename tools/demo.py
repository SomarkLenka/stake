#!/usr/bin/env python3
"""
Simple demonstration of the probability engine
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from probability_engine import ProbabilityEngine

def main():
    print("\n" + "="*60)
    print("PROBABILITY ENGINE DEMONSTRATION")
    print("="*60)
    
    engine = ProbabilityEngine()
    
    # Run a simulation with the requested strategy
    summary = engine.run_simulation(
        strategy_type="streak_multiplier",
        starting_bankroll=1000,
        base_bet=10,
        target_bankroll=2000,
        win_probability=0.49,  # 49% win chance (slight house edge)
        expected_return=2.0,    # 2x payout on win
        num_simulations=1000,   # Run 1000 simulations
        max_rounds=10000,       # Max 10k rounds per simulation
        streak_threshold=3,     # Switch multipliers at 3-loss streak
        multiplier1=2.0,        # Double bet on losses < 3 streak
        multiplier2=1.5         # 1.5x bet on losses >= 3 streak
    )
    
    print(summary)
    
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)
    
    # Compare different strategies
    strategies = [
        {
            'name': 'Your Requested Strategy (2x/1.5x)',
            'type': 'streak_multiplier',
            'params': {
                'streak_threshold': 3,
                'multiplier1': 2.0,
                'multiplier2': 1.5
            }
        },
        {
            'name': 'Conservative (1.5x/1.2x)',
            'type': 'streak_multiplier',
            'params': {
                'streak_threshold': 4,
                'multiplier1': 1.5,
                'multiplier2': 1.2
            }
        },
        {
            'name': 'Classic Martingale',
            'type': 'martingale',
            'params': {'multiplier': 2.0}
        }
    ]
    
    base_params = {
        'starting_bankroll': 1000,
        'base_bet': 10,
        'target_bankroll': 2000,
        'win_probability': 0.49,
        'expected_return': 2.0,
        'max_rounds': 10000
    }
    
    results = engine.compare_strategies(
        strategies,
        base_params,
        num_simulations=500  # Fewer simulations for quick demo
    )
    
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    for name, summary in results.items():
        print(f"\n{name}:")
        print(f"  Survival Rate:      {summary.survival_rate:.2%}")
        print(f"  Target Success:     {summary.target_success_rate:.2%}")
        print(f"  Avg Rounds:         {summary.average_rounds:.0f}")
        print(f"  Avg Final Bankroll: ${summary.average_final_bankroll:.2f}")
        print(f"  25th Percentile:    ${summary.percentile_25:.2f}")
        print(f"  75th Percentile:    ${summary.percentile_75:.2f}")

if __name__ == "__main__":
    main()
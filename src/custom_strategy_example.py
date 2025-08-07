#!/usr/bin/env python3
"""
Example of creating a custom strategy with the modular framework
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from probability_engine import ProbabilityEngine
from base_strategy import (
    Condition, Action, Rule,
    EventType, ComparisonOperator, ActionType
)

def create_fibonacci_strategy_rules():
    """
    Create a Fibonacci-inspired betting strategy:
    - On win: Reset to base bet
    - On 1st loss: Keep same bet
    - On 2nd loss: Add previous bet (1+1=2 units)
    - On 3rd loss: Add previous bet (2+1=3 units)
    - On 4th loss: Add previous bet (3+2=5 units)
    - On 5+ losses: Reset to prevent catastrophic loss
    """
    return [
        # On any win, reset to base
        Rule(
            condition=Condition(event_type=EventType.WIN),
            action=Action(action_type=ActionType.RESET_BET),
            priority=10
        ),
        
        # On 5+ loss streak, reset (stop loss)
        Rule(
            condition=Condition(
                event_type=EventType.LOSE,
                streak_operator=ComparisonOperator.GREATER_EQUAL,
                streak_threshold=5
            ),
            action=Action(action_type=ActionType.RESET_BET),
            priority=9
        ),
        
        # Fibonacci-like progression for losses
        Rule(
            condition=Condition(
                event_type=EventType.LOSE,
                streak_operator=ComparisonOperator.EQUAL,
                streak_threshold=1
            ),
            action=Action(action_type=ActionType.MULTIPLY_BET, value=1.0),  # Keep same
            priority=8
        ),
        Rule(
            condition=Condition(
                event_type=EventType.LOSE,
                streak_operator=ComparisonOperator.EQUAL,
                streak_threshold=2
            ),
            action=Action(action_type=ActionType.MULTIPLY_BET, value=2.0),  # Double
            priority=8
        ),
        Rule(
            condition=Condition(
                event_type=EventType.LOSE,
                streak_operator=ComparisonOperator.EQUAL,
                streak_threshold=3
            ),
            action=Action(action_type=ActionType.MULTIPLY_BET, value=1.5),  # 1.5x
            priority=8
        ),
        Rule(
            condition=Condition(
                event_type=EventType.LOSE,
                streak_operator=ComparisonOperator.EQUAL,
                streak_threshold=4
            ),
            action=Action(action_type=ActionType.MULTIPLY_BET, value=1.67),  # ~5/3
            priority=8
        ),
    ]

def create_dynamic_probability_strategy_rules():
    """
    Create a strategy that adjusts win probability based on streaks:
    - Assumes the game has "hot" and "cold" streaks
    - After wins, slightly reduce win probability (house adjusts)
    - After losses, slightly increase win probability (regression to mean)
    """
    return [
        # After 3+ wins, reduce win probability and reset bet
        Rule(
            condition=Condition(
                event_type=EventType.WIN,
                streak_operator=ComparisonOperator.GREATER_EQUAL,
                streak_threshold=3
            ),
            action=Action(action_type=ActionType.MULTIPLY_WIN_PROB, value=0.98),
            priority=10
        ),
        Rule(
            condition=Condition(
                event_type=EventType.WIN,
                streak_operator=ComparisonOperator.GREATER_EQUAL,
                streak_threshold=3
            ),
            action=Action(action_type=ActionType.RESET_BET),
            priority=9
        ),
        
        # After 3+ losses, increase win probability slightly
        Rule(
            condition=Condition(
                event_type=EventType.LOSE,
                streak_operator=ComparisonOperator.GREATER_EQUAL,
                streak_threshold=3
            ),
            action=Action(action_type=ActionType.MULTIPLY_WIN_PROB, value=1.02),
            priority=10
        ),
        
        # Standard bet adjustments
        Rule(
            condition=Condition(event_type=EventType.WIN),
            action=Action(action_type=ActionType.RESET_BET),
            priority=5
        ),
        Rule(
            condition=Condition(event_type=EventType.LOSE),
            action=Action(action_type=ActionType.MULTIPLY_BET, value=1.5),
            priority=5
        ),
    ]

def main():
    print("\n" + "="*60)
    print("CUSTOM STRATEGY EXAMPLES")
    print("="*60)
    
    engine = ProbabilityEngine()
    
    # Test Fibonacci-inspired strategy
    print("\n1. FIBONACCI-INSPIRED STRATEGY")
    print("-" * 40)
    
    fib_rules = create_fibonacci_strategy_rules()
    summary1 = engine.run_simulation(
        strategy_type="custom",
        starting_bankroll=1000,
        base_bet=10,
        target_bankroll=2000,
        win_probability=0.49,
        expected_return=2.0,
        num_simulations=500,
        max_rounds=10000,
        rules=fib_rules
    )
    
    print(f"Survival Rate:       {summary1.survival_rate:.1%}")
    print(f"Target Success Rate: {summary1.target_success_rate:.1%}")
    print(f"Average Final Bank:  ${summary1.average_final_bankroll:.2f}")
    
    # Test dynamic probability strategy
    print("\n2. DYNAMIC PROBABILITY STRATEGY")
    print("-" * 40)
    
    dynamic_rules = create_dynamic_probability_strategy_rules()
    summary2 = engine.run_simulation(
        strategy_type="custom",
        starting_bankroll=1000,
        base_bet=10,
        target_bankroll=2000,
        win_probability=0.50,  # Start with 50/50
        expected_return=2.0,
        num_simulations=500,
        max_rounds=10000,
        rules=dynamic_rules
    )
    
    print(f"Survival Rate:       {summary2.survival_rate:.1%}")
    print(f"Target Success Rate: {summary2.target_success_rate:.1%}")
    print(f"Average Final Bank:  ${summary2.average_final_bankroll:.2f}")
    
    # Compare all strategies
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)
    
    strategies = [
        {
            'name': 'Your Original (Streak Multiplier)',
            'type': 'streak_multiplier',
            'params': {
                'streak_threshold': 3,
                'multiplier1': 2.0,
                'multiplier2': 1.5
            }
        },
        {
            'name': 'Fibonacci-Inspired',
            'type': 'custom',
            'params': {'rules': fib_rules}
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
    
    print("\nRunning comparison (200 simulations each)...")
    results = engine.compare_strategies(
        strategies,
        base_params,
        num_simulations=200
    )
    
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    
    for name, summary in results.items():
        print(f"\n{name}:")
        print(f"  Survival:  {summary.survival_rate:.1%}")
        print(f"  Success:   {summary.target_success_rate:.1%}")
        print(f"  Avg Bank:  ${summary.average_final_bankroll:.2f}")

if __name__ == "__main__":
    main()
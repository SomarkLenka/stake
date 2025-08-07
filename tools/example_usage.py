#!/usr/bin/env python3
"""
Example usage of the probability engine
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from probability_engine import ProbabilityEngine
from base_strategy import (
    Condition, Action, Rule,
    EventType, ComparisonOperator, ActionType
)
from streak_multiplier_strategy import CustomRuleStrategy
from logger_utils import init_logger, logLev

logger = init_logger(name="example_usage", level=logLev.INFO, is_orc=True)


def example_basic_simulation():
    """Run a basic simulation with default strategy"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Simulation with Streak Multiplier Strategy")
    print("="*60)
    
    engine = ProbabilityEngine()
    
    summary = engine.run_simulation(
        strategy_type="streak_multiplier",
        starting_bankroll=1000,
        base_bet=10,
        target_bankroll=2000,
        win_probability=0.49,
        expected_return=2.0,
        num_simulations=1000,
        streak_threshold=3,
        multiplier1=2.0,
        multiplier2=1.5
    )
    
    print(summary)


def example_strategy_comparison():
    """Compare multiple strategies"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Strategy Comparison")
    print("="*60)
    
    engine = ProbabilityEngine()
    
    strategies = [
        {
            'name': 'Conservative Streak (1.5x/1.2x)',
            'type': 'streak_multiplier',
            'params': {
                'streak_threshold': 4,
                'multiplier1': 1.5,
                'multiplier2': 1.2
            }
        },
        {
            'name': 'Aggressive Streak (3x/2x)',
            'type': 'streak_multiplier',
            'params': {
                'streak_threshold': 2,
                'multiplier1': 3.0,
                'multiplier2': 2.0
            }
        },
        {
            'name': 'Classic Martingale',
            'type': 'martingale',
            'params': {'multiplier': 2.0}
        },
        {
            'name': 'Reverse Martingale (3 wins max)',
            'type': 'reverse_martingale',
            'params': {'multiplier': 2.0, 'max_streak': 3}
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
        num_simulations=1000
    )
    
    # Print comparison
    print("\n" + "="*60)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*60)
    
    for name, summary in results.items():
        print(f"\n{name}:")
        print(f"  Survival Rate: {summary.survival_rate:.2%}")
        print(f"  Target Success: {summary.target_success_rate:.2%}")
        print(f"  Avg Final Bankroll: ${summary.average_final_bankroll:.2f}")


def example_custom_strategy():
    """Create a fully custom strategy with specific rules"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Strategy with Complex Rules")
    print("="*60)
    
    # Define custom rules for a strategy that:
    # - Increases bet by 50% on wins (momentum betting)
    # - On first loss, keeps same bet
    # - On 2+ loss streak, uses fibonacci-like progression
    
    custom_rules = [
        # On win, multiply bet by 1.5 (but cap at 100)
        Rule(
            condition=Condition(event_type=EventType.WIN),
            action=Action(action_type=ActionType.MULTIPLY_BET, value=1.5),
            priority=10
        ),
        # On loss with streak >= 5, reset to base (stop loss)
        Rule(
            condition=Condition(
                event_type=EventType.LOSE,
                streak_operator=ComparisonOperator.GREATER_EQUAL,
                streak_threshold=5
            ),
            action=Action(action_type=ActionType.RESET_BET),
            priority=9
        ),
        # On loss with streak 2-4, use fibonacci-like multiplier
        Rule(
            condition=Condition(
                event_type=EventType.LOSE,
                streak_operator=ComparisonOperator.GREATER_EQUAL,
                streak_threshold=2
            ),
            action=Action(action_type=ActionType.MULTIPLY_BET, value=1.618),
            priority=8
        ),
        # On first loss, keep same bet
        Rule(
            condition=Condition(
                event_type=EventType.LOSE,
                streak_operator=ComparisonOperator.LESS,
                streak_threshold=2
            ),
            action=Action(action_type=ActionType.MULTIPLY_BET, value=1.0),
            priority=7
        )
    ]
    
    engine = ProbabilityEngine()
    
    # Need to register the custom strategy
    from streak_multiplier_strategy import CustomRuleStrategy
    
    summary = engine.run_simulation(
        strategy_type="custom",
        starting_bankroll=1000,
        base_bet=10,
        target_bankroll=2000,
        win_probability=0.49,
        expected_return=2.0,
        num_simulations=1000,
        rules=custom_rules
    )
    
    print(summary)


def example_parameter_optimization():
    """Optimize strategy parameters"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Parameter Optimization")
    print("="*60)
    
    engine = ProbabilityEngine()
    
    base_params = {
        'starting_bankroll': 1000,
        'base_bet': 10,
        'target_bankroll': 2000,
        'win_probability': 0.49,
        'expected_return': 2.0,
        'max_rounds': 5000
    }
    
    # Test different parameter combinations
    param_ranges = {
        'streak_threshold': [2, 3, 4],
        'multiplier1': [1.5, 2.0, 2.5],
        'multiplier2': [1.2, 1.5, 1.8]
    }
    
    print("Testing parameter combinations...")
    results = engine.optimize_parameters(
        strategy_type='streak_multiplier',
        base_params=base_params,
        param_ranges=param_ranges,
        num_simulations_per=500,
        target_metric='survival_rate'
    )
    
    # Find best combination
    best_params = None
    best_survival = 0
    
    for params, summary in results.items():
        if summary.survival_rate > best_survival:
            best_survival = summary.survival_rate
            best_params = params
            
    print(f"\nBest parameters found:")
    print(f"  Streak threshold: {best_params[0]}")
    print(f"  Multiplier 1: {best_params[1]}")
    print(f"  Multiplier 2: {best_params[2]}")
    print(f"  Survival rate: {best_survival:.2%}")


def example_probability_conditions():
    """Example with win probability adjustments"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Dynamic Win Probability Strategy")
    print("="*60)
    
    # Create a strategy that adjusts win probability based on streaks
    custom_rules = [
        # After 3 wins, reduce win probability (house adjusts)
        Rule(
            condition=Condition(
                event_type=EventType.WIN,
                streak_operator=ComparisonOperator.GREATER_EQUAL,
                streak_threshold=3
            ),
            action=Action(action_type=ActionType.MULTIPLY_WIN_PROB, value=0.95),
            priority=10
        ),
        # After 3 losses, slightly increase win probability (mercy rule)
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
        )
    ]
    
    engine = ProbabilityEngine()
    
    summary = engine.run_simulation(
        strategy_type="custom",
        starting_bankroll=1000,
        base_bet=10,
        target_bankroll=2000,
        win_probability=0.50,  # Start with 50/50
        expected_return=2.0,
        num_simulations=1000,
        rules=custom_rules
    )
    
    print(summary)


if __name__ == "__main__":
    # Run all examples
    example_basic_simulation()
    example_strategy_comparison()
    example_custom_strategy()
    example_parameter_optimization()
    example_probability_conditions()
#!/usr/bin/env python3
"""Test script for stop-loss feature in StreakMultiplierStrategy"""

from probability_engine import ProbabilityEngine
from monte_carlo_engine import MonteCarloEngine
from streak_multiplier_strategy import StreakMultiplierStrategy

def test_stop_loss_configurations():
    """Test different stop-loss configurations"""
    
    print("\n" + "="*60)
    print(" STOP-LOSS FEATURE TEST")
    print("="*60)
    print("\nComparing strategies with and without stop-loss\n")
    
    # Base configuration
    base_config = {
        'starting_bankroll': 1000,
        'base_bet': 10,
        'target_bankroll': 2000,
        'win_probability': 0.495,
        'expected_return': 2.0,
        'streak_threshold': 3,
        'multiplier1': 2.0,
        'multiplier2': 1.5,
        'num_simulations': 1000,
        'max_rounds': 10000
    }
    
    engine = ProbabilityEngine()
    
    # Test configurations
    configs = [
        {'name': 'No Stop-Loss', 'max_loss_streak': None},
        {'name': 'Stop at 5 losses', 'max_loss_streak': 5},
        {'name': 'Stop at 7 losses', 'max_loss_streak': 7},
        {'name': 'Stop at 10 losses', 'max_loss_streak': 10},
    ]
    
    results = []
    
    for config in configs:
        print(f"Testing: {config['name']}")
        
        # Create engine directly to avoid multiprocessing issues
        engine_direct = MonteCarloEngine(
            strategy_class=StreakMultiplierStrategy,
            strategy_params={
                'starting_bankroll': base_config['starting_bankroll'],
                'base_bet': base_config['base_bet'],
                'target_bankroll': base_config['target_bankroll'],
                'win_probability': base_config['win_probability'],
                'expected_return': base_config['expected_return'],
                'streak_threshold': base_config['streak_threshold'],
                'multiplier1': base_config['multiplier1'],
                'multiplier2': base_config['multiplier2'],
                'max_loss_streak': config['max_loss_streak']
            },
            max_rounds=base_config['max_rounds']
        )
        
        summary = engine_direct.run_simulations(
            num_simulations=base_config['num_simulations'],
            parallel=False  # Use sequential to avoid multiprocessing issues
        )
        
        results.append({
            'name': config['name'],
            'max_loss_streak': config['max_loss_streak'],
            'survival_rate': summary.survival_rate,
            'target_success_rate': summary.target_success_rate,
            'avg_final_bankroll': summary.average_final_bankroll,
            'percentile_75': summary.percentile_75
        })
        
        print(f"  Survival Rate: {summary.survival_rate:.2%}")
        print(f"  Target Success: {summary.target_success_rate:.2%}")
        print(f"  Avg Bankroll: ${summary.average_final_bankroll:.2f}\n")
    
    # Display comparison
    print("\n" + "="*60)
    print(" RESULTS COMPARISON")
    print("="*60)
    print(f"{'Strategy':<20} {'Survival':<12} {'Target':<12} {'Avg Bank':<12} {'75th %ile':<12}")
    print("-"*80)
    
    for result in results:
        name = result['name']
        survival = f"{result['survival_rate']:.1%}"
        target = f"{result['target_success_rate']:.1%}"
        avg_bank = f"${result['avg_final_bankroll']:.0f}"
        p75 = f"${result['percentile_75']:.0f}"
        print(f"{name:<20} {survival:<12} {target:<12} {avg_bank:<12} {p75:<12}")
    
    # Analysis
    print("\n" + "="*60)
    print(" ANALYSIS")
    print("="*60)
    
    baseline = results[0]  # No stop-loss
    
    for result in results[1:]:
        if result['max_loss_streak']:
            survival_diff = (result['survival_rate'] - baseline['survival_rate']) * 100
            target_diff = (result['target_success_rate'] - baseline['target_success_rate']) * 100
            
            print(f"\nStop at {result['max_loss_streak']} losses vs No Stop-Loss:")
            print(f"  Survival: {'+' if survival_diff >= 0 else ''}{survival_diff:.1f}%")
            print(f"  Target Success: {'+' if target_diff >= 0 else ''}{target_diff:.1f}%")
            
            if survival_diff > 5:
                print(f"  ✓ Significantly improves survival rate")
            elif survival_diff > 0:
                print(f"  ✓ Slightly improves survival rate")
            else:
                print(f"  ✗ Reduces survival rate")
                
            if abs(target_diff) < 2:
                print(f"  → Minimal impact on target success")
            elif target_diff > 0:
                print(f"  ✓ Improves target success")
            else:
                print(f"  ✗ Reduces target success")


def test_specific_scenarios():
    """Test specific loss streak scenarios"""
    
    print("\n" + "="*60)
    print(" SPECIFIC SCENARIO TEST")
    print("="*60)
    print("\nTesting behavior with extreme losing streaks\n")
    
    # Create a strategy with stop-loss at 6 losses
    strategy = StreakMultiplierStrategy(
        starting_bankroll=1000,
        base_bet=10,
        target_bankroll=2000,
        win_probability=0.495,
        expected_return=2.0,
        streak_threshold=3,
        multiplier1=2.0,
        multiplier2=1.5,
        max_loss_streak=6
    )
    
    print("Initial state:")
    print(f"  Bankroll: ${strategy.state.current_bankroll:.2f}")
    print(f"  Base bet: ${strategy.base_bet:.2f}")
    print(f"  Stop-loss at: {strategy.max_loss_streak} losses\n")
    
    # Simulate a losing streak
    print("Simulating losing streak:")
    for i in range(8):
        bet_before = strategy.state.current_bet
        next_bet, new_bankroll = strategy.process_round(won=False)
        
        print(f"Loss #{i+1}:")
        print(f"  Bet was: ${bet_before:.2f}")
        print(f"  Bankroll: ${new_bankroll:.2f}")
        print(f"  Next bet: ${next_bet:.2f}")
        print(f"  Streak: {strategy.state.streak_count}")
        
        if i == 5:
            print("  >>> Stop-loss triggered! Bet reset to base")
        print()
    
    # Now simulate a win
    print("Now simulating a win:")
    bet_before = strategy.state.current_bet
    next_bet, new_bankroll = strategy.process_round(won=True)
    print(f"  Bet was: ${bet_before:.2f}")
    print(f"  Bankroll: ${new_bankroll:.2f}")
    print(f"  Next bet: ${next_bet:.2f}")
    print(f"  Streak reset to: {strategy.state.streak_count}")


if __name__ == "__main__":
    test_stop_loss_configurations()
    test_specific_scenarios()
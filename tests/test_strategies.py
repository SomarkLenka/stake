#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
from base_strategy import (
    BettingState, Condition, Action, Rule,
    EventType, ComparisonOperator, ActionType
)
from streak_multiplier_strategy import (
    StreakMultiplierStrategy,
    MartingaleStrategy,
    ReverseMartingaleStrategy,
    CustomRuleStrategy
)
from logger_utils import init_logger, logLev

logger = init_logger(name="test_strategies", level=logLev.DEBUG, is_orc=False)


class TestBettingState(unittest.TestCase):
    """Test BettingState functionality"""
    
    def setUp(self):
        self.state = BettingState(
            current_bet=10,
            current_bankroll=1000,
            win_probability=0.5,
            expected_return=2.0
        )
        
    def test_initial_state(self):
        """Test initial state values"""
        self.assertEqual(self.state.current_bet, 10)
        self.assertEqual(self.state.current_bankroll, 1000)
        self.assertEqual(self.state.streak_count, 0)
        self.assertIsNone(self.state.last_event)
        
    def test_update_stats_win(self):
        """Test state update on win"""
        self.state.update_stats(won=True, amount=10)
        self.assertEqual(self.state.current_bankroll, 1010)
        self.assertEqual(self.state.total_wins, 1)
        self.assertEqual(self.state.total_losses, 0)
        self.assertEqual(self.state.total_rounds, 1)
        
    def test_update_stats_loss(self):
        """Test state update on loss"""
        self.state.update_stats(won=False, amount=10)
        self.assertEqual(self.state.current_bankroll, 990)
        self.assertEqual(self.state.total_wins, 0)
        self.assertEqual(self.state.total_losses, 1)
        self.assertEqual(self.state.total_rounds, 1)
        
    def test_max_min_tracking(self):
        """Test max/min bankroll tracking"""
        self.state.update_stats(won=True, amount=100)
        self.assertEqual(self.state.max_bankroll, 1100)
        
        self.state.update_stats(won=False, amount=200)
        self.assertEqual(self.state.min_bankroll, 900)


class TestConditions(unittest.TestCase):
    """Test Condition evaluation"""
    
    def test_simple_event_condition(self):
        """Test condition that only checks event type"""
        condition = Condition(event_type=EventType.WIN)
        state = BettingState(
            current_bet=10,
            current_bankroll=1000,
            win_probability=0.5,
            expected_return=2.0
        )
        
        self.assertTrue(condition.evaluate(state, EventType.WIN))
        self.assertFalse(condition.evaluate(state, EventType.LOSE))
        
    def test_streak_condition(self):
        """Test streak-based conditions"""
        condition = Condition(
            event_type=EventType.LOSE,
            streak_operator=ComparisonOperator.GREATER,
            streak_threshold=3
        )
        
        state = BettingState(
            current_bet=10,
            current_bankroll=1000,
            win_probability=0.5,
            expected_return=2.0,
            streak_count=4,
            last_event=EventType.LOSE
        )
        
        # Should match: LOSE event with streak > 3
        self.assertTrue(condition.evaluate(state, EventType.LOSE))
        
        # Should not match: streak not > 3
        state.streak_count = 2
        self.assertFalse(condition.evaluate(state, EventType.LOSE))
        
        # Should not match: wrong event type
        state.streak_count = 4
        self.assertFalse(condition.evaluate(state, EventType.WIN))


class TestActions(unittest.TestCase):
    """Test Action execution"""
    
    def setUp(self):
        self.state = BettingState(
            current_bet=10,
            current_bankroll=1000,
            win_probability=0.5,
            expected_return=2.0
        )
        
    def test_set_bet_action(self):
        """Test SET_BET action"""
        action = Action(action_type=ActionType.SET_BET, value=25)
        action.execute(self.state, base_bet=10, base_win_prob=0.5)
        self.assertEqual(self.state.current_bet, 25)
        
    def test_multiply_bet_action(self):
        """Test MULTIPLY_BET action"""
        action = Action(action_type=ActionType.MULTIPLY_BET, value=2.0)
        action.execute(self.state, base_bet=10, base_win_prob=0.5)
        self.assertEqual(self.state.current_bet, 20)
        
    def test_reset_bet_action(self):
        """Test RESET_BET action"""
        self.state.current_bet = 50
        action = Action(action_type=ActionType.RESET_BET)
        action.execute(self.state, base_bet=10, base_win_prob=0.5)
        self.assertEqual(self.state.current_bet, 10)
        
    def test_win_prob_actions(self):
        """Test win probability actions"""
        action = Action(action_type=ActionType.SET_WIN_PROB, value=0.6)
        action.execute(self.state, base_bet=10, base_win_prob=0.5)
        self.assertEqual(self.state.win_probability, 0.6)
        
        action = Action(action_type=ActionType.RESET_WIN_PROB)
        action.execute(self.state, base_bet=10, base_win_prob=0.5)
        self.assertEqual(self.state.win_probability, 0.5)


class TestStreakMultiplierStrategy(unittest.TestCase):
    """Test StreakMultiplierStrategy"""
    
    def setUp(self):
        self.strategy = StreakMultiplierStrategy(
            starting_bankroll=1000,
            base_bet=10,
            target_bankroll=2000,
            win_probability=0.5,
            expected_return=2.0,
            streak_threshold=3,
            multiplier1=2.0,
            multiplier2=1.5
        )
        
    def test_initialization(self):
        """Test strategy initialization"""
        self.assertEqual(self.strategy.starting_bankroll, 1000)
        self.assertEqual(self.strategy.base_bet, 10)
        self.assertEqual(len(self.strategy.rules), 3)
        
    def test_win_resets_bet(self):
        """Test that winning resets bet to base"""
        self.strategy.state.current_bet = 40
        self.strategy.process_round(won=True)
        self.assertEqual(self.strategy.state.current_bet, 10)
        
    def test_loss_below_threshold(self):
        """Test multiplier1 applied when streak < threshold"""
        # First loss
        self.strategy.process_round(won=False)
        self.assertEqual(self.strategy.state.current_bet, 20)  # 10 * 2.0
        
        # Second loss
        self.strategy.process_round(won=False)
        self.assertEqual(self.strategy.state.current_bet, 40)  # 20 * 2.0
        
    def test_loss_above_threshold(self):
        """Test multiplier2 applied when streak >= threshold"""
        # Build up streak to threshold
        for _ in range(3):
            self.strategy.process_round(won=False)
            
        # At threshold, should use multiplier2
        current_bet = self.strategy.state.current_bet
        self.strategy.process_round(won=False)
        self.assertEqual(self.strategy.state.current_bet, current_bet * 1.5)
        
    def test_bet_capped_at_bankroll(self):
        """Test that bet never exceeds bankroll"""
        self.strategy.state.current_bankroll = 15
        self.strategy.state.current_bet = 20
        self.strategy.process_round(won=False)
        self.assertLessEqual(self.strategy.state.current_bet, 15)


class TestMartingaleStrategy(unittest.TestCase):
    """Test classic Martingale strategy"""
    
    def setUp(self):
        self.strategy = MartingaleStrategy(
            starting_bankroll=1000,
            base_bet=10,
            target_bankroll=2000,
            win_probability=0.5,
            expected_return=2.0,
            multiplier=2.0
        )
        
    def test_double_on_loss(self):
        """Test that bet doubles on loss"""
        self.strategy.process_round(won=False)
        self.assertEqual(self.strategy.state.current_bet, 20)
        
        self.strategy.process_round(won=False)
        self.assertEqual(self.strategy.state.current_bet, 40)
        
    def test_reset_on_win(self):
        """Test that bet resets on win"""
        self.strategy.state.current_bet = 80
        self.strategy.process_round(won=True)
        self.assertEqual(self.strategy.state.current_bet, 10)


class TestReverseMartingaleStrategy(unittest.TestCase):
    """Test Reverse Martingale (Paroli) strategy"""
    
    def setUp(self):
        self.strategy = ReverseMartingaleStrategy(
            starting_bankroll=1000,
            base_bet=10,
            target_bankroll=2000,
            win_probability=0.5,
            expected_return=2.0,
            multiplier=2.0,
            max_streak=3
        )
        
    def test_double_on_win(self):
        """Test that bet doubles on win (up to max_streak)"""
        self.strategy.process_round(won=True)
        self.assertEqual(self.strategy.state.current_bet, 20)
        
        self.strategy.process_round(won=True)
        self.assertEqual(self.strategy.state.current_bet, 40)
        
    def test_reset_on_loss(self):
        """Test that bet resets on loss"""
        self.strategy.state.current_bet = 80
        self.strategy.process_round(won=False)
        self.assertEqual(self.strategy.state.current_bet, 10)
        
    def test_max_streak_limit(self):
        """Test that bet resets after max_streak wins"""
        # Win 3 times (max_streak)
        for _ in range(3):
            self.strategy.process_round(won=True)
            
        # Should reset on next win
        self.strategy.process_round(won=True)
        self.assertEqual(self.strategy.state.current_bet, 10)


class TestCustomRuleStrategy(unittest.TestCase):
    """Test custom rule strategy"""
    
    def test_custom_rules(self):
        """Test strategy with custom rules"""
        # Create custom rules
        rules = [
            Rule(
                condition=Condition(event_type=EventType.WIN),
                action=Action(action_type=ActionType.MULTIPLY_BET, value=1.5),
                priority=10
            ),
            Rule(
                condition=Condition(event_type=EventType.LOSE),
                action=Action(action_type=ActionType.ADD_BET, value=5),
                priority=10
            )
        ]
        
        strategy = CustomRuleStrategy(
            starting_bankroll=1000,
            base_bet=10,
            target_bankroll=2000,
            win_probability=0.5,
            expected_return=2.0,
            rules=rules
        )
        
        # Test win multiplies by 1.5
        strategy.process_round(won=True)
        self.assertEqual(strategy.state.current_bet, 15)
        
        # Test loss adds 5
        strategy.state.current_bet = 10
        strategy.process_round(won=False)
        self.assertEqual(strategy.state.current_bet, 15)


if __name__ == '__main__':
    unittest.main()
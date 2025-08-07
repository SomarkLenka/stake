from typing import List, Tuple
from base_strategy import (
    BaseStrategy, Condition, Action, Rule,
    EventType, ComparisonOperator, ActionType
)
from logger_utils import init_logger, logLev

# Only initialize logger in main process, not in spawned processes
import multiprocessing
if multiprocessing.current_process().name == 'MainProcess':
    logger = init_logger(name="streak_multiplier_strategy", level=logLev.INFO, is_orc=False)
else:
    import logging
    logger = logging.getLogger("streak_multiplier_strategy")


class StreakMultiplierStrategy(BaseStrategy):
    """
    Strategy that multiplies bet based on losing streaks:
    - On win: reset bet to base_bet
    - On lose where streak < X: multiply bet by multiplier1
    - On lose where streak >= X: multiply bet by multiplier2
    - After max_loss_streak losses: reset bet and streak (stop-loss)
    """
    
    def __init__(
        self,
        starting_bankroll: float,
        base_bet: float,
        target_bankroll: float,
        win_probability: float = 0.495,
        expected_return: float = 2.0,
        streak_threshold: int = 3,
        multiplier1: float = 2.0,
        multiplier2: float = 1.5,
        max_loss_streak: int = None,  # Stop-loss: reset after this many consecutive losses
        *args,
        **kwargs
    ):
        self.streak_threshold = streak_threshold
        self.multiplier1 = multiplier1
        self.multiplier2 = multiplier2
        self.max_loss_streak = max_loss_streak  # None means no stop-loss
        
        super().__init__(
            starting_bankroll,
            base_bet,
            target_bankroll,
            win_probability,
            expected_return,
            *args,
            **kwargs
        )
        
        # logger.info(f"Initialized StreakMultiplierStrategy with threshold={streak_threshold}, "
        #            f"multiplier1={multiplier1}, multiplier2={multiplier2}")
        
    def _initialize_rules(self):
        """Initialize the streak-based multiplier rules"""
        
        # Rule 1: On win, reset bet to base_bet (highest priority)
        self.rules.append(Rule(
            condition=Condition(event_type=EventType.WIN),
            action=Action(action_type=ActionType.RESET_BET),
            priority=10
        ))
        
        # Rule 2: Stop-loss - On lose with streak >= max_loss_streak, reset everything
        if self.max_loss_streak is not None:
            self.rules.append(Rule(
                condition=Condition(
                    event_type=EventType.LOSE,
                    streak_operator=ComparisonOperator.GREATER_EQUAL,
                    streak_threshold=self.max_loss_streak
                ),
                action=Action(action_type=ActionType.RESET_BET),
                priority=8  # Higher priority than multiplier rules
            ))
        
        # Rule 3: On lose with streak >= threshold (but < max_loss_streak), multiply by multiplier2
        self.rules.append(Rule(
            condition=Condition(
                event_type=EventType.LOSE,
                streak_operator=ComparisonOperator.GREATER_EQUAL,
                streak_threshold=self.streak_threshold
            ),
            action=Action(
                action_type=ActionType.MULTIPLY_BET,
                value=self.multiplier2  # Conservative multiplier for long streaks
            ),
            priority=5
        ))
        
        # Rule 4: On lose with streak < threshold, multiply by multiplier1 (aggressive)
        self.rules.append(Rule(
            condition=Condition(
                event_type=EventType.LOSE,
                streak_operator=ComparisonOperator.LESS,
                streak_threshold=self.streak_threshold
            ),
            action=Action(
                action_type=ActionType.MULTIPLY_BET,
                value=self.multiplier1  # Aggressive multiplier for short streaks
            ),
            priority=4
        ))
        
        # logger.debug(f"Initialized {len(self.rules)} rules for StreakMultiplierStrategy")
    
    def process_round(self, won: bool) -> Tuple[float, float]:
        """Override to handle stop-loss streak reset"""
        event = EventType.WIN if won else EventType.LOSE
        
        # Check if this loss would exceed stop-loss threshold
        # If we're about to hit the stop-loss on THIS loss
        if (self.max_loss_streak is not None and 
            event == EventType.LOSE and 
            self.state.last_event == EventType.LOSE and
            self.state.streak_count >= self.max_loss_streak - 1):
            # After this loss processes with the stop-loss rule,
            # we want the NEXT loss to start fresh
            # The parent will increment to max_loss_streak, triggering reset
            # Then we override streak back to 0 for next round
            result = super().process_round(won)
            # After stop-loss reset, set streak to 0 so next loss starts at 1
            if self.state.current_bet == self.base_bet:
                self.state.streak_count = 0
            return result
        
        # Normal processing
        return super().process_round(won)


class MartingaleStrategy(BaseStrategy):
    """Classic Martingale strategy - double on loss, reset on win"""
    
    def __init__(
        self,
        starting_bankroll: float,
        base_bet: float,
        target_bankroll: float,
        win_probability: float,
        expected_return: float,
        multiplier: float = 2.0,
        *args,
        **kwargs
    ):
        self.multiplier = multiplier
        super().__init__(
            starting_bankroll,
            base_bet,
            target_bankroll,
            win_probability,
            expected_return,
            *args,
            **kwargs
        )
        
    def _initialize_rules(self):
        """Initialize classic Martingale rules"""
        
        # On win, reset bet
        self.rules.append(Rule(
            condition=Condition(event_type=EventType.WIN),
            action=Action(action_type=ActionType.RESET_BET),
            priority=10
        ))
        
        # On loss, multiply bet
        self.rules.append(Rule(
            condition=Condition(event_type=EventType.LOSE),
            action=Action(
                action_type=ActionType.MULTIPLY_BET,
                value=self.multiplier
            ),
            priority=5
        ))


class ReverseMartingaleStrategy(BaseStrategy):
    """Reverse Martingale (Paroli) - double on win, reset on loss"""
    
    def __init__(
        self,
        starting_bankroll: float,
        base_bet: float,
        target_bankroll: float,
        win_probability: float,
        expected_return: float,
        multiplier: float = 2.0,
        max_streak: int = 3,
        *args,
        **kwargs
    ):
        self.multiplier = multiplier
        self.max_streak = max_streak
        super().__init__(
            starting_bankroll,
            base_bet,
            target_bankroll,
            win_probability,
            expected_return,
            *args,
            **kwargs
        )
        
    def _initialize_rules(self):
        """Initialize reverse Martingale rules"""
        
        # On loss, reset bet
        self.rules.append(Rule(
            condition=Condition(event_type=EventType.LOSE),
            action=Action(action_type=ActionType.RESET_BET),
            priority=10
        ))
        
        # On win with streak >= max_streak, reset bet
        self.rules.append(Rule(
            condition=Condition(
                event_type=EventType.WIN,
                streak_operator=ComparisonOperator.GREATER_EQUAL,
                streak_threshold=self.max_streak
            ),
            action=Action(action_type=ActionType.RESET_BET),
            priority=8
        ))
        
        # On win with streak < max_streak, multiply bet
        self.rules.append(Rule(
            condition=Condition(
                event_type=EventType.WIN,
                streak_operator=ComparisonOperator.LESS,
                streak_threshold=self.max_streak
            ),
            action=Action(
                action_type=ActionType.MULTIPLY_BET,
                value=self.multiplier
            ),
            priority=5
        ))


class CustomRuleStrategy(BaseStrategy):
    """Fully customizable strategy where rules are passed in"""
    
    def __init__(
        self,
        starting_bankroll: float,
        base_bet: float,
        target_bankroll: float,
        win_probability: float,
        expected_return: float,
        rules: List[Rule] = None,
        *args,
        **kwargs
    ):
        self.custom_rules = rules or []
        super().__init__(
            starting_bankroll,
            base_bet,
            target_bankroll,
            win_probability,
            expected_return,
            *args,
            **kwargs
        )
        
    def _initialize_rules(self):
        """Use custom rules passed in"""
        self.rules = self.custom_rules
        # logger.info(f"Initialized CustomRuleStrategy with {len(self.rules)} custom rules")
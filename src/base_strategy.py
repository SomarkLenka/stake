from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from logger_utils import init_logger, logLev

# Only initialize logger in main process, not in spawned processes
import multiprocessing
if multiprocessing.current_process().name == 'MainProcess':
    logger = init_logger(name="base_strategy", level=logLev.INFO, is_orc=False)
else:
    import logging
    logger = logging.getLogger("base_strategy")


class EventType(Enum):
    WIN = "win"
    LOSE = "lose"
    
    
class ComparisonOperator(Enum):
    GREATER = ">"
    LESS = "<"
    EQUAL = "="
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    

class ActionType(Enum):
    SET_BET = "set_bet"
    MULTIPLY_BET = "multiply_bet"
    ADD_BET = "add_bet"
    RESET_BET = "reset_bet"
    SET_WIN_PROB = "set_win_prob"
    MULTIPLY_WIN_PROB = "multiply_win_prob"
    ADD_WIN_PROB = "add_win_prob"
    RESET_WIN_PROB = "reset_win_prob"


@dataclass
class BettingState:
    """Represents the current state of betting"""
    current_bet: float
    current_bankroll: float
    win_probability: float
    expected_return: float
    streak_count: int = 0
    last_event: Optional[EventType] = None
    total_rounds: int = 0
    total_wins: int = 0
    total_losses: int = 0
    max_bankroll: float = 0
    min_bankroll: float = float('inf')
    
    def update_stats(self, won: bool, amount: float):
        """Update statistics after a round"""
        self.total_rounds += 1
        if won:
            self.total_wins += 1
            self.current_bankroll += amount
        else:
            self.total_losses += 1
            self.current_bankroll -= amount
            
        self.max_bankroll = max(self.max_bankroll, self.current_bankroll)
        self.min_bankroll = min(self.min_bankroll, self.current_bankroll)
        
        
@dataclass
class Condition:
    """Represents a condition that triggers an action"""
    event_type: EventType
    streak_operator: Optional[ComparisonOperator] = None
    streak_threshold: Optional[int] = None
    
    def evaluate(self, state: BettingState, event: EventType) -> bool:
        """Check if condition is met"""
        if event != self.event_type:
            return False
            
        if self.streak_operator is None:
            return True
            
        # Prevent streak evaluation when switching event types
        # This ensures a "lose streak >= 3" condition won't trigger on a win
        if state.last_event != event:
            return False
            
        return self._compare_streak(state.streak_count, self.streak_threshold)
        
    def _compare_streak(self, streak: int, threshold: int) -> bool:
        """Compare streak against threshold using operator"""
        if self.streak_operator == ComparisonOperator.GREATER:
            return streak > threshold
        elif self.streak_operator == ComparisonOperator.LESS:
            return streak < threshold
        elif self.streak_operator == ComparisonOperator.EQUAL:
            return streak == threshold
        elif self.streak_operator == ComparisonOperator.GREATER_EQUAL:
            return streak >= threshold
        elif self.streak_operator == ComparisonOperator.LESS_EQUAL:
            return streak <= threshold
        return False
        

@dataclass 
class Action:
    """Represents an action to take when condition is met"""
    action_type: ActionType
    value: Optional[float] = None
    
    def execute(self, state: BettingState, base_bet: float, base_win_prob: float) -> None:
        """Execute the action on the betting state"""
        if self.action_type == ActionType.SET_BET:
            state.current_bet = self.value
        elif self.action_type == ActionType.MULTIPLY_BET:
            state.current_bet *= self.value
        elif self.action_type == ActionType.ADD_BET:
            state.current_bet += self.value
        elif self.action_type == ActionType.RESET_BET:
            state.current_bet = base_bet
        elif self.action_type == ActionType.SET_WIN_PROB:
            state.win_probability = self.value
            self._adjust_expected_return(state)
        elif self.action_type == ActionType.MULTIPLY_WIN_PROB:
            state.win_probability *= self.value
            self._adjust_expected_return(state)
        elif self.action_type == ActionType.ADD_WIN_PROB:
            state.win_probability += self.value
            self._adjust_expected_return(state)
        elif self.action_type == ActionType.RESET_WIN_PROB:
            state.win_probability = base_win_prob
            self._adjust_expected_return(state)
            
        # logger.debug(f"Executed action: {self.action_type} with value {self.value}")
        
    def _adjust_expected_return(self, state: BettingState):
        """Adjust expected return based on new win probability"""
        # Simple linear adjustment - can be customized
        base_return = 2.0  # Assuming 2x return as base
        state.expected_return = base_return * (0.5 / state.win_probability)
        

@dataclass
class Rule:
    """Combines a condition with an action"""
    condition: Condition
    action: Action
    priority: int = 0
    
    
class BaseStrategy(ABC):
    """Abstract base class for betting strategies"""
    
    def __init__(
        self,
        starting_bankroll: float,
        base_bet: float,
        target_bankroll: float,
        win_probability: float,
        expected_return: float,
        *args,
        **kwargs
    ):
        self.starting_bankroll = starting_bankroll
        self.base_bet = base_bet
        self.target_bankroll = target_bankroll
        self.base_win_probability = win_probability
        self.base_expected_return = expected_return
        
        # Initialize state
        self.state = BettingState(
            current_bet=base_bet,
            current_bankroll=starting_bankroll,
            win_probability=win_probability,
            expected_return=expected_return,
            max_bankroll=starting_bankroll,
            min_bankroll=starting_bankroll
        )
        
        # Rules will be defined by subclasses
        self.rules: List[Rule] = []
        
        # Process additional args/kwargs
        self._process_additional_params(*args, **kwargs)
        
        # Initialize strategy-specific rules
        self._initialize_rules()
        
        # logger.info(f"Initialized {self.__class__.__name__} strategy")
        
    @abstractmethod
    def _initialize_rules(self):
        """Initialize the rules for this strategy"""
        pass
        
    def _process_additional_params(self, *args, **kwargs):
        """Process strategy-specific parameters"""
        pass
        
    def process_round(self, won: bool) -> Tuple[float, float]:
        """Process a betting round and return (bet_amount, new_bankroll)"""
        event = EventType.WIN if won else EventType.LOSE
        
        # Update streak
        # Note: We set streak_count = 1 (not 0) when event changes because:
        # 1. The first occurrence IS a streak of 1
        # 2. Allows conditions like "on any win" (streak >= 1) to work
        # 3. Condition.evaluate() already prevents cross-event streak checks
        if self.state.last_event == event:
            self.state.streak_count += 1
        else:
            self.state.streak_count = 1
        self.state.last_event = event
        
        # Calculate winnings/losses
        if won:
            amount = self.state.current_bet * (self.state.expected_return - 1)
        else:
            amount = self.state.current_bet
            
        # Update state
        self.state.update_stats(won, amount)
        
        # Apply rules in priority order
        for rule in sorted(self.rules, key=lambda r: r.priority, reverse=True):
            if rule.condition.evaluate(self.state, event):
                rule.action.execute(
                    self.state,
                    self.base_bet,
                    self.base_win_probability
                )
                break  # Only apply first matching rule
                
        # Ensure bet doesn't exceed bankroll
        self.state.current_bet = min(self.state.current_bet, self.state.current_bankroll)
        
        # logger.debug(f"Round {self.state.total_rounds}: {'WIN' if won else 'LOSE'}, "
        #             f"Bankroll: {self.state.current_bankroll:.2f}, "
        #             f"Next bet: {self.state.current_bet:.2f}")
        
        return self.state.current_bet, self.state.current_bankroll
        
    def reset(self):
        """Reset strategy to initial state"""
        self.state = BettingState(
            current_bet=self.base_bet,
            current_bankroll=self.starting_bankroll,
            win_probability=self.base_win_probability,
            expected_return=self.base_expected_return,
            max_bankroll=self.starting_bankroll,
            min_bankroll=self.starting_bankroll
        )
        # logger.debug("Strategy reset to initial state")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            "total_rounds": self.state.total_rounds,
            "total_wins": self.state.total_wins,
            "total_losses": self.state.total_losses,
            "current_bankroll": self.state.current_bankroll,
            "max_bankroll": self.state.max_bankroll,
            "min_bankroll": self.state.min_bankroll,
            "win_rate": self.state.total_wins / max(1, self.state.total_rounds),
            "current_bet": self.state.current_bet,
            "current_streak": self.state.streak_count,
            "last_event": self.state.last_event.value if self.state.last_event else None
        }
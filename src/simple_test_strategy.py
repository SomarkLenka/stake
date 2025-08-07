"""Simple test strategy for Monte Carlo engine testing"""

from base_strategy import BaseStrategy
from logger_utils import init_logger, logLev

logger = init_logger(name="simple_test_strategy", level=logLev.INFO, is_orc=False)


class SimpleTestStrategy(BaseStrategy):
    """
    A simple strategy that maintains a fixed fraction of bankroll as bet size.
    Compatible with Monte Carlo engine's parameter passing.
    """
    
    def __init__(
        self,
        initial_bankroll: float = 10000,
        target_bankroll: float = 20000,
        win_probability: float = 0.49,
        expected_return: float = 2.0,
        base_bet_fraction: float = 0.02,
        **kwargs
    ):
        """Initialize with common parameter names used by Monte Carlo engine"""
        
        # Calculate base bet from fraction
        base_bet = initial_bankroll * base_bet_fraction
        
        # Call parent with expected parameters
        super().__init__(
            starting_bankroll=initial_bankroll,
            base_bet=base_bet,
            target_bankroll=target_bankroll,
            win_probability=win_probability,
            expected_return=expected_return,
            **kwargs
        )
        
        self.base_bet_fraction = base_bet_fraction
        logger.debug(f"Initialized SimpleTestStrategy with bankroll={initial_bankroll}, "
                    f"bet_fraction={base_bet_fraction}, target={target_bankroll}")
    
    def get_custom_rules(self) -> list:
        """Return empty rules list for simple fixed betting"""
        return []
    
    def _initialize_rules(self) -> list:
        """Return empty rules list for simple fixed betting"""
        return []
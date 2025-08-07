#!/usr/bin/env python3
"""
Probability Engine - Main orchestrator for Monte Carlo betting simulations
"""

import argparse
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from monte_carlo_engine import MonteCarloEngine, SimulationSummary
from streak_multiplier_strategy import (
    StreakMultiplierStrategy,
    MartingaleStrategy,
    ReverseMartingaleStrategy,
    CustomRuleStrategy
)
from base_strategy import (
    Condition, Action, Rule,
    EventType, ComparisonOperator, ActionType
)
from logger_utils import init_logger, logLev

# Only initialize logger in main process, not in spawned processes
import multiprocessing
if multiprocessing.current_process().name == 'MainProcess':
    logger = init_logger(name="probability_engine", level=logLev.INFO, is_orc=True)
else:
    import logging
    logger = logging.getLogger("probability_engine")


AVAILABLE_STRATEGIES = {
    "streak_multiplier": StreakMultiplierStrategy,
    "martingale": MartingaleStrategy,
    "reverse_martingale": ReverseMartingaleStrategy,
    "custom": CustomRuleStrategy
}


class ProbabilityEngine:
    """Main engine for running probability simulations"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = {}
        if config_file:
            self.load_config(config_file)
        logger.info("Initialized ProbabilityEngine")
        
    def load_config(self, config_file: str):
        """Load configuration from JSON file"""
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        logger.info(f"Loaded configuration from {config_file}")
        
    def create_strategy(
        self,
        strategy_type: str,
        starting_bankroll: float,
        base_bet: float,
        target_bankroll: float,
        win_probability: float,
        expected_return: float,
        **kwargs
    ):
        """Create a strategy instance"""
        if strategy_type not in AVAILABLE_STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy_type}. "
                           f"Available: {list(AVAILABLE_STRATEGIES.keys())}")
        
        strategy_class = AVAILABLE_STRATEGIES[strategy_type]
        
        return strategy_class(
            starting_bankroll=starting_bankroll,
            base_bet=base_bet,
            target_bankroll=target_bankroll,
            win_probability=win_probability,
            expected_return=expected_return,
            **kwargs
        )
        
    def run_simulation(
        self,
        strategy_type: str = "streak_multiplier",
        starting_bankroll: float = 1500,
        base_bet: float = 10,
        target_bankroll: float = 4500,
        win_probability: float = 0.495,
        expected_return: float = 2.0,
        num_simulations: int = 500000,
        max_rounds: int = 500000,
        strategy_params: Dict[str, Any] = None,
        **kwargs
    ) -> SimulationSummary:
        """Run a Monte Carlo simulation with given parameters"""
        
        strategy_params = strategy_params or {}
        
        # Merge all parameters
        all_params = {
            'starting_bankroll': starting_bankroll,
            'base_bet': base_bet,
            'target_bankroll': target_bankroll,
            'win_probability': win_probability,
            'expected_return': expected_return,
            **strategy_params,
            **kwargs
        }
        
        logger.info(f"Running simulation with strategy: {strategy_type}")
        logger.info(f"Parameters: bankroll=${starting_bankroll}, bet=${base_bet}, "
                   f"target=${target_bankroll}, P(win)={win_probability:.2%}")
        
        # Get strategy class
        if strategy_type not in AVAILABLE_STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy_type}")
        
        strategy_class = AVAILABLE_STRATEGIES[strategy_type]
        
        # Create engine
        engine = MonteCarloEngine(
            strategy_class=strategy_class,
            strategy_params=all_params,
            max_rounds=max_rounds,
            stop_on_bankruptcy=True,
            stop_on_target=True
        )
        
        # Run simulations
        summary = engine.run_simulations(
            num_simulations=num_simulations,
            parallel=True
        )
        
        return summary
        
    def compare_strategies(
        self,
        strategies: List[Dict[str, Any]],
        base_params: Dict[str, Any],
        num_simulations: int = 1000000
    ) -> Dict[str, SimulationSummary]:
        """Compare multiple strategies with same base parameters"""
        
        results = {}
        
        for strategy_config in strategies:
            strategy_name = strategy_config.get('name', strategy_config['type'])
            strategy_type = strategy_config['type']
            strategy_params = strategy_config.get('params', {})
            
            logger.info(f"Testing strategy: {strategy_name}")
            
            summary = self.run_simulation(
                strategy_type=strategy_type,
                num_simulations=num_simulations,
                strategy_params=strategy_params,
                **base_params
            )
            
            results[strategy_name] = summary
            print(f"\nStrategy: {strategy_name}")
            print(summary)
            
        return results
        
    def optimize_parameters(
        self,
        strategy_type: str,
        base_params: Dict[str, Any],
        param_ranges: Dict[str, List[Any]],
        num_simulations_per: int = 100000,
        target_metric: str = "survival_rate"
    ) -> Dict[tuple, SimulationSummary]:
        """Optimize strategy parameters by testing different combinations"""
        
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        combinations = list(product(*param_values))
        
        logger.info(f"Testing {len(combinations)} parameter combinations")
        
        results = {}
        best_result = None
        best_params = None
        best_metric = 0
        
        for combo in combinations:
            # Create parameter dict
            test_params = dict(zip(param_names, combo))
            
            logger.info(f"Testing parameters: {test_params}")
            
            # Run simulation
            summary = self.run_simulation(
                strategy_type=strategy_type,
                num_simulations=num_simulations_per,
                strategy_params=test_params,
                **base_params
            )
            
            # Store result
            results[tuple(combo)] = summary
            
            # Check if this is the best so far
            metric_value = getattr(summary, target_metric)
            if metric_value > best_metric:
                best_metric = metric_value
                best_result = summary
                best_params = test_params
                
            logger.info(f"  → {target_metric}: {metric_value:.2%}")
            
        logger.info(f"\nBest parameters: {best_params}")
        logger.info(f"Best {target_metric}: {best_metric:.2%}")
        
        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Probability Engine for Betting Simulations")
    
    parser.add_argument(
        '--strategy', '-s',
        default='streak_multiplier',
        choices=list(AVAILABLE_STRATEGIES.keys()),
        help='Strategy to use'
    )
    
    parser.add_argument('--bankroll', '-b', type=float, default=1500,
                       help='Starting bankroll')
    parser.add_argument('--bet', type=float, default=10,
                       help='Base bet amount')
    parser.add_argument('--target', '-t', type=float, default=5000,
                       help='Target bankroll')
    parser.add_argument('--probability', '-p', type=float, default=0.495,
                       help='Win probability')
    parser.add_argument('--return', '-r', type=float, default=2.0,
                       dest='expected_return', help='Expected return multiplier')
    parser.add_argument('--simulations', '-n', type=int, default=100000,
                       help='Number of simulations')
    parser.add_argument('--max-rounds', type=int, default=100000,
                       help='Maximum rounds per simulation')
    
    # Strategy-specific parameters
    parser.add_argument('--streak-threshold', type=int, default=3,
                       help='Streak threshold for streak_multiplier strategy')
    parser.add_argument('--multiplier1', type=float, default=2.0,
                       help='Multiplier for losses below threshold')
    parser.add_argument('--multiplier2', type=float, default=1.5,
                       help='Multiplier for losses above threshold')
    parser.add_argument('--multiplier', type=float, default=2.0,
                       help='General multiplier for martingale strategies')
    
    # Operational parameters
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple strategies')
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize strategy parameters')
    parser.add_argument('--config', type=str,
                       help='Load configuration from JSON file')
    
    args = parser.parse_args()
    
    # Create engine
    engine = ProbabilityEngine(config_file=args.config)
    
    if args.compare:
        # Compare multiple strategies
        strategies = [
            {
                'name': 'Streak Multiplier (2x/1.5x)',
                'type': 'streak_multiplier',
                'params': {
                    'streak_threshold': 3,
                    'multiplier1': 2.0,
                    'multiplier2': 1.5
                }
            },
            {
                'name': 'Classic Martingale',
                'type': 'martingale',
                'params': {'multiplier': 2.0}
            },
            {
                'name': 'Reverse Martingale',
                'type': 'reverse_martingale',
                'params': {'multiplier': 2.0, 'max_streak': 3}
            }
        ]
        
        base_params = {
            'starting_bankroll': args.bankroll,
            'base_bet': args.bet,
            'target_bankroll': args.target,
            'win_probability': args.probability,
            'expected_return': args.expected_return,
            'max_rounds': args.max_rounds
        }
        
        results = engine.compare_strategies(
            strategies,
            base_params,
            num_simulations=args.simulations
        )
        
    elif args.optimize:
        # Optimize parameters
        base_params = {
            'starting_bankroll': args.bankroll,
            'base_bet': args.bet,
            'target_bankroll': args.target,
            'win_probability': args.probability,
            'expected_return': args.expected_return,
            'max_rounds': args.max_rounds
        }
        
        # Define parameter ranges to test
        param_ranges = {
            'streak_threshold': [2, 3, 4, 5],
            'multiplier1': [1.5, 2.0, 2.5],
            'multiplier2': [1.2, 1.5, 1.8]
        }
        
        results = engine.optimize_parameters(
            strategy_type=args.strategy,
            base_params=base_params,
            param_ranges=param_ranges,
            num_simulations_per=1000,
            target_metric='survival_rate'
        )
        
    else:
        # Single simulation
        strategy_params = {}
        
        if args.strategy == 'streak_multiplier':
            strategy_params = {
                'streak_threshold': args.streak_threshold,
                'multiplier1': args.multiplier1,
                'multiplier2': args.multiplier2
            }
        elif args.strategy in ['martingale', 'reverse_martingale']:
            strategy_params = {'multiplier': args.multiplier}
            
        summary = engine.run_simulation(
            strategy_type=args.strategy,
            starting_bankroll=args.bankroll,
            base_bet=args.bet,
            target_bankroll=args.target,
            win_probability=args.probability,
            expected_return=args.expected_return,
            num_simulations=args.simulations,
            max_rounds=args.max_rounds,
            strategy_params=strategy_params
        )
        
        print(summary)


if __name__ == "__main__":
    main()
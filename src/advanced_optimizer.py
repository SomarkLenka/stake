#!/usr/bin/env python3
"""
Advanced Parameter Optimizer for Betting Strategies
Optimizes: base_bet, multipliers, streak_threshold for target success or survival
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from itertools import product
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from probability_engine import ProbabilityEngine
from monte_carlo_engine import MonteCarloEngine
from streak_multiplier_strategy import StreakMultiplierStrategy
from logger_utils import init_logger, logLev
import time

logger = init_logger(name="advanced_optimizer", level=logLev.INFO, is_orc=True)


@dataclass
class OptimizationResult:
    """Result from an optimization run"""
    parameters: Dict[str, Any]
    target_success_rate: float
    survival_rate: float
    avg_final_bankroll: float
    avg_rounds: float
    percentile_75: float
    score: float  # Computed based on optimization goal
    
    def __str__(self):
        return (
            f"Parameters: {self.parameters}\n"
            f"  Target Success: {self.target_success_rate:.2%}\n"
            f"  Survival Rate: {self.survival_rate:.2%}\n"
            f"  Avg Bankroll: ${self.avg_final_bankroll:.2f}\n"
            f"  Score: {self.score:.4f}"
        )


@dataclass
class OptimizationConfig:
    """Configuration for optimization run"""
    # Fixed parameters
    starting_bankroll: float = 1000
    target_bankroll: float = 2000
    win_probability: float = 0.49
    expected_return: float = 2.0
    max_rounds: int = 10000
    
    # Parameter ranges to optimize
    base_bet_range: List[float] = None
    streak_threshold_range: List[int] = None
    multiplier1_range: List[float] = None  # For streak < threshold
    multiplier2_range: List[float] = None  # For streak >= threshold
    max_loss_streak_range: List[Optional[int]] = None  # Stop-loss after X losses
    
    # Optimization settings
    simulations_per_test: int = 500
    optimization_goal: str = "target_success"  # or "survival"
    max_workers: int = 4
    
    def __post_init__(self):
        """Set default ranges if not provided"""
        if self.base_bet_range is None:
            # Test bets from 0.5% to 5% of bankroll
            self.base_bet_range = [
                self.starting_bankroll * pct 
                for pct in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
            ]
        
        if self.streak_threshold_range is None:
            self.streak_threshold_range = [2, 3, 4, 5, 6]
            
        if self.multiplier1_range is None:
            # Aggressive multipliers for short streaks
            self.multiplier1_range = [1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
            
        if self.multiplier2_range is None:
            # Conservative multipliers for long streaks
            self.multiplier2_range = [1.1, 1.2, 1.3, 1.4, 1.5, 1.75]
            
        if self.max_loss_streak_range is None:
            # Stop-loss thresholds (None means no stop-loss)
            self.max_loss_streak_range = [None, 6, 8, 10]


class AdvancedOptimizer:
    """Advanced parameter optimizer for betting strategies"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.engine = ProbabilityEngine()
        logger.info("Initialized AdvancedOptimizer")
        logger.info(f"Optimization goal: {self.config.optimization_goal}")
        
    def generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations to test"""
        combinations = []
        
        # Create all combinations
        for params in product(
            self.config.base_bet_range,
            self.config.streak_threshold_range,
            self.config.multiplier1_range,
            self.config.multiplier2_range,
            self.config.max_loss_streak_range
        ):
            base_bet, streak_threshold, mult1, mult2, max_loss_streak = params
            
            # Skip invalid combinations
            # mult1 should generally be >= mult2 (more aggressive early)
            if mult1 < mult2 - 0.5:  # Allow some flexibility
                continue
                
            # Skip if bet is too large relative to bankroll
            if base_bet > self.config.starting_bankroll * 0.1:
                continue
                
            # Additional validation for stop-loss
            if max_loss_streak is not None and max_loss_streak <= streak_threshold:
                continue  # Stop-loss should be after the threshold switch
            
            combinations.append({
                'base_bet': base_bet,
                'streak_threshold': streak_threshold,
                'multiplier1': mult1,
                'multiplier2': mult2,
                'max_loss_streak': max_loss_streak
            })
        
        logger.info(f"Generated {len(combinations)} parameter combinations to test")
        return combinations
    
    def evaluate_parameters(self, params: Dict[str, Any]) -> OptimizationResult:
        """Evaluate a single parameter combination"""
        try:
            # Run simulation with these parameters
            summary = self.engine.run_simulation(
                strategy_type="streak_multiplier",
                starting_bankroll=self.config.starting_bankroll,
                base_bet=params['base_bet'],
                target_bankroll=self.config.target_bankroll,
                win_probability=self.config.win_probability,
                expected_return=self.config.expected_return,
                num_simulations=self.config.simulations_per_test,
                max_rounds=self.config.max_rounds,
                streak_threshold=params['streak_threshold'],
                multiplier1=params['multiplier1'],
                multiplier2=params['multiplier2'],
                max_loss_streak=params.get('max_loss_streak')
            )
            
            # Calculate score based on optimization goal
            if self.config.optimization_goal == "target_success":
                # Prioritize reaching target, with survival as secondary
                score = summary.target_success_rate * 1.0 + summary.survival_rate * 0.2
            elif self.config.optimization_goal == "survival":
                # Prioritize survival, with target success as secondary
                score = summary.survival_rate * 1.0 + summary.target_success_rate * 0.3
            elif self.config.optimization_goal == "balanced":
                # Balance both goals
                score = summary.target_success_rate * 0.5 + summary.survival_rate * 0.5
            else:
                score = summary.target_success_rate
            
            return OptimizationResult(
                parameters=params,
                target_success_rate=summary.target_success_rate,
                survival_rate=summary.survival_rate,
                avg_final_bankroll=summary.average_final_bankroll,
                avg_rounds=summary.average_rounds,
                percentile_75=summary.percentile_75,
                score=score
            )
            
        except Exception as e:
            logger.error(f"Failed to evaluate parameters {params}: {e}")
            return OptimizationResult(
                parameters=params,
                target_success_rate=0.0,
                survival_rate=0.0,
                avg_final_bankroll=0.0,
                avg_rounds=0.0,
                percentile_75=0.0,
                score=0.0
            )
    
    def optimize(self, top_n: int = 10) -> List[OptimizationResult]:
        """
        Run optimization and return top N parameter combinations
        
        Args:
            top_n: Number of top results to return
            
        Returns:
            List of top optimization results sorted by score
        """
        logger.info("Starting parameter optimization...")
        start_time = time.time()
        
        # Generate all combinations
        combinations = self.generate_parameter_combinations()
        
        # Evaluate in parallel
        results = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all evaluations
            futures = {
                executor.submit(self.evaluate_parameters, params): params 
                for params in combinations
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=60)
                    results.append(result)
                    completed += 1
                    
                    # Progress reporting
                    if completed % 10 == 0 or completed == len(combinations):
                        pct = completed / len(combinations) * 100
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (len(combinations) - completed) / rate if rate > 0 else 0
                        
                        logger.info(
                            f"Progress: {completed}/{len(combinations)} ({pct:.1f}%) - "
                            f"Rate: {rate:.1f}/sec - ETA: {eta:.0f}s"
                        )
                        
                        # Show best so far
                        if results:
                            best_so_far = max(results, key=lambda x: x.score)
                            logger.info(f"Best score so far: {best_so_far.score:.4f}")
                            
                except Exception as e:
                    logger.error(f"Evaluation failed: {e}")
        
        # Sort by score and return top N
        results.sort(key=lambda x: x.score, reverse=True)
        top_results = results[:top_n]
        
        elapsed = time.time() - start_time
        logger.info(f"Optimization completed in {elapsed:.1f} seconds")
        logger.info(f"Evaluated {len(results)} parameter combinations")
        
        return top_results
    
    def optimize_for_target(self, top_n: int = 10) -> List[OptimizationResult]:
        """Optimize specifically for reaching target bankroll"""
        self.config.optimization_goal = "target_success"
        logger.info("Optimizing for TARGET SUCCESS RATE")
        return self.optimize(top_n)
    
    def optimize_for_survival(self, top_n: int = 10) -> List[OptimizationResult]:
        """Optimize specifically for survival rate"""
        self.config.optimization_goal = "survival"
        logger.info("Optimizing for SURVIVAL RATE")
        return self.optimize(top_n)
    
    def optimize_balanced(self, top_n: int = 10) -> List[OptimizationResult]:
        """Optimize for balance between target and survival"""
        self.config.optimization_goal = "balanced"
        logger.info("Optimizing for BALANCED performance")
        return self.optimize(top_n)
    
    def print_results(self, results: List[OptimizationResult], title: str = "Optimization Results"):
        """Print formatted optimization results"""
        print("\n" + "="*80)
        print(f" {title}")
        print("="*80)
        
        for i, result in enumerate(results, 1):
            print(f"\n#{i} - Score: {result.score:.4f}")
            print("-"*40)
            print(f"  Base Bet:          ${result.parameters['base_bet']:.2f} "
                  f"({result.parameters['base_bet']/self.config.starting_bankroll*100:.1f}% of bankroll)")
            print(f"  Streak Threshold:  {result.parameters['streak_threshold']} losses")
            print(f"  Multiplier 1:      {result.parameters['multiplier1']:.2f}x (streak < {result.parameters['streak_threshold']})")
            print(f"  Multiplier 2:      {result.parameters['multiplier2']:.2f}x (streak >= {result.parameters['streak_threshold']})")
            if result.parameters.get('max_loss_streak'):
                print(f"  Stop-Loss:         Reset after {result.parameters['max_loss_streak']} losses")
            else:
                print(f"  Stop-Loss:         Disabled")
            print(f"  Target Success:    {result.target_success_rate:.2%}")
            print(f"  Survival Rate:     {result.survival_rate:.2%}")
            print(f"  Avg Final Bank:    ${result.avg_final_bankroll:.2f}")
            print(f"  75th Percentile:   ${result.percentile_75:.2f}")
    
    def save_results(self, results: List[OptimizationResult], filename: str = "optimization_results.json"):
        """Save optimization results to JSON file"""
        data = {
            'config': {
                'starting_bankroll': self.config.starting_bankroll,
                'target_bankroll': self.config.target_bankroll,
                'win_probability': self.config.win_probability,
                'expected_return': self.config.expected_return,
                'optimization_goal': self.config.optimization_goal,
                'simulations_per_test': self.config.simulations_per_test
            },
            'results': [
                {
                    'rank': i,
                    'parameters': r.parameters,
                    'metrics': {
                        'target_success_rate': r.target_success_rate,
                        'survival_rate': r.survival_rate,
                        'avg_final_bankroll': r.avg_final_bankroll,
                        'avg_rounds': r.avg_rounds,
                        'percentile_75': r.percentile_75,
                        'score': r.score
                    }
                }
                for i, r in enumerate(results, 1)
            ],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {filename}")


def main():
    """Main entry point for optimizer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Parameter Optimizer for Betting Strategies")
    
    # Fixed parameters
    parser.add_argument('--bankroll', type=float, default=1000,
                       help='Starting bankroll')
    parser.add_argument('--target', type=float, default=2000,
                       help='Target bankroll')
    parser.add_argument('--probability', type=float, default=0.49,
                       help='Win probability')
    parser.add_argument('--return', type=float, default=2.0,
                       dest='expected_return', help='Expected return multiplier')
    
    # Optimization parameters
    parser.add_argument('--goal', choices=['target', 'survival', 'balanced'],
                       default='balanced', help='Optimization goal')
    parser.add_argument('--simulations', type=int, default=500,
                       help='Simulations per parameter test')
    parser.add_argument('--top', type=int, default=10,
                       help='Number of top results to show')
    parser.add_argument('--workers', type=int, default=4,
                       help='Parallel workers')
    
    # Parameter ranges (comma-separated)
    parser.add_argument('--bet-range', type=str,
                       help='Comma-separated bet amounts (e.g., "5,10,15,20")')
    parser.add_argument('--threshold-range', type=str,
                       help='Comma-separated streak thresholds (e.g., "2,3,4,5")')
    parser.add_argument('--mult1-range', type=str,
                       help='Comma-separated multiplier1 values (e.g., "1.5,2.0,2.5")')
    parser.add_argument('--mult2-range', type=str,
                       help='Comma-separated multiplier2 values (e.g., "1.2,1.5,1.8")')
    
    # Output
    parser.add_argument('--save', type=str, help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Create configuration
    config = OptimizationConfig(
        starting_bankroll=args.bankroll,
        target_bankroll=args.target,
        win_probability=args.probability,
        expected_return=args.expected_return,
        simulations_per_test=args.simulations,
        optimization_goal=args.goal,
        max_workers=args.workers
    )
    
    # Parse custom ranges if provided
    if args.bet_range:
        config.base_bet_range = [float(x) for x in args.bet_range.split(',')]
    if args.threshold_range:
        config.streak_threshold_range = [int(x) for x in args.threshold_range.split(',')]
    if args.mult1_range:
        config.multiplier1_range = [float(x) for x in args.mult1_range.split(',')]
    if args.mult2_range:
        config.multiplier2_range = [float(x) for x in args.mult2_range.split(',')]
    
    # Create optimizer
    optimizer = AdvancedOptimizer(config)
    
    # Run optimization based on goal
    if args.goal == 'target':
        results = optimizer.optimize_for_target(args.top)
    elif args.goal == 'survival':
        results = optimizer.optimize_for_survival(args.top)
    else:
        results = optimizer.optimize_balanced(args.top)
    
    # Display results
    optimizer.print_results(results, f"Top {args.top} Results - Optimized for {args.goal.upper()}")
    
    # Save if requested
    if args.save:
        optimizer.save_results(results, args.save)
    
    # Show best configuration
    if results:
        best = results[0]
        print("\n" + "="*80)
        print(" RECOMMENDED CONFIGURATION")
        print("="*80)
        print(f"""
To use the optimal parameters, run:

python probability_engine.py \\
    --strategy streak_multiplier \\
    --bankroll {config.starting_bankroll} \\
    --bet {best.parameters['base_bet']:.2f} \\
    --target {config.target_bankroll} \\
    --probability {config.win_probability} \\
    --return {config.expected_return} \\
    --streak-threshold {best.parameters['streak_threshold']} \\
    --multiplier1 {best.parameters['multiplier1']} \\
    --multiplier2 {best.parameters['multiplier2']} \\
    --max-loss-streak {best.parameters.get('max_loss_streak', 'none')} \\
    --simulations 10000

Expected Performance:
- Target Success Rate: {best.target_success_rate:.2%}
- Survival Rate: {best.survival_rate:.2%}
        """)


if __name__ == "__main__":
    main()
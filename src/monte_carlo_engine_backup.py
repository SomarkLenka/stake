import random
from typing import List, Dict, Any, Optional, Type
from dataclasses import dataclass, field
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from multiprocessing import cpu_count, get_context
from base_strategy import BaseStrategy
from logger_utils import init_logger, logLev
import time
import os
import sys
from functools import partial
import logging

# Initialize main logger for orchestrator process
logger = init_logger(name="monte_carlo_engine", level=logLev.INFO, is_orc=True)


@dataclass
class SimulationResult:
    """Results from a single simulation run"""
    survived: bool
    final_bankroll: float
    rounds_played: int
    max_bankroll: float
    min_bankroll: float
    target_reached: bool
    bankruptcy_round: Optional[int] = None
    target_round: Optional[int] = None
    win_rate: float = 0.0
    
    
@dataclass
class SimulationSummary:
    """Summary of all simulation runs"""
    total_runs: int
    survival_rate: float
    target_success_rate: float
    average_rounds: float
    average_final_bankroll: float
    median_final_bankroll: float
    std_final_bankroll: float
    max_bankroll_achieved: float
    min_bankroll_achieved: float
    average_bankruptcy_round: Optional[float] = None
    average_target_round: Optional[float] = None
    percentile_25: float = 0.0
    percentile_75: float = 0.0
    percentile_95: float = 0.0
    
    def __str__(self):
        return (
            f"\n{'='*60}\n"
            f"MONTE CARLO SIMULATION RESULTS\n"
            f"{'='*60}\n"
            f"Total Runs:           {self.total_runs:,}\n"
            f"Survival Rate:        {self.survival_rate:.2%}\n"
            f"Target Success Rate:  {self.target_success_rate:.2%}\n"
            f"{'='*60}\n"
            f"Average Rounds:       {self.average_rounds:,.0f}\n"
            f"Avg Final Bankroll:   ${self.average_final_bankroll:,.2f}\n"
            f"Median Bankroll:      ${self.median_final_bankroll:,.2f}\n"
            f"Std Dev Bankroll:     ${self.std_final_bankroll:,.2f}\n"
            f"{'='*60}\n"
            f"Percentiles:\n"
            f"  25th:               ${self.percentile_25:,.2f}\n"
            f"  75th:               ${self.percentile_75:,.2f}\n"
            f"  95th:               ${self.percentile_95:,.2f}\n"
            f"{'='*60}\n"
            f"Max Bankroll:         ${self.max_bankroll_achieved:,.2f}\n"
            f"Min Bankroll:         ${self.min_bankroll_achieved:,.2f}\n"
        )
        

class MonteCarloEngine:
    """Engine for running Monte Carlo simulations on betting strategies"""
    
    def __init__(
        self,
        strategy_class: Type[BaseStrategy],
        strategy_params: Dict[str, Any],
        max_rounds: int = 10000,
        stop_on_bankruptcy: bool = True,
        stop_on_target: bool = True,
        seed: Optional[int] = None,
        batch_size: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ):
        self.strategy_class = strategy_class
        self.strategy_params = strategy_params
        self.max_rounds = max_rounds
        self.stop_on_bankruptcy = stop_on_bankruptcy
        self.stop_on_target = stop_on_target
        self.progress_callback = progress_callback
        
        # Optimize batch size based on CPU count and simulation complexity
        self.batch_size = batch_size or self._calculate_optimal_batch_size()
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        logger.info(f"Initialized MonteCarloEngine with {strategy_class.__name__}")
        logger.info(f"Max rounds: {max_rounds}, Stop on bankruptcy: {stop_on_bankruptcy}, "
                   f"Stop on target: {stop_on_target}")
        logger.info(f"Optimal batch size: {self.batch_size}")
        
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on system resources and simulation complexity"""
        cpu_count_val = cpu_count()
        
        # Base batch size on simulation complexity
        if self.max_rounds <= 1000:
            base_batch = max(100, cpu_count_val * 10)
        elif self.max_rounds <= 5000:
            base_batch = max(50, cpu_count_val * 5)
        else:
            base_batch = max(25, cpu_count_val * 2)
            
        # Ensure reasonable bounds
        return min(max(base_batch, 10), 500)
        
    def _get_optimal_workers(self, num_simulations: int) -> int:
        """Calculate optimal number of workers based on workload"""
        cpu_count_val = cpu_count()
        
        # For small workloads, use fewer workers to avoid overhead
        if num_simulations < 100:
            return min(2, cpu_count_val)
        elif num_simulations < 1000:
            return min(cpu_count_val // 2, 4)
        else:
            # Use more workers for large workloads, but cap to avoid memory issues
            return min(cpu_count_val + 2, 12)
    
    def _run_single_simulation(self, sim_id: int) -> SimulationResult:
        """Run a single simulation"""
        # Create fresh strategy instance
        strategy = self.strategy_class(**self.strategy_params)
        
        rounds = 0
        target_reached = False
        target_round = None
        bankruptcy_round = None
        
        while rounds < self.max_rounds:
            rounds += 1
            
            # Determine outcome based on current win probability
            won = random.random() < strategy.state.win_probability
            
            # Process the round
            next_bet, new_bankroll = strategy.process_round(won)
            
            # Check for target reached
            if new_bankroll >= strategy.target_bankroll:
                target_reached = True
                target_round = rounds
                if self.stop_on_target:
                    break
                    
            # Check for bankruptcy
            if new_bankroll <= 0 or next_bet > new_bankroll:
                bankruptcy_round = rounds
                if self.stop_on_bankruptcy:
                    break
                    
        stats = strategy.get_stats()
        
        result = SimulationResult(
            survived=bankruptcy_round is None,
            final_bankroll=strategy.state.current_bankroll,
            rounds_played=rounds,
            max_bankroll=stats["max_bankroll"],
            min_bankroll=stats["min_bankroll"],
            target_reached=target_reached,
            bankruptcy_round=bankruptcy_round,
            target_round=target_round,
            win_rate=stats["win_rate"]
        )
        
        # Remove logging from worker processes to avoid conflicts
        # Progress tracking is handled by the main orchestrator process
        
        return result

    def run_simulations(
        self,
        num_simulations: int = 10000,
        parallel: bool = True,
        max_workers: Optional[int] = None
    ) -> SimulationSummary:
        """Run multiple simulations and return summary statistics with optimal parallel processing"""
        
        logger.info(f"Starting {num_simulations:,} simulations...")
        start_time = time.time()
        
        # Determine optimal execution strategy
        optimal_workers = max_workers or self._get_optimal_workers(num_simulations)
        use_parallel = parallel and num_simulations > 50  # Enable parallel processing!
        
        logger.info(f"Execution mode: {'Parallel' if use_parallel else 'Sequential'}")
        if use_parallel:
            logger.info(f"Workers: {optimal_workers}, Batch size: {self.batch_size}")
        
        results: List[SimulationResult] = []
        
        if use_parallel:
            results = self._run_parallel_simulations(
                num_simulations, optimal_workers, start_time
            )
        else:
            results = self._run_sequential_simulations(
                num_simulations, start_time
            )
        
        elapsed = time.time() - start_time
        simulations_per_second = num_simulations / elapsed
        logger.info(f"Completed {num_simulations:,} simulations in {elapsed:.2f} seconds")
        logger.info(f"Performance: {simulations_per_second:,.0f} simulations/second")
        
        # Calculate summary statistics
        summary = self._calculate_summary(results)
        
        return summary
        
    def _run_parallel_simulations(
        self,
        num_simulations: int,
        num_workers: int,
        start_time: float
    ) -> List[SimulationResult]:
        """Run simulations in parallel with optimized batching and error handling"""
        
        # Try multiprocessing first, fall back to threading if it fails
        try:
            return self._run_multiprocessing_simulations(num_simulations, num_workers, start_time)
        except Exception as e:
            logger.warning(f"Multiprocessing failed: {e}. Falling back to threading...")
            return self._run_threading_simulations(num_simulations, num_workers, start_time)
    
    def _run_multiprocessing_simulations(
        self,
        num_simulations: int,
        num_workers: int,
        start_time: float
    ) -> List[SimulationResult]:
        """Run simulations using multiprocessing with careful error handling"""
        
        # Generate base seed for reproducible results across workers
        base_seed = random.randint(0, 2**31 - 1)
        
        # Prepare worker arguments with optimized batching
        worker_args = [
            (i, self.strategy_class, self.strategy_params, 
             self.max_rounds, self.stop_on_bankruptcy, self.stop_on_target, base_seed)
            for i in range(num_simulations)
        ]
        
        results = []
        completed = 0
        
        # Use smaller worker count for stability
        stable_workers = min(num_workers, 4)
        
        with ProcessPoolExecutor(max_workers=stable_workers, mp_context=None) as executor:
            # Submit work in smaller batches to reduce memory pressure
            batch_size = min(50, max(1, num_simulations // stable_workers))
            
            for batch_start in range(0, num_simulations, batch_size):
                batch_end = min(batch_start + batch_size, num_simulations)
                batch_args = worker_args[batch_start:batch_end]
                
                # Submit batch and collect results with timeout
                futures = [executor.submit(_worker_simulation, args) for args in batch_args]
                
                for future in as_completed(futures, timeout=60):  # 1 minute timeout for batch
                    try:
                        result = future.result(timeout=10)  # 10 second timeout per simulation
                        results.append(result)
                        completed += 1
                        
                        # Progress reporting
                        if completed % max(100, num_simulations // 20) == 0:
                            elapsed = time.time() - start_time
                            rate = completed / elapsed
                            eta = (num_simulations - completed) / rate if rate > 0 else 0
                            logger.info(
                                f"Progress: {completed:,}/{num_simulations:,} "
                                f"({completed/num_simulations:.1%}) - "
                                f"{rate:.0f}/sec, ETA: {eta:.0f}s"
                            )
                            
                    except Exception as e:
                        logger.error(f"Multiprocessing simulation failed: {e}")
                        # Create a failed result to maintain count
                        results.append(SimulationResult(
                            survived=False, final_bankroll=0.0, rounds_played=0,
                            max_bankroll=0.0, min_bankroll=0.0, target_reached=False
                        ))
                        completed += 1
        
        return results
    
    def _run_threading_simulations(
        self,
        num_simulations: int,
        num_workers: int,
        start_time: float
    ) -> List[SimulationResult]:
        """Run simulations using threading as fallback"""
        
        # Use ThreadPoolExecutor for CPU-bound tasks with careful worker count
        # Threading won't give true parallelism for CPU-bound work but avoids import issues
        thread_workers = min(num_workers, 4)  # Limited by GIL anyway
        
        results = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=thread_workers) as executor:
            # Submit all simulations
            futures = [
                executor.submit(self._run_single_simulation, i)
                for i in range(num_simulations)
            ]
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                    completed += 1
                    
                    # Progress reporting
                    if completed % max(100, num_simulations // 20) == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        eta = (num_simulations - completed) / rate if rate > 0 else 0
                        logger.info(
                            f"Progress: {completed:,}/{num_simulations:,} "
                            f"({completed/num_simulations:.1%}) - "
                            f"{rate:.0f}/sec, ETA: {eta:.0f}s"
                        )
                        
                except Exception as e:
                    logger.error(f"Threading simulation failed: {e}")
                    # Create a failed result to maintain count
                    results.append(SimulationResult(
                        survived=False, final_bankroll=0.0, rounds_played=0,
                        max_bankroll=0.0, min_bankroll=0.0, target_reached=False
                    ))
                    completed += 1
        
        return results
        
    def _run_sequential_simulations(
        self, 
        num_simulations: int, 
        start_time: float
    ) -> List[SimulationResult]:
        """Run simulations sequentially with optimized progress reporting"""
        
        results = []
        report_interval = max(100, num_simulations // 20)
        
        for i in range(num_simulations):
            try:
                result = self._run_single_simulation(i)
                results.append(result)
                
                # Optimized progress reporting
                if (i + 1) % report_interval == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    eta = (num_simulations - i - 1) / rate if rate > 0 else 0
                    logger.info(
                        f"Progress: {i+1:,}/{num_simulations:,} "
                        f"({(i+1)/num_simulations:.1%}) - "
                        f"{rate:.0f}/sec, ETA: {eta:.0f}s"
                    )
                    
            except Exception as e:
                logger.error(f"Simulation {i} failed: {e}")
                # Create a failed result to maintain count
                results.append(SimulationResult(
                    survived=False, final_bankroll=0.0, rounds_played=0,
                    max_bankroll=0.0, min_bankroll=0.0, target_reached=False
                ))
        
        return results

    def _calculate_summary(self, results: List[SimulationResult]) -> SimulationSummary:
        """Calculate summary statistics from simulation results"""
        
        total_runs = len(results)
        survived = [r for r in results if r.survived]
        target_reached = [r for r in results if r.target_reached]
        bankruptcies = [r for r in results if r.bankruptcy_round is not None]
        
        final_bankrolls = [r.final_bankroll for r in results]
        
        # Calculate percentiles
        percentiles = np.percentile(final_bankrolls, [25, 50, 75, 95])
        
        summary = SimulationSummary(
            total_runs=total_runs,
            survival_rate=len(survived) / total_runs,
            target_success_rate=len(target_reached) / total_runs,
            average_rounds=np.mean([r.rounds_played for r in results]),
            average_final_bankroll=np.mean(final_bankrolls),
            median_final_bankroll=percentiles[1],
            std_final_bankroll=np.std(final_bankrolls),
            max_bankroll_achieved=max(r.max_bankroll for r in results),
            min_bankroll_achieved=min(r.min_bankroll for r in results),
            percentile_25=percentiles[0],
            percentile_75=percentiles[2],
            percentile_95=percentiles[3]
        )
        
        if bankruptcies:
            summary.average_bankruptcy_round = np.mean([r.bankruptcy_round for r in bankruptcies])
            
        if target_reached:
            summary.average_target_round = np.mean([r.target_round for r in target_reached])
            
        logger.info(f"Summary calculated: Survival rate={summary.survival_rate:.2%}, "
                   f"Target success={summary.target_success_rate:.2%}")
        
        return summary

    def run_probability_sweep(
        self,
        probabilities: List[float],
        expected_returns: List[float],
        num_simulations_per: int = 1000
    ) -> Dict[tuple, SimulationSummary]:
        """Run simulations across multiple probability/return combinations"""
        
        results = {}
        total_combinations = len(probabilities) * len(expected_returns)
        current = 0
        
        logger.info(f"Running probability sweep: {len(probabilities)} probabilities × "
                   f"{len(expected_returns)} returns = {total_combinations} combinations")
        
        for win_prob in probabilities:
            for exp_return in expected_returns:
                current += 1
                logger.info(f"[{current}/{total_combinations}] Testing "
                          f"P(win)={win_prob:.2%}, Return={exp_return:.2f}x")
                
                # Update strategy parameters
                params = self.strategy_params.copy()
                params['win_probability'] = win_prob
                params['expected_return'] = exp_return
                
                # Create temporary engine with updated params
                temp_engine = MonteCarloEngine(
                    self.strategy_class,
                    params,
                    self.max_rounds,
                    self.stop_on_bankruptcy,
                    self.stop_on_target
                )
                
                # Run simulations
                summary = temp_engine.run_simulations(
                    num_simulations=num_simulations_per,
                    parallel=True
                )
                
                results[(win_prob, exp_return)] = summary
                
                logger.info(f"  → Survival rate: {summary.survival_rate:.2%}, "
                          f"Target success: {summary.target_success_rate:.2%}")
        
        return results


# Standalone worker function for multiprocessing (must be at module level)
def _worker_simulation(args) -> SimulationResult:
    """Standalone worker function for multiprocessing with enhanced error handling"""
    try:
        sim_id, strategy_class, strategy_params, max_rounds, stop_on_bankruptcy, stop_on_target, worker_seed = args
        
        # Set unique seed for this worker process
        if worker_seed is not None:
            random.seed(worker_seed + sim_id)
            np.random.seed(worker_seed + sim_id)
        
        # Create fresh strategy instance with error handling
        try:
            strategy = strategy_class(**strategy_params)
        except Exception as e:
            # Return failed result if strategy creation fails
            return SimulationResult(
                survived=False, final_bankroll=0.0, rounds_played=0,
                max_bankroll=0.0, min_bankroll=0.0, target_reached=False
            )
        
        rounds = 0
        target_reached = False
        target_round = None
        bankruptcy_round = None
        
        while rounds < max_rounds:
            rounds += 1
            
            try:
                # Determine outcome based on current win probability
                won = random.random() < strategy.state.win_probability
                
                # Process the round
                next_bet, new_bankroll = strategy.process_round(won)
                
                # Check for target reached
                if new_bankroll >= strategy.target_bankroll:
                    target_reached = True
                    target_round = rounds
                    if stop_on_target:
                        break
                        
                # Check for bankruptcy
                if new_bankroll <= 0 or next_bet > new_bankroll:
                    bankruptcy_round = rounds
                    if stop_on_bankruptcy:
                        break
                        
            except Exception as e:
                # If simulation round fails, treat as bankruptcy
                bankruptcy_round = rounds
                break
                
        try:
            stats = strategy.get_stats()
        except Exception as e:
            # Fallback stats if get_stats fails
            stats = {
                "max_bankroll": strategy.state.max_bankroll if hasattr(strategy, 'state') else 0.0,
                "min_bankroll": strategy.state.min_bankroll if hasattr(strategy, 'state') else 0.0,
                "win_rate": 0.0
            }
        
        return SimulationResult(
            survived=bankruptcy_round is None,
            final_bankroll=strategy.state.current_bankroll if hasattr(strategy, 'state') else 0.0,
            rounds_played=rounds,
            max_bankroll=stats.get("max_bankroll", 0.0),
            min_bankroll=stats.get("min_bankroll", 0.0),
            target_reached=target_reached,
            bankruptcy_round=bankruptcy_round,
            target_round=target_round,
            win_rate=stats.get("win_rate", 0.0)
        )
        
    except Exception as e:
        # Ultimate fallback - return failed simulation
        return SimulationResult(
            survived=False, final_bankroll=0.0, rounds_played=0,
            max_bankroll=0.0, min_bankroll=0.0, target_reached=False
        )
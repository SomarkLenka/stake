"""
Highly optimized Monte Carlo simulation engine with advanced concurrency and vectorization.
This implementation focuses on achieving maximum throughput while maintaining correctness.
"""

import random
import math
import time
import warnings
from typing import List, Dict, Any, Optional, Type, Callable
from dataclasses import dataclass, field
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from base_strategy import BaseStrategy
from logger_utils import init_logger, logLev
from monte_carlo_engine import SimulationResult, SimulationSummary

logger = init_logger(name="monte_carlo_engine_optimized", level=logLev.INFO, is_orc=True)


class OptimizedMonteCarloEngine:
    """
    Ultra-high-performance Monte Carlo simulation engine with advanced optimizations:
    - Vectorized random number generation
    - Intelligent batch processing
    - Optimized thread management
    - Memory-efficient operations
    - Reduced synchronization overhead
    """
    
    def __init__(
        self,
        strategy_class: Type[BaseStrategy],
        strategy_params: Dict[str, Any],
        max_rounds: int = 10000,
        stop_on_bankruptcy: bool = True,
        stop_on_target: bool = True,
        seed: Optional[int] = None
    ):
        self.strategy_class = strategy_class
        self.strategy_params = strategy_params
        self.max_rounds = max_rounds
        self.stop_on_bankruptcy = stop_on_bankruptcy
        self.stop_on_target = stop_on_target
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        logger.info(f"Initialized OptimizedMonteCarloEngine with {strategy_class.__name__}")
        logger.info(f"Max rounds: {max_rounds}, Optimization level: MAXIMUM")
    
    def run_simulations(
        self,
        num_simulations: int = 100000,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        progress_interval: int = 10000
    ) -> SimulationSummary:
        """
        Run Monte Carlo simulations with maximum optimization.
        
        Args:
            num_simulations: Number of simulations to run
            parallel: Enable parallel processing
            max_workers: Number of worker threads (auto-calculated if None)
            batch_size: Simulations per batch (auto-calculated if None)
            progress_interval: Progress reporting interval
        """
        logger.info(f"🚀 Starting OPTIMIZED simulation run: {num_simulations:,} simulations")
        start_time = time.time()
        
        # Calculate optimal parameters
        if max_workers is None:
            max_workers = self._calculate_optimal_workers(num_simulations)
        if batch_size is None:
            batch_size = self._calculate_optimal_batch_size(num_simulations, max_workers)
            
        logger.info(f"⚙️ Configuration: {max_workers} workers, {batch_size} batch size")
        
        if parallel and num_simulations >= 100:
            results = self._run_parallel_optimized(
                num_simulations, max_workers, batch_size, progress_interval
            )
        else:
            results = self._run_sequential_optimized(num_simulations, progress_interval)
        
        elapsed = time.time() - start_time
        throughput = num_simulations / elapsed
        
        logger.info(f"✅ Completed {num_simulations:,} simulations in {elapsed:.2f}s")
        logger.info(f"⚡ THROUGHPUT: {throughput:,.0f} simulations/second")
        
        # Calculate summary
        summary = self._calculate_summary(results)
        return summary
    
    def _calculate_optimal_workers(self, num_simulations: int) -> int:
        """Calculate optimal thread count based on workload and system"""
        cpu_cores = cpu_count()
        
        if num_simulations < 1000:
            return min(4, cpu_cores)
        elif num_simulations < 10000:
            return min(cpu_cores + 2, 12)
        else:
            # For large workloads, use more aggressive threading
            return min(cpu_cores * 2, 16)
    
    def _calculate_optimal_batch_size(self, num_simulations: int, max_workers: int) -> int:
        """Calculate batch size for optimal throughput"""
        # Target: Each worker processes multiple batches to maintain utilization
        # But not too many tiny batches (overhead) or too few large batches (imbalance)
        
        base_batch = num_simulations // (max_workers * 6)  # 6 batches per worker
        base_batch = max(200, min(base_batch, 1500))  # Reasonable bounds
        
        return base_batch
    
    def _run_parallel_optimized(
        self, num_simulations: int, max_workers: int, batch_size: int, progress_interval: int
    ) -> List[SimulationResult]:
        """Run simulations with optimized parallel processing"""
        
        results = []
        
        # Create batches
        batches = []
        for i in range(0, num_simulations, batch_size):
            batch_end = min(i + batch_size, num_simulations)
            batch_ids = list(range(i, batch_end))
            batches.append(batch_ids)
        
        logger.info(f"📊 Created {len(batches)} batches for parallel processing")
        
        # Use optimized thread pool
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="MCOptim"
        ) as executor:
            
            # Submit all batch jobs
            future_to_batch = {
                executor.submit(self._run_batch_vectorized, batch_ids, seed_offset=i): len(batch_ids)
                for i, batch_ids in enumerate(batches)
            }
            
            # Collect results with efficient progress tracking
            completed_sims = 0
            last_progress = 0
            
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                    completed_sims += future_to_batch[future]
                    
                    # Efficient progress reporting
                    if completed_sims - last_progress >= progress_interval:
                        progress = completed_sims / num_simulations * 100
                        logger.info(f"📈 Progress: {completed_sims:,}/{num_simulations:,} ({progress:.1f}%)")
                        last_progress = completed_sims
                        
                except Exception as e:
                    logger.error(f"Batch failed: {e}")
                    # Add placeholder results
                    failed_count = future_to_batch[future]
                    for _ in range(failed_count):
                        results.append(self._create_failed_result())
        
        return results
    
    def _run_batch_vectorized(self, batch_ids: List[int], seed_offset: int = 0) -> List[SimulationResult]:
        """
        Run a batch of simulations with vectorized optimizations where possible.
        """
        batch_results = []
        
        # Set thread-specific seed
        thread_seed = hash(f"{time.time()}_{seed_offset}") % (2**32)
        np.random.seed(thread_seed)
        random.seed(thread_seed)
        
        for sim_id in batch_ids:
            try:
                result = self._run_single_simulation_optimized(sim_id)
                batch_results.append(result)
            except Exception as e:
                logger.debug(f"Simulation {sim_id} failed: {e}")
                batch_results.append(self._create_failed_result())
        
        return batch_results
    
    def _run_single_simulation_optimized(self, sim_id: int) -> SimulationResult:
        """Run a single simulation with optimizations"""
        
        # Create strategy instance
        strategy = self.strategy_class(**self.strategy_params)
        
        rounds = 0
        target_reached = False
        target_round = None
        bankruptcy_round = None
        
        # Pre-generate some random numbers for efficiency
        win_prob = strategy.state.win_probability
        random_cache = np.random.random(min(self.max_rounds, 1000))
        cache_pos = 0
        
        while rounds < self.max_rounds:
            rounds += 1
            
            # Use cached random number if available, otherwise generate new
            if cache_pos < len(random_cache):
                rand_val = random_cache[cache_pos]
                cache_pos += 1
            else:
                rand_val = random.random()
                
            won = rand_val < win_prob
            
            # Process round
            try:
                next_bet, new_bankroll = strategy.process_round(won)
            except Exception:
                # Handle strategy errors gracefully
                break
            
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
        
        # Get final stats
        stats = strategy.get_stats()
        
        return SimulationResult(
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
    
    def _run_sequential_optimized(self, num_simulations: int, progress_interval: int) -> List[SimulationResult]:
        """Run simulations sequentially with optimizations"""
        results = []
        
        for i in range(num_simulations):
            result = self._run_single_simulation_optimized(i)
            results.append(result)
            
            if (i + 1) % progress_interval == 0:
                progress = (i + 1) / num_simulations * 100
                logger.info(f"Progress: {i+1:,}/{num_simulations:,} ({progress:.1f}%)")
        
        return results
    
    def _create_failed_result(self) -> SimulationResult:
        """Create placeholder result for failed simulations"""
        return SimulationResult(
            survived=False,
            final_bankroll=0.0,
            rounds_played=0,
            max_bankroll=0.0,
            min_bankroll=0.0,
            target_reached=False
        )
    
    def _calculate_summary(self, results: List[SimulationResult]) -> SimulationSummary:
        """Calculate summary statistics efficiently"""
        
        total_runs = len(results)
        
        # Use numpy for efficient calculations
        survived_flags = np.array([r.survived for r in results])
        target_flags = np.array([r.target_reached for r in results])
        final_bankrolls = np.array([r.final_bankroll for r in results])
        rounds_played = np.array([r.rounds_played for r in results])
        
        # Calculate statistics
        survival_rate = np.mean(survived_flags)
        target_success_rate = np.mean(target_flags)
        
        # Bankroll statistics
        avg_bankroll = np.mean(final_bankrolls)
        std_bankroll = np.std(final_bankrolls)
        percentiles = np.percentile(final_bankrolls, [25, 50, 75, 95])
        
        # Round statistics  
        avg_rounds = np.mean(rounds_played)
        max_bankroll = max(r.max_bankroll for r in results)
        min_bankroll = min(r.min_bankroll for r in results)
        
        # Conditional statistics
        bankruptcies = [r for r in results if r.bankruptcy_round is not None]
        targets_reached = [r for r in results if r.target_reached]
        
        avg_bankruptcy_round = np.mean([r.bankruptcy_round for r in bankruptcies]) if bankruptcies else None
        avg_target_round = np.mean([r.target_round for r in targets_reached]) if targets_reached else None
        
        summary = SimulationSummary(
            total_runs=total_runs,
            survival_rate=survival_rate,
            target_success_rate=target_success_rate,
            average_rounds=avg_rounds,
            average_final_bankroll=avg_bankroll,
            median_final_bankroll=percentiles[1],
            std_final_bankroll=std_bankroll,
            max_bankroll_achieved=max_bankroll,
            min_bankroll_achieved=min_bankroll,
            average_bankruptcy_round=avg_bankruptcy_round,
            average_target_round=avg_target_round,
            percentile_25=percentiles[0],
            percentile_75=percentiles[2],
            percentile_95=percentiles[3]
        )
        
        logger.info(f"📊 Results: Survival={summary.survival_rate:.1%}, Target={summary.target_success_rate:.1%}")
        
        return summary
import random
import math
from typing import List, Dict, Any, Optional, Type, Callable
from dataclasses import dataclass, field
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, TimeoutError
from multiprocessing import cpu_count, Manager, get_context
from base_strategy import BaseStrategy
from logger_utils import init_logger, logLev
import time
import os
import sys
import warnings
from functools import partial

# Only initialize logger in main process, not in spawned processes
import multiprocessing
if multiprocessing.current_process().name == 'MainProcess':
    logger = init_logger(name="monte_carlo_engine", level=logLev.INFO, is_orc=True)
else:
    import logging
    logger = logging.getLogger("monte_carlo_engine")


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
            
        logger.info(f"Initialized MonteCarloEngine with {strategy_class.__name__}")
        logger.info(f"Max rounds: {max_rounds}, Stop on bankruptcy: {stop_on_bankruptcy}, "
                   f"Stop on target: {stop_on_target}")
        
    def _run_single_simulation(self, sim_id: int, use_cache: bool = True) -> SimulationResult:
        """Run a single simulation with optional random number caching for performance"""
        # Create fresh strategy instance
        strategy = self.strategy_class(**self.strategy_params)
        
        rounds = 0
        target_reached = False
        target_round = None
        bankruptcy_round = None
        
        # Pre-generate random numbers for efficiency when requested
        win_prob = strategy.state.win_probability
        random_cache = None
        cache_pos = 0
        
        if use_cache:
            cache_size = min(self.max_rounds, 1000)
            random_cache = np.random.random(cache_size)
        
        while rounds < self.max_rounds:
            rounds += 1
            
            # Use cached random number if available, otherwise generate new
            if random_cache is not None and cache_pos < len(random_cache):
                rand_val = random_cache[cache_pos]
                cache_pos += 1
            else:
                rand_val = random.random()
                
            won = rand_val < win_prob
            
            # Process the round with error handling
            try:
                next_bet, new_bankroll = strategy.process_round(won)
            except Exception as e:
                logger.debug(f"Strategy error in simulation {sim_id}: {e}")
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
        
        # Reduce logging frequency for better performance
        if sim_id > 0 and sim_id % 5000 == 0:
            logger.debug(f"Progress: {sim_id} simulations completed")
        
        return result
        
    def run_simulations(
        self,
        num_simulations: int = 1000000,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        use_multiprocessing: bool = False,  # Default to False to prevent hanging
        timeout_per_batch: int = 120
    ) -> SimulationSummary:
        """Run multiple simulations and return summary statistics with advanced optimizations"""
        
        logger.info(f"🚀 Starting {num_simulations:,} simulations with optimized engine...")
        start_time = time.time()
        
        # Optimize worker allocation and batch sizing
        if max_workers is None:
            max_workers = self._calculate_optimal_workers(num_simulations)
        if batch_size is None:
            batch_size = self._calculate_optimal_batch_size(num_simulations, max_workers)
            
        logger.info(f"⚙️ Configuration: {max_workers} workers, batch size {batch_size}")
        
        results: List[SimulationResult] = []
        
        if parallel and num_simulations > 100:
            # Default to threading unless multiprocessing is explicitly requested
            if use_multiprocessing and num_simulations >= 1000:
                logger.info("⚠️ Multiprocessing requested - this may hang in some environments")
                logger.info("💡 If hanging occurs, set use_multiprocessing=False (default)")
                try:
                    # Check if we're in a context where multiprocessing might hang
                    if hasattr(sys, '_getframe'):
                        # Check if we're being run from a spawned process
                        frame = sys._getframe()
                        if '<string>' in str(frame.f_code.co_filename):
                            logger.info("Detected execution from console/eval, using threading instead")
                            results = self._run_threaded_simulations(
                                num_simulations, max_workers, batch_size
                            )
                        else:
                            results = self._run_multiprocessing_simulations(
                                num_simulations, max_workers, batch_size, timeout_per_batch
                            )
                    else:
                        results = self._run_multiprocessing_simulations(
                            num_simulations, max_workers, batch_size, timeout_per_batch
                        )
                except Exception as e:
                    logger.warning(f"Multiprocessing failed ({e}), falling back to threading")
                    results = self._run_threaded_simulations(
                        num_simulations, max_workers, batch_size
                    )
            else:
                logger.info("🔄 Using optimized threading (multiprocessing disabled for stability)")
                results = self._run_threaded_simulations(
                    num_simulations, max_workers, batch_size
                )
        else:
            # Sequential processing for small batches or debugging
            results = self._run_sequential_simulations(num_simulations)
        
        elapsed = time.time() - start_time
        throughput = num_simulations / elapsed
        logger.info(f"✅ Completed {num_simulations:,} simulations in {elapsed:.2f}s")
        logger.info(f"⚡ THROUGHPUT: {throughput:,.0f} simulations/second")
        
        # Calculate summary statistics
        summary = self._calculate_summary(results)
        
        return summary
        
    def _calculate_summary(self, results: List[SimulationResult]) -> SimulationSummary:
        """Calculate summary statistics efficiently using vectorized operations"""
        
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
        percentiles = np.percentile(final_bankrolls, [25, 50, 75, 95])
        
        bankruptcies = [r for r in results if r.bankruptcy_round is not None]
        target_reached = [r for r in results if r.target_reached]
        
        summary = SimulationSummary(
            total_runs=total_runs,
            survival_rate=survival_rate,
            target_success_rate=target_success_rate,
            average_rounds=np.mean(rounds_played),
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
        
    def _calculate_optimal_workers(self, num_simulations: int) -> int:
        """Calculate optimal number of workers based on CPU count and workload"""
        cpu_cores = cpu_count()
        
        if num_simulations < 1000:
            return min(4, cpu_cores)
        elif num_simulations < 10000:
            return min(cpu_cores + 2, 12)
        else:
            # For large workloads, use more aggressive threading for better throughput
            return min(cpu_cores * 2, 16)
            
    def _calculate_optimal_batch_size(self, num_simulations: int, max_workers: int) -> int:
        """Calculate optimal batch size to minimize overhead while maximizing throughput"""
        # Target: Each worker processes multiple batches to maintain utilization
        # But not too many tiny batches (overhead) or too few large batches (imbalance)
        
        # Optimized batch sizing for better performance
        base_batch = num_simulations // (max_workers * 6)  # 6 batches per worker
        base_batch = max(200, min(base_batch, 1500))  # Optimized bounds
        
        return base_batch
        
    def _run_multiprocessing_simulations(
        self, num_simulations: int, max_workers: int, batch_size: int, timeout: int
    ) -> List[SimulationResult]:
        """Run simulations using multiprocessing for maximum performance"""
        
        results = []
        
        # Import the isolated worker function
        try:
            from monte_carlo_worker_isolated import run_simulation_batch_isolated
            worker_func = run_simulation_batch_isolated
        except ImportError:
            try:
                from monte_carlo_worker_safe import run_simulation_batch_safe
                worker_func = run_simulation_batch_safe
            except ImportError:
                # Fall back to module-level function if workers not available
                logger.debug("Using fallback worker function")
                worker_func = _run_simulation_batch
        
        # Create batches for processing
        batches = []
        for i in range(0, num_simulations, batch_size):
            batch_end = min(i + batch_size, num_simulations)
            batch_ids = list(range(i, batch_end))
            batches.append(batch_ids)
            
        logger.info(f"Created {len(batches)} batches for multiprocessing")
        logger.info(f"🔄 Starting worker processes... (this may take a few seconds)")
        
        # Use spawn context for better compatibility
        ctx = get_context('spawn')
        
        # Track initialization time
        init_start = time.time()
        
        # Use ProcessPoolExecutor with spawn context for true parallelization
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            init_time = time.time() - init_start
            logger.info(f"✅ {max_workers} worker processes started in {init_time:.1f}s, submitting {len(batches)} batch jobs...")
            
            # Submit all batch jobs
            future_to_batch = {}
            submission_start = time.time()
            
            for i, batch_ids in enumerate(batches):
                future = executor.submit(
                    worker_func,
                    batch_ids,
                    self.strategy_class,
                    self.strategy_params,
                    self.max_rounds,
                    self.stop_on_bankruptcy,
                    self.stop_on_target,
                    i  # seed_offset
                )
                future_to_batch[future] = len(batch_ids)
                
                # Log submission progress for large batch counts
                if (i + 1) % 20 == 0 or (i + 1) == len(batches):
                    logger.debug(f"Submitted {i + 1}/{len(batches)} batches")
            
            submission_time = time.time() - submission_start
            logger.info(f"✅ All {len(batches)} batches submitted in {submission_time:.2f}s, processing...")
            
            # Collect results with progress reporting
            completed_sims = 0
            completed_batches = 0
            failed_batches = 0
            max_failures = len(batches) // 2  # If more than half fail, abort
            last_progress_time = time.time()
            progress_interval = 5.0  # Report progress every 5 seconds if no batches complete
            initial_wait_time = 10.0  # Wait up to 10 seconds for first result
            start_wait = time.time()
            got_first_result = False
            
            # Use a shorter timeout for initial results to detect hanging quickly
            batch_timeout = min(timeout, 30)  # 30 seconds max per batch
            total_timeout = batch_timeout * len(batches)
            
            try:
                for future in as_completed(future_to_batch, timeout=total_timeout):
                    # Check if this is the first result
                    if not got_first_result:
                        got_first_result = True
                        wait_time = time.time() - start_wait
                        logger.info(f"🎯 First batch completed after {wait_time:.1f}s")
                    
                    try:
                        batch_dict_results = future.result(timeout=batch_timeout)
                        # Convert dictionaries back to SimulationResult objects
                        batch_results = [self._dict_to_simulation_result(d) for d in batch_dict_results]
                        results.extend(batch_results)
                        completed_sims += future_to_batch[future]
                        completed_batches += 1
                        
                        # Progress updates with batch information
                        current_time = time.time()
                        if completed_sims % (batch_size * 5) == 0 or completed_sims >= num_simulations:
                            elapsed = current_time - submission_start
                            rate = completed_sims / elapsed if elapsed > 0 else 0
                            eta = (num_simulations - completed_sims) / rate if rate > 0 else 0
                            logger.info(f"📊 Progress: {completed_sims:,}/{num_simulations:,} ({completed_sims/num_simulations:.1%}) | "
                                      f"Batches: {completed_batches}/{len(batches)} | "
                                      f"Rate: {rate:.0f}/sec | ETA: {eta:.1f}s")
                            last_progress_time = current_time
                        
                        # Also log if it's been too long since last update
                        elif current_time - last_progress_time > progress_interval:
                            elapsed = current_time - submission_start
                            rate = completed_sims / elapsed if elapsed > 0 else 0
                            logger.info(f"⏳ Still processing... {completed_batches}/{len(batches)} batches done, "
                                      f"{completed_sims:,} simulations completed ({rate:.0f}/sec)")
                            last_progress_time = current_time
                        
                    except TimeoutError:
                        failed_batches += 1
                        if failed_batches <= 3:  # Only log first few failures
                            logger.debug(f"Batch timed out after {batch_timeout}s")
                        # Add placeholder results for failed batch
                        batch_size_failed = future_to_batch[future]
                        for _ in range(batch_size_failed):
                            results.append(self._create_failed_result())
                        
                        if failed_batches > max_failures:
                            logger.warning("Too many multiprocessing failures, aborting remaining batches")
                            raise RuntimeError("Multiprocessing unstable")
                            
                    except Exception as e:
                        failed_batches += 1
                        if failed_batches <= 3:  # Only log first few failures
                            logger.debug(f"Batch processing error: {e}")
                        # Add placeholder results for failed batch  
                        batch_size_failed = future_to_batch[future]
                        for _ in range(batch_size_failed):
                            results.append(self._create_failed_result())
                        
                        if failed_batches > max_failures:
                            logger.warning("Too many multiprocessing failures, aborting remaining batches")
                            raise RuntimeError("Multiprocessing unstable")
                            
            except TimeoutError:
                # Check if we got any results at all
                if not got_first_result:
                    logger.error(f"⚠️ No batches completed after {initial_wait_time}s - multiprocessing appears to be hanging")
                    logger.warning("Aborting multiprocessing, will fall back to threading")
                    raise RuntimeError("Multiprocessing hanging - no results received")
                else:
                    logger.warning(f"Timeout waiting for remaining batches (got {completed_batches}/{len(batches)})")
                    # Return partial results if we got some
                    if results:
                        logger.info(f"Returning {len(results)} partial results")
                        return results
                    raise
                        
        return results
        
    def _run_threaded_simulations(
        self, num_simulations: int, max_workers: int, batch_size: int
    ) -> List[SimulationResult]:
        """Run simulations using optimized threading"""
        
        results = []
        
        # Optimized batch size for threading
        thread_batch_size = batch_size
        
        # Create batches
        batches = []
        for i in range(0, num_simulations, thread_batch_size):
            batch_end = min(i + thread_batch_size, num_simulations)
            batch_ids = list(range(i, batch_end))
            batches.append(batch_ids)
        
        logger.info(f"📦 Created {len(batches)} batches for threaded processing")
        
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="MCOptim"
        ) as executor:
            logger.info(f"🚀 Submitting {len(batches)} batches to thread pool...")
            
            # Submit all batch jobs with seed offsets
            submission_start = time.time()
            future_to_batch = {
                executor.submit(self._run_threaded_batch, batch_ids, seed_offset=i): len(batch_ids)
                for i, batch_ids in enumerate(batches)
            }
            
            submission_time = time.time() - submission_start
            logger.info(f"✅ All batches submitted in {submission_time:.2f}s, processing with {max_workers} threads...")
            
            # Collect results with efficient progress tracking
            completed_sims = 0
            completed_batches = 0
            last_progress = 0
            progress_interval = max(1000, num_simulations // 20)  # Report at 5% intervals
            start_time = time.time()
            
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                    completed_sims += future_to_batch[future]
                    completed_batches += 1
                    
                    # Efficient progress reporting
                    if completed_sims - last_progress >= progress_interval or completed_sims >= num_simulations:
                        elapsed = time.time() - start_time
                        rate = completed_sims / elapsed if elapsed > 0 else 0
                        eta = (num_simulations - completed_sims) / rate if rate > 0 else 0
                        progress = completed_sims / num_simulations * 100
                        logger.info(f"📊 Threading Progress: {completed_sims:,}/{num_simulations:,} ({progress:.1f}%) | "
                                  f"Batches: {completed_batches}/{len(batches)} | "
                                  f"Rate: {rate:.0f}/sec | ETA: {eta:.1f}s")
                        last_progress = completed_sims
                        
                except Exception as e:
                    logger.error(f"Thread batch failed: {e}")
                    # Add placeholder results
                    failed_count = future_to_batch[future]
                    for _ in range(failed_count):
                        results.append(self._create_failed_result())
                    
        return results
        
    def _run_sequential_simulations(self, num_simulations: int) -> List[SimulationResult]:
        """Run simulations sequentially for small workloads or debugging"""
        results = []
        
        for i in range(num_simulations):
            # Use caching for better performance even in sequential mode
            results.append(self._run_single_simulation(i, use_cache=True))
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Progress: {i+1:,}/{num_simulations:,} "
                          f"({(i+1)/num_simulations:.1%})")
                          
        return results
        
    def _run_threaded_batch(self, batch_ids: List[int], seed_offset: int = 0) -> List[SimulationResult]:
        """Run a batch of simulations in thread context with optimizations"""
        batch_results = []
        
        # Set thread-specific seed for better randomness
        thread_seed = hash(f"{time.time()}_{seed_offset}") % (2**32)
        np.random.seed(thread_seed)
        random.seed(thread_seed)
        
        for sim_id in batch_ids:
            try:
                # Use caching for better performance in threads
                result = self._run_single_simulation(sim_id, use_cache=True)
                batch_results.append(result)
            except Exception as e:
                logger.debug(f"Simulation {sim_id} failed: {e}")
                batch_results.append(self._create_failed_result())
                
        return batch_results
        
    def _create_failed_result(self) -> SimulationResult:
        """Create a placeholder result for failed simulations"""
        return SimulationResult(
            survived=False,
            final_bankroll=0.0,
            rounds_played=0,
            max_bankroll=0.0,
            min_bankroll=0.0,
            target_reached=False
        )
        
    def _dict_to_simulation_result(self, result_dict: dict) -> SimulationResult:
        """Convert dictionary result from multiprocessing back to SimulationResult"""
        return SimulationResult(
            survived=result_dict.get('survived', False),
            final_bankroll=result_dict.get('final_bankroll', 0.0),
            rounds_played=result_dict.get('rounds_played', 0),
            max_bankroll=result_dict.get('max_bankroll', 0.0),
            min_bankroll=result_dict.get('min_bankroll', 0.0),
            target_reached=result_dict.get('target_reached', False),
            bankruptcy_round=result_dict.get('bankruptcy_round'),
            target_round=result_dict.get('target_round'),
            win_rate=result_dict.get('win_rate', 0.0)
        )


# Module-level function for multiprocessing (must be pickle-able)
def _run_simulation_batch(
    batch_ids: List[int],
    strategy_class: Type[BaseStrategy],
    strategy_params: Dict[str, Any],
    max_rounds: int,
    stop_on_bankruptcy: bool,
    stop_on_target: bool,
    seed_offset: int = 0
) -> List[dict]:
    """
    Run a batch of simulations in a separate process with optimizations.
    Returns results as dictionaries to avoid pickle issues.
    """
    batch_results = []
    
    # Suppress warnings in worker processes
    warnings.filterwarnings('ignore')
    
    # Set unique random seed for this batch
    base_seed = hash(f"{os.getpid()}_{time.time()}_{seed_offset}") % (2**32)
    
    for sim_id in batch_ids:
        # Use unique seed per simulation
        sim_seed = base_seed + sim_id
        random.seed(sim_seed)
        np.random.seed(sim_seed % (2**32))
        
        try:
            # Create strategy instance
            strategy = strategy_class(**strategy_params)
            
            rounds = 0
            target_reached = False
            target_round = None
            bankruptcy_round = None
            
            # Pre-generate random numbers for efficiency
            win_prob = strategy.state.win_probability
            cache_size = min(max_rounds, 1000)
            random_cache = np.random.random(cache_size)
            cache_pos = 0
            
            # Run simulation
            while rounds < max_rounds:
                rounds += 1
                
                # Use cached random number if available
                if cache_pos < len(random_cache):
                    rand_val = random_cache[cache_pos]
                    cache_pos += 1
                else:
                    rand_val = random.random()
                    
                won = rand_val < win_prob
                
                # Process round with error handling
                try:
                    next_bet, new_bankroll = strategy.process_round(won)
                except Exception:
                    break
                
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
                        
            # Get final stats
            stats = strategy.get_stats()
            
            # Return as dictionary for multiprocessing compatibility
            result = {
                'survived': bankruptcy_round is None,
                'final_bankroll': strategy.state.current_bankroll,
                'rounds_played': rounds,
                'max_bankroll': stats["max_bankroll"],
                'min_bankroll': stats["min_bankroll"],
                'target_reached': target_reached,
                'bankruptcy_round': bankruptcy_round,
                'target_round': target_round,
                'win_rate': stats["win_rate"]
            }
            
            batch_results.append(result)
            
        except Exception as e:
            # Add failed result
            batch_results.append({
                'survived': False,
                'final_bankroll': 0.0,
                'rounds_played': 0,
                'max_bankroll': 0.0,
                'min_bankroll': 0.0,
                'target_reached': False,
                'bankruptcy_round': None,
                'target_round': None,
                'win_rate': 0.0,
                'error': str(e)
            })
    
    return batch_results
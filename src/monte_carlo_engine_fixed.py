import random
from typing import List, Dict, Any, Optional, Type
from dataclasses import dataclass, field
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from base_strategy import BaseStrategy
from logger_utils import init_logger, logLev
import time
from multiprocessing import cpu_count

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
        
        # Only log every 100th simulation to reduce noise
        if sim_id > 0 and sim_id % 100 == 0:
            logger.info(f"Progress: {sim_id} simulations completed")
        
        return result
        
    def run_simulations(
        self,
        num_simulations: int = 10000,
        parallel: bool = True,
        max_workers: Optional[int] = None
    ) -> SimulationSummary:
        """Run multiple simulations and return summary statistics"""
        
        logger.info(f"Starting {num_simulations:,} simulations...")
        start_time = time.time()
        
        results: List[SimulationResult] = []
        
        if parallel and num_simulations > 100:
            # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid pickle issues
            # Threading is fine since we're mostly doing CPU-bound work with Python objects
            if max_workers is None:
                max_workers = min(cpu_count(), 8)  # Cap at 8 to avoid too much overhead
                
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self._run_single_simulation, i)
                    for i in range(num_simulations)
                ]
                
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        logger.error(f"Simulation failed: {e}")
                        # Add a failed result to maintain count
                        results.append(SimulationResult(
                            survived=False,
                            final_bankroll=0.0,
                            rounds_played=0,
                            max_bankroll=0.0,
                            min_bankroll=0.0,
                            target_reached=False
                        ))
                    
                    if len(results) % 1000 == 0:
                        logger.info(f"Progress: {len(results):,}/{num_simulations:,} "
                                  f"({len(results)/num_simulations:.1%})")
        else:
            # Sequential processing for small batches or debugging
            for i in range(num_simulations):
                results.append(self._run_single_simulation(i))
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"Progress: {i+1:,}/{num_simulations:,} "
                              f"({(i+1)/num_simulations:.1%})")
        
        elapsed = time.time() - start_time
        logger.info(f"Completed {num_simulations:,} simulations in {elapsed:.2f} seconds")
        
        # Calculate summary statistics
        summary = self._calculate_summary(results)
        
        return summary
        
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
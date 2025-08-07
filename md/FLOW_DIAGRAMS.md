# Probability Engine - Detailed Flow Diagrams

## Table of Contents
1. [System Architecture Flow](#system-architecture-flow)
2. [Simulation Execution Flow](#simulation-execution-flow)
3. [Strategy Processing Flow](#strategy-processing-flow)
4. [Rule Evaluation Flow](#rule-evaluation-flow)
5. [Statistical Analysis Flow](#statistical-analysis-flow)
6. [Optimization Flow](#optimization-flow)

---

## System Architecture Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INTERFACE                          │
│  • Command Line Args  • JSON Config  • Python API           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   PROBABILITY ENGINE                         │
│  • Config Loading    • Strategy Factory                      │
│  • Simulation Orchestration  • Comparison & Optimization     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  MONTE CARLO ENGINE                          │
│  • Simulation Runner  • Parallel Processing                  │
│  • Result Aggregation • Statistical Analysis                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    STRATEGY LAYER                            │
│  ┌──────────────┬──────────────┬──────────────┐            │
│  │StreakMulti  │  Martingale  │    Custom    │            │
│  │  Strategy    │   Strategy   │   Strategy   │            │
│  └──────┬───────┴──────┬───────┴──────┬───────┘            │
│         └──────────────┼──────────────┘                    │
│                        ▼                                    │
│                  BASE STRATEGY                              │
│         • State Management  • Rule Processing               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                CONDITION-ACTION FRAMEWORK                    │
│  ┌─────────────┬─────────────┬─────────────┐              │
│  │  Conditions │   Actions   │    Rules    │              │
│  │  • Events   │  • Bet Mods │  • Priority │              │
│  │  • Streaks  │  • Prob Mods│  • Binding  │              │
│  └─────────────┴─────────────┴─────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

---

## Simulation Execution Flow

```
START: ProbabilityEngine.run_simulation()
│
├─[1] PARAMETER VALIDATION
│     ├─ Validate bankroll > 0
│     ├─ Validate 0 < win_probability < 1
│     ├─ Validate bet <= bankroll
│     └─ Validate expected_return > 0
│
├─[2] STRATEGY CREATION
│     ├─ Look up strategy class in AVAILABLE_STRATEGIES
│     ├─ Merge base_params with strategy_params
│     └─ Instantiate strategy with parameters
│
├─[3] ENGINE INITIALIZATION
│     └─ MonteCarloEngine(strategy_class, params, max_rounds)
│
├─[4] SIMULATION LOOP (n = num_simulations)
│     │
│     ├─[4.1] For each simulation i:
│     │       │
│     │       ├─ Create fresh strategy instance
│     │       ├─ Initialize: rounds = 0, target_reached = False
│     │       │
│     │       ├─[4.2] ROUND LOOP (while rounds < max_rounds)
│     │       │       │
│     │       │       ├─ Generate outcome: random() < win_probability
│     │       │       ├─ Call strategy.process_round(won)
│     │       │       │   ├─ Update streak
│     │       │       │   ├─ Calculate winnings/losses
│     │       │       │   ├─ Apply matching rule
│     │       │       │   └─ Return (next_bet, bankroll)
│     │       │       │
│     │       │       ├─ Check stop conditions:
│     │       │       │   ├─ If bankroll >= target:
│     │       │       │   │   ├─ target_reached = True
│     │       │       │   │   └─ If stop_on_target: BREAK
│     │       │       │   │
│     │       │       │   └─ If bankroll <= 0:
│     │       │       │       ├─ bankruptcy = True
│     │       │       │       └─ If stop_on_bankruptcy: BREAK
│     │       │       │
│     │       │       └─ rounds++
│     │       │
│     │       └─[4.3] Create SimulationResult
│     │               ├─ survived = (bankruptcy == None)
│     │               ├─ final_bankroll
│     │               ├─ rounds_played
│     │               ├─ target_reached
│     │               └─ statistics
│     │
│     └─[4.4] Collect all SimulationResults
│
├─[5] STATISTICAL ANALYSIS
│     ├─ Calculate survival_rate = survived / total
│     ├─ Calculate target_success_rate = reached / total
│     ├─ Calculate percentiles (25th, 50th, 75th, 95th)
│     ├─ Calculate mean, median, std deviation
│     └─ Create SimulationSummary
│
└─[6] RETURN SimulationSummary
```

---

## Strategy Processing Flow

```
CALL: strategy.process_round(won=True/False)
│
├─[1] DETERMINE EVENT TYPE
│     └─ event = EventType.WIN if won else EventType.LOSE
│
├─[2] UPDATE STREAK TRACKING
│     │
│     ├─ If state.last_event == event:
│     │   └─ state.streak_count++
│     │
│     └─ Else:
│         ├─ state.streak_count = 1
│         └─ state.last_event = event
│
├─[3] CALCULATE ROUND OUTCOME
│     │
│     ├─ If won:
│     │   └─ amount = current_bet * (expected_return - 1)
│     │
│     └─ Else:
│         └─ amount = current_bet
│
├─[4] UPDATE STATISTICS
│     └─ state.update_stats(won, amount)
│         ├─ total_rounds++
│         ├─ Update wins/losses counter
│         ├─ Adjust current_bankroll
│         └─ Update max/min bankroll
│
├─[5] APPLY STRATEGY RULES
│     │
│     ├─ Sort rules by priority (descending)
│     │
│     └─ For each rule (highest priority first):
│         │
│         ├─ If rule.condition.evaluate(state, event):
│         │   ├─ rule.action.execute(state, base_bet, base_prob)
│         │   └─ BREAK (only first match applies)
│         │
│         └─ Continue to next rule
│
├─[6] ENFORCE CONSTRAINTS
│     └─ current_bet = min(current_bet, current_bankroll)
│
└─[7] RETURN (current_bet, current_bankroll)
```

---

## Rule Evaluation Flow

```
CALL: condition.evaluate(state, event)
│
├─[1] CHECK EVENT TYPE MATCH
│     │
│     ├─ If event != self.event_type:
│     │   └─ RETURN False
│     │
│     └─ Continue
│
├─[2] CHECK STREAK CONDITIONS
│     │
│     ├─ If self.streak_operator is None:
│     │   └─ RETURN True (no streak condition)
│     │
│     ├─ If state.last_event != event:
│     │   └─ RETURN False (not continuing streak)
│     │
│     └─ Continue
│
└─[3] COMPARE STREAK VALUE
      │
      └─ RETURN _compare_streak(state.streak_count, threshold)
          │
          ├─ GREATER: return count > threshold
          ├─ LESS: return count < threshold
          ├─ EQUAL: return count == threshold
          ├─ GREATER_EQUAL: return count >= threshold
          └─ LESS_EQUAL: return count <= threshold

CALL: action.execute(state, base_bet, base_win_prob)
│
├─[1] DETERMINE ACTION TYPE
│     │
│     ├─ SET_BET: state.current_bet = value
│     ├─ MULTIPLY_BET: state.current_bet *= value
│     ├─ ADD_BET: state.current_bet += value
│     ├─ RESET_BET: state.current_bet = base_bet
│     │
│     ├─ SET_WIN_PROB: 
│     │   ├─ state.win_probability = value
│     │   └─ _adjust_expected_return(state)
│     │
│     ├─ MULTIPLY_WIN_PROB:
│     │   ├─ state.win_probability *= value
│     │   └─ _adjust_expected_return(state)
│     │
│     ├─ ADD_WIN_PROB:
│     │   ├─ state.win_probability += value
│     │   └─ _adjust_expected_return(state)
│     │
│     └─ RESET_WIN_PROB:
│         ├─ state.win_probability = base_win_prob
│         └─ _adjust_expected_return(state)
│
└─[2] LOG ACTION (if enabled)
```

---

## Statistical Analysis Flow

```
CALL: MonteCarloEngine._calculate_summary(results)
│
├─[1] CATEGORIZE RESULTS
│     ├─ survived = [r for r if r.survived]
│     ├─ target_reached = [r for r if r.target_reached]
│     ├─ bankruptcies = [r for r if r.bankruptcy_round != None]
│     └─ final_bankrolls = [r.final_bankroll for all r]
│
├─[2] CALCULATE BASIC STATISTICS
│     ├─ total_runs = len(results)
│     ├─ survival_rate = len(survived) / total_runs
│     ├─ target_success_rate = len(target_reached) / total_runs
│     ├─ average_rounds = mean([r.rounds_played])
│     ├─ average_final_bankroll = mean(final_bankrolls)
│     └─ std_final_bankroll = std(final_bankrolls)
│
├─[3] CALCULATE PERCENTILES
│     └─ percentiles = numpy.percentile(final_bankrolls, 
│                                       [25, 50, 75, 95])
│         ├─ percentile_25 = percentiles[0]
│         ├─ median_final_bankroll = percentiles[1]
│         ├─ percentile_75 = percentiles[2]
│         └─ percentile_95 = percentiles[3]
│
├─[4] CALCULATE EXTREMES
│     ├─ max_bankroll_achieved = max([r.max_bankroll])
│     └─ min_bankroll_achieved = min([r.min_bankroll])
│
├─[5] CALCULATE CONDITIONAL AVERAGES
│     │
│     ├─ If bankruptcies exist:
│     │   └─ avg_bankruptcy_round = mean([r.bankruptcy_round])
│     │
│     └─ If target_reached exist:
│         └─ avg_target_round = mean([r.target_round])
│
└─[6] CREATE & RETURN SimulationSummary
      └─ Package all statistics into dataclass
```

---

## Optimization Flow

```
CALL: ProbabilityEngine.optimize_parameters()
│
├─[1] GENERATE PARAMETER SPACE
│     │
│     ├─ Extract parameter names and ranges
│     │   param_names = ['streak_threshold', 'multiplier1', ...]
│     │   param_values = [[2,3,4], [1.5,2.0,2.5], ...]
│     │
│     └─ Create all combinations (Cartesian product)
│         combinations = itertools.product(*param_values)
│         Example: [(2, 1.5), (2, 2.0), (3, 1.5), ...]
│
├─[2] INITIALIZE TRACKING
│     ├─ results = {}
│     ├─ best_result = None
│     ├─ best_params = None
│     └─ best_metric = 0
│
├─[3] TEST EACH COMBINATION
│     │
│     └─ For each combo in combinations:
│         │
│         ├─[3.1] Create parameter dictionary
│         │       test_params = dict(zip(param_names, combo))
│         │
│         ├─[3.2] Run simulation
│         │       summary = run_simulation(
│         │           strategy_type=strategy_type,
│         │           strategy_params=test_params,
│         │           num_simulations=num_simulations_per,
│         │           **base_params
│         │       )
│         │
│         ├─[3.3] Store result
│         │       results[combo] = summary
│         │
│         └─[3.4] Check if best
│                 metric_value = getattr(summary, target_metric)
│                 If metric_value > best_metric:
│                     ├─ best_metric = metric_value
│                     ├─ best_result = summary
│                     └─ best_params = test_params
│
├─[4] LOG BEST PARAMETERS
│     └─ logger.info(f"Best {target_metric}: {best_metric}")
│
└─[5] RETURN all results
      └─ Caller can analyze full parameter space
```

---

## Parallel Processing Flow

```
PARALLEL EXECUTION (when enabled and num_simulations > 100)
│
├─[1] CREATE PROCESS POOL
│     └─ ProcessPoolExecutor(max_workers=max_workers)
│
├─[2] SUBMIT TASKS
│     │
│     └─ For i in range(num_simulations):
│         └─ futures.append(
│               executor.submit(_run_single_simulation, i)
│           )
│
├─[3] COLLECT RESULTS
│     │
│     └─ For future in as_completed(futures):
│         │
│         ├─ result = future.result()
│         ├─ results.append(result)
│         │
│         └─ If len(results) % 1000 == 0:
│             └─ Log progress
│
└─[4] CLEANUP
      └─ Executor automatically closed via context manager

SEQUENTIAL EXECUTION (fallback)
│
└─ For i in range(num_simulations):
    ├─ result = _run_single_simulation(i)
    ├─ results.append(result)
    └─ Log progress every 1000
```

---

## Strategy Comparison Flow

```
CALL: ProbabilityEngine.compare_strategies()
│
├─[1] INITIALIZE RESULTS CONTAINER
│     └─ results = {}
│
├─[2] ITERATE THROUGH STRATEGIES
│     │
│     └─ For each strategy_config in strategies:
│         │
│         ├─[2.1] Extract configuration
│         │       ├─ name = config.get('name', config['type'])
│         │       ├─ type = config['type']
│         │       └─ params = config.get('params', {})
│         │
│         ├─[2.2] Run simulation
│         │       summary = run_simulation(
│         │           strategy_type=type,
│         │           strategy_params=params,
│         │           num_simulations=num_simulations,
│         │           **base_params
│         │       )
│         │
│         ├─[2.3] Store result
│         │       results[name] = summary
│         │
│         └─[2.4] Display summary
│                 print(f"Strategy: {name}")
│                 print(summary)
│
└─[3] RETURN comparison results
      └─ Dictionary: {strategy_name: SimulationSummary}
```

---

## Data Flow Through System

```
INPUT DATA
│
├─ User Parameters
│   ├─ starting_bankroll: 1000
│   ├─ base_bet: 10
│   ├─ target_bankroll: 2000
│   ├─ win_probability: 0.49
│   └─ expected_return: 2.0
│
└─ Strategy Config
    ├─ strategy_type: "streak_multiplier"
    └─ strategy_params:
        ├─ streak_threshold: 3
        ├─ multiplier1: 2.0
        └─ multiplier2: 1.5

PROCESSING PIPELINE
│
├─[1] Strategy Instantiation
│     └─ Creates BettingState with initial values
│
├─[2] Simulation Loop (per simulation)
│     │
│     ├─ Round Generation
│     │   └─ Random outcomes based on win_probability
│     │
│     ├─ State Updates
│     │   ├─ Bankroll changes
│     │   ├─ Bet adjustments
│     │   └─ Streak tracking
│     │
│     └─ Rule Applications
│         └─ Modifies state based on conditions
│
├─[3] Result Collection
│     └─ SimulationResult per simulation
│
└─[4] Statistical Aggregation
      └─ SimulationSummary

OUTPUT DATA
│
└─ SimulationSummary
    ├─ survival_rate: 0.359
    ├─ target_success_rate: 0.359
    ├─ average_final_bankroll: 719.46
    ├─ percentiles:
    │   ├─ 25th: 0.00
    │   ├─ 50th: 0.00
    │   ├─ 75th: 2002.50
    │   └─ 95th: 2007.50
    └─ ... other statistics
```

---

## Error Handling Flow

```
ERROR DETECTION & HANDLING
│
├─[1] INPUT VALIDATION ERRORS
│     │
│     ├─ Invalid Strategy Type
│     │   └─ ValueError("Unknown strategy: {type}")
│     │
│     ├─ Invalid Parameters
│     │   ├─ Negative bankroll/bet
│     │   ├─ Probability outside [0,1]
│     │   └─ ValueError("Invalid parameter: {param}")
│     │
│     └─ Missing Required Parameters
│         └─ TypeError("Missing required parameter: {param}")
│
├─[2] RUNTIME ERRORS
│     │
│     ├─ Division by Zero
│     │   └─ Handle in _adjust_expected_return()
│     │
│     ├─ Memory Errors
│     │   └─ Reduce simulation batch size
│     │
│     └─ Parallel Processing Errors
│         └─ Fallback to sequential processing
│
└─[3] RECOVERY STRATEGIES
      │
      ├─ Validation Errors
      │   └─ Return error message to user
      │
      ├─ Runtime Errors
      │   ├─ Log error details
      │   ├─ Attempt recovery
      │   └─ Graceful degradation
      │
      └─ Critical Errors
          ├─ Save partial results
          └─ Clean shutdown
```

---

## Memory Management Flow

```
MEMORY ALLOCATION & CLEANUP
│
├─[1] INITIALIZATION PHASE
│     ├─ Strategy class definition: ~1KB
│     ├─ Engine initialization: ~10KB
│     └─ Parameter storage: ~1KB
│
├─[2] SIMULATION PHASE (per simulation)
│     ├─ Strategy instance: ~5KB
│     ├─ BettingState: ~200 bytes
│     ├─ Rules list: ~1KB
│     └─ Temporary variables: ~500 bytes
│
├─[3] RESULT STORAGE
│     ├─ SimulationResult: ~100 bytes each
│     ├─ Results list (1000 sims): ~100KB
│     └─ Results list (10000 sims): ~1MB
│
├─[4] STATISTICAL ANALYSIS
│     ├─ NumPy arrays: 8 bytes * num_simulations
│     ├─ Percentile calculation: temporary ~2x array size
│     └─ Summary object: ~500 bytes
│
└─[5] CLEANUP
      ├─ Strategy instances: garbage collected per sim
      ├─ Results: kept until summary created
      └─ Summary: returned to caller
```
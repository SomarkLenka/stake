# Probability Engine - Complete Architecture Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [File Structure](#file-structure)
3. [Core Components](#core-components)
4. [Detailed Class Documentation](#detailed-class-documentation)
5. [Call Flow Diagrams](#call-flow-diagrams)
6. [Usage Examples](#usage-examples)

## System Overview

The Probability Engine is a modular Monte Carlo simulation framework for testing betting strategies. It uses a condition-action rule system that allows for flexible strategy definition and evaluation.

### Key Design Principles
- **Modularity**: Each component (conditions, actions, rules, strategies) is independent
- **Extensibility**: New strategies can be created by combining existing components
- **Performance**: Supports parallel processing for large-scale simulations
- **Analytics**: Comprehensive statistical analysis with percentile distributions

## File Structure

```
stake/
├── src/
│   ├── base_strategy.py          # Core abstract classes and data structures
│   ├── streak_multiplier_strategy.py  # Concrete strategy implementations
│   ├── monte_carlo_engine.py     # Simulation engine
│   ├── probability_engine.py     # Main orchestrator
│   ├── simple_demo.py           # Basic demonstration
│   └── custom_strategy_example.py # Custom strategy examples
├── tests/
│   └── test_strategies.py        # Unit tests
├── tools/
│   ├── demo.py                   # Extended demo
│   └── example_usage.py          # Usage examples
└── md/
    ├── CHANGELOG.md              # Change tracking
    └── ARCHITECTURE.md           # This document
```

## Core Components

### 1. Base Strategy Module (`base_strategy.py`)
Foundation classes that define the strategy framework.

### 2. Strategy Implementations (`streak_multiplier_strategy.py`)
Concrete betting strategies built on the base framework.

### 3. Monte Carlo Engine (`monte_carlo_engine.py`)
Simulation runner that executes strategies over multiple iterations.

### 4. Probability Engine (`probability_engine.py`)
High-level orchestrator for running simulations and comparisons.

## Detailed Class Documentation

---

# base_strategy.py

## Enums

### `EventType`
Represents the outcome of a betting round.

**Values:**
- `WIN = "win"` - Player won the round
- `LOSE = "lose"` - Player lost the round

### `ComparisonOperator`
Operators for comparing values in conditions.

**Values:**
- `GREATER = ">"` - Greater than comparison
- `LESS = "<"` - Less than comparison
- `EQUAL = "="` - Equal to comparison
- `GREATER_EQUAL = ">="` - Greater than or equal
- `LESS_EQUAL = "<="` - Less than or equal

### `ActionType`
Types of actions that can be taken when a condition is met.

**Values:**
- `SET_BET = "set_bet"` - Set bet to specific value
- `MULTIPLY_BET = "multiply_bet"` - Multiply current bet
- `ADD_BET = "add_bet"` - Add value to current bet
- `RESET_BET = "reset_bet"` - Reset to base bet
- `SET_WIN_PROB = "set_win_prob"` - Set win probability
- `MULTIPLY_WIN_PROB = "multiply_win_prob"` - Multiply win probability
- `ADD_WIN_PROB = "add_win_prob"` - Add to win probability
- `RESET_WIN_PROB = "reset_win_prob"` - Reset to base probability

---

## Data Classes

### `BettingState`
Maintains the current state of the betting session.

**Attributes:**
- `current_bet: float` - Current bet amount
- `current_bankroll: float` - Current available funds
- `win_probability: float` - Current probability of winning (0-1)
- `expected_return: float` - Multiplier for winnings (e.g., 2.0 for 2x)
- `streak_count: int = 0` - Current streak length
- `last_event: Optional[EventType] = None` - Previous round outcome
- `total_rounds: int = 0` - Total rounds played
- `total_wins: int = 0` - Total wins
- `total_losses: int = 0` - Total losses
- `max_bankroll: float = 0` - Highest bankroll achieved
- `min_bankroll: float = float('inf')` - Lowest bankroll reached

**Methods:**

#### `update_stats(won: bool, amount: float) -> None`
Updates statistics after a betting round.

**Parameters:**
- `won`: Whether the round was won
- `amount`: Amount won or lost

**Side Effects:**
- Updates `total_rounds`, `total_wins`/`total_losses`
- Adjusts `current_bankroll`
- Updates `max_bankroll` and `min_bankroll`

---

### `Condition`
Defines a condition that triggers an action.

**Attributes:**
- `event_type: EventType` - Event that must occur (WIN/LOSE)
- `streak_operator: Optional[ComparisonOperator]` - How to compare streak
- `streak_threshold: Optional[int]` - Streak value to compare against

**Methods:**

#### `evaluate(state: BettingState, event: EventType) -> bool`
Checks if the condition is satisfied.

**Parameters:**
- `state`: Current betting state
- `event`: Event that just occurred

**Returns:**
- `True` if condition is met, `False` otherwise

**Logic:**
1. Check if event matches `event_type`
2. If no streak operator, return True
3. Check if continuing a streak
4. Compare streak count against threshold using operator

#### `_compare_streak(streak: int, threshold: int) -> bool`
Private method to compare streak values.

**Parameters:**
- `streak`: Current streak count
- `threshold`: Threshold to compare against

**Returns:**
- Result of comparison based on `streak_operator`

---

### `Action`
Represents an action to take when a condition is met.

**Attributes:**
- `action_type: ActionType` - Type of action to perform
- `value: Optional[float]` - Value to use in action (if applicable)

**Methods:**

#### `execute(state: BettingState, base_bet: float, base_win_prob: float) -> None`
Executes the action on the betting state.

**Parameters:**
- `state`: Current betting state to modify
- `base_bet`: Original base bet amount
- `base_win_prob`: Original win probability

**Side Effects:**
- Modifies `state` based on `action_type`
- Adjusts expected return when changing win probability

#### `_adjust_expected_return(state: BettingState) -> None`
Private method to recalculate expected return after probability change.

**Parameters:**
- `state`: Betting state to update

**Logic:**
- Uses inverse relationship: `return = base_return * (0.5 / win_prob)`
- Maintains fair odds when probability changes

---

### `Rule`
Combines a condition with an action.

**Attributes:**
- `condition: Condition` - Condition to check
- `action: Action` - Action to execute if condition is met
- `priority: int = 0` - Higher priority rules are evaluated first

---

## Abstract Base Class

### `BaseStrategy`
Abstract base class for all betting strategies.

**Constructor Parameters:**
- `starting_bankroll: float` - Initial funds
- `base_bet: float` - Default bet amount
- `target_bankroll: float` - Goal to reach
- `win_probability: float` - Probability of winning each round
- `expected_return: float` - Payout multiplier
- `*args, **kwargs` - Additional strategy-specific parameters

**Attributes:**
- `starting_bankroll: float` - Initial bankroll
- `base_bet: float` - Base betting amount
- `target_bankroll: float` - Target to reach
- `base_win_probability: float` - Original win probability
- `base_expected_return: float` - Original expected return
- `state: BettingState` - Current betting state
- `rules: List[Rule]` - Strategy rules (populated by subclass)

**Abstract Methods:**

#### `_initialize_rules() -> None`
Must be implemented by subclasses to define strategy rules.

**Methods:**

#### `_process_additional_params(*args, **kwargs) -> None`
Optional method for subclasses to handle extra parameters.

#### `process_round(won: bool) -> Tuple[float, float]`
Processes a betting round and updates state.

**Parameters:**
- `won`: Whether the round was won

**Returns:**
- Tuple of (next_bet_amount, current_bankroll)

**Logic Flow:**
1. Determine event type (WIN/LOSE)
2. Update streak count
3. Calculate winnings/losses
4. Update statistics
5. Apply first matching rule (by priority)
6. Cap bet at available bankroll
7. Return next bet and bankroll

#### `reset() -> None`
Resets strategy to initial state.

**Side Effects:**
- Creates new `BettingState` with initial values

#### `get_stats() -> Dict[str, Any]`
Returns current statistics.

**Returns Dictionary:**
- `total_rounds`: Rounds played
- `total_wins`: Number of wins
- `total_losses`: Number of losses
- `current_bankroll`: Current funds
- `max_bankroll`: Peak bankroll
- `min_bankroll`: Lowest bankroll
- `win_rate`: Win percentage
- `current_bet`: Next bet amount
- `current_streak`: Current streak length
- `last_event`: Previous outcome

---

# streak_multiplier_strategy.py

## Concrete Strategy Implementations

### `StreakMultiplierStrategy(BaseStrategy)`
Strategy that adjusts bet multipliers based on losing streak length.

**Constructor Parameters:**
- All `BaseStrategy` parameters
- `streak_threshold: int = 3` - Streak length to switch multipliers
- `multiplier1: float = 2.0` - Multiplier for streak < threshold
- `multiplier2: float = 1.5` - Multiplier for streak >= threshold

**Rule Implementation:**
1. **Priority 10**: On WIN → Reset bet to base
2. **Priority 5**: On LOSE with streak >= threshold → Multiply by multiplier2
3. **Priority 4**: On LOSE with streak < threshold → Multiply by multiplier1

**Behavior:**
- Aggressive betting on short losing streaks
- Conservative betting on long losing streaks
- Reset on any win

---

### `MartingaleStrategy(BaseStrategy)`
Classic Martingale doubling strategy.

**Constructor Parameters:**
- All `BaseStrategy` parameters
- `multiplier: float = 2.0` - Bet multiplier on loss

**Rule Implementation:**
1. **Priority 10**: On WIN → Reset bet to base
2. **Priority 5**: On LOSE → Multiply bet by multiplier

**Behavior:**
- Double (or multiply) bet after each loss
- Reset to base bet after win
- High risk, high reward

---

### `ReverseMartingaleStrategy(BaseStrategy)`
Paroli system - increase bets on wins.

**Constructor Parameters:**
- All `BaseStrategy` parameters
- `multiplier: float = 2.0` - Bet multiplier on win
- `max_streak: int = 3` - Maximum win streak before reset

**Rule Implementation:**
1. **Priority 10**: On LOSE → Reset bet to base
2. **Priority 8**: On WIN with streak >= max_streak → Reset bet
3. **Priority 5**: On WIN with streak < max_streak → Multiply bet

**Behavior:**
- Increase bets during winning streaks
- Reset on any loss
- Cap winnings at max_streak

---

### `CustomRuleStrategy(BaseStrategy)`
Fully customizable strategy with user-defined rules.

**Constructor Parameters:**
- All `BaseStrategy` parameters
- `rules: List[Rule]` - Custom rule list

**Implementation:**
- Uses provided rules directly
- No predefined behavior
- Maximum flexibility

---

# monte_carlo_engine.py

## Data Classes

### `SimulationResult`
Results from a single simulation run.

**Attributes:**
- `survived: bool` - Whether bankroll stayed positive
- `final_bankroll: float` - Ending bankroll
- `rounds_played: int` - Total rounds in simulation
- `max_bankroll: float` - Peak bankroll achieved
- `min_bankroll: float` - Lowest bankroll reached
- `target_reached: bool` - Whether target was achieved
- `bankruptcy_round: Optional[int]` - Round of bankruptcy (if occurred)
- `target_round: Optional[int]` - Round target was reached (if achieved)
- `win_rate: float` - Percentage of rounds won

---

### `SimulationSummary`
Aggregated statistics from all simulations.

**Attributes:**
- `total_runs: int` - Number of simulations
- `survival_rate: float` - Percentage that avoided bankruptcy
- `target_success_rate: float` - Percentage that reached target
- `average_rounds: float` - Mean rounds per simulation
- `average_final_bankroll: float` - Mean ending bankroll
- `median_final_bankroll: float` - Median ending bankroll
- `std_final_bankroll: float` - Standard deviation of bankrolls
- `max_bankroll_achieved: float` - Highest bankroll across all sims
- `min_bankroll_achieved: float` - Lowest bankroll across all sims
- `average_bankruptcy_round: Optional[float]` - Mean round of bankruptcy
- `average_target_round: Optional[float]` - Mean round target reached
- `percentile_25: float` - 25th percentile bankroll
- `percentile_75: float` - 75th percentile bankroll
- `percentile_95: float` - 95th percentile bankroll

**Methods:**

#### `__str__() -> str`
Formats summary as readable report.

---

## Main Class

### `MonteCarloEngine`
Engine for running Monte Carlo simulations.

**Constructor Parameters:**
- `strategy_class: Type[BaseStrategy]` - Strategy class to simulate
- `strategy_params: Dict[str, Any]` - Parameters for strategy
- `max_rounds: int = 10000` - Maximum rounds per simulation
- `stop_on_bankruptcy: bool = True` - Stop if bankrupt
- `stop_on_target: bool = True` - Stop if target reached
- `seed: Optional[int] = None` - Random seed for reproducibility

**Methods:**

#### `_run_single_simulation(sim_id: int) -> SimulationResult`
Executes one simulation run.

**Parameters:**
- `sim_id`: Simulation identifier for logging

**Returns:**
- `SimulationResult` with simulation outcomes

**Logic Flow:**
1. Create fresh strategy instance
2. Loop up to max_rounds:
   - Generate random outcome based on win_probability
   - Process round through strategy
   - Check for target reached
   - Check for bankruptcy
   - Break if stop condition met
3. Compile and return results

#### `run_simulations(num_simulations: int, parallel: bool, max_workers: Optional[int]) -> SimulationSummary`
Runs multiple simulations.

**Parameters:**
- `num_simulations`: Number of simulations to run
- `parallel`: Whether to use multiprocessing
- `max_workers`: Maximum parallel workers

**Returns:**
- `SimulationSummary` with aggregated statistics

**Logic Flow:**
1. Initialize results list
2. If parallel and num_simulations > 100:
   - Use ProcessPoolExecutor
   - Submit simulations to worker pool
   - Collect results as completed
3. Else run sequentially
4. Calculate summary statistics
5. Return summary

#### `_calculate_summary(results: List[SimulationResult]) -> SimulationSummary`
Aggregates individual results into summary.

**Parameters:**
- `results`: List of simulation results

**Returns:**
- `SimulationSummary` with statistics

**Calculations:**
- Survival rate = survived / total
- Success rate = reached_target / total
- Percentiles using numpy
- Averages for rounds, bankrolls, bankruptcy/target rounds

#### `run_probability_sweep(probabilities: List[float], expected_returns: List[float], num_simulations_per: int) -> Dict[tuple, SimulationSummary]`
Tests multiple probability/return combinations.

**Parameters:**
- `probabilities`: Win probabilities to test
- `expected_returns`: Payout multipliers to test
- `num_simulations_per`: Simulations per combination

**Returns:**
- Dictionary mapping (probability, return) to summary

**Logic:**
- Nested loop through probabilities and returns
- Create temporary engine for each combination
- Run simulations and store results

---

# probability_engine.py

## Constants

### `AVAILABLE_STRATEGIES`
Dictionary mapping strategy names to classes.

```python
{
    "streak_multiplier": StreakMultiplierStrategy,
    "martingale": MartingaleStrategy,
    "reverse_martingale": ReverseMartingaleStrategy,
    "custom": CustomRuleStrategy
}
```

## Main Class

### `ProbabilityEngine`
High-level orchestrator for simulations.

**Constructor Parameters:**
- `config_file: Optional[str] = None` - JSON config file path

**Methods:**

#### `load_config(config_file: str) -> None`
Loads configuration from JSON file.

**Parameters:**
- `config_file`: Path to configuration file

**Side Effects:**
- Updates `self.config` dictionary

#### `create_strategy(strategy_type: str, **params) -> BaseStrategy`
Factory method to create strategy instances.

**Parameters:**
- `strategy_type`: Name of strategy ("martingale", "custom", etc.)
- `starting_bankroll`: Initial funds
- `base_bet`: Base bet amount
- `target_bankroll`: Target to reach
- `win_probability`: Win probability
- `expected_return`: Payout multiplier
- `**kwargs`: Additional strategy-specific parameters

**Returns:**
- Configured strategy instance

**Raises:**
- `ValueError` if unknown strategy type

#### `run_simulation(**params) -> SimulationSummary`
Runs a Monte Carlo simulation.

**Parameters:**
- `strategy_type: str` - Strategy to use
- `starting_bankroll: float` - Initial funds
- `base_bet: float` - Base bet
- `target_bankroll: float` - Target goal
- `win_probability: float` - Win probability
- `expected_return: float` - Payout multiplier
- `num_simulations: int` - Number of runs
- `max_rounds: int` - Max rounds per simulation
- `strategy_params: Dict` - Strategy-specific params
- `**kwargs`: Additional parameters

**Returns:**
- `SimulationSummary` with results

**Logic Flow:**
1. Merge all parameters
2. Get strategy class
3. Create MonteCarloEngine
4. Run simulations
5. Return summary

#### `compare_strategies(strategies: List[Dict], base_params: Dict, num_simulations: int) -> Dict[str, SimulationSummary]`
Compares multiple strategies.

**Parameters:**
- `strategies`: List of strategy configurations
- `base_params`: Common parameters for all strategies
- `num_simulations`: Simulations per strategy

**Returns:**
- Dictionary mapping strategy names to summaries

**Strategy Configuration Format:**
```python
{
    'name': 'Strategy Name',
    'type': 'strategy_type',
    'params': {...}  # Strategy-specific parameters
}
```

#### `optimize_parameters(strategy_type: str, base_params: Dict, param_ranges: Dict, num_simulations_per: int, target_metric: str) -> Dict[tuple, SimulationSummary]`
Optimizes strategy parameters.

**Parameters:**
- `strategy_type`: Strategy to optimize
- `base_params`: Base configuration
- `param_ranges`: Parameter ranges to test
- `num_simulations_per`: Simulations per combination
- `target_metric`: Metric to optimize ("survival_rate", etc.)

**Returns:**
- Dictionary mapping parameter tuples to summaries

**Logic:**
1. Generate all parameter combinations
2. Test each combination
3. Track best performing combination
4. Return all results

#### `main() -> None`
Command-line interface entry point.

**Command-line Arguments:**
- `--strategy, -s`: Strategy type
- `--bankroll, -b`: Starting bankroll
- `--bet`: Base bet amount
- `--target, -t`: Target bankroll
- `--probability, -p`: Win probability
- `--return, -r`: Expected return
- `--simulations, -n`: Number of simulations
- `--max-rounds`: Maximum rounds
- `--streak-threshold`: Streak threshold (streak_multiplier)
- `--multiplier1`: First multiplier (streak_multiplier)
- `--multiplier2`: Second multiplier (streak_multiplier)
- `--multiplier`: General multiplier (martingale)
- `--compare`: Compare multiple strategies
- `--optimize`: Optimize parameters
- `--config`: Load from config file

---

# Call Flow Diagrams

## 1. Single Simulation Flow

```
User → ProbabilityEngine.run_simulation()
    ├→ Create MonteCarloEngine(strategy_class, params)
    └→ MonteCarloEngine.run_simulations(n)
        ├→ Loop n times:
        │   └→ _run_single_simulation(i)
        │       ├→ Create strategy instance
        │       └→ Loop max_rounds:
        │           ├→ Generate random outcome
        │           ├→ strategy.process_round(won)
        │           │   ├→ Update streak
        │           │   ├→ Calculate winnings/losses
        │           │   ├→ Update statistics
        │           │   └→ Apply matching rule:
        │           │       ├→ condition.evaluate()
        │           │       └→ action.execute()
        │           └→ Check stop conditions
        └→ _calculate_summary(results)
            └→ Return SimulationSummary
```

## 2. Strategy Processing Flow

```
BaseStrategy.process_round(won)
    ├→ Determine EventType (WIN/LOSE)
    ├→ Update streak count
    │   ├→ If same as last_event: increment
    │   └→ Else: reset to 1
    ├→ Calculate amount won/lost
    ├→ state.update_stats(won, amount)
    ├→ Sort rules by priority
    ├→ For each rule:
    │   ├→ rule.condition.evaluate(state, event)
    │   │   ├→ Check event_type match
    │   │   ├→ Check streak operator (if any)
    │   │   └→ Return True/False
    │   └→ If True:
    │       ├→ rule.action.execute(state)
    │       └→ Break (only first match)
    └→ Cap bet at bankroll
```

## 3. Rule Evaluation Flow

```
Condition.evaluate(state, event)
    ├→ Check: event == self.event_type?
    │   └→ No: return False
    ├→ Check: self.streak_operator is None?
    │   └→ Yes: return True
    ├→ Check: state.last_event == event?
    │   └→ No: return False (not continuing streak)
    └→ _compare_streak(state.streak_count, self.streak_threshold)
        └→ Apply operator comparison
```

## 4. Strategy Comparison Flow

```
ProbabilityEngine.compare_strategies(strategies, base_params, n)
    ├→ For each strategy config:
    │   ├→ Extract strategy type and params
    │   └→ run_simulation(type, params, base_params, n)
    │       └→ [Single Simulation Flow]
    ├→ Collect all summaries
    └→ Return results dictionary
```

## 5. Parameter Optimization Flow

```
ProbabilityEngine.optimize_parameters(strategy, base, ranges, n, metric)
    ├→ Generate parameter combinations (itertools.product)
    ├→ For each combination:
    │   ├→ Create test_params dictionary
    │   ├→ run_simulation(strategy, test_params, base, n)
    │   ├→ Extract metric value
    │   └→ Track if best so far
    └→ Return all results with best highlighted
```

---

# Usage Examples

## Basic Usage

```python
from probability_engine import ProbabilityEngine

# Create engine
engine = ProbabilityEngine()

# Run simple simulation
summary = engine.run_simulation(
    strategy_type="streak_multiplier",
    starting_bankroll=1000,
    base_bet=10,
    target_bankroll=2000,
    win_probability=0.49,
    expected_return=2.0,
    num_simulations=1000,
    streak_threshold=3,
    multiplier1=2.0,
    multiplier2=1.5
)

print(f"Survival Rate: {summary.survival_rate:.1%}")
```

## Creating Custom Strategies

```python
from base_strategy import Condition, Action, Rule, EventType, ActionType, ComparisonOperator

# Define custom rules
rules = [
    # On win, reduce bet by half
    Rule(
        condition=Condition(event_type=EventType.WIN),
        action=Action(action_type=ActionType.MULTIPLY_BET, value=0.5),
        priority=10
    ),
    # On loss streak > 5, reset
    Rule(
        condition=Condition(
            event_type=EventType.LOSE,
            streak_operator=ComparisonOperator.GREATER,
            streak_threshold=5
        ),
        action=Action(action_type=ActionType.RESET_BET),
        priority=9
    )
]

# Use custom strategy
summary = engine.run_simulation(
    strategy_type="custom",
    rules=rules,
    # ... other parameters
)
```

## Comparing Strategies

```python
strategies = [
    {'name': 'Conservative', 'type': 'streak_multiplier', 
     'params': {'multiplier1': 1.5, 'multiplier2': 1.2}},
    {'name': 'Aggressive', 'type': 'martingale',
     'params': {'multiplier': 3.0}}
]

results = engine.compare_strategies(
    strategies=strategies,
    base_params={'starting_bankroll': 1000, 'base_bet': 10},
    num_simulations=1000
)
```

## Parameter Optimization

```python
# Find optimal parameters
results = engine.optimize_parameters(
    strategy_type='streak_multiplier',
    base_params={'starting_bankroll': 1000},
    param_ranges={
        'streak_threshold': [2, 3, 4, 5],
        'multiplier1': [1.5, 2.0, 2.5],
        'multiplier2': [1.1, 1.3, 1.5]
    },
    num_simulations_per=500,
    target_metric='survival_rate'
)
```

---

# Testing Documentation

## test_strategies.py

### Test Classes

#### `TestBettingState`
Tests the BettingState data class.

**Test Methods:**
- `test_initial_state()` - Verifies initial values
- `test_update_stats_win()` - Tests win statistics update
- `test_update_stats_loss()` - Tests loss statistics update
- `test_max_min_tracking()` - Tests bankroll tracking

#### `TestConditions`
Tests Condition evaluation logic.

**Test Methods:**
- `test_simple_event_condition()` - Tests event-only conditions
- `test_streak_condition()` - Tests streak-based conditions

#### `TestActions`
Tests Action execution.

**Test Methods:**
- `test_set_bet_action()` - Tests SET_BET action
- `test_multiply_bet_action()` - Tests MULTIPLY_BET action
- `test_reset_bet_action()` - Tests RESET_BET action
- `test_win_prob_actions()` - Tests probability actions

#### `TestStreakMultiplierStrategy`
Tests the StreakMultiplierStrategy.

**Test Methods:**
- `test_initialization()` - Verifies proper setup
- `test_win_resets_bet()` - Tests win behavior
- `test_loss_below_threshold()` - Tests multiplier1 application
- `test_loss_above_threshold()` - Tests multiplier2 application
- `test_bet_capped_at_bankroll()` - Tests bankroll constraint

#### `TestMartingaleStrategy`
Tests classic Martingale.

**Test Methods:**
- `test_double_on_loss()` - Tests doubling behavior
- `test_reset_on_win()` - Tests win reset

#### `TestReverseMartingaleStrategy`
Tests Paroli system.

**Test Methods:**
- `test_double_on_win()` - Tests win doubling
- `test_reset_on_loss()` - Tests loss reset
- `test_max_streak_limit()` - Tests streak cap

#### `TestCustomRuleStrategy`
Tests custom rule strategies.

**Test Methods:**
- `test_custom_rules()` - Tests arbitrary rule combinations

---

# Performance Characteristics

## Time Complexity

### Strategy Operations
- `process_round()`: O(n) where n = number of rules
- `condition.evaluate()`: O(1)
- `action.execute()`: O(1)
- `get_stats()`: O(1)

### Simulation Operations
- Single simulation: O(r) where r = rounds played
- Multiple simulations: O(s * r) where s = simulations
- Parallel simulations: O(s * r / p) where p = processors
- Summary calculation: O(s) for statistics, O(s log s) for percentiles

## Space Complexity

### Per Strategy Instance
- O(n) where n = number of rules
- BettingState: O(1) fixed size

### Per Simulation
- SimulationResult: O(1) fixed size
- Strategy instance: O(n) rules

### Monte Carlo Engine
- Results storage: O(s) where s = simulations
- Summary: O(1) fixed size

## Optimization Opportunities

1. **Rule Caching**: Pre-sort rules by priority
2. **Vectorization**: Use NumPy for bulk operations
3. **Memory Pool**: Reuse strategy instances
4. **Lazy Evaluation**: Calculate statistics on-demand
5. **Progressive Sampling**: Stop early if results converge

---

# Configuration Files

## Example JSON Configuration

```json
{
  "simulation": {
    "num_simulations": 10000,
    "max_rounds": 10000,
    "stop_on_bankruptcy": true,
    "stop_on_target": true
  },
  "base_params": {
    "starting_bankroll": 1000,
    "base_bet": 10,
    "target_bankroll": 2000,
    "win_probability": 0.49,
    "expected_return": 2.0
  },
  "strategies": [
    {
      "name": "Optimized Streak",
      "type": "streak_multiplier",
      "params": {
        "streak_threshold": 3,
        "multiplier1": 2.0,
        "multiplier2": 1.5
      }
    }
  ]
}
```

---

# Error Handling

## Common Exceptions

### ValueError
- Unknown strategy type
- Invalid parameter ranges
- Negative bankroll/bet values

### TypeError
- Missing required parameters
- Wrong parameter types

### RuntimeError
- Simulation failures
- Parallel processing errors

## Validation

### Input Validation
- Bankroll > 0
- Bet > 0 and <= bankroll
- 0 < win_probability < 1
- expected_return > 0
- num_simulations > 0

### State Validation
- Bet never exceeds bankroll
- Streak count >= 0
- Statistics consistency

---

# Extending the System

## Adding New Conditions

1. Create new ComparisonOperator values if needed
2. Extend Condition class with new evaluation logic
3. Update condition.evaluate() method

## Adding New Actions

1. Add new ActionType enum value
2. Implement execution in action.execute()
3. Handle state modifications appropriately

## Adding New Strategies

1. Subclass BaseStrategy
2. Implement _initialize_rules()
3. Add to AVAILABLE_STRATEGIES dictionary
4. Optionally override _process_additional_params()

## Adding New Metrics

1. Add fields to SimulationResult
2. Calculate in _run_single_simulation()
3. Aggregate in _calculate_summary()
4. Add to SimulationSummary

---

# Best Practices

## Strategy Design
- Keep rules simple and testable
- Order rules by priority carefully
- Avoid conflicting conditions
- Test edge cases thoroughly

## Simulation Setup
- Use sufficient simulations (1000+ for statistics)
- Set reasonable max_rounds to prevent infinite loops
- Consider parallel processing for large batches
- Use seeds for reproducible results

## Performance
- Minimize logging in tight loops
- Use parallel processing for >100 simulations
- Cache frequently accessed values
- Profile before optimizing

## Testing
- Unit test each component separately
- Integration test strategy behavior
- Validate statistical properties
- Test edge cases and error conditions
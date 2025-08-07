# Probability Engine - API Reference

## Quick Start

```python
from probability_engine import ProbabilityEngine

# Initialize engine
engine = ProbabilityEngine()

# Run simulation with default Streak Multiplier strategy
summary = engine.run_simulation(
    strategy_type="streak_multiplier",
    starting_bankroll=1000,
    base_bet=10,
    target_bankroll=2000,
    win_probability=0.49,
    expected_return=2.0,
    num_simulations=1000
)

print(f"Survival Rate: {summary.survival_rate:.1%}")
print(f"Target Success: {summary.target_success_rate:.1%}")
```

---

## Module: `base_strategy`

### Enumerations

#### `EventType`
```python
class EventType(Enum):
    WIN = "win"   # Round won
    LOSE = "lose" # Round lost
```

#### `ComparisonOperator`
```python
class ComparisonOperator(Enum):
    GREATER = ">"          # Greater than
    LESS = "<"            # Less than  
    EQUAL = "="           # Equal to
    GREATER_EQUAL = ">="  # Greater or equal
    LESS_EQUAL = "<="     # Less or equal
```

#### `ActionType`
```python
class ActionType(Enum):
    SET_BET = "set_bet"                      # Set to specific value
    MULTIPLY_BET = "multiply_bet"            # Multiply current bet
    ADD_BET = "add_bet"                      # Add to current bet
    RESET_BET = "reset_bet"                  # Reset to base bet
    SET_WIN_PROB = "set_win_prob"            # Set win probability
    MULTIPLY_WIN_PROB = "multiply_win_prob"  # Multiply win probability
    ADD_WIN_PROB = "add_win_prob"            # Add to win probability
    RESET_WIN_PROB = "reset_win_prob"        # Reset to base probability
```

### Classes

#### `BettingState`
```python
@dataclass
class BettingState:
    current_bet: float              # Current bet amount
    current_bankroll: float         # Available funds
    win_probability: float          # Win probability (0-1)
    expected_return: float          # Payout multiplier
    streak_count: int = 0           # Current streak length
    last_event: Optional[EventType] = None  # Previous outcome
    total_rounds: int = 0           # Total rounds played
    total_wins: int = 0             # Total wins
    total_losses: int = 0           # Total losses
    max_bankroll: float = 0         # Peak bankroll
    min_bankroll: float = float('inf')  # Lowest bankroll
    
    def update_stats(self, won: bool, amount: float) -> None:
        """Update statistics after a round"""
```

#### `Condition`
```python
@dataclass
class Condition:
    event_type: EventType                        # Required event
    streak_operator: Optional[ComparisonOperator] = None  # Comparison
    streak_threshold: Optional[int] = None       # Threshold value
    
    def evaluate(self, state: BettingState, event: EventType) -> bool:
        """Check if condition is satisfied"""
```

#### `Action`
```python
@dataclass
class Action:
    action_type: ActionType          # Action to perform
    value: Optional[float] = None    # Value for action
    
    def execute(self, state: BettingState, base_bet: float, 
                base_win_prob: float) -> None:
        """Execute action on betting state"""
```

#### `Rule`
```python
@dataclass
class Rule:
    condition: Condition  # Condition to check
    action: Action       # Action if condition met
    priority: int = 0    # Higher priority evaluated first
```

#### `BaseStrategy` (Abstract)
```python
class BaseStrategy(ABC):
    def __init__(self,
                 starting_bankroll: float,
                 base_bet: float,
                 target_bankroll: float,
                 win_probability: float,
                 expected_return: float,
                 *args, **kwargs):
        """Initialize betting strategy"""
    
    @abstractmethod
    def _initialize_rules(self) -> None:
        """Define strategy rules (implement in subclass)"""
    
    def process_round(self, won: bool) -> Tuple[float, float]:
        """Process betting round
        Returns: (next_bet, current_bankroll)"""
    
    def reset(self) -> None:
        """Reset to initial state"""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
```

---

## Module: `streak_multiplier_strategy`

### Classes

#### `StreakMultiplierStrategy`
```python
class StreakMultiplierStrategy(BaseStrategy):
    def __init__(self,
                 starting_bankroll: float,
                 base_bet: float,
                 target_bankroll: float,
                 win_probability: float,
                 expected_return: float,
                 streak_threshold: int = 3,    # Switch at this streak
                 multiplier1: float = 2.0,      # For streak < threshold
                 multiplier2: float = 1.5,      # For streak >= threshold
                 *args, **kwargs):
        """Streak-based multiplier strategy"""
```

**Behavior:**
- WIN: Reset to base bet
- LOSE (streak < threshold): Multiply by `multiplier1`
- LOSE (streak >= threshold): Multiply by `multiplier2`

#### `MartingaleStrategy`
```python
class MartingaleStrategy(BaseStrategy):
    def __init__(self,
                 starting_bankroll: float,
                 base_bet: float,
                 target_bankroll: float,
                 win_probability: float,
                 expected_return: float,
                 multiplier: float = 2.0,  # Bet multiplier on loss
                 *args, **kwargs):
        """Classic Martingale doubling strategy"""
```

**Behavior:**
- WIN: Reset to base bet
- LOSE: Multiply bet by `multiplier`

#### `ReverseMartingaleStrategy`
```python
class ReverseMartingaleStrategy(BaseStrategy):
    def __init__(self,
                 starting_bankroll: float,
                 base_bet: float,
                 target_bankroll: float,
                 win_probability: float,
                 expected_return: float,
                 multiplier: float = 2.0,   # Bet multiplier on win
                 max_streak: int = 3,       # Max wins before reset
                 *args, **kwargs):
        """Paroli system - increase on wins"""
```

**Behavior:**
- LOSE: Reset to base bet
- WIN (streak < max_streak): Multiply by `multiplier`
- WIN (streak >= max_streak): Reset to base bet

#### `CustomRuleStrategy`
```python
class CustomRuleStrategy(BaseStrategy):
    def __init__(self,
                 starting_bankroll: float,
                 base_bet: float,
                 target_bankroll: float,
                 win_probability: float,
                 expected_return: float,
                 rules: List[Rule] = None,  # Custom rules
                 *args, **kwargs):
        """Fully customizable strategy"""
```

---

## Module: `monte_carlo_engine`

### Classes

#### `SimulationResult`
```python
@dataclass
class SimulationResult:
    survived: bool                    # Avoided bankruptcy
    final_bankroll: float            # Ending bankroll
    rounds_played: int               # Total rounds
    max_bankroll: float              # Peak bankroll
    min_bankroll: float              # Lowest bankroll
    target_reached: bool             # Reached target
    bankruptcy_round: Optional[int]  # When bankrupt
    target_round: Optional[int]      # When target reached
    win_rate: float = 0.0           # Win percentage
```

#### `SimulationSummary`
```python
@dataclass
class SimulationSummary:
    total_runs: int                  # Number of simulations
    survival_rate: float             # % avoiding bankruptcy
    target_success_rate: float       # % reaching target
    average_rounds: float            # Mean rounds
    average_final_bankroll: float   # Mean ending bankroll
    median_final_bankroll: float    # Median bankroll
    std_final_bankroll: float       # Std deviation
    max_bankroll_achieved: float    # Highest across all
    min_bankroll_achieved: float    # Lowest across all
    average_bankruptcy_round: Optional[float]  # Mean bankruptcy
    average_target_round: Optional[float]      # Mean target
    percentile_25: float            # 25th percentile
    percentile_75: float            # 75th percentile
    percentile_95: float            # 95th percentile
    
    def __str__(self) -> str:
        """Format as readable report"""
```

#### `MonteCarloEngine`
```python
class MonteCarloEngine:
    def __init__(self,
                 strategy_class: Type[BaseStrategy],
                 strategy_params: Dict[str, Any],
                 max_rounds: int = 10000,
                 stop_on_bankruptcy: bool = True,
                 stop_on_target: bool = True,
                 seed: Optional[int] = None):
        """Initialize Monte Carlo engine"""
    
    def run_simulations(self,
                       num_simulations: int = 10000,
                       parallel: bool = True,
                       max_workers: Optional[int] = None
                       ) -> SimulationSummary:
        """Run multiple simulations
        Returns: Aggregated summary statistics"""
    
    def run_probability_sweep(self,
                             probabilities: List[float],
                             expected_returns: List[float],
                             num_simulations_per: int = 1000
                             ) -> Dict[tuple, SimulationSummary]:
        """Test multiple probability/return combinations
        Returns: Dict mapping (prob, return) to summary"""
```

---

## Module: `probability_engine`

### Classes

#### `ProbabilityEngine`
```python
class ProbabilityEngine:
    def __init__(self, config_file: Optional[str] = None):
        """Initialize probability engine
        Args:
            config_file: Optional JSON configuration file"""
    
    def load_config(self, config_file: str) -> None:
        """Load configuration from JSON file"""
    
    def create_strategy(self,
                       strategy_type: str,
                       starting_bankroll: float,
                       base_bet: float,
                       target_bankroll: float,
                       win_probability: float,
                       expected_return: float,
                       **kwargs) -> BaseStrategy:
        """Factory method to create strategies
        
        Args:
            strategy_type: One of ['streak_multiplier', 'martingale', 
                          'reverse_martingale', 'custom']
            **kwargs: Strategy-specific parameters
        
        Returns:
            Configured strategy instance"""
    
    def run_simulation(self,
                      strategy_type: str = "streak_multiplier",
                      starting_bankroll: float = 1000,
                      base_bet: float = 10,
                      target_bankroll: float = 2000,
                      win_probability: float = 0.49,
                      expected_return: float = 2.0,
                      num_simulations: int = 10000,
                      max_rounds: int = 10000,
                      strategy_params: Dict[str, Any] = None,
                      **kwargs) -> SimulationSummary:
        """Run Monte Carlo simulation
        
        Returns:
            SimulationSummary with results"""
    
    def compare_strategies(self,
                          strategies: List[Dict[str, Any]],
                          base_params: Dict[str, Any],
                          num_simulations: int = 10000
                          ) -> Dict[str, SimulationSummary]:
        """Compare multiple strategies
        
        Args:
            strategies: List of strategy configs
            base_params: Common parameters
            num_simulations: Runs per strategy
        
        Returns:
            Dict mapping names to summaries"""
    
    def optimize_parameters(self,
                           strategy_type: str,
                           base_params: Dict[str, Any],
                           param_ranges: Dict[str, List[Any]],
                           num_simulations_per: int = 1000,
                           target_metric: str = "survival_rate"
                           ) -> Dict[tuple, SimulationSummary]:
        """Optimize strategy parameters
        
        Args:
            strategy_type: Strategy to optimize
            base_params: Base configuration
            param_ranges: Parameters to test
            num_simulations_per: Runs per combo
            target_metric: Metric to maximize
        
        Returns:
            Dict mapping param tuples to summaries"""
```

---

## Complete Example: Custom Strategy

```python
from probability_engine import ProbabilityEngine
from base_strategy import (
    Condition, Action, Rule,
    EventType, ComparisonOperator, ActionType
)

# Create complex custom rules
rules = [
    # Rule 1: Reset on win
    Rule(
        condition=Condition(event_type=EventType.WIN),
        action=Action(action_type=ActionType.RESET_BET),
        priority=10
    ),
    
    # Rule 2: Stop loss at 5 losses
    Rule(
        condition=Condition(
            event_type=EventType.LOSE,
            streak_operator=ComparisonOperator.GREATER_EQUAL,
            streak_threshold=5
        ),
        action=Action(action_type=ActionType.SET_BET, value=5),
        priority=9
    ),
    
    # Rule 3: Progressive increase on losses
    Rule(
        condition=Condition(
            event_type=EventType.LOSE,
            streak_operator=ComparisonOperator.LESS,
            streak_threshold=5
        ),
        action=Action(action_type=ActionType.MULTIPLY_BET, value=1.5),
        priority=8
    )
]

# Run simulation
engine = ProbabilityEngine()
summary = engine.run_simulation(
    strategy_type="custom",
    starting_bankroll=1000,
    base_bet=10,
    target_bankroll=2000,
    win_probability=0.49,
    expected_return=2.0,
    num_simulations=5000,
    rules=rules  # Pass custom rules
)

# Display results
print(f"Survival Rate: {summary.survival_rate:.2%}")
print(f"Success Rate: {summary.target_success_rate:.2%}")
print(f"Median Bankroll: ${summary.median_final_bankroll:.2f}")
```

---

## Command Line Interface

### Basic Usage
```bash
python probability_engine.py \
    --strategy streak_multiplier \
    --bankroll 1000 \
    --bet 10 \
    --target 2000 \
    --probability 0.49 \
    --return 2.0 \
    --simulations 10000
```

### Strategy-Specific Parameters
```bash
# Streak Multiplier
python probability_engine.py \
    --strategy streak_multiplier \
    --streak-threshold 3 \
    --multiplier1 2.0 \
    --multiplier2 1.5 \
    --simulations 1000

# Martingale
python probability_engine.py \
    --strategy martingale \
    --multiplier 2.0 \
    --simulations 1000

# Reverse Martingale
python probability_engine.py \
    --strategy reverse_martingale \
    --multiplier 2.0 \
    --max-streak 3 \
    --simulations 1000
```

### Comparison Mode
```bash
python probability_engine.py --compare \
    --bankroll 1000 \
    --bet 10 \
    --target 2000 \
    --probability 0.49 \
    --simulations 5000
```

### Optimization Mode
```bash
python probability_engine.py --optimize \
    --strategy streak_multiplier \
    --bankroll 1000 \
    --bet 10 \
    --target 2000 \
    --simulations 1000
```

### Configuration File
```bash
python probability_engine.py --config config.json
```

---

## Configuration File Format

### JSON Schema
```json
{
  "simulation": {
    "num_simulations": 10000,
    "max_rounds": 10000,
    "stop_on_bankruptcy": true,
    "stop_on_target": true,
    "seed": 42
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
      "name": "Conservative",
      "type": "streak_multiplier",
      "params": {
        "streak_threshold": 4,
        "multiplier1": 1.5,
        "multiplier2": 1.2
      }
    },
    {
      "name": "Aggressive",
      "type": "martingale",
      "params": {
        "multiplier": 3.0
      }
    }
  ],
  "optimization": {
    "target_metric": "survival_rate",
    "param_ranges": {
      "streak_threshold": [2, 3, 4, 5],
      "multiplier1": [1.5, 2.0, 2.5],
      "multiplier2": [1.1, 1.3, 1.5]
    }
  }
}
```

---

## Error Codes and Handling

### Common Exceptions

| Exception | Cause | Solution |
|-----------|-------|----------|
| `ValueError: Unknown strategy` | Invalid strategy type | Use one of: streak_multiplier, martingale, reverse_martingale, custom |
| `ValueError: Invalid parameters` | Negative values | Ensure bankroll, bet, probabilities are positive |
| `TypeError: Missing required` | Missing parameters | Check all required parameters are provided |
| `RuntimeError: Simulation failed` | Internal error | Check logs, reduce simulation count |

### Validation Rules

1. **Bankroll**: Must be > 0
2. **Bet**: Must be > 0 and <= bankroll
3. **Win Probability**: Must be 0 < p < 1
4. **Expected Return**: Must be > 0
5. **Simulations**: Must be > 0
6. **Max Rounds**: Must be > 0

---

## Performance Guidelines

### Recommended Settings

| Simulations | Use Case | Parallel | Time |
|------------|----------|----------|------|
| 100 | Quick test | No | <0.1s |
| 1,000 | Development | No | ~0.5s |
| 10,000 | Analysis | Yes | ~5s |
| 100,000 | Research | Yes | ~60s |

### Memory Usage

| Simulations | Memory |
|------------|--------|
| 1,000 | ~10 MB |
| 10,000 | ~100 MB |
| 100,000 | ~1 GB |

### Optimization Tips

1. **Use parallel processing** for > 1000 simulations
2. **Reduce max_rounds** if games typically end quickly
3. **Use seeds** for reproducible results
4. **Disable logging** for production runs
5. **Batch parameter testing** rather than sequential

---

## Statistical Interpretation

### Metrics Guide

| Metric | Good | Average | Poor |
|--------|------|---------|------|
| Survival Rate | >40% | 20-40% | <20% |
| Target Success | >30% | 15-30% | <15% |
| Avg Rounds | 500-2000 | 200-500 | <200 |

### Distribution Analysis

- **Bimodal**: Most strategies show bankruptcy or success
- **Median = 0**: Indicates >50% bankruptcy rate
- **High Std Dev**: High variance in outcomes
- **95th Percentile**: Best-case scenarios

---

## Advanced Usage

### Custom Condition Functions
```python
class CustomCondition(Condition):
    def evaluate(self, state: BettingState, event: EventType) -> bool:
        # Custom logic
        if state.current_bankroll < 100:
            return True
        return super().evaluate(state, event)
```

### Dynamic Strategy Switching
```python
def adaptive_strategy(bankroll_level):
    if bankroll_level < 500:
        return "conservative"
    elif bankroll_level > 1500:
        return "aggressive"
    return "balanced"
```

### Batch Processing
```python
# Test multiple configurations
configs = [
    {"win_probability": p, "expected_return": r}
    for p in [0.45, 0.48, 0.49, 0.50]
    for r in [1.8, 2.0, 2.2]
]

results = {}
for config in configs:
    results[tuple(config.values())] = engine.run_simulation(**config)
```

---

## Troubleshooting

### Issue: Logging Too Verbose
**Solution**: Adjust logging level in strategy files
```python
logger = init_logger(name="module", level=logLev.WARNING)
```

### Issue: Parallel Processing Fails
**Solution**: Disable parallel processing
```python
summary = engine.run_simulations(parallel=False)
```

### Issue: Memory Error
**Solution**: Reduce simulation batch size
```python
# Run in batches
total_results = []
for i in range(0, 100000, 10000):
    batch = engine.run_simulations(10000)
    total_results.append(batch)
```

### Issue: Inconsistent Results
**Solution**: Set random seed
```python
engine = MonteCarloEngine(seed=42)
```
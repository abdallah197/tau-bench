# Partial Scoring for Tool Calling

## Overview
The partial scoring module provides a more nuanced evaluation of agent performance by giving partial credit when agents use the correct tool but with incorrect arguments.

## How It Works
The scoring algorithm assigns:
- **40% weight** to using the correct tool name
- **60% weight** to having correct arguments

For arguments, the algorithm recursively compares expected vs actual values:
- Exact matches get full credit
- Dictionaries: credit based on matching key-value pairs
- Lists: credit based on matching items
- Primitives: binary match (0 or 1)

## Usage

### Basic Usage
```bash
python partial_scoring.py <results.json>
```

This will:
1. Calculate partial scores for all tasks
2. Display statistics comparing binary vs partial scoring
3. Save enhanced results to `<results>_partial.json`

### Command Line Options
```bash
python partial_scoring.py <results.json> [-o output.json] [--no-stats]
```

- `-o, --output`: Specify output file (default: adds `_partial` suffix)
- `--no-stats`: Skip printing statistics

### Examples

Process airline results:
```bash
python partial_scoring.py /path/to/airline_results.json
```

Process with custom output:
```bash
python partial_scoring.py results.json -o enhanced_results.json
```

## Interpreting Results

### Partial Score Ranges
- **1.0**: Perfect match - correct tool and all arguments
- **0.9-1.0**: Near miss - correct tool, minor argument differences
- **0.4-0.9**: Partial success - correct tool, some argument issues
- **0.0-0.4**: Wrong approach - incorrect tool selection

### Key Insights
The partial scoring reveals:
1. **Near-misses**: Tasks where agents were very close to success
2. **Tool selection accuracy**: How often agents choose the right tool
3. **Argument precision**: How well agents understand parameter requirements

## Example Output
```
=== SCORING STATISTICS ===
Total tasks: 200

Binary Scoring:
  Success rate: 93/200 (46.5%)
  Average: 0.465

Partial Scoring:
  Average: 0.644
  Min: 0.000
  Max: 1.000

Partial Score Distribution:
  [0.0-0.2):  40 tasks ( 20.0%)
  [0.2-0.4):  10 tasks (  5.0%)
  [0.4-0.6):  29 tasks ( 14.5%)
  [0.6-0.8):  19 tasks (  9.5%)
  [0.8-1.0):  32 tasks ( 16.0%)
  [1.0]:  70 tasks ( 35.0%)

Interesting Cases:
  Binary fail but partial > 0.7: 50 tasks
    Task 0: partial=0.96
    Task 3: partial=0.83
    Task 6: partial=0.93
```

## Integration with tau-bench
The module is designed as a standalone post-processor that:
- Works with existing tau-bench JSON output files
- Doesn't require modifications to the core tau-bench code
- Can be easily integrated into the main pipeline if desired

## Benefits
1. **More granular feedback**: Understand where agents are failing
2. **Progress tracking**: See improvement even when binary scores don't change
3. **Debugging aid**: Identify specific issues (tool selection vs argument errors)
4. **Fair evaluation**: Give credit for partially correct attempts
"""
Partial scoring module for tau-bench tool calling evaluation.
Provides partial credit for tool calls that are partially correct.
"""

import json
from typing import List, Dict, Any, Tuple


def extract_tool_calls(trajectory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract all tool calls from a trajectory.

    Args:
        trajectory: List of messages in the conversation

    Returns:
        List of tool calls with name and arguments
    """
    tool_calls = []

    for message in trajectory:
        if message.get('tool_calls'):
            for call in message['tool_calls']:
                func = call.get('function', {})
                name = func.get('name', '')
                args_str = func.get('arguments', '{}')

                try:
                    arguments = json.loads(args_str)
                except:
                    arguments = {}

                tool_calls.append({
                    'name': name,
                    'arguments': arguments
                })

    return tool_calls


def compare_values(expected: Any, actual: Any) -> float:
    """
    Compare two values and return similarity score (0.0 to 1.0).

    Args:
        expected: Expected value
        actual: Actual value

    Returns:
        Similarity score between 0 and 1
    """
    # Exact match
    if expected == actual:
        return 1.0

    # Type mismatch
    if type(expected) != type(actual):
        return 0.0

    # Dictionary comparison
    if isinstance(expected, dict):
        if not actual:
            return 0.0

        matching_keys = 0
        total_keys = len(expected)

        if total_keys == 0:
            return 1.0

        for key, exp_value in expected.items():
            if key in actual:
                matching_keys += compare_values(exp_value, actual[key])

        return matching_keys / total_keys

    # List comparison
    if isinstance(expected, list):
        if not actual:
            return 0.0 if expected else 1.0
        if not expected:
            return 1.0

        # Count how many expected items have a match in actual
        matches = 0
        for exp_item in expected:
            # Find best match in actual list
            best_match = 0
            for act_item in actual:
                score = compare_values(exp_item, act_item)
                best_match = max(best_match, score)
            matches += best_match

        return matches / len(expected)

    # For other types, no partial credit
    return 0.0


def calculate_partial_score(expected_actions: List[Dict], actual_calls: List[Dict]) -> float:
    """
    Calculate partial success score for tool calling.

    Scoring:
    - 40% for correct tool name
    - 60% for correct arguments

    Args:
        expected_actions: List of expected actions with 'name' and 'kwargs'
        actual_calls: List of actual tool calls with 'name' and 'arguments'

    Returns:
        Score between 0.0 and 1.0
    """
    if not expected_actions:
        # No actions expected, perfect if no calls made
        return 1.0 if not actual_calls else 0.0

    total_score = 0.0

    for expected in expected_actions:
        exp_name = expected.get('name', '')
        exp_kwargs = expected.get('kwargs', {})

        best_score = 0.0

        # Find the best matching actual call
        for actual in actual_calls:
            act_name = actual.get('name', '')
            act_args = actual.get('arguments', actual.get('kwargs', {}))

            score = 0.0

            # Tool name match (40%)
            if exp_name == act_name:
                score += 0.4

                # Arguments match (60%)
                arg_similarity = compare_values(exp_kwargs, act_args)
                score += 0.6 * arg_similarity

            best_score = max(best_score, score)

        total_score += best_score

    # Average score across all expected actions
    return total_score / len(expected_actions)


def process_results_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Process a tau-bench results JSON file and add partial scores.

    Args:
        filepath: Path to the JSON results file

    Returns:
        List of results with added partial_score field
    """
    with open(filepath, 'r') as f:
        results = json.load(f)

    enhanced_results = []

    for task in results:
        # Check if this is an error result
        if 'error' in task.get('info', {}):
            # For error cases, set partial score to 0
            task['partial_score'] = 0.0
            enhanced_results.append(task)
            continue

        # Extract expected actions
        try:
            expected = task['info']['task']['actions']
        except KeyError:
            # If structure is unexpected, skip with score 0
            task['partial_score'] = 0.0
            enhanced_results.append(task)
            continue

        # Extract actual tool calls from trajectory
        actual = extract_tool_calls(task['traj'])

        # Calculate partial score
        partial_score = calculate_partial_score(expected, actual)

        # Add to result
        task['partial_score'] = partial_score
        enhanced_results.append(task)

    return enhanced_results


def print_statistics(results: List[Dict[str, Any]]) -> None:
    """
    Print statistics comparing binary and partial scores.

    Args:
        results: List of task results with 'reward' and 'partial_score'
    """
    binary_scores = [r['reward'] for r in results]
    partial_scores = [r['partial_score'] for r in results]

    print("\n=== SCORING STATISTICS ===")
    print(f"Total tasks: {len(results)}")
    print()

    # Binary statistics
    binary_success = sum(1 for s in binary_scores if s > 0.9)
    print("Binary Scoring:")
    print(f"  Success rate: {binary_success}/{len(results)} ({binary_success/len(results)*100:.1f}%)")
    print(f"  Average: {sum(binary_scores)/len(binary_scores):.3f}")
    print()

    # Partial statistics
    print("Partial Scoring:")
    print(f"  Average: {sum(partial_scores)/len(partial_scores):.3f}")
    print(f"  Min: {min(partial_scores):.3f}")
    print(f"  Max: {max(partial_scores):.3f}")
    print()

    # Distribution
    print("Partial Score Distribution:")
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    for low, high in bins:
        count = sum(1 for s in partial_scores if low <= s < high)
        pct = count / len(partial_scores) * 100
        print(f"  [{low:.1f}-{high:.1f}): {count:3d} tasks ({pct:5.1f}%)")

    # Perfect scores
    perfect = sum(1 for s in partial_scores if s >= 1.0)
    print(f"  [1.0]: {perfect:3d} tasks ({perfect/len(partial_scores)*100:5.1f}%)")
    print()

    # Find interesting cases
    print("Interesting Cases:")

    # Binary fail but high partial
    high_partial_fails = [(i, r) for i, r in enumerate(results)
                          if r['reward'] < 0.1 and r['partial_score'] > 0.7]
    if high_partial_fails:
        print(f"  Binary fail but partial > 0.7: {len(high_partial_fails)} tasks")
        for i, r in high_partial_fails[:3]:
            print(f"    Task {r['task_id']}: partial={r['partial_score']:.2f}")

    # Binary success but low partial (shouldn't happen often)
    low_partial_success = [(i, r) for i, r in enumerate(results)
                          if r['reward'] > 0.9 and r['partial_score'] < 0.8]
    if low_partial_success:
        print(f"  Binary success but partial < 0.8: {len(low_partial_success)} tasks")
        for i, r in low_partial_success[:3]:
            print(f"    Task {r['task_id']}: partial={r['partial_score']:.2f}")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description='Calculate partial scores for tau-bench tool calling results'
    )
    parser.add_argument(
        'input_file',
        help='Path to the tau-bench results JSON file'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file for enhanced results (default: adds _partial suffix)',
        default=None
    )
    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='Skip printing statistics'
    )

    args = parser.parse_args()

    try:
        # Process the file
        print(f"Processing: {args.input_file}")
        results = process_results_file(args.input_file)

        # Print statistics unless disabled
        if not args.no_stats:
            print_statistics(results)

        # Save enhanced results
        if args.output:
            output_path = args.output
        else:
            # Default: add _partial suffix before .json
            output_path = args.input_file.replace('.json', '_partial.json')

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nEnhanced results saved to: {output_path}")

    except FileNotFoundError:
        print(f"Error: File not found: {args.input_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file: {args.input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
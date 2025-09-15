#!/usr/bin/env python3

import json
import argparse
from typing import Dict, List, Any, Tuple
from pathlib import Path


def extract_tool_calls_from_trajectory(traj: List[Dict]) -> List[Dict]:
    """Extract all tool calls from a trajectory."""
    tool_calls = []
    for entry in traj:
        if entry.get("role") == "assistant" and "tool_calls" in entry:
            if entry["tool_calls"] is not None:
                for tool_call in entry["tool_calls"]:
                    tool_calls.append({
                        "name": tool_call["function"]["name"],
                        "arguments": json.loads(tool_call["function"]["arguments"])
                    })
    return tool_calls


def normalize_args(args: Dict) -> Dict:
    """Normalize arguments for comparison (convert to same types, etc.)"""
    normalized = {}
    for key, value in args.items():
        # Convert to string for comparison to handle type mismatches
        if value is None:
            normalized[key] = None
        elif isinstance(value, bool):
            normalized[key] = value
        elif isinstance(value, (int, float)):
            normalized[key] = value
        else:
            normalized[key] = str(value)
    return normalized


def compare_arguments(expected_args: Dict, actual_args: Dict) -> float:
    """Compare two argument dictionaries and return a similarity score."""
    if not expected_args and not actual_args:
        return 1.0

    # Normalize arguments
    expected_norm = normalize_args(expected_args)
    actual_norm = normalize_args(actual_args)

    # Get all keys from both dictionaries
    all_keys = set(expected_norm.keys()) | set(actual_norm.keys())

    if not all_keys:
        return 1.0

    matches = 0
    for key in all_keys:
        expected_val = expected_norm.get(key)
        actual_val = actual_norm.get(key)

        if expected_val == actual_val:
            matches += 1

    return matches / len(all_keys)


def match_actions_to_tool_calls(
    gold_actions: List[Dict],
    tool_calls: List[Dict]
) -> List[Tuple[Dict, Dict, float]]:
    """
    Match gold actions to tool calls using a greedy approach.
    Returns list of (gold_action, matched_tool_call, score) tuples.
    """
    matches = []
    used_tool_calls = set()

    for gold_action in gold_actions:
        best_match = None
        best_score = 0
        best_idx = -1

        for idx, tool_call in enumerate(tool_calls):
            if idx in used_tool_calls:
                continue

            # Calculate name match score
            name_match = 1.0 if gold_action["name"] == tool_call["name"] else 0.0

            # Calculate argument match score
            gold_args = gold_action.get("kwargs", {})
            tool_args = tool_call.get("arguments", {})
            args_match = compare_arguments(gold_args, tool_args)

            # Calculate total score for this tool call
            score = 0.5 * name_match + 0.5 * args_match

            # Keep track of best match
            if score > best_score:
                best_score = score
                best_match = tool_call
                best_idx = idx

        if best_match is not None:
            matches.append((gold_action, best_match, best_score))
            used_tool_calls.add(best_idx)
        else:
            # No match found for this gold action
            matches.append((gold_action, None, 0.0))

    return matches


def calculate_tool_call_reward(task: Dict) -> Tuple[float, Dict]:
    """
    Calculate the tool call reward for a single task.
    Returns (tool_call_reward, detailed_info)
    """
    # Check if task has valid info and reward_info
    if (task.get("info") is None or
        task["info"].get("reward_info") is None or
        task["info"]["reward_info"].get("actions") is None):
        # Return 0 reward if no gold actions available
        return 0.0, {
            "gold_actions": [],
            "extracted_tool_calls": [],
            "matches": [],
            "tool_call_reward": 0.0,
            "error": "No reward_info or actions available"
        }

    # Extract gold actions
    gold_actions = task["info"]["reward_info"]["actions"]

    # Extract tool calls from trajectory
    tool_calls = extract_tool_calls_from_trajectory(task["traj"])

    # Match actions to tool calls
    matches = match_actions_to_tool_calls(gold_actions, tool_calls)

    # Calculate average score
    if not matches:
        tool_call_reward = 0.0
    else:
        scores = [score for _, _, score in matches]
        tool_call_reward = sum(scores) / len(scores)

    # Prepare detailed info
    detailed_info = {
        "gold_actions": gold_actions,
        "extracted_tool_calls": tool_calls,
        "matches": [
            {
                "gold_action": gold_action,
                "matched_tool_call": matched_tool if matched_tool else None,
                "score": score
            }
            for gold_action, matched_tool, score in matches
        ],
        "tool_call_reward": tool_call_reward
    }

    return tool_call_reward, detailed_info


def process_file(file_path: Path, verbose: bool = False) -> Dict:
    """Process a single JSON file and calculate partial rewards."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    results = []
    total_original_reward = 0
    total_tool_call_reward = 0
    total_partial_reward = 0

    for task in data:
        task_id = task["task_id"]
        original_reward = task["reward"]

        # Calculate tool call reward
        tool_call_reward, detailed_info = calculate_tool_call_reward(task)

        # Calculate partial reward
        partial_reward = 0.5 * original_reward + 0.5 * tool_call_reward

        result = {
            "task_id": task_id,
            "original_reward": original_reward,
            "tool_call_reward": tool_call_reward,
            "partial_reward": partial_reward
        }

        if verbose:
            result["detailed_info"] = detailed_info

        results.append(result)

        # Update totals
        total_original_reward += original_reward
        total_tool_call_reward += tool_call_reward
        total_partial_reward += partial_reward

    # Calculate averages
    num_tasks = len(results)
    summary = {
        "file": str(file_path),
        "num_tasks": num_tasks,
        "avg_original_reward": total_original_reward / num_tasks if num_tasks > 0 else 0,
        "avg_tool_call_reward": total_tool_call_reward / num_tasks if num_tasks > 0 else 0,
        "avg_partial_reward": total_partial_reward / num_tasks if num_tasks > 0 else 0,
        "tasks": results
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description='Calculate partial scores for TAU-bench results')
    parser.add_argument('input_file', type=str, help='Path to the JSON results file')
    parser.add_argument('-o', '--output', type=str, help='Output file path (optional)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Include detailed matching info')

    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File {input_path} does not exist")
        return

    # Process the file
    print(f"Processing {input_path}...")
    results = process_file(input_path, verbose=args.verbose)

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"File: {results['file']}")
    print(f"Number of tasks: {results['num_tasks']}")
    print(f"Average original reward: {results['avg_original_reward']:.4f}")
    print(f"Average tool call reward: {results['avg_tool_call_reward']:.4f}")
    print(f"Average partial reward: {results['avg_partial_reward']:.4f}")

    # Save results if output path specified
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {output_path}")

    # Print per-task summary only if verbose mode
    if args.verbose:
        print("\n=== PER-TASK RESULTS ===")
        print(f"{'Task ID':<10} {'Original':<10} {'Tool Call':<10} {'Partial':<10}")
        print("-" * 40)
        for task in results['tasks']:
            print(f"{task['task_id']:<10} {task['original_reward']:<10.4f} {task['tool_call_reward']:<10.4f} {task['partial_reward']:<10.4f}")


if __name__ == "__main__":
    main()
## How to Run

### 1. Installation

It is recommended to use a virtual environment.

```bash
pip install .
```

### 2. Set API Key

You need to set your XAI API key as an environment variable.

```bash
export XAI_API_KEY='your-xai-api-key'
```

### 3. Run Benchmarks

Here are the commands to run the `airline` and `retail` benchmarks using Grok 3.

#### Airline

```bash
python run.py \
  --agent-strategy tool-calling \
  --env airline \
  --model grok-3 \
  --model-provider xai \
  --user-model grok-3 \
  --user-model-provider xai \
  --user-strategy llm \
  --max-concurrency 10
```

#### Retail

```bash
python run.py \
  --agent-strategy tool-calling \
  --env retail \
  --model grok-3 \
  --model-provider xai \
  --user-model grok-3 \
  --user-model-provider xai \
  --user-strategy llm \
  --max-concurrency 10
```

## Improvements Instructions 

To run the tool calling metric and partial scores metric improvements. 

```bash
 python improvements/partial_scoring.py outputs/airline/full_resuls_airline_k=4.json -o                                          │
│   outputs/airline/airline_scores_with_partial.json -v
``` 
the script have the logic for improved metric. 

## Results 
For task 1, running grok 3 results are saved in `outputs/`

Outputs layout
```bash
tau-bench/outputs/: folder where all benchmark run artifacts are saved.
tau-bench/outputs/airline/: results for the Airline environment.
tau-bench/outputs/retail/: results for the Retail environment.
```

#### Files inside

- full_results_*.json: full JSON output from a benchmark run \ (complete traces, actions, rewards, tool calls, overall scores)

- *_scores_with_partial.json : derived summary with partial reward components and tool-call success rate. result of running improvements/partial_scoring.py

```

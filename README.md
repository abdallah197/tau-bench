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

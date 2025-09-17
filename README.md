# Health-Coach: Agent-Augmented Reinforcement Learning for Clinical Decision Support

This repository contains the implementation of an agent-augmented reinforcement learning system for heart disease risk assessment and clinical decision support, as described in our paper "Agent-Augmented Reinforcement Learning for Clinical Decision Support".

## Overview

The system combines:
- **Tabular Q-learning** with dynamic exploration strategy selection
- **LLM-based agents** for context-aware decision making and explainable narratives
- **SHAP-based explainability** for model predictions
- **Per-patient personalization** through adaptive thresholds and reporting

The architecture uses dependency injection and typed interfaces throughout, enabling safe integration of AI agents within a clinical decision support workflow.

## System Requirements

- Python 3.12+
- NVIDIA GPU with > 6GB VRAM (for local LLM hosting, optional)
- Linux-based OS (for README instructions, optional)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/IsaacLatta/Health-Coach.git
cd Health-Coach
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate 
```

3. Install dependencies:
```bash
python3 -m pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp example.env .env
# Edit .env with your configuration
```

### Environment Configuration

Create a `.env` file in the project root with the following variables:

```bash
# LLM Configuration (uncomment one)
#MODEL=ollama/qwen3:latest
#MODEL=gpt-4o-mini
#MODEL=model-of-your-choosing

# API Keys (replace with your own)
OPENAI_API_KEY=your_openai_key_here

# Server Configuration
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8080
FRONTEND_HOST=127.0.0.1
FRONTEND_PORT=8501

# Model and Database Paths (adjust to your setup)
PRED_MODEL_PATH=/path/to/Health-Coach/models/predictor.pkl
DB_PATH=/path/to/Health-Coach/data/db/health_coach.db
DB_TIMEOUT=5
DB_JOURNAL_MODE=WAL
DB_SYNCHRONOUS=NORMAL

# RL Configuration
# These are only the fallbacks for bad selected actions,
# The updates will happen based on the configured tuples in
# src/health_coach/config.py for each explorer
Q_STATES=10
Q_ACTIONS=6
```

## Running the Application

### Quick Start - Web Application

1. Start the application:
```bash
./run_app.sh
```

2. Open your browser and navigate to:
```
http://127.0.0.1:8501
```

3. The frontend is pre-populated with a mock patient (P-0002) for demonstration. Simply click "Run" to see the system in action.

### Features Available in the Web UI

- **Risk Assessment**: View calibrated heart disease risk with uncertainty bands
- **SHAP Explanations**: See top-k feature contributions to risk prediction
- **RL Insights**: Monitor the reinforcement learning agent's decisions and performance
- **Interactive Chat**: Ask questions about the patient's risk profile and RL history

## Running the Comparison Harness

The comparison harness evaluates different exploration strategies and the agent selector performance.

### Basic Usage

```bash
cd src
python3 -m health_coach.compare.main_compare [OPTIONS]
```

### Command Line Options

- `--explorer`: Run explorer optimization
- `--pure`: Run pure RL comparison (without agent selection)
- `--agent`: Run agent-augmented RL comparison
- `--all`: Run all comparisons
- `--help`: See this help message

### Examples

```bash
# Run only the agent comparison (default if no args)
python3 -m health_coach.compare.main_compare

# Run explorer optimization
python3 -m health_coach.compare.main_compare --explorer

# Run all comparisons
python3 -m health_coach.compare.main_compare --all
```

### Configuration

The comparison harness is configured via `src/health_coach/compare/config.py`. Key parameters:

- `NUM_EPISODES`: Number of training episodes (default: 5)
- `EPISODE_LENGTH`: Steps per episode (default: 5)
- `TRAIN_FRACTION`: Train/validation split (default: 0.8)
- `Q_STATES`: Number of discrete states (default: 10)
- `Q_ACTIONS`: Number of available actions (default: 6)

**NOTE**: There are many other parameters available within `config.p`, by default they will be set to the optimal ones chosen during our evaluation.

For comprehensive evaluation, modify these values in `config.py` before running.

## Results

Pre-computed results from our evaluation and explorer tuning are available in:
```
results/*.json
```

## Enabling LLM Features

The LLM integration is currently commented out for ease of initial setup. To enable:

1. **Update LLM configuration** in `backend/flows/llm.py`:
   - Uncomment the appropriate return statements in `get_llm()` and `get_embedder()`
   
2. **Set environment variables**:
   - Set `MODEL` in your `.env` file
   - Ensure API keys are configured

3. **Update agent constructors** to use `llm=llm.get_llm()`

4. **For local Qwen model**:
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Pull Qwen model
   ollama pull qwen3:latest
   ```
**NOTE**: Our testing was run with Qwen-3B, Gemma3, or gpt-4o-mini. For other LLMs it is encouraged to consult the crewai LLM docs for their specific requirements, located [here](https://docs.crewai.com/en/concepts/llms). Then update `get_llm()` or `get_embedder()` in `src/health_coach/backend/flows/llm.py` for the comparison harness, or the agent constructors in `src/health_coach/backend/flows/*/agents.py` for integration within the application.

## Architecture Notes

### Code Organization

- **Frontend** (`src/health_coach/frontend/`): Streamlit-based UI
- **Backend** (`src/health_coach/backend/`):
  - `flows/`: Agent orchestration (ReportingFlow, RLFlow, etc.)
  - `services/`: Core services (PredictionService, SHAPService, etc.)
  - `stores/`: Data persistence layers
  - `rl/`: Reinforcement learning implementation
- **Comparison Harness** (`src/health_coach/compare/`):
  - Modified versions of ContextService and RLEngine for evaluation
  - Synthetic data generation
  - Explorer implementations and tuning

## Troubleshooting

1. **Module not found errors**: Ensure you're running from the `src` directory and have activated the virtual environment with the necessary python packages.

2. **Database errors**: Check that the DB_PATH in `.env` points to a valid location with write permissions.

3. **GPU memory issues**: Reduce batch size or use CPU-only mode by setting appropriate environment variables.

4. **Port conflicts**: Modify BACKEND_PORT and FRONTEND_PORT in `.env` if defaults are in use.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{latta2025agent,
  title={Agent-Augmented Reinforcement Learning for Clinical Decision Support},
  author={Latta, Isaac and others},
  journal={Thompson Rivers University},
  year={2025}
}
```

## Contact

For questions or issues, please open a GitHub issue.
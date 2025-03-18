# MiniAGI

A lightweight implementation of an Artificial General Intelligence framework combining neural and symbolic approaches to reinforcement learning.

## Features

- **Global Workspace Architecture**: Dynamic threshold adjustment for cognitive processing
- **Symbolic Reasoning**: Knowledge graph-based inference with weighted rules
- **Meta-Reasoning**: Self-adjusting cognition based on knowledge complexity
- **Curiosity-Driven Learning**: Intrinsic motivation through novelty detection
- **Neuro-Symbolic Integration**: BFS-based symbolic rules enhance neural policy learning
- **GAE-PPO Implementation**: Advanced policy optimization with Generalized Advantage Estimation
- **Multimodal Processing**: Handles both visual (5x5 grid) and textual inputs
- **Multi-head Attention**: Efficient feature extraction across modalities

## Environments

- **LargeGridPOEnv**: 10x10 grid with partial observability (5x5 view window) and BFS distance metrics

## Components

- DynamicGlobalWorkspace: Central broadcast mechanism with adaptive thresholds
- SymbolicReasoningModule: Rule-based inference engine
- MetaReasoningModule: Cognitive self-regulation
- CuriosityModule: Intrinsic reward calculation
- PPOActorCritic: Neural policy network with GAE implementation

## Ablation Studies

Includes comparative analysis of different component combinations:
- Base agent (no enhancements)
- Curiosity-only agent
- Meta+Symbolic agent
- Full integration (Curiosity+Meta+Symbolic)

## Usage

```python
# Create a complete agent with all components
agent = CompleteMiniAGI(use_curiosity=True, use_meta=True, use_symbolic=True)

# Run training for 10 episodes per epoch
reward = agent.run_epoch(episodes=10)
```

## Requirements

- PyTorch
- NumPy
- Matplotlib

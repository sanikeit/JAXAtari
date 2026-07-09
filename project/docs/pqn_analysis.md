# PQN Agent Analysis (Proximal Q-Network)

This document summarizes the investigation of the built-in PQN (Proximal Q-Network) agents within the JAXAtari repository, specifically examining `pqn_agent.py` and `baselines/pqn_atari.py`.

## 1. What PQN stands for
PQN stands for **Proximal Q-Network** (or Proximal Q-Learning). It acts as a hybrid between DQN and PPO. Instead of a traditional asynchronous replay buffer (like in standard Q-Learning), it uses a PPO-style synchronous rollout buffer. It collects transitions over multiple parallel environments, computes generalized advantage-like TD($\lambda$) returns, and then updates the Q-Network using MSE loss over multiple epochs.

## 2. Observation Formats
*   **`pqn_atari.py` (PyTorch Baseline):** Uses stacked pixel observations (4 frames, 84x84 grayscale) processed by a CNN.
*   **`pqn_agent.py` (JAX version):** Supports dual formats natively. 
    *   It can use standard stacked pixel frames processed by a CNN.
    *   It supports an `OBJECT_CENTRIC` format (enabled via config) where observations are flattened by the `FlattenObservationWrapper` and processed by a standard MLP instead of a CNN.

## 3. Training Loop
Both implementations use an on-policy style collection loop for Q-learning:
1.  **Rollout:** Collect exactly `num_steps` of transitions across `num_envs` parallel environments using an $\epsilon$-greedy policy.
2.  **Targets:** Compute Q-learning targets using TD($\lambda$) / lambda returns.
3.  **Update:** Flatten the rollout buffer into a batch, shuffle it, and update the Q-Network using MSE loss over multiple `update_epochs` (PPO style).

## 4. Does it solve MsPacman?
**No.** Both scripts are general-purpose RL training loops. 
*   `pqn_atari.py` defaults to running `Breakout-v5`.
*   `pqn_agent.py` reads the environment name dynamically from a configuration file (`config["ENV_NAME"]`). 

While they have the structural capability to train on MsPacman, they do not contain hardcoded "solved" logic, pre-trained MsPacman weights, or MsPacman-specific heuristic tuning.

## 5. Evaluation & Visualization Scripts
*   **`pqn_agent_visualisation.py`**: The repository provides a dedicated PyGame-based script for loading trained JAX PQN models (`.safetensors`) and visualizing them playing the game. **This is highly reusable**; we can adapt its PyGame rendering loop to visualize our A* planner's trajectories in real-time.
*   **W&B Logging:** `pqn_agent.py` also has built-in evaluation hooks to automatically generate and log mp4 videos to Weights & Biases at the end of training.

## 6. Relevance to Group 17
*   **Phase 3/4 Relevance:** The `pqn_agent_visualisation.py` script is extremely valuable to us. By modifying it to accept our `MBv2Planner` actions instead of neural network logits, we can build a real-time debugging visualizer to watch our planner get trapped in corridor oscillations.
*   **Neuro-Symbolic Future (Phase 4):** If we decide to train a learned predictor later, we can borrow the `OBJECT_CENTRIC` MLP architecture and PPO-style rollout buffer from `pqn_agent.py` rather than building an RL loop from scratch.

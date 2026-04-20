# Drug-Dosage-Optimisation-using-DQN


A **Deep Q-Network (DQN)** based Reinforcement Learning agent that learns to recommend optimal insulin dosage levels for diabetic patients using clinical features from the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

---

## 🧠 Overview

This project applies **Deep Reinforcement Learning** to a simulated clinical decision-making environment. The agent observes a patient's glucose level, BMI, and age, then selects an insulin dose (Low, Medium, or High) to bring glucose into a healthy range (80–120 mg/dL).

The agent is trained over **3,000 episodes** using the DQN algorithm with Experience Replay and a Target Network — two key techniques from DeepMind's seminal DQN paper.

---

## 🏗️ Architecture

```
State  (3 features)  →  Online DQN  →  Q-values (3 actions)
                         ↕ periodically synced
                        Target DQN   →  Stable Bellman targets
```

### Components

| Component | Description |
|---|---|
| `DQN` | 3-layer fully-connected neural network (Linear → ReLU → Linear → ReLU → Linear) |
| `ReplayBuffer` | Experience Replay buffer storing `(s, a, r, s', done)` tuples (capacity: 10,000) |
| `DQNAgent` | ε-greedy agent with online + target networks, Adam optimizer, MSE loss |
| `step()` | Simulated environment — applies dose, adds glucose noise, returns reward |
| `reward_function()` | Reward shaping: +10 for normal range, +2 for near-normal, −5 for dangerous levels |
| `predict_dosage()` | Interactive inference function for new patient data |

---

## 🔁 Reinforcement Learning Setup

| Parameter | Value |
|---|---|
| State space | `[glucose_norm, bmi_norm, age_norm]` (normalized to [0,1]) |
| Action space | `{0: Low Dose, 1: Medium Dose, 2: High Dose}` |
| Dose effects | `[−2.0, −5.0, −8.0]` glucose reduction |
| Discount factor γ | 0.9 |
| Learning rate | 0.001 |
| Epsilon start / min | 1.0 → 0.05 (decay: 0.995) |
| Batch size | 64 |
| Target update freq | Every 100 steps |
| Episodes | 3,000 |
| Steps per episode | 30 |

---

## 📊 Training Outputs

After training, three plots are generated and saved as `DQN_training_results.png`:

1. **Reward Learning Curve** — Episode rewards with 50-episode moving average
2. **Training Loss (MSE)** — Bellman error across training with 30-episode moving average
3. **Reward Distribution** — Histogram of the final 500 episodes

A performance summary is also printed showing overall average reward, final 100-episode average, best episode reward, and final epsilon.

---

## 📁 Project Structure

```
RLA_Program.ipynb       # Main notebook with all code
diabetes.csv            # Input dataset (Pima Indians Diabetes)
DQN_training_results.png  # Generated after training
README.md               # This file
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib torch
```

### Dataset

Download the **Pima Indians Diabetes Dataset** from Kaggle and place it at the path referenced in the notebook:

```
/content/diabetes.csv
```

Or update the path in this line:
```python
data = pd.read_csv(r"/content/diabetes.csv")
```

### Run

Open and run `RLA_Program.ipynb` in **Google Colab** or **Jupyter Notebook** top to bottom. After training completes, an interactive loop will prompt you to enter patient vitals and receive a dosage recommendation:

```
Enter Patient Details
Enter Glucose Level : 145
Enter BMI           : 32.5
Enter Age           : 45

===================================
      DQN DOSAGE PREDICTION
===================================
  Glucose Level : 145.0
  BMI           : 32.5
  Age           : 45
  Q-values      : Low=3.12 | Med=5.87 | High=7.43
  Recommended   : High Dose
  Est. Glucose after dose : 137.0
```

---

## 🔬 How DQN Works Here

1. **Initialization** — A random patient is sampled from the dataset to start each episode.
2. **Action selection** — The agent uses ε-greedy: random action with probability ε, else the action with the highest Q-value.
3. **Environment step** — The chosen dose reduces glucose; stochastic noise is added to simulate real-world variation.
4. **Reward** — The agent receives reward based on whether the resulting glucose is in the healthy range.
5. **Experience Replay** — The transition `(s, a, r, s', done)` is stored in the replay buffer and a random mini-batch is sampled for training.
6. **Target Network** — The Bellman target uses a periodically-frozen copy of the network to stabilize training.
7. **Epsilon decay** — Exploration gradually decreases as the agent becomes more confident.

---

## ⚠️ Disclaimer

This project is a **research/educational simulation only**. It is **not** intended for real medical use. Insulin dosage in practice must be determined by qualified healthcare professionals.

---

## 📚 References

- Mnih et al. (2015) — [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236), DeepMind / Nature
- [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), UCI ML Repository
- PyTorch Documentation — https://pytorch.org/docs/

---


Built as part of a Reinforcement Learning Applications (RLA) program exploring RL in healthcare decision-making.

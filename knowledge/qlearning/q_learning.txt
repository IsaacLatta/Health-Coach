## 1. Introduction

Reinforcement learning (RL) is a framework for sequential decision-making in which an agent interacts with an environment to maximize a cumulative reward signal. In our research, we compare **pure Q-learning**—a classical, model-free RL algorithm—with **agent-augmented Q-learning**, where a learned “shaper” agent uses dynamic context to modulate both action selection and reward signals. The goal of this knowledge source is to ground a ContextProvidingAgent in:

1. **Fundamental RL concepts** (MDP, Q-function, Bellman updates).  
2. **Performance objectives** (convergence, cumulative reward, sample efficiency).  
3. **Exploration–exploitation trade-offs** that shape effective learning.

By understanding these principles, the agent can choose at the fly which shaping strategies (for example, decaying epsilon-greedy vs. softmax, potential-based vs. trend-based reward shaping) are most appropriate for a given learning phase and environment.

## 2. Q-Learning Fundamentals

### 2.1 Markov Decision Process (MDP)

A **Markov Decision Process (MDP)** formalizes an RL problem as a tuple of:

- **States**: the set of all possible situations the agent can be in.  
- **Actions**: the choices available to the agent in each state.  
- **Transition probabilities**: for each state and action, a probability distribution over next states.  
- **Reward function**: the immediate reward received after taking an action in a state.  
- **Discount factor** (named “gamma”): a number between 0 and 1 that weights the importance of future rewards versus immediate rewards.

> “A Markov decision process is a mathematical framework for modeling decision making in situations where outcomes are partly random and partly under the control of a decision maker.” [1]

### 2.2 Q-Function and Q-Table

The **Q-function** represents the expected cumulative reward when the agent takes a given action in a given state and thereafter follows the optimal policy. Q-learning aims to learn the **optimal Q-function**, denoted Q*(state, action), without requiring a model of the environment.

To implement Q-learning in a finite domain, we use a **Q-table**, which maps each (state, action) pair to a numerical value. Initially this table is often filled with zeros and updated over time as the agent gains experience.

> “Q-learning is a model-free reinforcement learning algorithm that trains an agent to assign values to its possible actions based on its current state, without requiring a model of the environment.” [2]

### 2.3 Bellman Optimality Update Rule

Each time the agent takes an action in a state and observes a reward and next state, the Q-table is updated according to the **Bellman optimality rule**. Using:

- **learning rate** (named “alpha”): how much new information overrides old;  
- **discount factor** (named “gamma”): how much future rewards are worth compared to immediate rewards;

the update is:

Q(state, action) = Q(state, action)+ alpha * (reward + gamma * max over a' of Q(next_state, a') − Q(state, action))


This rule blends the current estimate with the observed reward plus the best possible future value.

> “The core of the algorithm is a Bellman equation as a simple value-iteration update, using the weighted average of the current value and the new information.” [2], [5]

## 3. Learning Objectives & Evaluation Metrics

When comparing pure Q-learning to agent-augmented variants, we track:

- **Convergence**  
  The degree to which the Q-values stabilize over time and the derived policy stops changing. Proven guarantees exist for convergence under a decaying learning rate and sufficient exploration [2].

- **Cumulative Reward per Episode**  
  The total reward the agent collects in one complete run from start to terminal state. Higher cumulative reward indicates better performance.

- **Sample Efficiency**  
  How rapidly the agent improves its performance given a limited number of interactions.  
  > “An algorithm is sample efficient if it can make good use of every single piece of experience it happens to generate and rapidly improve its policy.” [3]

- **Exploration–Exploitation Trade-Off**  
  Balancing between trying new actions (exploration) and choosing the current best action (exploitation). Measured via regret or rate of improvement.  
  > “The exploration–exploitation dilemma… arises when an agent must choose between exploiting the best known option and exploring new options that may yield higher future rewards.” [4]

These metrics, measured under identical environments, allow a direct comparison of the impact of dynamic, context-driven shaping versus the pure Q-learning baseline.

## 4. Q-Learning System Context Signals

To empower the ContextProvidingAgent with actionable insights into the learning process, the following system-level context signals are most useful:

### 4.1 Episode Progress  
- **Current Step**: Index of the current time step within the episode.  
- **Episode Length**: Total number of steps per episode.  
  - Knowing how far along in an episode allows the agent to adjust exploration pressure (e.g., more exploration early, more exploitation near the end) [9].

### 4.2 Cumulative Reward & Moving Average  
- **Cumulative Reward**: Sum of all rewards collected so far in the episode.  
- **Moving Average Reward**: Smoothed average of recent rewards over a fixed window.  
  - Moving averages reduce noise and highlight performance trends, helping the agent decide when to tighten or relax exploration [6], [9].

### 4.3 State-Action Visit & Transition Counts  
- **Visit Count(s, a)**: Number of times action a has been taken in state s.  
- **Transition Count(s → s′)**: Number of times the environment moved from s to s′.  
  - Low counts indicate high uncertainty; these can trigger count-based bonuses to drive systematic exploration [7], [9].

### 4.4 Q-Value Distribution Metrics  
- **Maximum Q-Value**: Highest Q(s, a) over all actions in the current state.  
- **Mean & Variance of Q-Values**: Statistical moments of the Q-values for state s.  
  - Large variance suggests conflicting value estimates, indicating the need for more targeted exploration or learning-rate adjustment [2], [9].

### 4.5 Temporal Difference (TD) Error Magnitude  

- **TD Error**:  
```
td_error = observed_reward + gamma * max_future_Q − current_Q
```

- **Average TD Error**: Mean magnitude of recent TD errors.  
- High TD error indicates significant prediction error, suggesting the agent should increase its learning rate or exploration in that region [8], [9].


## 5. Putting It All Together: Dynamic Adaptation

An expert shaper agent continually monitors learning progress and context signals (Section 5) to select and tune shaping strategies at each phase of training. Below is a step-by-step guide to how such an agent might apply dynamic shaping.

### 5.1 Initial Exploration Phase  
1. **High uncertainty, sparse knowledge**  
   - **Use count-based bonuses** to encourage visiting uncharted state–action pairs:  
     ```plaintext
     bonus(s,a) = β / sqrt(visit_count(s,a) + 1)
     ```  
     This drives systematic coverage when visit counts are low .  
   - **Apply UCB action selection** to balance reward estimates with uncertainty:  
     ```plaintext
     score(a) = Q(s,a) + sqrt( (2 · ln total_steps) / visit_count(s,a) )
     ```  
     Ensures both high-value and under-explored actions are tried .

2. **Maintain diversity**  
   - **Max-entropy intrinsic reward**  
     ```plaintext
     r_intrinsic = – Σₐ π(a|s) · log π(a|s)
     ```
     Added to raw reward to keep policy spread wide, guarding against premature convergence .

### 5.2 Focused Learning Phase  
As visit counts grow and moving-average reward begins to rise:

1. **Transition to smoother exploration**  
   - **Decaying epsilon-greedy**:  
     ```plaintext
     epsilon = max(epsilon_min, epsilon_initial · decay_rate^episode)
     ```  
     Balances exploration early with exploitation later .  
   - **Softmax (Boltzmann) selection** with moderate temperature (tau):  
     ```plaintext
     P(a|s) = exp(Q(s,a)/tau) / Σ_b exp(Q(s,b)/tau)
     ```  
     Enables graded exploration when Q-values are close .

2. **Introduce potential-based reward shaping**  
   - When a heuristic potential Φ(s) is available (e.g., estimated distance to goal), add:  
     ```plaintext
     r'(s,a,s') = r(s,a,s') + gamma·Φ(s') – Φ(s)
     ```  
     This guides learning without altering optimal policies .

### 5.3 Convergence & Fine-Tuning Phase  
Near policy stabilization (low TD error variance, stable Q-values):

1. **Minimize shaping influence**  
   - Gradually reduce shaping weights (β for bonuses, potential scaling) so the final policy aligns with true environment reward.  
2. **Tighten exploitation**  
   - Lower epsilon toward epsilon_min and increase softmax inverse temperature (1/τ) to focus on best actions.

### 5.4 Dynamic Reward Shaping Adjustments  
Throughout all phases, the agent watches for performance plateaus or high noise:

- **Trend-based shaping** if moving-average reward plateaus:  
  ```plaintext
  r'(s,a,s') = r(s,a,s') + w_trend · (MA_reward_current – MA_reward_previous)

Boosts rewards when recent performance is below expected trend.

* **Amplification/Damping of shaping signals** based on variance:

  ```plaintext
  w(s,a) = f(Var[TD_errors])
  r'(s,a,s') = r(s,a,s') + w(s,a)·F(s,a,s')
  ```

  Where F is the chosen shaping function and w increases when variance is low (trust shaping) and decreases when variance is high (avoid noise amplification) .

### 5.5 Example Dynamic Rules

```python
if visit_count(s,a) < 5:
    apply count-based bonus with β = 1.0
elif moving_average_reward < threshold_low:
    increase epsilon by 0.05
elif td_error_variance > high_variance:
    switch to trend-based shaping with weight = 0.5
elif episode_progress > 0.8:
    set epsilon = epsilon_min, reduce shaping weights by half
```

### 5.6 Practical Considerations

* **Computational overhead**: limit expensive context calculations (e.g., count-based bonuses) to every N steps.
* **Preserve convergence guarantees**: ensure potential-based shaping is used when optimal-policy invariance is required .
* **Avoid conflicting signals**: normalize and bound shaping terms so they never dominate raw rewards.

By following these expert‐driven, context‐aware rules, the shaper agent can steer Q-learning dynamically—accelerating exploration early, focusing learning mid-training, and fine-tuning exploitation toward optimal policies.


---

## References

[1] “Markov decision process,” *Wikipedia*. [Online]. Available: https://en.wikipedia.org/wiki/Markov_decision_process. Accessed: Jul. 10, 2025.  
[2] “Q-learning,” *Wikipedia*. [Online]. Available: https://en.wikipedia.org/wiki/Q-learning. Accessed: Jul. 10, 2025.  
[3] “What is sample efficiency in RL?,” *Milvus*. [Online]. Available: https://milvus.io/ai-quick-reference/what-is-sample-efficiency-in-rl. Accessed: Jul. 10, 2025.  
[4] “Exploration–exploitation dilemma,” *Wikipedia*. [Online]. Available: https://en.wikipedia.org/wiki/Exploration%E2%80%93exploitation_dilemma. Accessed: Jul. 10, 2025.  
[5] “Bellman equation,” *Wikipedia*. [Online]. Available: https://en.wikipedia.org/wiki/Bellman_equation. Accessed: Jul. 10, 2025.  


[2] “Q-learning,” *Wikipedia*. [Online]. Available: https://en.wikipedia.org/wiki/Q-learning. Accessed: Jul. 10, 2025.  
[6] “Moving average,” *Wikipedia*. [Online]. Available: https://en.wikipedia.org/wiki/Moving_average. Accessed: Jul. 10, 2025.  
[7] “Count-based exploration,” *Wikipedia*. [Online]. Available: https://en.wikipedia.org/wiki/Count-based_exploration. Accessed: Jul. 10, 2025.  
[8] “Temporal difference learning,” *Wikipedia*. [Online]. Available: https://en.wikipedia.org/wiki/Temporal_difference_learning. Accessed: Jul. 10, 2025.  
[9] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018.  
```

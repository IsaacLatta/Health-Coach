
## 1. Proven Strategies for Dynamically Optimizing Q-Learning

### 1.1 Adaptive Exploration

**ε-greedy selection**
With probability 1 − epsilon, the agent exploits by choosing

```plaintext
a = argmaxₐ Q(s,a)
```

and with probability epsilon, it explores by selecting a random action:

```plaintext
if random() < epsilon:
    a = random_action()
else:
    a = argmaxₐ Q(s,a)
```

Here 0 < epsilon < 1 controls the explore–exploit trade-off. Epsilon can be fixed, decay over time (e.g., epsilon\_t = epsilon₀·decay^t), or be adapted based on performance metrics ([Wikipedia][6]).

**Softmax (Boltzmann) selection**
Actions are sampled according to a Gibbs distribution over Q-values:

```plaintext
P(a|s) = exp(Q(s,a)/tau) / Σ_b exp(Q(s,b)/tau)
```

where tau > 0 is a “temperature” parameter: high tau encourages uniform exploration, low tau concentrates on high-value actions ([Wikipedia][7]).

**Upper Confidence Bound (UCB1)**
Balances exploitation with an optimism bonus:

```plaintext
UCB1_i(t) = Q_i + sqrt((2 · ln t) / N_i)
```

* Q\_i: empirical mean reward of arm i
* N\_i: count of pulls of arm i
* t: total number of trials
  Arms with low N\_i get larger bonuses, driving systematic exploration under the “optimism in the face of uncertainty” principle ([Wikipedia][8]).

**Thompson Sampling**
A Bayesian probability-matching approach. Maintain a posterior P(θ|D) over model parameters θ; each round:

```plaintext
θ* ~ P(θ|D)
a = argmaxₐ E[r | s, a, θ*]
```

By sampling θ\* from the posterior, the agent naturally explores actions in proportion to their probability of being optimal ([Wikipedia][9]).

**Count-based novelty bonus**
Augment the environment reward with an intrinsic bonus inversely proportional to the square root of state-action visit counts:

```plaintext
bonus(s,a) = β / sqrt(N(s,a) + 1)
r' = r + bonus(s,a)
```

This encourages visiting rarely-seen pairs first. For continuous or high-dimensional spaces, “pseudo-counts” derived from density models can be used ([NeurIPS Papers][10]).

**Maximum-entropy exploration**
Introduce an intrinsic reward equal to the negative policy entropy:

```plaintext
r_intrinsic = - Σₐ π(a|s) · log π(a|s)
r' = r + η · r_intrinsic
```

Maximizing this term encourages a more uniform, diverse action distribution ([Wikipedia][11]).

---

### 1.2 Reward Shaping

**Potential-based shaping**
Adds a shaping term derived from a potential function Φ(s) that provably preserves the optimal policy:

```plaintext
r'(s,a,s') = r(s,a,s') + γ · Φ(s') - Φ(s)
```

Here γ is the discount factor. The difference in potential effectively “guides” learning without changing the set of optimal policies ([Artificial Intelligence Stack Exchange][12]).

**Progress-based (trend) shaping**
Incorporates intermediate progress signals—e.g., subtask completion metrics or formal temporal-logic progress—into the reward:

```plaintext
r'(s,a,s') = r(s,a,s') + w(s) · Progress(s')
```

where Progress(s') quantifies advancement toward subgoals and w(s) may be updated dynamically based on recent performance trends ([arXiv][13]).

**Magnitude amplification vs. damping**
Learns a state- or action-dependent weight w(s,a) that scales the shaping term to amplify helpful rewards and dampen misleading ones:

```plaintext
r'(s,a,s') = r(s,a,s') + w(s,a) · F(s,a,s')
```

The weight function w(s,a) can itself be optimized (e.g., via gradient methods) to maximize the true cumulative return ([arXiv][13]).

[6]: https://en.wikipedia.org/wiki/Reinforcement_learning "Reinforcement learning - Wikipedia"
[7]: https://en.wikipedia.org/wiki/Softmax_function?utm_source=chatgpt.com "Softmax function"
[8]: https://en.wikipedia.org/wiki/Upper_Confidence_Bound?utm_source=chatgpt.com "Upper Confidence Bound"
[9]: https://en.wikipedia.org/wiki/Thompson_sampling?utm_source=chatgpt.com "Thompson sampling"
[10]: https://papers.neurips.cc/paper/6868-exploration-a-study-of-count-based-exploration-for-deep-reinforcement-learning.pdf?utm_source=chatgpt.com "[PDF] A Study of Count-Based Exploration for Deep Reinforcement Learning"
[11]: https://en.wikipedia.org/wiki/Exploration%E2%80%93exploitation_dilemma?utm_source=chatgpt.com "Exploration–exploitation dilemma"
[12]: https://ai.stackexchange.com/questions/10213/why-does-potential-based-reward-shaping-seem-to-alter-the-optimal-policy-in-this?utm_source=chatgpt.com "Why does potential-based reward shaping seem to alter the optimal ..."
[13]: https://arxiv.org/abs/2412.10917 "[2412.10917] Adaptive Reward Design for Reinforcement Learning"
[14]: https://arxiv.org/abs/2011.02669?utm_source=chatgpt.com "Learning to Utilize Shaping Rewards: A New Approach of Reward Shaping"

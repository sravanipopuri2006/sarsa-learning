# SARSA Learning Algorithm


## AIM
To implement the SARSA (State–Action–Reward–State–Action) learning algorithm to find the optimal policy and value function for a given environment and compare its performance with the Monte Carlo method.

## PROBLEM STATEMENT
Reinforcement Learning aims to train an agent to make a sequence of decisions in an environment to maximize cumulative rewards. In this experiment, the goal is to train an agent using the SARSA algorithm, an on-policy Temporal Difference (TD) method, where the agent learns the action-value function based on the policy it follows. The environment (e.g., FrozenLake-v1 or gym-walk) provides discrete states and actions. The agent must learn the best policy to reach the goal while minimizing penalties or negative rewards..

## SARSA LEARNING ALGORITHM
1.Initialize

2.Initialize Q(s, a) arbitrarily for all state–action pairs.

3.Set parameters:
Learning rate α
Discount factor γ
Exploration rate ε

4.For each episode:
Initialize the starting state s.

5.Choose an action a from state s using an ε-greedy policy derived from Q.

6.For each step of the episode:
Take action a, observe reward r and next state s′.

7.Choose the next action a′ from s′ using ε-greedy(Q).
Update the action-value function: [ Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma Q(s',a') - Q(s,a)] ]

8.Set s ← s′, a ← a′.

9.Repeat until s is a terminal state.

## SARSA LEARNING FUNCTION
### Name:POPURI SRAVANI
### Register Number:212223240117

```
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):

    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action = lambda state, Q, epsilon: (np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state])))
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)
    for e in tqdm(range(n_episodes), leave=False):
      state, done = env.reset(), False
      action = select_action(state, Q, epsilons[e])
      while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = select_action(next_state, Q, epsilons[e])
            td_target = reward + gamma * Q[next_state][next_action] * (not done)
            td_error = td_target - Q[state][action]
            Q[state][action] = Q[state][action] + alphas[e] * td_error

  
            state, action = next_state, next_action
      Q_track[e] = Q
      pi_track.append(np.argmax(Q, axis=1))
    V = np.max(Q, axis=1)
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:
<img width="579" height="715" alt="image" src="https://github.com/user-attachments/assets/61e2dddf-a3cd-4160-98e2-fc8f6185d77e" />
<img width="1120" height="715" alt="image" src="https://github.com/user-attachments/assets/8d0132cb-7503-4f30-ab17-813631014970" />

<img width="1189" height="720" alt="image" src="https://github.com/user-attachments/assets/7224b759-8b1f-4304-b830-fa2ce7f9f63c" />
<img width="1487" height="782" alt="image" src="https://github.com/user-attachments/assets/4bdc3f6f-860f-4bc7-a6d0-ee69f2d1d102" />






## RESULT:

The SARSA algorithm successfully learned the optimal policy and value function for the given environment. The learned policy closely approximated the optimal policy derived from Monte Carlo methods. The comparison plot shows that SARSA achieves stable convergence with slightly more bias due to its on-policy nature.


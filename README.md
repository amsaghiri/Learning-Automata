# Learning Automata

A small, readable implementation of a **variable-structure learning automaton** for teaching and
self-study. It is used as course material by Dr. Ali Mohammad Saghiri
([homepage](https://amsaghiri.github.io/)).

A learning automaton is a simple decision maker. It keeps one probability per action, picks an action
at random according to those probabilities, receives a reward or a penalty from an unknown environment,
and shifts the probabilities toward actions that were rewarded. Repeating this loop, it learns which
action is best without ever being told how the environment works.

## What is in the repository

| File | What it is |
| --- | --- |
| `Automata.py` | The `Automata` class and a complete demo: one automaton with 7 actions learning in a simple environment. |
| `cala.m` | A MATLAB experiment in which an automaton chooses between two actions from a context vector. |

## Run the demo

Requires Python 3 with NumPy and Matplotlib.

```bash
pip install numpy matplotlib
python Automata.py
```

The script prints the average reward and saves two figures in the current folder:

- `favor_action.png` – the action the automaton currently prefers, at each step
- `entropy.png` – the entropy of its action probabilities, which falls as it becomes confident

With the default settings the automaton settles on action 3, the action the environment favours, and
the average reward over 10,000 steps is about 0.98.

## How it works

At step *k* the automaton selects action *i* and receives a response *B(k)*, 1 for reward and 0 for
penalty. The probabilities are then updated with the linear reward–penalty scheme:

```
selected action i:   p_i(k+1) = p_i(k) + λ1·B·(1 − p_i(k)) − λ2·(1 − B)·p_i(k)
every other action j: p_j(k+1) = p_j(k) − λ1·B·p_j(k) + λ2·(1 − B)·(1/(r − 1) − p_j(k))
```

`λ1` is the reward rate and `λ2` is the penalty rate. With `λ2 = 0`, the default, penalties are ignored
and the scheme is linear reward–inaction (L<sub>R-I</sub>).

## Things to try

All settings are at the bottom of `Automata.py`.

| Setting | Default | Try |
| --- | --- | --- |
| `r` | 7 | Fewer or more actions. How does learning time change? |
| `lambda_1` | 0.01 | 0.1 learns faster but can lock onto a wrong action. Compare the entropy plots. |
| `lambda_2` | 0 | A small value such as 0.001 gives L<sub>R-εP</sub>. Does it still converge fully? |
| `num_iteration` | 10000 | Shorten it and see whether the automaton has converged yet. |

## Related repositories

- [Goore-Game](https://github.com/amsaghiri/Goore-Game) – many automata cooperating without communicating
- [Multi-armed-bandit](https://github.com/amsaghiri/Multi-armed-bandit) – the same learning problem with ε-greedy agents
- [Adaptive-Dropout-Algorithm](https://github.com/amsaghiri/Adaptive-Dropout-Algorithm) – hybrid learning automata

## Further reading

- K. S. Narendra and M. A. L. Thathachar, *Learning Automata: An Introduction*.
- A. Rezvanian, A. M. Saghiri, S. M. Vahidipour, M. Esnaashari, M. R. Meybodi,
  [*Recent Advances in Learning Automata*](https://link.springer.com/book/10.1007/978-3-319-72428-7), Springer, 2018.

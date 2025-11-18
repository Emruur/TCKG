# Thresholded Constrained Knowledge Gradient (TCKG)

## Thresholded feasible region

TCKG defines a **thresholded feasible region**:

$$
\mathcal{F}_n(\tau)
= \{x \in \mathcal{X} : \text{PF}_n(x) \ge \tau\},
$$

where $\tau \in [0,1]$ is a user-defined *confidence threshold* (e.g., $\tau = 0.5$).

Only points with sufficiently high posterior feasibility probability are treated as "feasible."

---

## Feasible optimum estimate

**Current feasible optimum estimate** (for maximization) is:

$$
V_n(\tau) = \max_{x \in \mathcal{F}_n(\tau)} \mu^f_n(x).
$$

This represents the highest posterior mean among points that are currently likely to be feasible.

---

## The TCKG acquisition function

When evaluating a new candidate $x_c$, we consider its *expected contribution* to improving the feasible optimum after observing new data.

Let $D_{n+1} = D_n \cup \{(x_c, y_f(x_c), y_c(x_c))\}$ denote the updated dataset, and define:

$$
V_{n+1}(\tau)
= \max_{x \in \mathcal{F}_{n+1}(\tau)} \mu^f_{n+1}(x),
$$

where 

$ 
\mathcal{F}_{n+1}(\tau) = \{x:\text{PF}_{n+1}(x) \ge \tau\} 
$

is the thresholded feasible set after the hypothetical new observation.

Then, the **Thresholded Constrained Knowledge Gradient** acquisition function is defined as:

$$
\alpha_{\text{TCKG}}(x_c)= \mathbb{E}_{y_f, y_c \mid D_n}\!V_{n+1}(\tau)- V_n(\tau)
$$

For minimization, the sign is reversed: $\alpha_{\text{TCKG}} = V_n - \mathbb{E}[V_{n+1}]$.

The expectation is approximated via Monte Carlo sampling (“fantasy” GP updates)
using random draws of $(y_f, y_c)$ at the candidate $x_c$.

---

## Monte Carlo approximation

In practice, we approximate:

$$
\alpha_{\text{TCKG}}(x_c)
\approx
\frac{1}{M} \sum_{m=1}^{M}
\Big[
\max_{x \in \mathcal{F}_{n+1}^{(m)}(\tau)} \mu^{f,(m)}_{n+1}(x)
\Big] -
\max_{x \in \mathcal{F}_n(\tau)} \mu^f_n(x)
$$

where each Monte Carlo sample $m$ corresponds to a fantasy update
of the GP posteriors conditioned on a synthetic observation at $x_c$.

---

## Intuition

- $V_n(\tau)$ measures the best *posterior mean* in the currently feasible region.
- After sampling at $x_c$, the posterior changes, potentially improving:
  - the objective $f(x)$ in already feasible regions, or
  - the shape of the feasible region itself (via constraint updates).
- TCKG quantifies the **expected increase in the best feasible value** given this new information.

---

## Conceptual comparison

| Method | Feasibility handling | Objective measure |
|:--|:--|:--|
| CEI / EIC | Weighted by PF (soft feasibility) | Expected improvement × PF |
| CKG (Ungredda & Branke, 2022) | Integrates PF probabilistically | Expected value of information |
| **TCKG (this work)** | Hard threshold PF ≥ τ inside the expectation | Expected improvement in feasible optimum |
| SafeOpt / StageOpt | Hard safe set from confidence bounds | Safe exploration with guarantees |

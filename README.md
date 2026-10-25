# Portfolio Optimisation with Frank–Wolfe Methods

**Optimization for Data Science · University of Padua · Group 31**

This university group project investigates **projection-free convex optimisation** for the classical **Markowitz mean–variance portfolio selection problem**. We study the convergence, computational efficiency and sparsity of three Frank–Wolfe algorithms, compare them with Projected Gradient Descent, and evaluate their behaviour on historical financial datasets and in a rolling-window setting.

## The optimisation problem

The goal is to allocate capital across assets while balancing expected return and risk, subject to a fully invested, long-only portfolio:

$$
\begin{aligned}
\min_{x \in \mathbb{R}^n}\quad & \gamma x^\top \Sigma x - \bar r^\top x \\
\text{subject to}\quad & \sum_{i=1}^{n}x_i=1,\qquad x_i\geq 0.
\end{aligned}
$$

Here, $x$ is the vector of portfolio weights, $\bar r$ is the expected-return vector, $\Sigma$ is the return covariance matrix, and $\gamma$ controls risk aversion.

## Methods investigated

- **Frank–Wolfe (FW):** a conditional-gradient method that uses a linear minimisation oracle instead of projecting onto the probability simplex.
- **Away-Step Frank–Wolfe (ASFW):** allows weight to be removed from unfavourable active assets, mitigating slow convergence near boundary solutions.
- **Pairwise Frank–Wolfe (PWFW):** transfers weight directly between active vertices and promising new vertices.
- **Projected Gradient Descent (PGD):** a projection-based benchmark for solution quality and computational cost.

The experiments also compare **exact line search** and **diminishing step sizes**, with convergence assessed using the **Frank–Wolfe duality gap**.

## Datasets

We use historical **weekly asset-return datasets** associated with three equity indices (Bruni et al., 2016):

| Dataset | Assets | Weekly observations |
| --- | ---: | ---: |
| Dow Jones | 28 | 1,363 |
| FTSE 100 | 83 | 717 |
| S&P 500 | 442 | 595 |

## Main results

With risk-aversion parameter $\gamma = 1$ and duality-gap tolerance $10^{-6}$, the full-sample experiments produce the following iteration counts:

| Dataset | FW | ASFW | PWFW |
| --- | ---: | ---: | ---: |
| Dow Jones | 29 | 9 | 13 |
| FTSE 100 | 4,172 | 11 | 12 |
| S&P 500 | 3,829 | 16 | 14 |

**Key findings:**

- **Faster convergence on larger problems:** ASFW and PWFW reach the stopping tolerance in fewer than 20 iterations for all three datasets, while standard FW requires thousands of iterations on FTSE 100 and S&P 500.
- **Comparable solution quality:** all three algorithms achieve very similar objective values and final duality gaps below $10^{-6}$.
- **Sparse allocations:** solutions typically contain only a few assets, despite investment universes of up to 442 assets.
- **Projection-free efficiency:** in the reported operation-level comparison, simplex projection for PGD is approximately **7.5–8.5 times** more expensive than the Frank–Wolfe linear minimisation oracle.
- **Step-size trade-offs:** exact line search performs better on Dow Jones, whereas diminishing step sizes converge in fewer iterations on FTSE 100 and S&P 500 in the reported experiments.

## Rolling-window analysis

To examine performance under repeated portfolio rebalancing, the three datasets are restricted to their **most recent 595 observations**. At each rebalancing point, the previous **52 weekly observations** are used to estimate returns and covariance; the resulting portfolio is then held for **four periods**.

Six risk-aversion values are examined: $\gamma \in \{0.01, 0.1, 0.5, 1, 5, 10\}$.

For the S&P 500, using the selected risk-aversion parameter, the mean number of optimisation iterations per rolling window is approximately **8,910 for FW**, **47 for ASFW**, and **35 for PWFW**. The results reinforce the advantages of active-set variants in higher-dimensional settings.

## Limitations

The rolling-window wealth calculations **exclude transaction costs**, and the reported outcomes depend on the historical sample and chosen risk-aversion parameter. Highly sparse portfolios may also be less diversified than some investors require. These experiments are a methodological comparison, **not investment advice or a production trading strategy**.

## Report

The full mathematical derivations, experimental design, figures and detailed results are available in the [**final project report (PDF)**](Final_Optimisation_Project%282%29.pdf).

## Academic context

Completed as a **group project** for *Optimization for Data Science* at the **University of Padua** (Group 31). The work and results presented here are collaborative.

### Selected references

- Bruni, R., Cesarone, F., Scozzari, A., & Tardella, F. (2016). *Real-world datasets for portfolio selection and solutions of some stochastic dominance portfolio models*. Data in Brief.
- Jaggi, M. (2013). *Revisiting Frank–Wolfe: Projection-Free Sparse Convex Optimization*. ICML.
- Lacoste-Julien, S., & Jaggi, M. (2015). *On the Global Linear Convergence of Frank–Wolfe Optimization Variants*. NeurIPS.

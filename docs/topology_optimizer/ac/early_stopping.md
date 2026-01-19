# Early Stopping Strategy for N-1 Overload Analysis

This document outlines a validation strategy for early stopping during N-1 contingency analysis in power grid optimization. The approach leverages worst-case overloads from both AC and DC grid models to efficiently reject suboptimal topologies.

---

## 1. Calculate Worst-k N-1 AC Overloads for the Unsplit Grid (`g0(AC)`)

- **Definition:**
    For the original (unsplit) AC grid, identify the `k` worst N-1 contingency cases based on total overload.
- **Parameters:**
    - `worst_k_contingency_cases`: List of contingency case identifiers (e.g., `[a, b, c]`. Here k=3)
    - `top_k_overloads_n_1`: Total overload across these cases (e.g., `X`)

- **Example:**
    ```json
    {
        "worst_k_contingency_cases": ["a", "b", "c"],
        "overload": 120.5,
    }
    ```
    This means cases `a`, `b`, and `c` together result in a total overload of 120.5 units for 3 failures.

---

## 2. Calculate Worst-k DC N-1 Overloads for the Split Grid (`g1(DC)`)

- **Definition:**
    For the split grid in DC approximation, identify the `k` worst N-1 contingency cases.
- **Parameters:**
    - `worst_k_contingency_cases`: List of contingency case identifiers (e.g., `[a, d, e]`)
    - `top_k_overloads_n_1`: Total overload across these cases (e.g., `Y`)
- **Example:**
    ```json
    {
        "worst_k_contingency_cases": ["a", "d", "e"],
        "overload": 95.0,
    }
    ```
    Here, cases `a`, `d`, and `e` together result in a total overload of 95.0 units for 3 failures.

---

## 3. Early Stopping in AC Optimizer's Scoring Function

- **Process:**
    1. **Perform N-1 Analysis:**
         For a candidate AC topology, analyze N-1 cases: `[a, b, c, d, e]` (union of cases from `g0(AC)` and `g1(DC)`).
    2. **Compare Overloads:**
         - If the total overload from these cases exceeds `X` (the worst-case overload from `g0(AC)`), **stop** the analysis and **reject** the topology.
         - If the overload is less than or equal to `X`, **continue** the analysis and **store** the topology in the repertoire.
- **Note:**
    This is a greedy validation strategy. While it may not always yield the optimal solution, it efficiently filters out poor candidates and therefore we do this only for AC validation and not for AC optimisation

- **Example:**
    - Suppose `g0(AC)` yields an overload of `120.5` for `[a, b, c]`.
    - For a new topology, the N-1 analysis on `[a, b, c, d, e]` results in an overload of `130.0`.
    - Since `130.0 > 120.5`, the topology is **rejected** early.

---

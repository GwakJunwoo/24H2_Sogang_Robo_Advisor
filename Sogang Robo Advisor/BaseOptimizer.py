from typing import Any, List, Optional, Tuple, Union, Callable, Dict
import numpy as np
import cvxpy as cp
import collections

class BaseOptimizer:
    def __init__(self, n_assets, tickers=None):
        self.n_assets = n_assets
        self.tickers = tickers if tickers else list(range(n_assets))
        self.weights = None  # This will hold the asset weights

    def set_weights(self, input_weights: Dict[str, float]) -> None:
        """Sets the internal weights array from a dictionary."""
        self.weights = np.array([input_weights[ticker] for ticker in self.tickers])

    def clean_weights(self, cutoff=1e-4, rounding=None) -> Dict[str, float]:
        """Removes very small weights and optionally rounds the weights."""
        if self.weights is None:
            raise AttributeError("Weights not yet computed")

        self.weights[np.abs(self.weights) < cutoff] = 0
        if rounding is not None:
            self.weights = np.round(self.weights, rounding)

        return self._make_output_weights()

    def _make_output_weights(self, weights=None) -> Dict[str, float]:
        return collections.OrderedDict(zip(self.tickers, weights if weights is not None else self.weights))


class BaseConvexOptimizer(BaseOptimizer):
    def __init__(self, n_assets, tickers=None, weight_bounds=(0, 1), solver=None, verbose=False):
        super().__init__(n_assets, tickers)
        self.weight_bounds = weight_bounds
        self._solver = solver
        self._verbose = verbose

        self._w = cp.Variable(n_assets)
        self._objective = None
        self._constraints = []

        # Add weight bounds as constraints
        self.add_weight_bounds()

    def add_weight_bounds(self):
        """Adds weight bounds as constraints to the optimization problem."""
        if isinstance(self.weight_bounds, tuple):
            lower_bound, upper_bound = self.weight_bounds
            self._constraints.append(self._w >= lower_bound)
            self._constraints.append(self._w <= upper_bound)
        elif isinstance(self.weight_bounds, list):
            for i, (lower_bound, upper_bound) in enumerate(self.weight_bounds):
                self._constraints.append(self._w[i] >= lower_bound)
                self._constraints.append(self._w[i] <= upper_bound)

    def add_constraint(self, constraint_function: Any) -> None:
        """Adds custom constraints."""
        self._constraints.append(constraint_function(self._w))

    def _solve_cvxpy_opt_problem(self) -> Dict[str, float]:
        problem = cp.Problem(cp.Minimize(self._objective), self._constraints)
        problem.solve(solver=self._solver, verbose=self._verbose)
        self.weights = self._w.value
        return self._make_output_weights()

    def convex_objective(self, custom_objective: Callable, weights_sum_to_one=True, **kwargs) -> Dict[str, float]:
        self._objective = custom_objective(self._w, **kwargs)
        if weights_sum_to_one:
            self.add_constraint(lambda w: cp.sum(w) == 1)

        return self._solve_cvxpy_opt_problem()
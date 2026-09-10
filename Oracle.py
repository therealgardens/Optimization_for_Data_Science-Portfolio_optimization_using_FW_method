import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from typing import Any, Union

def compute_var_and_means(data: np.ndarray, window_size: int = 52, offset: int = 1) -> tuple[np.ndarray, np.ndarray]:
    means = []
    covariances = []

    for k in range(0, data.shape[0] - window_size, offset):
        window = data[k:k+ window_size]
        means.append(np.mean(window, axis=0))
        covariances.append(np.cov(window, rowvar=False))

    return np.asarray(means), np.asarray(covariances)

class PortfolioOptimisation_Whole_Dataset:
    def __init__(self, Sigma: np.ndarray, r_bar: np.ndarray, gamma: float = 1.0) -> None:
        self.Sigma = Sigma #covariance matrix
        self.r_bar = r_bar #expected returns vector
        self.gamma = gamma #risk aversion parameter
        self.n = len(r_bar) #number of assets

    def objective(self, x: np.ndarray) -> float:
        return self.gamma * np.dot(x, np.dot(self.Sigma, x)) - np.dot(self.r_bar, x) #objective function of Markowitz

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2 * self.gamma * np.dot(self.Sigma, x) - self.r_bar #gradient of OF

    def exact_line_search(self, x: np.ndarray, d: np.ndarray, gamma_max: float, tol: float = 1e-16) -> float:
        grad = self.gradient(x)
        d_Sigma_d = self.gamma * np.dot(d, np.dot(self.Sigma, d))  #curvature
        if d_Sigma_d <= tol: #checks whether curvature is negligible
            return gamma_max #gamma_max = 1 for FW
        opt_gamma = -np.dot(grad, d) / (2 * d_Sigma_d) #quadratic curvature
        return np.clip(opt_gamma, 0.0, gamma_max) #ensures the step size is valid

    def fw(self, max_iter: int = 1000, tol: float = 1e-6, use_exact_line_search: bool = True) -> tuple[np.ndarray, pd.DataFrame, float]:
        start_time = time.time() #starts cpu time
        x = np.zeros(self.n)
        x[0] = 1.0 #invests 100 percent in first asset
        history = []

        for k in range(max_iter):
            grad = self.gradient(x)
            s_idx = np.argmin(grad) #finds the index corresponding to the smallest gradient component
            s = np.zeros(self.n)
            s[s_idx] = 1.0 #makes s a simplex vertex

            direction = s - x #fw direction
            gap = -np.dot(grad, direction) #duality gap

            history.append({
                'Iteration': k,
                'Objective': self.objective(x),
                'Duality Gap': gap,
                'Active Set Size': np.sum(x > tol) #how many portfolio weights are larger than the tolerance
            })

            if gap <= tol: #current solution is sufficiently close to optimality according to the FW gap
                break

            if use_exact_line_search:
                gamma_k = self.exact_line_search(x, direction, gamma_max=1.0) #exact line search
            else:
                gamma_k = 2.0 / (k + 2.0) #fixed step size

            x = (1 - gamma_k) * x + gamma_k * s #updates FW

        execution_time = time.time() - start_time #stops time
        return x, pd.DataFrame(history), execution_time

    def asfw(self, max_iter: int = 1000, tol: float = 1e-6) -> tuple[np.ndarray, pd.DataFrame, float]:
        start_time = time.time() #starts cpu time
        active_set = {0: 1.0} #current active vertices and their weights
        x = np.zeros(self.n)
        x[0] = 1.0 #invests 100 percent in first asset
        history = []

        for k in range(max_iter):
            grad = self.gradient(x)
            s_idx = np.argmin(grad) #finds the index corresponding to the smallest gradient component
            d_FW = np.zeros(self.n)
            d_FW[s_idx] = 1.0
            d_FW -= x #constructs d_FW = s - x
            g_FW = -np.dot(grad, d_FW) #calculates FW gap

            active_idx = list(active_set.keys()) #all currently active assets
            v_idx = active_idx[np.argmax([grad[i] for i in active_idx])] #find asset with the largest gradient component
            d_ASFW = x.copy()
            d_ASFW[v_idx] -= 1.0 #creates away direction
            g_ASFW = -np.dot(grad, d_ASFW) #improvement associated with the away direction
            fw_gap = g_FW #stores gap

            history.append({
                'Iteration': k,
                'Objective': self.objective(x),
                'Duality Gap': fw_gap,
                'Active Set Size': len(active_set)
            })

            if fw_gap <= tol:
                break

            if g_FW >= g_ASFW: #compares the two possible directions
                d = d_FW #use normal FW direction
                gamma_max = 1.0 #gamma max for FW is 1
                is_away = False #not an away step
            else:
                d = d_ASFW #use the away direction
                gamma_max = active_set[v_idx] / (1.0 - active_set[v_idx] + 1e-16) #max away feasible step
                is_away = True #records that it is away step

            gamma_t = self.exact_line_search(x, d, gamma_max) #finds optimal step size along the selected direction
            x = x + gamma_t * d #updates portfolio

            #updating the active set
            if not is_away:
                for idx in list(active_set.keys()):
                    active_set[idx] *= (1.0 - gamma_t) #reduces the existing weights by 1 - gamma
                active_set[s_idx] = active_set.get(s_idx, 0.0) + gamma_t #add gamma to the newly selected FW vertex
            else:
                for idx in list(active_set.keys()):
                    active_set[idx] *= (1.0 + gamma_t) #scales active weights upward
                active_set[v_idx] -= gamma_t #reduces the away vertex weight

            active_set = {idx: w for idx, w in active_set.items() if w > tol} #remove unnecessary vertices from the active set

        execution_time = time.time() - start_time
        return x, pd.DataFrame(history), execution_time

    def pwfw(self, max_iter: int = 1000, tol: float = 1e-6) -> tuple[np.ndarray, pd.DataFrame, float]:
        start_time = time.time()
        active_set = {0: 1.0}
        x = np.zeros(self.n)
        x[0] = 1.0
        history = []

        for k in range(max_iter):
            grad = self.gradient(x)

            s_idx = int(np.argmin(grad))
            active_idx = list(active_set.keys())
            v_idx = active_idx[int(np.argmax([grad[i] for i in active_idx]))]

            # Standard FW gap is used consistently as the stopping criterion.
            s = np.zeros(self.n)
            s[s_idx] = 1.0
            d_fw = s - x
            fw_gap = -np.dot(grad, d_fw)

            history.append({
                "Iteration": k,
                "Objective": self.objective(x),
                "Duality Gap": fw_gap,
                "Active Set Size": len(active_set),
            })

            if fw_gap <= tol:
                break

            # Pairwise direction transfers weight directly from away vertex v
            # to FW vertex s. If both vertices coincide, fall back to FW.
            if s_idx == v_idx:
                d = d_fw
                gamma_max = 1.0
                is_pairwise = False
            else:
                d = np.zeros(self.n)
                d[s_idx] = 1.0
                d[v_idx] = -1.0
                gamma_max = active_set[v_idx]
                is_pairwise = True

            gamma_t = self.exact_line_search(x, d, gamma_max)
            x = x + gamma_t * d

            if is_pairwise:
                active_set[s_idx] = active_set.get(s_idx, 0.0) + gamma_t
                active_set[v_idx] -= gamma_t
            else:
                for idx in list(active_set.keys()):
                    active_set[idx] *= (1.0 - gamma_t)
                active_set[s_idx] = active_set.get(s_idx, 0.0) + gamma_t

            active_set = {idx: w for idx, w in active_set.items() if w > tol}

        execution_time = time.time() - start_time
        return x, pd.DataFrame(history), execution_time

    def run_all_fw(self, max_iter: int = 1000, tol: float = 1e-6) -> dict[str, Dict[str, Any]]:
        results = {}

        fw_x, fw_history, fw_time = self.fw(max_iter, tol)
        results['FW'] = {
            'x': fw_x,
            'history': fw_history,
            'time'  : fw_time
        }

        asfw_x, asfw_history, asfw_time = self.asfw(max_iter, tol)
        results['ASFW'] = {
            'x': asfw_x,
            'history': asfw_history,
            'time': asfw_time
        }

        pwfw_x, pwfw_history, pwfw_time = self.pwfw(max_iter, tol)
        results['PWFW'] = {
            'x': pwfw_x,
            'history': pwfw_history,
            'time': pwfw_time
        }

        return results
    def project_simplex(self, y: np.ndarray) -> np.ndarray:
        u = np.sort(y)[::-1] #y is sorted from largest to smallest as necessary for the simplex prjection algorithm
        cs= np.cumsum(u)  #cumulative sum
        rho = np.nonzero(u * np.arange(1, len(y) + 1) > (cs - 1))[0][-1] #find the position where the projection threshold should be calculated
        theta = (cs[rho] - 1) / (rho + 1) #threshold
        return np.maximum(y - theta, 0)

    def projected_gradient(self, step_size: float = None, max_iter: int = 1000, tol: float = 1e-6) -> tuple[np.ndarray, pd.DataFrame, float]:
        if step_size is None:
            L = 2 * self.gamma * np.max(np.linalg.eigvalsh(self.Sigma))
            step_size = 1 / L

        start_time = time.time()
        x = np.ones(self.n) / self.n
        history = []
        for k in range(max_iter):
            grad = self.gradient(x)
            y = x - step_size * grad
            x_new = self.project_simplex(y)

            # Compute FW gap for PGD
            s_idx = np.argmin(grad)
            s = np.zeros(self.n)
            s[s_idx] = 1.0
            fw_gap = -np.dot(grad, s - x_new)

            history.append({
                "Iteration": k,
                "Objective": self.objective(x_new),
                "Duality Gap": fw_gap,
                "Active Set Size": np.sum(x_new > tol)

            })

            if np.linalg.norm(x_new - x) < tol:
                x = x_new
                break
            x = x_new

        execution_time = time.time() - start_time
        return x, pd.DataFrame(history), execution_time


class PortfolioOptimisation_rolling(PortfolioOptimisation_Whole_Dataset):
    def __init__(self, data: np.ndarray, window_size: int = 52, data_offset: int = 1, gamma: float = 1.0) -> None:
        self.data = np.asarray(data, dtype=float)
        self.window_size = window_size
        self.data_offset = data_offset
        self.gamma = float(gamma)

        self.n_assets = self.data.shape[1]
        self.means, self.covariances = compute_var_and_means(
            self.data,
            window_size=self.window_size,
            offset=self.data_offset,
        )
        self.T = len(self.means)

    def set_window(self, k: int) -> None:
        self.r_bar = self.means[k]
        self.Sigma = 0.5 * (self.covariances[k] + self.covariances[k].T)
        self.n = len(self.r_bar)

    def rolling_backtest(self, max_iter: int = 10000, tol: float = 1e-6, forecast_horizon: int = 4, methods=("FW", "ASFW", "PWFW")) -> pd.DataFrame:
        records = []
        n_windows = len(self.data) - self.window_size - forecast_horizon + 1
        budgets = {method: 1.0 for method in methods}

        for k in range(0, n_windows, forecast_horizon):
            self.set_window(k)
            optimised = {}

            for method in methods:
                solver = getattr(self, method.lower())
                x, history, runtime = solver(max_iter=max_iter, tol=tol)
                optimised[method] = (x, history, runtime)

            for t in range(1, forecast_horizon + 1):
                test_idx = k + self.window_size + t - 1
                if test_idx >= len(self.data):
                    break

                realised_asset_returns = self.data[test_idx]

                for method, (x, history, runtime) in optimised.items():
                    predicted_return = float(np.dot(self.r_bar, x))
                    realised_return = float(np.dot(realised_asset_returns, x))
                    budgets[method] *= 1.0 + realised_return

                    records.append({
                        "window": k,
                        "rebalance_period": k // forecast_horizon,
                        "train_start": k,
                        "train_end": k + self.window_size - 1,
                        "forecast_horizon": t,
                        "test_index": test_idx,
                        "method": method,
                        "predicted_return": predicted_return,
                        "realised_return": realised_return,
                        "prediction_error": realised_return - predicted_return,
                        "absolute_error": abs(realised_return - predicted_return),
                        "squared_error": (realised_return - predicted_return) ** 2,
                        "direction_correct": (
                            np.sign(predicted_return) == np.sign(realised_return)
                        ),
                        "objective": float(self.objective(x)),
                        "iterations": len(history),
                        "final_gap": float(history["Duality Gap"].iloc[-1]),
                        "active_assets": int(np.sum(x > tol)),
                        "runtime": runtime,
                        "budget": budgets[method],
                    })

        return pd.DataFrame(records)
    def backtest_summary(backtest: pd.DataFrame) -> pd.DataFrame:
        summaries = []

        for method, df in backtest.groupby("method"):
            errors = df["prediction_error"].to_numpy()
            realised = df["realised_return"].to_numpy()

            summaries.append({
                "method": method,
                "weeks_tested": len(df),
                "MAE": float(np.mean(np.abs(errors))),
                "RMSE": float(np.sqrt(np.mean(errors ** 2))),
                "Mean predicted return": float(df["predicted_return"].mean()),
                "Mean realised return": float(df["realised_return"].mean()),
                "Directional accuracy": float(df["direction_correct"].mean()),
                "Realised volatility": float(df["realised_return"].std(ddof=1)),
                "Cumulative realised return": float(np.prod(1.0 + realised) - 1.0),
            })

        return pd.DataFrame(summaries).sort_values("method").reset_index(drop=True)
    def run_all(self, max_iter: int = 10000, tol: float = 1e-6) -> dict[int, dict[str, dict[str, Union[np.ndarray, pd.DataFrame, float]]]]:
        all_results = {}

        for k in range(self.T):
            self.set_window(k)
            all_results[k] = self.run_all_fw(max_iter=max_iter, tol=tol)

        return all_results




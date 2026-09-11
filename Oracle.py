import numpy as np
import pandas as pd
import time


def comp_stats(data: np.ndarray, wind_size: int = 52, offset: int = 1) -> tuple[np.ndarray, np.ndarray]:
    means = []
    cov = []
    for k in range(0, data.shape[0] - wind_size, offset):
        wind = data[k:k+ wind_size]
        #calculate mean return and covariance for each window
        means.append(np.mean(wind, axis=0))
        cov.append(np.cov(wind, rowvar=False))

    return np.asarray(means), np.asarray(cov)

class PortfolioOptimisation_Whole_Dataset:
    def __init__(self, Sigma: np.ndarray, r_bar: np.ndarray, gamma: float = 1) -> None:
        #here we initialise all of the key components of the markowitz problmem 
        self.Sigma = Sigma 
        self.r_bar = r_bar
        self.gamma = gamma 
        self.n = len(r_bar) 

    def obj(self, x: np.ndarray) -> float:
        return self.gamma * np.dot(x, np.dot(self.Sigma, x)) - np.dot(self.r_bar, x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return 2 * self.gamma * np.dot(self.Sigma, x) - self.r_bar 

    def line_search(self, x: np.ndarray, d: np.ndarray, gamma_max: float, tol: float = 1e-16) -> float:
        #important for selecting step size 
        grad = self.grad(x)
        d_Sigma_d = self.gamma * np.dot(d, np.dot(self.Sigma, d)) 
        #if almost no curvature, use the maximum step
        if d_Sigma_d <= tol: 
            return gamma_max 
        opt_gamma = -np.dot(grad, d) / (2 * d_Sigma_d) 
        return max(0 , min(opt_gamma, gamma_max)) # want to make sure optimal step size allowed

    def fw(self, max_iter: int = 1000, tol: float = 1e-6, use_line_search: bool = True) -> tuple[np.ndarray, pd.DataFrame, float]:
        s_t = time.time() 
        #init asset 0 to having 100 percent of the wealth
        x = np.zeros(self.n)
        x[0] = 1 
        hist = []

        for k in range(max_iter):
            grad = self.grad(x)
            #choose asset with smallest grad
            s_idx = np.argmin(grad) 
            s = np.zeros(self.n)
            s[s_idx] = 1  
            #calculate fw gap and direction
            d_fw  = s - x 
            fw_gap = -np.dot(grad, d_fw) 

            hist.append({
                'iter': k,
                'obj': self.obj(x),
                'gap': fw_gap,
                'active': np.sum(x > tol) 
            })

            if fw_gap <= tol:
                    break
            #useful for the dim vs line search analysis
            if use_line_search:
                gamma_k = self.line_search(x, d_fw, gamma_max=1) 
            else:
                #dim step size
                gamma_k = 2  / (k + 2) 
            #update portfolio in direction of selected asset
            x = (1 - gamma_k) * x + gamma_k * s
        e_t = time.time() - s_t 
        return x, pd.DataFrame(hist), e_t

    def asfw(self, max_iter: int = 1000, tol: float = 1e-6) -> tuple[np.ndarray, pd.DataFrame, float]:
        #same as in fw
        s_t = time.time() 
        a_set = {0: 1} 
        x = np.zeros(self.n)
        x[0] = 1  
        hist = []
        for k in range(max_iter):
            grad = self.grad(x)
            #choose the asset with the smallest gradient
            s_idx = np.argmin(grad) 
            d_fw = np.zeros(self.n)
            d_fw[s_idx] = 1 
            d_fw -= x 
            #improvement from fw direction
            fw_gap = -np.dot(grad, d_fw)
            
            a_idx = list(a_set.keys()) 
            v_idx = a_idx[np.argmax([grad[i] for i in a_idx])] 
            d_asfw = x.copy() #dont want to change structure of x directly
            d_asfw[v_idx] = d_asfw[v_idx] - 1  
            g_asfw = -np.dot(grad, d_asfw) 
            
            hist.append({
                'iter': k,
                'obj': self.obj(x),
                'gap': fw_gap,
                'active': len(a_set)
            })

            if fw_gap <= tol:
                break
            #compare fw and aw directions
            if fw_gap >= g_asfw:
                d = d_fw
                gamma_max = 1 
                is_away = False
            else:
                d = d_asfw 
                #add 1e-16 so that no division by zero
                gamma_max = a_set[v_idx] / (1  - a_set[v_idx] + 1e-16) 
                is_away = True

            gamma_t = self.line_search(x, d, gamma_max)
            x = x + gamma_t * d 

        
            if not is_away:
                for idx in list(a_set.keys()):
                    a_set[idx] = a_set[idx] * (1  - gamma_t) 
                a_set[s_idx] = a_set.get(s_idx, 0) + gamma_t 
            else:
                for idx in list(a_set.keys()):
                    a_set[idx] = a_set[idx] * (1  + gamma_t) 
                a_set[v_idx] = a_set[v_idx] - gamma_t 
            #remove assets with effectively zero weight
            a_set = {idx: w for idx, w in a_set.items() if w > tol} 

        e_t = time.time() - s_t
        return x, pd.DataFrame(hist), e_t

    def pwfw(self, max_iter: int = 1000, tol: float = 1e-6) -> tuple[np.ndarray, pd.DataFrame, float]:
        s_t = time.time()
        a_set = {0: 1}
        x = np.zeros(self.n)
        x[0] = 1 
        hist = []

        for k in range(max_iter):
            grad = self.grad(x)

            s_idx = int(np.argmin(grad))
            a_idx = list(a_set.keys())
            #active asset to remove weight from
            v_idx = a_idx[int(np.argmax([grad[i] for i in a_idx]))]

            s = np.zeros(self.n)
            s[s_idx] = 1 
            d_fw = s - x
            fw_gap = -np.dot(grad, d_fw)

            hist.append({
                'iter': k,
                'obj': self.obj(x),
                'gap': fw_gap,
                'active': len(a_set)
            })


            if fw_gap <= tol:
                break

            if s_idx == v_idx:
                d = d_fw
                gamma_max = 1 
                is_pw = False
            else:
                #transfer weight directly from the away asset to the new asset
                d = np.zeros(self.n)
                d[s_idx] = 1 
                d[v_idx] = -1 
                #cannot remove more weight than the asset currently has
                gamma_max = a_set[v_idx]
                is_pw = True

            gamma_t = self.line_search(x, d, gamma_max)
            #update portfolio
            x = x + gamma_t * d

            if is_pw:
                #transfer weight from v_idx to s_idx
                a_set[s_idx] = a_set.get(s_idx, 0) + gamma_t
                a_set[v_idx] = a_set[v_idx] - gamma_t
            else:
                for idx in list(a_set.keys()):
                    a_set[idx] = a_set[idx] * (1  - gamma_t)
                a_set[s_idx] = a_set.get(s_idx, 0) + gamma_t

            a_set = {idx: w for idx, w in a_set.items() if w > tol}

        e_t = time.time() - s_t
        return x, pd.DataFrame(hist), e_t

    def run_all_fw(self, max_iter: int = 1000, tol: float = 1e-6) -> dict[str, dict[str, Any]]:
        results = {}

        fw_x, fw_hist, fw_time = self.fw(max_iter, tol)
        results['fw'] = {
            'x': fw_x,
            'hist': fw_hist,
            'time'  : fw_time
        }

        asfw_x, asfw_hist, asfw_time = self.asfw(max_iter, tol)
        results['asfw'] = {
            'x': asfw_x,
            'hist': asfw_hist,
            'time': asfw_time
        }

        pwfw_x, pwfw_hist, pwfw_time = self.pwfw(max_iter, tol)
        results['pwfw'] = {
            'x': pwfw_x,
            'hist': pwfw_hist,
            'time': pwfw_time
        }

        return results
    def project_simplex(self, y: np.ndarray) -> np.ndarray:
        #find projection thresh
        u = np.sort(y)[::-1] 
        cs= np.cumsum(u)  
        #sum up to one
        rho = np.nonzero(u * np.arange(1, len(y) + 1) > (cs - 1))[0][-1] 
        theta = (cs[rho] - 1) / (rho + 1) 
        return np.maximum(y - theta, 0)

    def projected_grad(self, step_size: float = None, max_iter: int = 1000, tol: float = 1e-6) -> tuple[np.ndarray, pd.DataFrame, float]:
        if step_size is None:
            #use the largest eigenvalue to choose the step size
            L = 2 * self.gamma * np.max(np.linalg.eigvalsh(self.Sigma))
            step_size = 1 / L

        s_t = time.time()
        #start with equal weights between assets
        x = np.ones(self.n) / self.n
        hist = []
        for k in range(max_iter):
            grad = self.grad(x)
            #take a gradient step before projecting back onto the simplex
            y = x - step_size * grad
            x_new = self.project_simplex(y)

            s_idx = np.argmin(grad)
            s = np.zeros(self.n)
            s[s_idx] = 1 
            fw_gap = -np.dot(grad, s - x_new)

            hist.append({
                'iter': k,
                'obj': self.obj(x),
                'gap': fw_gap,
                'active': np.sum(x > tol) 
            })

            if np.linalg.norm(x_new - x) < tol:
                x = x_new
                break
            x = x_new

        e_t = time.time() - s_t
        return x, pd.DataFrame(hist), e_t

#new class oracle so that we can initialise other vars
class PortfolioOptimisation_rolling(PortfolioOptimisation_Whole_Dataset):
    def __init__(self, data: np.ndarray, wind_size: int = 52, data_offset: int = 1, gamma: float = 1) -> None:
        self.data = np.asarray(data, dtype=float)
        self.wind_size = wind_size
        self.data_offset = data_offset
        self.gamma = float(gamma)

        self.n_assets = self.data.shape[1]
        self.means, self.cov = comp_stats(
            self.data,
            wind_size=self.wind_size,
            offset=self.data_offset,
       )
        self.T = len(self.means)

    def wind(self, k: int) -> None:
        #update model using the current rolling window
        self.r_bar = self.means[k]
        self.Sigma = 0.5 * (self.cov[k] + self.cov[k].T)
        self.n = len(self.r_bar)

    def rolling_backtest(self, max_iter: int = 10000, tol: float = 1e-6, f_horizon: int = 4, methods=("fw", "asfw", "pwfw")) -> pd.DataFrame:
        records = []
        #number of windows for testing
        n_wind = len(self.data) - self.wind_size - f_horizon + 1
        #cumulative wealth for each opt method 
        bugs = {method: 1  for method in methods}

        for k in range(0, n_wind, f_horizon):
            #model update using current trainign window
            self.wind(k)
            opt = {}

            #current window running
            for method in methods:
                solver = getattr(self, method.lower())
                x, hist, runtime = solver(max_iter=max_iter, tol=tol)
                opt[method] = (x, hist, runtime)
            #test the portfolios over the forecast horizon
            for t in range(1, f_horizon + 1):
                test_idx = k + self.wind_size + t - 1
                if test_idx >= len(self.data):
                    break

                #get the realised asset returns for current test period    
                real_a_r = self.data[test_idx]

                for method, (x, hist, runtime) in opt.items():
                    real_ret = float(np.dot(real_a_r, x))
                    bugs[method] = bugs[method] *(1  + real_ret)

                    records.append({
                        "wind": k,
                        "rebalance_period": k // f_horizon,
                        "train_start": k,
                        "train_end": k + self.wind_size - 1,
                        "f_horizon": t,
                        "test_index": test_idx,
                        "method": method,
                        "real_ret": real_ret,
                        "obj": float(self.obj(x)),
                        "iter": len(hist),
                        "gap": float(hist["gap"].iloc[-1]),
                        "active": int(np.sum(x > tol)),
                        "time": runtime,
                        "budget": bugs[method]
                    })

        return pd.DataFrame(records)




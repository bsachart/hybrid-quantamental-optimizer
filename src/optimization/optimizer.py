"""
Module for Mean-Variance Optimization (MVO).

Philosophy:
    - "Strategic programming": Design the optimizer to be extensible (e.g., adding constraints later).
    - Maximize Sharpe Ratio is the default objective.
    - Robustness: Handle edge cases where risk-free rate > portfolio return (though rare in long-only equity).

Reference:
    - Modern Portfolio Theory (Markowitz).
"""

from dataclasses import dataclass
import numpy as np
import scipy.optimize as sco
import numpy.typing as npt
from typing import Dict, Tuple, Optional, List


# def portfolio_performance(
#     weights: npt.NDArray[np.float64],
#     expected_returns: npt.NDArray[np.float64],
#     cov_matrix: npt.NDArray[np.float64],
#     risk_free_rate: float = 0.0,
# ) -> Tuple[float, float, float]:
#     """
#     Calculates portfolio return, volatility, and Sharpe ratio.

#     Args:
#         weights: Asset weights (sum to 1).
#         expected_returns: Annualized expected returns for each asset.
#         cov_matrix: Annualized covariance matrix.
#         risk_free_rate: Risk-free rate for Sharpe calculation.

#     Returns:
#         (return, volatility, sharpe_ratio)
#     """
#     weights = np.array(weights)
#     ret = np.sum(weights * expected_returns)
#     vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
#     sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0
#     return ret, vol, sharpe


# def negative_sharpe_ratio(
#     weights: npt.NDArray[np.float64],
#     expected_returns: npt.NDArray[np.float64],
#     cov_matrix: npt.NDArray[np.float64],
#     risk_free_rate: float = 0.0,
# ) -> float:
#     """
#     Objective function to minimize (negative Sharpe Ratio).
#     """
#     _, _, sharpe = portfolio_performance(
#         weights, expected_returns, cov_matrix, risk_free_rate
#     )
#     return -sharpe


# def optimize_portfolio(
#     expected_returns: npt.NDArray[np.float64],
#     cov_matrix: npt.NDArray[np.float64],
#     risk_free_rate: float = 0.02,
#     max_weight: float = 1.0,
# ) -> Dict[str, any]:
#     """
#     Finds the optimal portfolio weights that maximize the Sharpe Ratio.

#     Constraints:
#     1. Sum of weights = 1 (Fully invested).
#     2. Weights >= 0 (Long only).
#     3. Weights <= max_weight (Concentration limit).

#     Args:
#         expected_returns: 1D array of annualized expected returns.
#         cov_matrix: 2D array of annualized covariance.
#         risk_free_rate: Risk-free rate (default 2%).
#         max_weight: Maximum weight for a single asset (default 1.0 = no limit).

#     Returns:
#         Dictionary containing:
#         - 'weights': Optimal weights.
#         - 'return': Portfolio return.
#         - 'volatility': Portfolio volatility.
#         - 'sharpe': Sharpe ratio.
#         - 'success': Boolean indicating optimizer success.
#         - 'message': Optimizer status message.
#     """
#     num_assets = len(expected_returns)
#     args = (expected_returns, cov_matrix, risk_free_rate)

#     # Initial guess: Equal weights
#     initial_weights = num_assets * [
#         1.0 / num_assets,
#     ]

#     # Constraints
#     # 1. Sum of weights = 1
#     constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

#     # Bounds
#     # 0 <= weight <= max_weight
#     bounds = tuple((0.0, max_weight) for _ in range(num_assets))

#     result = sco.minimize(
#         negative_sharpe_ratio,
#         initial_weights,
#         args=args,
#         method="SLSQP",
#         bounds=bounds,
#         constraints=constraints,
#     )

#     if not result.success:
#         # Fallback or warning?
#         # For now, we return the result but mark success as False
#         pass

#     optimal_weights = result.x
#     ret, vol, sharpe = portfolio_performance(
#         optimal_weights, expected_returns, cov_matrix, risk_free_rate
#     )

#     return {
#         "weights": optimal_weights,
#         "return": ret,
#         "volatility": vol,
#         "sharpe": sharpe,
#         "success": result.success,
#         "message": result.message,
#     }


# def portfolio_volatility(
#     weights: npt.NDArray[np.float64], cov_matrix: npt.NDArray[np.float64]
# ) -> float:
#     """
#     Helper to calculate just volatility for minimization.
#     """
#     return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


# def generate_random_portfolios(
#     num_portfolios: int,
#     expected_returns: npt.NDArray[np.float64],
#     cov_matrix: npt.NDArray[np.float64],
#     frontier_points: Optional[List[Dict[str, float]]] = None,
# ) -> List[Dict[str, float]]:
#     """
#     Generates random portfolios to visualize the feasible set.
    
#     Simple approach: Dirichlet sampling with varied concentration parameters
#     to get a mix of concentrated and diversified portfolios.
#     """
#     results = []
#     num_assets = len(expected_returns)
    
#     # Simple Dirichlet sampling with varying concentration
#     # Low alpha = concentrated portfolios (sparse, near vertices)
#     # High alpha = diversified portfolios (near center)
#     alphas = [0.1, 0.3, 0.5, 1.0, 2.0]
#     n_per_alpha = num_portfolios // len(alphas)
    
#     for alpha in alphas:
#         for _ in range(n_per_alpha):
#             weights = np.random.dirichlet(np.ones(num_assets) * alpha)
#             ret = np.sum(weights * expected_returns)
#             vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
#             results.append({"return": ret, "volatility": vol, "weights": weights})
    
#     # Fill any remainder
#     remainder = num_portfolios - len(results)
#     for _ in range(remainder):
#         weights = np.random.dirichlet(np.ones(num_assets))
#         ret = np.sum(weights * expected_returns)
#         vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
#         results.append({"return": ret, "volatility": vol, "weights": weights})
    
#     return results


# def get_min_volatility_portfolio(
#     expected_returns: npt.NDArray[np.float64], cov_matrix: npt.NDArray[np.float64]
# ) -> Dict[str, float]:
#     """
#     Finds the global minimum volatility portfolio.
#     """
#     num_assets = len(expected_returns)
#     initial_weights = np.ones(num_assets) / num_assets
#     bounds = tuple((0.0, 1.0) for _ in range(num_assets))
#     constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

#     result = sco.minimize(
#         portfolio_volatility,
#         initial_weights,
#         args=(cov_matrix,),
#         method="SLSQP",
#         bounds=bounds,
#         constraints=constraints,
#     )

#     ret = np.sum(result.x * expected_returns)

#     return {"return": ret, "volatility": result.fun, "weights": result.x}


# def calculate_efficient_frontier(
#     expected_returns: npt.NDArray[np.float64],
#     cov_matrix: npt.NDArray[np.float64],
#     num_points: int = 20,
# ) -> List[Dict[str, float]]:
#     """
#     Calculates the efficient frontier by minimizing volatility for a range of target returns.

#     Args:
#         expected_returns: Annualized expected returns.
#         cov_matrix: Annualized covariance matrix.
#         num_points: Number of points to calculate on the frontier.

#     Returns:
#         List of dictionaries [{'return': r, 'volatility': v}, ...]
#     """
#     num_assets = len(expected_returns)

#     # Determine range of returns
#     # Start from Global Minimum Variance Portfolio
#     min_vol_port = get_min_volatility_portfolio(expected_returns, cov_matrix)
#     min_ret = min_vol_port["return"]
#     max_ret = np.max(expected_returns)

#     # Create target returns grid
#     # We add a small buffer to ensure feasible solutions
#     if min_ret >= max_ret:
#         # Edge case: all assets have same return or something weird
#         target_returns = np.array([min_ret])
#     else:
#         target_returns = np.linspace(min_ret, max_ret, num_points)

#     frontier_points = []

#     # Initial guess
#     initial_weights = np.ones(num_assets) / num_assets
#     bounds = tuple((0.0, 1.0) for _ in range(num_assets))

#     for target in target_returns:
#         # Constraints:
#         # 1. Sum of weights = 1
#         # 2. Portfolio return = target
#         # Note: Use default arg (t=target) to capture by value, not reference
#         constraints = (
#             {"type": "eq", "fun": lambda x: np.sum(x) - 1},
#             {"type": "eq", "fun": lambda x, t=target: np.sum(x * expected_returns) - t},
#         )

#         result = sco.minimize(
#             portfolio_volatility,
#             initial_weights,
#             args=(cov_matrix,),
#             method="SLSQP",
#             bounds=bounds,
#             constraints=constraints,
#         )

#         if result.success:
#             frontier_points.append({"return": target, "volatility": result.fun, "weights": result.x})

#     return frontier_points




@dataclass
class PortfolioMetrics:
    """
    Portfolio performance metrics.
    
    Includes diagnostic info from optimization:
    - success: Whether optimizer converged
    - message: Optimizer status message (empty string if not from optimization)
    """
    return_: float
    volatility: float
    sharpe_ratio: float
    weights: npt.NDArray[np.float64]
    success: bool = True
    message: str = ""


class PortfolioOptimizer:
    """
    Deep module for portfolio optimization.
    
    Hides complexity of:
    - Scipy optimization mechanics
    - Constraint formulation
    - Numerical edge cases
    
    Simple interface: Just provide returns/covariance and get optimal portfolios.
    """
    
    def __init__(
        self,
        expected_returns: npt.NDArray[np.float64],
        cov_matrix: npt.NDArray[np.float64],
        risk_free_rate: float = 0.02,
    ):
        """
        Initialize optimizer with market assumptions.
        
        Args:
            expected_returns: Annualized expected returns per asset
            cov_matrix: Annualized covariance matrix
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.num_assets = len(expected_returns)
    
    def calculate_metrics(self, weights: npt.NDArray[np.float64]) -> PortfolioMetrics:
        """
        Calculate portfolio metrics for given weights.
        
        Note: Sharpe ratio can be negative when portfolio return < risk-free rate.
        When volatility is zero (risk-free or zero-weight portfolio), Sharpe is 
        undefined; we return 0.0 by convention.
        """
        ret = weights @ self.expected_returns
        vol = np.sqrt(weights @ self.cov_matrix @ weights)
        sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0.0
        return PortfolioMetrics(ret, vol, sharpe, weights)
    
    def maximize_sharpe(self, max_weight: float = 1.0) -> PortfolioMetrics:
        """
        Find portfolio with maximum Sharpe ratio.
        
        Args:
            max_weight: Maximum allocation to single asset (concentration limit)
            
        Returns:
            PortfolioMetrics with optimal portfolio and optimization status
        """
        def objective(weights):
            metrics = self.calculate_metrics(weights)
            return -metrics.sharpe_ratio  # Minimize negative Sharpe
        
        result = self._optimize(
            objective=objective,
            bounds=(0.0, max_weight),
            target_return=None,
        )
        
        metrics = self.calculate_metrics(result.x)
        metrics.success = result.success
        metrics.message = result.message
        return metrics
    
    def minimize_volatility(
        self, 
        target_return: Optional[float] = None
    ) -> PortfolioMetrics:
        """
        Find minimum volatility portfolio, optionally with target return.
        
        Args:
            target_return: If provided, minimize vol subject to this return
            
        Returns:
            PortfolioMetrics with minimum volatility portfolio and optimization status
        """
        def objective(weights):
            return np.sqrt(weights @ self.cov_matrix @ weights)
        
        result = self._optimize(
            objective=objective,
            bounds=(0.0, 1.0),
            target_return=target_return,
        )
        
        metrics = self.calculate_metrics(result.x)
        metrics.success = result.success
        metrics.message = result.message
        return metrics
    
    def efficient_frontier(self, num_points: int = 20) -> List[PortfolioMetrics]:
        """
        Calculate efficient frontier by minimizing volatility across return range.
        
        Args:
            num_points: Number of portfolios to compute
            
        Returns:
            List of portfolios along efficient frontier (only successful optimizations)
        """
        # Get return range from min-vol portfolio to max individual return
        min_vol = self.minimize_volatility()
        min_return = min_vol.return_
        max_return = np.max(self.expected_returns)
        
        if min_return >= max_return:
            return [min_vol]
        
        target_returns = np.linspace(min_return, max_return, num_points)
        frontier = []
        
        for target in target_returns:
            portfolio = self.minimize_volatility(target_return=target)
            if portfolio.success:
                frontier.append(portfolio)
        
        return frontier
    
    def random_portfolios(self, num_portfolios: int = 1000) -> List[PortfolioMetrics]:
        """
        Generate random portfolios for visualization.
        
        Uses Dirichlet sampling with varied concentration to explore
        both concentrated and diversified allocations.
        
        Args:
            num_portfolios: Number of random portfolios to generate
            
        Returns:
            List of random portfolio metrics
        """
        portfolios = []
        alphas = [0.1, 0.3, 0.5, 1.0, 2.0]  # Concentration parameters
        n_per_alpha = num_portfolios // len(alphas)
        
        for alpha in alphas:
            for _ in range(n_per_alpha):
                weights = np.random.dirichlet(np.ones(self.num_assets) * alpha)
                portfolios.append(self.calculate_metrics(weights))
        
        # Fill remainder with uniform concentration
        for _ in range(num_portfolios - len(portfolios)):
            weights = np.random.dirichlet(np.ones(self.num_assets))
            portfolios.append(self.calculate_metrics(weights))
        
        return portfolios
    
    def _optimize(
        self,
        objective,
        bounds: tuple[float, float],
        target_return: Optional[float],
    ) -> sco.OptimizeResult:
        """
        Internal optimization routine. Encapsulates scipy complexity.
        
        This is the "deep" part - users never see constraints, methods, etc.
        """
        initial_weights = np.ones(self.num_assets) / self.num_assets
        bounds_tuple = tuple(bounds for _ in range(self.num_assets))
        
        # Always constrain: weights sum to 1
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        
        # Optionally constrain: portfolio return equals target
        if target_return is not None:
            constraints.append({
                "type": "eq",
                "fun": lambda x, t=target_return: x @ self.expected_returns - t
            })
        
        return sco.minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds_tuple,
            constraints=constraints,
        )


# Convenience function for single-shot optimization
def optimize_portfolio(
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    risk_free_rate: float = 0.02,
    max_weight: float = 1.0,
) -> PortfolioMetrics:
    """
    Convenience function: Find optimal portfolio that maximizes Sharpe ratio.
    
    For advanced use (efficient frontier, multiple optimizations), 
    create a PortfolioOptimizer instance directly.
    """
    optimizer = PortfolioOptimizer(expected_returns, cov_matrix, risk_free_rate)
    return optimizer.maximize_sharpe(max_weight)
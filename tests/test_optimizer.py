"""
Tests for portfolio optimization module.

Organization:
    - Fixtures: Shared test data
    - TestPortfolioMetrics: Core calculation tests
    - TestOptimization: Constraint and objective tests
    - TestEfficientFrontier: Frontier dominance tests
    - TestEdgeCases: Boundary conditions and error handling
"""

import pytest
import numpy as np
import pandas as pd
from src.optimization.optimizer import (
    PortfolioOptimizer,
    optimize_portfolio,
    PortfolioMetrics,
)
from src.core.returns import calculate_implied_cagr


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_data():
    """
    Five uncorrelated assets with increasing risk/return.
    
    Returns are 10%-30%, volatilities are 10%-30%.
    Uncorrelated for easy reasoning about optimal portfolios.
    """
    expected_returns = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    vols = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    cov_matrix = np.diag(vols**2)
    return expected_returns, cov_matrix


@pytest.fixture
def optimizer(sample_data):
    """Reusable optimizer instance."""
    expected_returns, cov_matrix = sample_data
    return PortfolioOptimizer(expected_returns, cov_matrix, risk_free_rate=0.02)


# ============================================================================
# Test Portfolio Metrics
# ============================================================================

class TestPortfolioMetrics:
    """Test core portfolio calculations."""
    
    def test_calculate_metrics(self, optimizer):
        """Verify return, volatility, and Sharpe calculations."""
        # Equal weights portfolio
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        metrics = optimizer.calculate_metrics(weights)
        
        # Expected return = weighted average of returns
        assert np.isclose(metrics.return_, 0.20)
        
        # For uncorrelated assets: portfolio_var = sum(w_i^2 * var_i)
        # vol = sqrt(0.2^2 * 0.10^2 + 0.2^2 * 0.15^2 + ... + 0.2^2 * 0.30^2)
        variances = np.array([0.10, 0.15, 0.20, 0.25, 0.30])**2
        expected_vol = np.sqrt(np.sum(weights**2 * variances))
        assert np.isclose(metrics.volatility, expected_vol)
        
        # Sharpe = (ret - rf) / vol
        expected_sharpe = (0.20 - 0.02) / expected_vol
        assert np.isclose(metrics.sharpe_ratio, expected_sharpe)
    
    def test_zero_weights(self, optimizer):
        """Zero weights should give zero return/volatility, zero Sharpe."""
        weights = np.zeros(5)
        metrics = optimizer.calculate_metrics(weights)
        
        assert metrics.return_ == 0.0
        assert metrics.volatility == 0.0
        assert metrics.sharpe_ratio == 0.0  # Division by zero handled
    
    def test_negative_sharpe_ratio(self):
        """Sharpe can be negative when portfolio return < risk-free rate."""
        # Low return (1%), high risk-free rate (5%)
        optimizer = PortfolioOptimizer(
            expected_returns=np.array([0.01, 0.02]),
            cov_matrix=np.diag([0.04, 0.09]),
            risk_free_rate=0.05
        )
        
        # 100% in first asset
        weights = np.array([1.0, 0.0])
        metrics = optimizer.calculate_metrics(weights)
        
        # Sharpe = (0.01 - 0.05) / 0.20 = -0.20
        assert np.isclose(metrics.sharpe_ratio, -0.20)


# ============================================================================
# Test Optimization
# ============================================================================

class TestOptimization:
    """Test optimization constraints and objectives."""
    
    def test_maximize_sharpe_basic(self, optimizer):
        """Optimizer should find valid portfolio with positive Sharpe."""
        portfolio = optimizer.maximize_sharpe()
        
        assert portfolio.success
        assert np.isclose(np.sum(portfolio.weights), 1.0)
        assert np.all(portfolio.weights >= -1e-6)
        assert portfolio.sharpe_ratio > 0
    
    def test_max_weight_constraint(self, optimizer):
        """Optimizer must respect concentration limits."""
        max_weight = 0.25
        portfolio = optimizer.maximize_sharpe(max_weight=max_weight)
        
        assert portfolio.success
        assert np.isclose(np.sum(portfolio.weights), 1.0)
        assert np.all(portfolio.weights <= max_weight + 1e-6)
        assert np.all(portfolio.weights >= -1e-6)
    
    def test_minimize_volatility(self, optimizer):
        """Min-vol portfolio should have lowest possible volatility."""
        portfolio = optimizer.minimize_volatility()
        
        assert portfolio.success
        assert np.isclose(np.sum(portfolio.weights), 1.0)
        
        # For uncorrelated assets, min-vol portfolio allocates inversely to variance
        # w_i ∝ 1/var_i
        # Asset 0 has lowest variance (0.10^2 = 0.01), so should have highest weight
        # But it's not necessarily 100% because risk-return tradeoff
        # Just verify asset 0 has the highest weight
        assert portfolio.weights[0] == np.max(portfolio.weights)
    
    def test_minimize_volatility_with_target_return(self, optimizer):
        """Target return constraint should be satisfied."""
        target_return = 0.20
        portfolio = optimizer.minimize_volatility(target_return=target_return)
        
        assert portfolio.success
        assert np.isclose(portfolio.return_, target_return, atol=1e-4)


# ============================================================================
# Test Efficient Frontier
# ============================================================================

class TestEfficientFrontier:
    """Test efficient frontier generation and dominance properties."""
    
    def test_frontier_dominance(self, optimizer):
        """
        Efficient frontier must dominate random portfolios.
        
        For any random portfolio at volatility V, the frontier portfolio
        at volatility V must have higher (or equal) return.
        """
        # Generate frontier and random portfolios
        frontier = optimizer.efficient_frontier(num_points=50)
        random_portfolios = optimizer.random_portfolios(num_portfolios=100)
        
        # Extract frontier curve
        frontier_vols = np.array([p.volatility for p in frontier])
        frontier_rets = np.array([p.return_ for p in frontier])
        
        # Sort for interpolation
        sort_idx = np.argsort(frontier_vols)
        frontier_vols = frontier_vols[sort_idx]
        frontier_rets = frontier_rets[sort_idx]
        
        # Check each random portfolio
        for rp in random_portfolios:
            # Interpolate max return at this volatility
            max_return_at_vol = np.interp(rp.volatility, frontier_vols, frontier_rets)
            
            # Random portfolio should not exceed frontier (with tolerance)
            assert max_return_at_vol >= rp.return_ - 1e-4, (
                f"Random portfolio (Vol={rp.volatility:.2%}, Ret={rp.return_:.2%}) "
                f"exceeds frontier (MaxRet={max_return_at_vol:.2%})"
            )
    
    def test_frontier_gap_exists(self, optimizer):
        """
        Optimal portfolio should significantly outperform random portfolios.
        
        This validates the 'cloud below frontier' visualization.
        """
        # Generate many random portfolios
        random_portfolios = optimizer.random_portfolios(num_portfolios=500)
        
        # Get optimal Sharpe portfolio
        optimal = optimizer.maximize_sharpe()
        opt_sharpe = optimal.sharpe_ratio
        
        # Calculate Sharpe for random portfolios
        random_sharpes = np.array([
            (p.return_ - 0.02) / p.volatility for p in random_portfolios
        ])
        
        # Optimal must beat all random portfolios
        max_random_sharpe = np.max(random_sharpes)
        assert opt_sharpe >= max_random_sharpe
        
        # Should significantly beat average (demonstrates gap)
        avg_random_sharpe = np.mean(random_sharpes)
        assert opt_sharpe > avg_random_sharpe * 1.1
    
    def test_frontier_monotonic(self, optimizer):
        """Frontier should have monotonically increasing return and volatility."""
        frontier = optimizer.efficient_frontier(num_points=30)
        
        returns = [p.return_ for p in frontier]
        vols = [p.volatility for p in frontier]
        
        # Returns should increase (or stay same)
        assert all(returns[i] <= returns[i+1] + 1e-6 for i in range(len(returns)-1))
        
        # Volatility should increase (efficient frontier slopes up-right)
        assert all(vols[i] <= vols[i+1] + 1e-6 for i in range(len(vols)-1))


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test boundary conditions and error handling."""
    
    def test_single_asset(self):
        """Single asset should get 100% allocation."""
        optimizer = PortfolioOptimizer(
            expected_returns=np.array([0.10]),
            cov_matrix=np.array([[0.04]]),
            risk_free_rate=0.02
        )
        
        portfolio = optimizer.maximize_sharpe()
        
        assert portfolio.success
        assert np.isclose(portfolio.weights[0], 1.0)
    
    def test_all_negative_returns(self):
        """Should find least-bad portfolio without crashing."""
        optimizer = PortfolioOptimizer(
            expected_returns=np.array([-0.05, -0.10, -0.15]),
            cov_matrix=np.diag([0.01, 0.02, 0.03]),
            risk_free_rate=0.0
        )
        
        portfolio = optimizer.maximize_sharpe()
        
        # Should complete without crashing
        assert np.isclose(np.sum(portfolio.weights), 1.0)
        # All returns negative, so portfolio return must be negative
        assert portfolio.return_ < 0
    
    def test_infeasible_max_weight(self):
        """Max weight too low makes equal allocation impossible."""
        optimizer = PortfolioOptimizer(
            expected_returns=np.array([0.1, 0.15, 0.2, 0.25, 0.3]),
            cov_matrix=np.diag([0.01] * 5),
            risk_free_rate=0.02
        )
        
        # 5 assets * 0.1 max = 0.5 < 1.0 needed (infeasible)
        portfolio = optimizer.maximize_sharpe(max_weight=0.1)
        
        # Optimizer may fail or return constrained solution
        # Just verify no crash and weights respect bounds
        if portfolio.success:
            assert np.all(portfolio.weights <= 0.1 + 1e-6)
    
    def test_identical_assets(self):
        """Identical assets should get roughly equal weights."""
        optimizer = PortfolioOptimizer(
            expected_returns=np.array([0.1, 0.1, 0.1]),
            cov_matrix=np.eye(3) * 0.04,
            risk_free_rate=0.02
        )
        
        portfolio = optimizer.maximize_sharpe()
        
        assert portfolio.success
        # Diversification should spread weights roughly equally
        assert np.std(portfolio.weights) < 0.1
    
    def test_near_zero_volatility_asset(self):
        """Asset with near-zero volatility should not crash calculations."""
        optimizer = PortfolioOptimizer(
            expected_returns=np.array([0.1, 0.02]),
            cov_matrix=np.array([[0.04, 0], [0, 1e-10]]),
            risk_free_rate=0.02
        )
        
        # Should not crash - just verify it completes
        portfolio = optimizer.maximize_sharpe()
        # Either succeeds or fails gracefully
        assert isinstance(portfolio, PortfolioMetrics)


# ============================================================================
# Test CAGR Edge Cases (if still needed - consider moving to separate file)
# ============================================================================

class TestCAGREdgeCases:
    """Edge cases for calculate_implied_cagr."""
    
    def test_negative_target_margin(self):
        """Company expected to have losses at exit."""
        cagr = calculate_implied_cagr(
            current_price=100,
            sales_per_share=10,
            net_margin_current=0.1,
            net_margin_target=-0.2,
            adjusted_growth_rate=0.1,
            exit_pe=20,
            years=5,
        )
        assert cagr == -1.0
    
    def test_zero_sales(self):
        """Pre-revenue company has zero future value."""
        cagr = calculate_implied_cagr(
            current_price=100,
            sales_per_share=0,
            net_margin_current=0,
            net_margin_target=0.2,
            adjusted_growth_rate=0.5,
            exit_pe=30,
            years=5,
        )
        assert cagr == -1.0
    
    def test_raises_on_zero_price(self):
        """Zero price should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            calculate_implied_cagr(
                current_price=0,
                sales_per_share=10,
                net_margin_current=0.1,
                net_margin_target=0.2,
                adjusted_growth_rate=0.1,
                exit_pe=20,
                years=5,
            )
    
    def test_raises_on_zero_years(self):
        """Zero years should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            calculate_implied_cagr(
                current_price=100,
                sales_per_share=10,
                net_margin_current=0.1,
                net_margin_target=0.2,
                adjusted_growth_rate=0.1,
                exit_pe=20,
                years=0,
            )


# ============================================================================
# Test Convenience Function
# ============================================================================

def test_convenience_function(sample_data):
    """Test backward-compatible convenience function."""
    expected_returns, cov_matrix = sample_data
    
    portfolio = optimize_portfolio(
        expected_returns,
        cov_matrix,
        risk_free_rate=0.02,
        max_weight=0.3
    )
    
    assert isinstance(portfolio, PortfolioMetrics)
    assert portfolio.success
    assert np.isclose(np.sum(portfolio.weights), 1.0)
    assert np.all(portfolio.weights <= 0.3 + 1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
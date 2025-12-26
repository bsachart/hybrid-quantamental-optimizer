
import pytest
import numpy as np
from src.optimization.optimizer import PortfolioOptimizer, PortfolioMetrics

class TestPortfolioOptimizer:
    """Test suite for portfolio optimization and cash allocation."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        expected_returns = np.array([0.10, 0.12, 0.08])
        cov_matrix = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.01],
            [0.005, 0.01, 0.02]
        ])
        risk_free_rate = 0.02
        return expected_returns, cov_matrix, risk_free_rate
    
    def test_no_cash_in_optimization(self, sample_data):
        """Test that cash is NOT included in the optimization."""
        expected_returns, cov_matrix, risk_free_rate = sample_data
        
        optimizer = PortfolioOptimizer(
            expected_returns, cov_matrix, risk_free_rate
        )
        
        # Verify number of assets
        assert optimizer.num_assets == 3, "Should have exactly 3 assets (no cash)"
        
        # Verify expected returns doesn't include cash
        assert len(optimizer.expected_returns) == 3
        
        # Verify covariance matrix shape
        assert optimizer.cov_matrix.shape == (3, 3)
        
        # Check that we can't access include_risk_free_asset
        assert not hasattr(optimizer, 'include_risk_free_asset')
    
    def test_maximize_sharpe_returns_valid_portfolio(self, sample_data):
        """Test that maximize_sharpe returns a valid portfolio."""
        expected_returns, cov_matrix, risk_free_rate = sample_data
        
        optimizer = PortfolioOptimizer(
            expected_returns, cov_matrix, risk_free_rate
        )
        result = optimizer.maximize_sharpe()
        
        # Check weights sum to 1
        assert np.isclose(np.sum(result.weights), 1.0)
        
        # Check weights are non-negative (long-only)
        assert np.all(result.weights >= -1e-6)
        
        # Check success
        assert result.success
        
        # Check Sharpe ratio is positive
        assert result.sharpe_ratio > 0
        
        # Verify no cash in weights
        assert len(result.weights) == 3
    
    def test_sharpe_ratio_calculation(self, sample_data):
        """Test that Sharpe ratio is calculated correctly."""
        expected_returns, cov_matrix, risk_free_rate = sample_data
        
        optimizer = PortfolioOptimizer(
            expected_returns, cov_matrix, risk_free_rate
        )
        
        # Test with known weights
        weights = np.array([0.4, 0.3, 0.3])
        metrics = optimizer.calculate_metrics(weights)
        
        # Manually calculate expected values
        expected_return = np.dot(weights, expected_returns)
        expected_vol = np.sqrt(weights @ cov_matrix @ weights)
        expected_sharpe = (expected_return - risk_free_rate) / expected_vol
        
        assert np.isclose(metrics.return_, expected_return)
        assert np.isclose(metrics.volatility, expected_vol)
        assert np.isclose(metrics.sharpe_ratio, expected_sharpe)
    
    def test_maximize_sharpe_better_than_equal_weight(self, sample_data):
        """Test that optimal portfolio beats equal-weight portfolio."""
        expected_returns, cov_matrix, risk_free_rate = sample_data
        
        optimizer = PortfolioOptimizer(
            expected_returns, cov_matrix, risk_free_rate
        )
        
        # Equal weight portfolio
        equal_weights = np.array([1/3, 1/3, 1/3])
        equal_metrics = optimizer.calculate_metrics(equal_weights)
        
        # Optimal portfolio
        optimal = optimizer.maximize_sharpe()
        
        # Optimal should have higher or equal Sharpe ratio
        assert optimal.sharpe_ratio >= equal_metrics.sharpe_ratio - 1e-6
    
    def test_allocate_with_cash_basic(self, sample_data):
        """Test cash allocation with target volatility."""
        expected_returns, cov_matrix, risk_free_rate = sample_data
        
        optimizer = PortfolioOptimizer(
            expected_returns, cov_matrix, risk_free_rate
        )
        
        # Get tangency portfolio
        tangency = optimizer.maximize_sharpe()
        
        # Target half the volatility
        target_vol = tangency.volatility * 0.5
        
        # Allocate with cash
        result = optimizer.allocate_with_cash(
            tangency, 
            target_vol,
            asset_names=["Stock A", "Stock B", "Stock C"]
        )
        
        # Check that we get ~50% risky, 50% cash
        assert np.isclose(result['risky_fraction'], 0.5, atol=0.01)
        assert np.isclose(result['cash_fraction'], 0.5, atol=0.01)
        
        # Check final volatility matches target
        assert np.isclose(result['final_volatility'], target_vol, atol=1e-6)
        
        # Check allocation table includes cash
        assert "CASH" in result['allocation_table']
        
        # Check all weights sum to 1
        total_weight = sum(result['allocation_table'].values())
        assert np.isclose(total_weight, 1.0)
    
    def test_allocate_with_cash_high_target_volatility(self, sample_data):
        """Test cash allocation when target vol exceeds tangency vol."""
        expected_returns, cov_matrix, risk_free_rate = sample_data
        
        optimizer = PortfolioOptimizer(
            expected_returns, cov_matrix, risk_free_rate
        )
        
        tangency = optimizer.maximize_sharpe()
        
        # Request higher volatility than tangency (should cap at 100% risky)
        target_vol = tangency.volatility * 1.5
        
        result = optimizer.allocate_with_cash(tangency, target_vol)
        
        # Should be 100% in risky portfolio
        assert np.isclose(result['risky_fraction'], 1.0)
        assert np.isclose(result['cash_fraction'], 0.0)
        
        # Final volatility should equal tangency volatility
        assert np.isclose(result['final_volatility'], tangency.volatility)
    
    def test_allocate_with_cash_zero_target_volatility(self, sample_data):
        """Test cash allocation with zero target volatility (100% cash)."""
        expected_returns, cov_matrix, risk_free_rate = sample_data
        
        optimizer = PortfolioOptimizer(
            expected_returns, cov_matrix, risk_free_rate
        )
        
        tangency = optimizer.maximize_sharpe()
        
        # Target zero volatility
        result = optimizer.allocate_with_cash(tangency, 0.0)
        
        # Should be 100% cash
        assert np.isclose(result['risky_fraction'], 0.0)
        assert np.isclose(result['cash_fraction'], 1.0)
        assert np.isclose(result['final_volatility'], 0.0)
        assert np.isclose(result['final_return'], risk_free_rate)
    
    def test_bounds_validation(self, sample_data):
        """Test that bounds are properly validated."""
        expected_returns, cov_matrix, risk_free_rate = sample_data
        
        optimizer = PortfolioOptimizer(
            expected_returns, cov_matrix, risk_free_rate
        )
        
        # Wrong number of bounds should raise error
        with pytest.raises(ValueError, match="Bounds length"):
            optimizer.maximize_sharpe(bounds=[(0, 1), (0, 1)])  # Only 2 bounds for 3 assets
    
    def test_efficient_frontier(self, sample_data):
        """Test efficient frontier generation."""
        expected_returns, cov_matrix, risk_free_rate = sample_data
        
        optimizer = PortfolioOptimizer(
            expected_returns, cov_matrix, risk_free_rate
        )
        
        frontier = optimizer.efficient_frontier(num_points=10)
        
        # Should return multiple points
        assert len(frontier) > 0
        
        # All points should be successful
        assert all(p.success for p in frontier)
        
        # Returns should be increasing
        returns = [p.return_ for p in frontier]
        assert returns == sorted(returns)
        
        # No cash in any weights
        assert all(len(p.weights) == 3 for p in frontier)

    def test_single_asset_portfolio(self):
        """Test with only one asset."""
        expected_returns = np.array([0.10])
        cov_matrix = np.array([[0.04]])
        
        optimizer = PortfolioOptimizer(expected_returns, cov_matrix, 0.02)
        result = optimizer.maximize_sharpe()
        
        assert result.success
        assert np.isclose(result.weights[0], 1.0)


    def test_negative_expected_returns(self):
        """Test behavior with some negative expected returns."""
        expected_returns = np.array([-0.05, 0.10, 0.08])
        cov_matrix = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.01],
            [0.005, 0.01, 0.02]
        ])
        
        optimizer = PortfolioOptimizer(expected_returns, cov_matrix, 0.02)
        result = optimizer.maximize_sharpe()
        
        # Should avoid or minimize negative return asset
        assert result.weights[0] < 0.1  # Should have low weight on negative return asset
        assert result.success

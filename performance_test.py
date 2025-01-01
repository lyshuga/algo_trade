import numpy as np
import pandas as pd
import time
from trading_agent import TradingStrategy
# Suppress pandas FutureWarning about integer indexing
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def generate_test_data(n_points=1000):
    """Generate sample OHLCV data"""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=n_points, freq='D')
    
    # Generate random walk prices
    close = 100 * (1 + np.random.randn(n_points).cumsum() * 0.02)
    high = close * (1 + abs(np.random.randn(n_points) * 0.01))
    low = close * (1 - abs(np.random.randn(n_points) * 0.01))
    open_price = close * (1 + np.random.randn(n_points) * 0.01)
    volume = np.random.randint(1000, 100000, n_points)
    
    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)
    
    return df

def benchmark_indicators():
    # Get test data
    data = generate_test_data(1000)
    
    # Create a standalone class with just the indicator methods
    class IndicatorTester:
        def __init__(self):
            self.data = data
            self.volume_profile_period = 20
            self.vwap_deviation = 1.5
            self.macd_signal_length = 9
            
            # Pre-calculate required indicators
            self.vwap = self._calculate_vwap()
            self.swing_highs = self._find_swing_points(high=True)
            self.swing_lows = self._find_swing_points(high=False)
        
        # Copy all the indicator methods from TradingStrategy
        _sma = TradingStrategy._sma
        _ema = TradingStrategy._ema
        _rsi = TradingStrategy._rsi
        _macd_line = TradingStrategy._macd_line
        _macd_signal = TradingStrategy._macd_signal
        _atr = TradingStrategy._atr
        _bb_upper = TradingStrategy._bb_upper
        _bb_middle = TradingStrategy._bb_middle
        _bb_lower = TradingStrategy._bb_lower
        _calculate_vwap = TradingStrategy._calculate_vwap
        _calculate_poc = TradingStrategy._calculate_poc
        _calculate_value_area = TradingStrategy._calculate_value_area
        _calculate_vwap_bands = TradingStrategy._calculate_vwap_bands
        _find_swing_points = TradingStrategy._find_swing_points
        _analyze_trend_structure = TradingStrategy._analyze_trend_structure
    
    strategy = IndicatorTester()
    
    # Dictionary to store execution times
    execution_times = {}
    
    # Test each indicator function
    indicators_to_test = {
        'SMA': (strategy._sma, [data['Close'].values, 20]),
        'EMA': (strategy._ema, [data['Close'].values, 20]),
        'RSI': (strategy._rsi, [data['Close'].values, 14]),
        'MACD Line': (strategy._macd_line, [data['Close'].values]),
        'MACD Signal': (strategy._macd_signal, [data['Close'].values]),
        'ATR': (strategy._atr, [data['High'].values, data['Low'].values, data['Close'].values, 14]),
        'Bollinger Upper': (strategy._bb_upper, [data['Close'].values, 20, 2]),
        'Bollinger Middle': (strategy._bb_middle, [data['Close'].values, 20]),
        'Bollinger Lower': (strategy._bb_lower, [data['Close'].values, 20, 2])
    }
    
    print("\nBenchmarking indicators (1000 data points):")
    print("-" * 50)
    print(f"{'Indicator':<20} {'Time (ms)':<10} {'Calls/sec':<10}")
    print("-" * 50)
    
    # Run each indicator 10 times and take average
    n_iterations = 10
    for name, (func, args) in indicators_to_test.items():
        start_time = time.perf_counter()
        
        for _ in range(n_iterations):
            _ = func(*args)
            
        end_time = time.perf_counter()
        
        avg_time = ((end_time - start_time) * 1000) / n_iterations  # Convert to milliseconds
        calls_per_sec = 1000 / avg_time  # Calculate calls per second
        
        execution_times[name] = avg_time
        print(f"{name:<20} {avg_time:>8.3f}ms {calls_per_sec:>8.0f}/s")
    
    # Volume profile indicators
    volume_indicators = {
        'VWAP': (strategy._calculate_vwap, []),
        'POC': (strategy._calculate_poc, []),
        'Value Area': (strategy._calculate_value_area, []),
        'VWAP Bands': (strategy._calculate_vwap_bands, [1]),
        'Swing Points': (strategy._find_swing_points, [True, 5]),
        'Trend Structure': (strategy._analyze_trend_structure, [20])
    }
    
    print("\nVolume Profile Indicators:")
    print("-" * 50)
    
    for name, (func, args) in volume_indicators.items():
        start_time = time.perf_counter()
        
        for _ in range(n_iterations):
            _ = func(*args)
            
        end_time = time.perf_counter()
        
        avg_time = ((end_time - start_time) * 1000) / n_iterations
        calls_per_sec = 1000 / avg_time
        
        execution_times[name] = avg_time
        print(f"{name:<20} {avg_time:>8.3f}ms {calls_per_sec:>8.0f}/s")

    return execution_times

if __name__ == "__main__":
    benchmark_indicators() 
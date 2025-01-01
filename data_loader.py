import pandas as pd
import numpy as np
import ccxt
import ta
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self, exchange='binance', symbol='BTC/USDT', timeframe='1h'):
        self.exchange = ccxt.binance()
        self.symbol = symbol
        self.timeframe = timeframe
        
    def fetch_historical_data(self, start_date, end_date):
        """Fetch historical data and add technical indicators"""
        # Convert dates to timestamps
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        # Fetch OHLCV data
        ohlcv = self.exchange.fetch_ohlcv(
            symbol=self.symbol,
            timeframe=self.timeframe,
            since=start_ts,
            limit=1000,
            until=end_ts  # Add end timestamp parameter
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv, 
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Add technical indicators
        self._add_indicators(df)
        
        return df
    
    def _add_indicators(self, df):
        """Add various technical indicators"""
        # Trend indicators
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # Volume indicators
        df['vwap'] = self._calculate_vwap(df)
        df['volume_sma'] = ta.volume.volume_weighted_average_price(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume']
        )
        
        # Momentum indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['macd_line'] = ta.trend.macd_diff(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        
        # Volatility indicators
        # Initialize Bollinger Bands indicator
        bb_indicator = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        
        # Add Bollinger Bands features
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        
        df['atr'] = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'], window=14
        )
        
        # Market structure
        df['support'] = self._calculate_support(df)
        df['resistance'] = self._calculate_resistance(df)
        
        return df
    
    def _calculate_vwap(self, df):
        """Calculate VWAP (Volume Weighted Average Price)"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    def _calculate_support(self, df, window=20):
        """Calculate dynamic support levels using local minima"""
        return df['low'].rolling(window=window).min()
    
    def _calculate_resistance(self, df, window=20):
        """Calculate dynamic resistance levels using local maxima"""
        return df['high'].rolling(window=window).max() 
from backtesting import Strategy
from backtesting.lib import crossover
import ta
import numpy as np
import pandas as pd

class TradingStrategy(Strategy):
    # Add these parameters near the top with other basic parameters
    use_max_position_duration = True
    max_hold_period = 48  # in hours
    # Basic parameters
    n_sma_fast = 20
    n_sma_slow = 50
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    atr_period = 14
    atr_multiplier = 2.0
    volume_sma_length = 20
    entry_threshold = 0.5
    
    # Feature toggles and parameters
    use_volume_profile = True
    volume_profile_period = 20
    vwap_deviation = 1.5
    
    use_market_structure = True
    swing_window = 5
    trend_window = 20
    
    use_bollinger_bands = True
    bb_window = 20
    bb_dev = 2
    
    use_macd = True
    macd_fast = 12
    macd_slow = 26
    macd_signal_length = 9
    
    use_rsi = True
    
    use_volume = True
    volume_surge_threshold = 1.2
    
    # Risk management parameters
    risk_per_trade = 0.02
    tp_atr_multiplier = 1.5
    
    # Weights as individual parameters
    weight_trend = 2.0
    weight_momentum = 1.0
    weight_volatility = 1.0
    weight_volume = 1.0
    weight_structure = 2.0

    use_breakeven_stop = True
    breakeven_threshold_atr = 1.0  # Move to breakeven after price moves this many ATRs in our favor
    breakeven_buffer_atr = 0.1 
    
    def init(self):
        # Basic price data
        close = np.array(self.data.Close)
        high = np.array(self.data.High)
        low = np.array(self.data.Low)
        volume = np.array(self.data.Volume)
        
        # Core trend indicators (always enabled)
        self.sma_fast = self.I(self._sma, close, self.n_sma_fast)
        self.sma_slow = self.I(self._sma, close, self.n_sma_slow)
        
        self.entry_price = None

        if self.use_max_position_duration:
            # Track position entry times
            self.position_entry_time = None

        # Optional EMA/MACD
        if self.use_macd:
            self.ema12 = self.I(self._ema, close, self.macd_fast)
            self.ema26 = self.I(self._ema, close, self.macd_slow)
            self.macd = self.I(self._macd_line, close)
            self.macd_signal = self.I(self._macd_signal, close)
        
        # Optional RSI
        if self.use_rsi:
            self.rsi = self.I(self._rsi, close, self.rsi_period)
        
        # Optional ATR
        self.atr = self.I(self._atr, high, low, close, self.atr_period)
        
        # Optional Bollinger Bands
        if self.use_bollinger_bands:
            self.bb_upper = self.I(self._bb_upper, close, window=self.bb_window, dev=self.bb_dev)
            self.bb_middle = self.I(self._bb_middle, close, window=self.bb_window)
            self.bb_lower = self.I(self._bb_lower, close, window=self.bb_window, dev=self.bb_dev)
        
        # Optional Volume indicators
        if self.use_volume:

            self.volume_sma = self.I(self._sma, volume, self.volume_sma_length)
        
        # Optional Volume Profile
        if self.use_volume_profile:
            self.vwap = self.I(self._calculate_vwap)
            self.volume_poc = self.I(self._calculate_poc)
            self.volume_var = self.I(self._calculate_value_area)
            self.vwap_upper = self.I(self._calculate_vwap_bands, direction=1)
            self.vwap_lower = self.I(self._calculate_vwap_bands, direction=-1)
        
        # Optional Market Structure
        if self.use_market_structure:
            self.swing_highs = self.I(self._find_swing_points, high=True, window=self.swing_window)
            self.swing_lows = self.I(self._find_swing_points, high=False, window=self.swing_window)
            self.trend_structure = self.I(self._analyze_trend_structure, window=self.trend_window)

    # # Helper indicator functions
    # def _sma(self, data, window):
    #     return pd.Series(data).rolling(window).mean().values
        
    def _ema(self, data, window):
        return pd.Series(data).ewm(span=window, adjust=False).mean().values
    
    # Helper indicator functions
    def _sma(self, data, window):
        # Using numpy's convolve for faster SMA calculation
        weights = np.ones(window) / window
        sma = np.convolve(data, weights, mode='valid')
        # Pad the beginning with NaN to match original length
        return np.concatenate([np.full(window-1, np.nan), sma])
        
    # def _ema(self, data, window):
    #     # Using numexpr for faster EMA calculation
    #     alpha = 2.0 / (window + 1)
    #     # Initialize output array
    #     ema = np.empty_like(data)
    #     ema[0] = data[0]  # First value is same as input
    #     # Calculate EMA
    #     for i in range(1, len(data)):
    #         ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    #     return ema
        
    def _rsi(self, data, window):
        series = pd.Series(data)
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return (100 - (100 / (1 + rs))).values
        
    def _macd_line(self, data):
        series = pd.Series(data)
        exp1 = series.ewm(span=12, adjust=False).mean()
        exp2 = series.ewm(span=26, adjust=False).mean()
        return (exp1 - exp2).values
        
    def _macd_signal(self, data):
        macd = self._macd_line(data)
        return pd.Series(macd).ewm(span=self.macd_signal_length, adjust=False).mean().values
        
    def _atr(self, high, low, close, window):
        """Optimized ATR calculation"""
        high = np.array(high)
        low = np.array(low)
        close = np.array(close)
        
        # Calculate True Range components using numpy operations
        high_low = high - low
        high_close = np.abs(high - np.roll(close, 1))
        low_close = np.abs(low - np.roll(close, 1))
        
        # Combine TR components
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        tr[0] = tr[1]  # Fix first value
        
        # Calculate ATR using exponential moving average
        atr = np.zeros_like(tr)
        atr[0] = tr[0]
        multiplier = 2 / (window + 1)
        
        for i in range(1, len(tr)):
            atr[i] = (tr[i] * multiplier) + (atr[i-1] * (1 - multiplier))
        
        return atr
        
    def _bb_upper(self, data, window=20, dev=2):
        series = pd.Series(data)
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        return (sma + (std * dev)).values
        
    def _bb_middle(self, data, window=20):
        return pd.Series(data).rolling(window=window).mean().values
        
    def _bb_lower(self, data, window=20, dev=2):
        series = pd.Series(data)
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        return (sma - (std * dev)).values
        
    def _calculate_vwap(self):
        high = np.array(self.data.High)
        low = np.array(self.data.Low)
        close = np.array(self.data.Close)
        volume = np.array(self.data.Volume)
        
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    def _calculate_poc(self):
        """Optimized Point of Control calculation using numpy"""
        prices = (self.data.High + self.data.Low) / 2
        volumes = self.data.Volume
        window = self.volume_profile_period
        poc = np.zeros(len(prices))
        
        # Pre-allocate arrays for the sliding window
        window_prices = np.zeros(window)
        window_volumes = np.zeros(window)
        
        # Use fixed number of bins for histogram
        n_bins = 50
        
        for i in range(window, len(prices)):
            # Update window data using numpy slicing
            window_prices = prices[i-window:i]
            window_volumes = volumes[i-window:i]
            
            # Fixed bin edges for better performance
            bins = np.linspace(window_prices.min(), window_prices.max(), n_bins)
            
            # Use numpy's histogram function with weights
            hist, _ = np.histogram(window_prices, bins=bins, weights=window_volumes)
            
            # Find the price level with highest volume
            poc[i] = bins[np.argmax(hist)]
        
        return poc
    
    def _calculate_value_area(self, value_area_pct=0.70):
        """Optimized Value Area calculation"""
        prices = (self.data.High + self.data.Low) / 2
        volumes = self.data.Volume
        window = self.volume_profile_period
        
        upper = np.zeros(len(prices))
        lower = np.zeros(len(prices))
        
        # Pre-allocate arrays
        window_prices = np.zeros(window)
        window_volumes = np.zeros(window)
        
        for i in range(window, len(prices)):
            # Update window data using numpy slicing
            window_prices = prices[i-window:i]
            window_volumes = volumes[i-window:i]
            
            # Sort using numpy's argsort
            sorted_indices = np.argsort(window_prices)
            sorted_volumes = window_volumes[sorted_indices]
            sorted_prices = window_prices[sorted_indices]
            
            # Calculate cumulative volume using numpy's cumsum
            cum_vol = np.cumsum(sorted_volumes)
            target_vol = cum_vol[-1] * value_area_pct
            
            # Find value area boundaries using binary search
            value_area_idx = np.searchsorted(cum_vol, target_vol)
            
            if value_area_idx < len(sorted_prices):
                upper[i] = sorted_prices[value_area_idx]
                lower[i] = sorted_prices[0]
            else:
                upper[i] = sorted_prices[-1]
                lower[i] = sorted_prices[0]
        
        return upper, lower
    
    def _calculate_vwap_bands(self, direction=1):
        """Optimized VWAP deviation bands calculation"""
        vwap = self.vwap
        window = self.volume_profile_period
        close = np.array(self.data.Close)
        
        # Calculate rolling standard deviation using numpy's stride tricks
        def rolling_std(arr, window):
            # Create a view into the array with rolling windows
            shape = (arr.shape[0] - window + 1, window)
            strides = (arr.strides[0], arr.strides[0])
            windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
            
            # Calculate std for each window
            return np.concatenate([np.full(window-1, np.nan), np.std(windows, axis=1)])
        
        std = rolling_std(close, window)
        return vwap + (direction * self.vwap_deviation * std)
    
    def _find_swing_points(self, high=True, window=5):
        """Find swing highs or lows"""
        prices = np.array(self.data.High if high else self.data.Low)
        swings = np.zeros(len(prices))
        
        for i in range(window, len(prices)-window):
            if high:
                if prices[i] == max(prices[i-window:i+window+1]):
                    swings[i] = prices[i]
            else:
                if prices[i] == min(prices[i-window:i+window+1]):
                    swings[i] = prices[i]
                    
        return swings
    
    def _analyze_trend_structure(self, window=20):
        """Analyze trend structure using swing points"""
        highs = self.swing_highs
        lows = self.swing_lows
        
        structure = np.zeros(len(highs))
        
        for i in range(window, len(highs)):
            # Get non-zero swing points in the window
            recent_highs = [h for h in highs[i-window:i] if h > 0]
            recent_lows = [l for l in lows[i-window:i] if l > 0]
            
            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                higher_highs = recent_highs[-1] > recent_highs[-2]
                higher_lows = recent_lows[-1] > recent_lows[-2]
                
                if higher_highs and higher_lows:
                    structure[i] = 1  # Uptrend
                elif not higher_highs and not higher_lows:
                    structure[i] = -1  # Downtrend
                else:
                    structure[i] = 0  # Consolidation
                    
        return structure

    def next(self):

        if self.use_breakeven_stop and self.position.size != 0 and self.entry_price is not None:
            current_price = self.data.Close[-1]
            atr = self.atr[-1]
            breakeven_buffer = atr * self.breakeven_buffer_atr
            
            if self.position.size > 0:  # Long position
                profit_target = self.entry_price + (atr * self.breakeven_threshold_atr)
                if current_price >= profit_target:
                    new_stop = self.entry_price + breakeven_buffer
                    if new_stop > self.position.sl:
                        self.position.sl = new_stop
            
            else:  # Short position
                profit_target = self.entry_price - (atr * self.breakeven_threshold_atr)
                if current_price <= profit_target:
                    new_stop = self.entry_price - breakeven_buffer
                    if new_stop < self.position.sl:
                        self.position.sl = new_stop
        

        # Add this at the start of next() method, before other position checks
        if self.use_max_position_duration and self.position.size != 0:
            # Check if we need to exit due to time limit
            current_time = self.data.index[-1]
            if self.position_entry_time is not None:
                hold_duration = current_time - self.position_entry_time
                if hold_duration.total_seconds() / 3600 >= self.max_hold_period:
                    self.position.close()
                    self.position_entry_time = None
                    return
        
        pos_size = self.position.size
        atr_stop = self.atr[-1] * self.atr_multiplier
        
        if pos_size == 0:
            # Calculate position size based on risk
            risk_pct = self.risk_per_trade  # 2% risk per trade
            price = self.data.Close[-1]
            
            # Calculate position size in units (whole numbers)
            account_risk = self.equity * risk_pct
            risk_per_unit = atr_stop * price  # Dollar risk per unit
            position_size = max(1, round(account_risk / risk_per_unit))  # At least 1 unit
            
            # Check entry conditions with probability scores
            long_prob = self._check_long_entry()
            short_prob = self._check_short_entry()
            
            # Scale position size based on probability (keeping it as whole numbers)
            # Add after successful entry (inside the if blocks after self.buy/self.sell)
            if long_prob >= self.entry_threshold:
                sl = price - atr_stop
                tp = price + (atr_stop * (self.tp_atr_multiplier + long_prob))
                final_size = max(1, round(position_size * long_prob))
                self.buy(size=final_size, sl=sl, tp=tp)
                if self.use_max_position_duration:
                    self.position_entry_time = self.data.index[-1]
                
                self.entry_price = price
                
            elif short_prob >= self.entry_threshold:
                sl = price + atr_stop
                tp = price - (atr_stop * (self.tp_atr_multiplier + short_prob))
                final_size = max(1, round(position_size * short_prob))
                self.sell(size=final_size, sl=sl, tp=tp)
                if self.use_max_position_duration:
                    self.position_entry_time = self.data.index[-1]
                self.entry_price = price

            # Reset entry price when position is closed
            if self.position.size == 0:
                self.entry_price = None


    def _check_long_entry(self):
        score = 0
        total_weight = 0
        
        # Trend confirmation (always included)
        if self.data.Close[-1] > self.sma_fast[-1]:
            score += 2 * self.weight_trend
        if self.sma_fast[-1] > self.sma_slow[-1]:
            score += 2 * self.weight_trend
        total_weight += 4 * self.weight_trend
        
        # MACD
        if self.use_macd:
            if self.ema12[-1] > self.ema26[-1]:
                score += 1 * self.weight_momentum
            if self.macd[-1] > self.macd_signal[-1]:
                score += 1 * self.weight_momentum
            if self.macd[-1] > 0:
                score += 1 * self.weight_momentum
            total_weight += 3 * self.weight_momentum
        
        # Market structure
        if self.use_market_structure:
            if self.trend_structure[-1] == 1:
                score += 2 * self.weight_structure
            
            recent_low = next((low for low in self.swing_lows[-20:] if low > 0), None)
            if recent_low and self.data.Close[-1] > recent_low:
                score += 1 * self.weight_structure
            
            recent_lows = [low for low in self.swing_lows[-20:] if low > 0]
            if len(recent_lows) >= 2 and recent_lows[-1] > recent_lows[-2]:
                score += 1 * self.weight_structure
            total_weight += 4 * self.weight_structure
        
        # Volume
        if self.use_volume:
            if self.data.Volume[-1] > self.volume_sma[-1] * self.volume_surge_threshold:
                score += 1 * self.weight_volume
            total_weight += self.weight_volume
        
        # RSI
        if self.use_rsi:
            # if self.rsi[-1] < 40:
            #     score += 1 * self.weight_momentum
            if self.rsi[-1] < self.rsi_oversold:
                score += 1 * self.weight_momentum
            total_weight += 1 * self.weight_momentum
        
        # Bollinger Bands
        if self.use_bollinger_bands:
            if self.data.Close[-1] <= self.bb_lower[-1]:
                score += 2 * self.weight_volatility
            elif self.data.Close[-1] <= self.bb_middle[-1]:
                score += 1 * self.weight_volatility
            total_weight += 2 * self.weight_volatility
        
        # Volume Profile
        if self.use_volume_profile:
            if self.data.Close[-1] > self.volume_poc[-1]:
                score += 1 * self.weight_volume
            if self.data.Close[-1] < self.vwap_upper[-1]:
                score += 1 * self.weight_volume
            if self.data.Close[-1] > self.vwap[-1]:
                score += 1 * self.weight_volume
            
            _, value_area_low = self.volume_var
            if abs(self.data.Close[-1] - value_area_low[-1]) / self.data.Close[-1] < 0.01:
                score += 2 * self.weight_volume
            total_weight += 5 * self.weight_volume
        
        # Calculate final probability
        probability = score / total_weight if total_weight > 0 else 0
        return probability

    def _check_short_entry(self):
        score = 0
        total_weight = 0
        
        # Trend confirmation (always included)
        if self.data.Close[-1] < self.sma_fast[-1]:
            score += 2 * self.weight_trend
        if self.sma_fast[-1] < self.sma_slow[-1]:
            score += 2 * self.weight_trend
        total_weight += 4 * self.weight_trend
        
        # MACD
        if self.use_macd:
            if self.ema12[-1] < self.ema26[-1]:
                score += 1 * self.weight_momentum
            if self.macd[-1] < self.macd_signal[-1]:
                score += 1 * self.weight_momentum
            if self.macd[-1] < 0:
                score += 1 * self.weight_momentum
            total_weight += 3 * self.weight_momentum
        
        # Market structure
        if self.use_market_structure:
            if self.trend_structure[-1] == -1:  # Confirmed downtrend
                score += 2 * self.weight_structure
            
            recent_high = next((high for high in self.swing_highs[-20:] if high > 0), None)
            if recent_high and self.data.Close[-1] < recent_high:
                score += 1 * self.weight_structure
            
            recent_highs = [high for high in self.swing_highs[-20:] if high > 0]
            if len(recent_highs) >= 2 and recent_highs[-1] < recent_highs[-2]:
                score += 1 * self.weight_structure
            total_weight += 4 * self.weight_structure
        
        # Volume
        if self.use_volume:
            if self.data.Volume[-1] > self.volume_sma[-1] * self.volume_surge_threshold:
                score += 1 * self.weight_volume
            total_weight += self.weight_volume
        
        # RSI
        if self.use_rsi:
            # if self.rsi[-1] > 60:
            #     score += 1 * self.weight_momentum
            if self.rsi[-1] > self.rsi_overbought:
                score += 1 * self.weight_momentum
            total_weight += 1 * self.weight_momentum
        
        # Bollinger Bands
        if self.use_bollinger_bands:
            if self.data.Close[-1] >= self.bb_upper[-1]:
                score += 2 * self.weight_volatility
            elif self.data.Close[-1] >= self.bb_middle[-1]:
                score += 1 * self.weight_volatility
            total_weight += 2 * self.weight_volatility
        
        # Volume Profile
        if self.use_volume_profile:
            if self.data.Close[-1] < self.volume_poc[-1]:
                score += 1 * self.weight_volume
            if self.data.Close[-1] > self.vwap_lower[-1]:
                score += 1 * self.weight_volume
            if self.data.Close[-1] < self.vwap[-1]:
                score += 1 * self.weight_volume
            
            value_area_high, _ = self.volume_var
            if abs(self.data.Close[-1] - value_area_high[-1]) / self.data.Close[-1] < 0.01:
                score += 2 * self.weight_volume
            total_weight += 5 * self.weight_volume
        
        # Calculate final probability
        probability = score / total_weight if total_weight > 0 else 0
        return probability 
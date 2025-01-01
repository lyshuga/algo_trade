# Cryptocurrency Trading Strategy Framework

A comprehensive backtesting framework for cryptocurrency trading strategies with advanced technical analysis and optimization capabilities.

## Overview

This project implements a flexible and modular cryptocurrency trading strategy framework that combines multiple technical indicators, market structure analysis, and volume profiling. It includes backtesting capabilities with parameter optimization and detailed performance analysis.

## Features

- **Multiple Technical Indicators**
  - SMA/EMA crossovers
  - RSI
  - MACD
  - Bollinger Bands
  - ATR for position sizing and stop losses
  - Volume analysis
  - VWAP and Volume Profile

- **Advanced Analysis**
  - Market structure detection
  - Dynamic support/resistance levels
  - Volume profile analysis
  - Multi-timeframe analysis
  - Probability-based entry signals

- **Risk Management**
  - Position sizing based on ATR
  - Dynamic stop-loss placement
  - Take-profit optimization
  - Maximum position duration
  - Breakeven stop functionality

- **Optimization & Analysis**
  - Parameter optimization using scikit-optimize
  - Detailed performance metrics
  - Visual analysis with interactive plots
  - Heat map visualization for parameter relationships

## Requirements

```
python
backtesting>=0.3.3
ccxt>=3.0.0
pandas>=1.3.0
numpy>=1.19.0
ta>=0.7.0
bokeh>=2.4.0
scikit-optimize>=0.9.0
```

## Installation

```
bash
git clone https://github.com/yourusername/crypto-trading-strategy.git
cd crypto-trading-strategy
pip install -r requirements.txt
```

## Usage

### Basic Usage

```
from backtesting import Backtest
from trading_agent import TradingStrategy
from data_loader import DataLoader
```

### Initialize data loader
```
loader = DataLoader(exchange='binance', symbol='BTC/USDT', timeframe='1h')
data = loader.fetch_historical_data('2024-01-01', '2024-12-31')
```

### Create and run backtest
```
bt = Backtest(data, TradingStrategy, cash=100000, commission=.002)
stats = bt.run()
```
### Analyze results
`print(stats)`

### Run optimization
```
stats, heatmap = bt.optimize(
opt_params,
maximize='Return (Ann.) [%]',
method='skopt',
max_tries=100,
return_heatmap=True
)
```


## Strategy Configuration

The trading strategy can be customized through various parameters:

- Basic trend parameters (SMA periods)
- RSI parameters (oversold/overbought levels)
- Volume analysis settings
- Feature toggles for different indicators
- Risk management parameters
- Scoring weights for different components

See `trading_agent.py` for all available parameters and their descriptions.

## Performance Analysis

The framework provides detailed performance metrics including:

- Overall returns and risk-adjusted metrics
- Trade analysis (win rate, profit factor, etc.)
- Position analysis by direction
- Time-based analysis
- Drawdown analysis
- Visual analysis through interactive plots

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.
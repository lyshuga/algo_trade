from backtesting import Backtest
from trading_agent import TradingStrategy
import pandas as pd
import ccxt
from bokeh.io import output_file, save
from bokeh.plotting import figure
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, HoverTool, ColorBar
from bokeh.transform import linear_cmap
from bokeh.palettes import RdYlBu11 as palette
import numpy as np

def fetch_data(start_date='2024-01-01', end_date='2024-12-31', timeframe='1h'):
    """Fetch historical data from Binance"""
    exchange = ccxt.binance()
    
    # Convert dates to timestamps
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
    
    # Fetch OHLCV data
    ohlcv = exchange.fetch_ohlcv(
        symbol='BTC/USDT',
        timeframe=timeframe,
        since=start_ts,
        limit=1000
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(
        ohlcv, 
        columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df

def plot_results(stats, filename='backtest_results.html'):
    """Custom plotting function"""
    output_file(filename)
    
    # Create figure for equity curve and price
    p1 = figure(
        title='Backtest Results',
        x_axis_type='datetime',
        width=1200,
        height=400
    )
    
    # # Plot price
    # price_data = stats._data.Close
    # p1.line(
    #     price_data.index,
    #     price_data,
    #     line_color='gray',
    #     alpha=0.7,
    #     legend_label='Price'
    # )
    
    # Plot equity curve
    equity_curve = stats._equity_curve
    p1.line(
        equity_curve.index,
        equity_curve.Equity,
        line_color='blue',
        legend_label='Equity'
    )
    
    # Add trades if any
    if not stats._trades.empty:
        trades = stats._trades
        
        # Plot entry points
        p1.circle(
            trades.EntryTime,
            trades.EntryPrice,
            size=8,
            color='green',
            legend_label='Entry'
        )
        
        # Plot exit points
        p1.circle(
            trades.ExitTime,
            trades.ExitPrice,
            size=8,
            color='red',
            legend_label='Exit'
        )
    
    # # Configure legend
    # p1.legend.location = "top_left"
    # p1.legend.click_policy = "hide"  # Makes legend items clickable
    
    return p1

def plot_optimization_grid(results_df, param_names, metric='Return (Ann.) [%]'):
    """Create grid plots for optimization results"""
    plots = []
    
    # Create pairs of parameters for grid plots
    param_pairs = [
        ('n_sma_fast', 'n_sma_slow'),
        ('rsi_oversold', 'rsi_overbought'),
        # ('n_sma_fast', 'atr_multiplier'),
        # ('rsi_oversold', 'atr_multiplier'),
        ('n_sma_fast', 'volume_sma_length'),
        # ('volume_sma_length', 'atr_multiplier'),
        ('rsi_oversold', 'volume_sma_length'),
        # ('entry_threshold', 'volume_sma_length'),
        # ('entry_threshold', 'atr_multiplier'),
        # ('weight_trend', 'weight_momentum'),
        # ('weight_trend', 'weight_structure'),
        # ('weight_volume', 'weight_momentum'),
        ('use_macd', 'use_rsi'),
        ('use_rsi', 'use_volume'),
        ('use_volume', 'use_bollinger_bands'),
        ('use_bollinger_bands', 'use_macd'),
        ('use_market_structure', 'use_bollinger_bands'),
        ('use_market_structure', 'use_macd'),
        # ('weight_volatility', 'weight_structure'),
        # ('n_sma_fast', 'entry_threshold'),
        # ('rsi_oversold', 'entry_threshold')
    ]
    
    for param1, param2 in param_pairs:
        try:
            # Create pivot table for the parameter pair
            pivot = results_df.pivot_table(
                values=metric,
                index=param1,
                columns=param2,
                aggfunc='mean'
            )
            
            # Create figure
            p = figure(
                title=f'{metric}: {param1} vs {param2}',
                x_axis_label=param2,
                y_axis_label=param1,
                width=500,
                height=400,
                tools='pan,box_zoom,reset,save,hover'
            )
            
            # Create color mapper
            mapper = linear_cmap(
                'value',
                palette,
                low=results_df[metric].min(),
                high=results_df[metric].max()
            )
            
            # Create heatmap
            x_coords, y_coords = np.meshgrid(pivot.columns, pivot.index)
            source = ColumnDataSource(dict(
                x=x_coords.ravel(),
                y=y_coords.ravel(),
                value=pivot.values.ravel()
            ))
            
            # Add rectangles for heatmap
            p.rect(
                x='x', y='y',
                width=1, height=1,
                source=source,
                fill_color=mapper,
                line_color=None
            )
            
            # Add hover tool
            hover = HoverTool(tooltips=[
                (param2, '@x'),
                (param1, '@y'),
                (metric, '@value{0.00}')
            ])
            p.add_tools(hover)
            
            # Add colorbar
            color_bar = ColorBar(
                color_mapper=mapper.transform,
                title=metric,
                location=(0, 0)
            )
            p.add_layout(color_bar, 'right')
            
            plots.append(p)
        except:
            pass
    
    # Arrange plots in a grid
    grid = gridplot(plots, ncols=2)
    return grid

def analyze_backtest_results(stats):
    """Analyze and print detailed backtest results"""
    
    # 1. Overall Performance Metrics
    print("\n=== Overall Performance ===")
    print(f"Start                     {stats['Start']}")
    print(f"End                       {stats['End']}")
    print(f"Duration                  {stats['Duration']}")
    print(f"Exposure Time [%]         {stats['Exposure Time [%]']:.2f}")
    print(f"Equity Final [$]          {stats['Equity Final [$]']:.2f}")
    print(f"Equity Peak [$]           {stats['Equity Peak [$]']:.2f}")
    print(f"Return [%]                {stats['Return [%]']:.2f}")
    print(f"Buy & Hold Return [%]     {stats['Buy & Hold Return [%]']:.2f}")
    print(f"Return (Ann.) [%]         {stats['Return (Ann.) [%]']:.2f}")
    
    # 2. Risk Metrics
    print("\n=== Risk Metrics ===")
    print(f"Volatility (Ann.) [%]     {stats['Volatility (Ann.) [%]']:.2f}")
    print(f"Sharpe Ratio              {stats['Sharpe Ratio']:.2f}")
    print(f"Sortino Ratio             {stats['Sortino Ratio']:.2f}")
    print(f"Calmar Ratio              {stats['Calmar Ratio']:.2f}")
    print(f"Max. Drawdown [%]         {stats['Max. Drawdown [%]']:.2f}")
    print(f"Avg. Drawdown [%]         {stats['Avg. Drawdown [%]']:.2f}")
    print(f"Max. Drawdown Duration    {stats['Max. Drawdown Duration']}")
    print(f"Avg. Drawdown Duration    {stats['Avg. Drawdown Duration']}")
    
    # 3. Trade Analysis
    print("\n=== Trade Analysis ===")
    print(f"# Trades                  {stats['# Trades']}")
    print(f"Win Rate [%]              {stats['Win Rate [%]']:.2f}")
    print(f"Best Trade [%]            {stats['Best Trade [%]']:.2f}")
    print(f"Worst Trade [%]           {stats['Worst Trade [%]']:.2f}")
    print(f"Avg. Trade [%]            {stats['Avg. Trade [%]']:.2f}")
    print(f"Max. Trade Duration       {stats['Max. Trade Duration']}")
    print(f"Avg. Trade Duration       {stats['Avg. Trade Duration']}")
    print(f"Profit Factor             {stats['Profit Factor']:.2f}")
    print(f"Expectancy [%]            {stats['Expectancy [%]']:.2f}")
    print(f"SQN                       {stats['SQN']:.2f}")
    
    # 4. Detailed Trade Analysis
    trades_df = stats._trades
    print("\n=== Detailed Trades ===")
    print("\nTrade Summary:")
    print(trades_df.describe())
    
    # 5. Position Analysis
    print("\n=== Position Analysis ===")
    print("\nLong Trades:")
    long_trades = trades_df[trades_df['Size'] > 0]
    print(f"Count: {len(long_trades)}")
    print(f"Win Rate: {(long_trades['PnL'] > 0).mean()*100:.2f}%")
    print(f"Average Return: {long_trades['ReturnPct'].mean()*100:.2f}%")
    
    print("\nShort Trades:")
    short_trades = trades_df[trades_df['Size'] < 0]
    print(f"Count: {len(short_trades)}")
    print(f"Win Rate: {(short_trades['PnL'] > 0).mean()*100:.2f}%")
    print(f"Average Return: {short_trades['ReturnPct'].mean()*100:.2f}%")
    
    # 6. Time Analysis
    if isinstance(trades_df.index, pd.DatetimeIndex):
        trades_df['Hour'] = trades_df.index.hour
        print("\n=== Time Analysis ===")
        print("\nTrades by Hour:")
        print(trades_df.groupby('Hour')['ReturnPct'].agg(['count', 'mean', 'std']))
    
    return {
        'overall_metrics': {
            'total_return': stats['Return [%]'],
            'annual_return': stats['Return (Ann.) [%]'],
            'sharpe_ratio': stats['Sharpe Ratio'],
            'max_drawdown': stats['Max. Drawdown [%]'],
            'win_rate': stats['Win Rate [%]'],
        },
        'trades': trades_df,
        'equity_curve': stats._equity_curve,
        'positions': stats.positions if hasattr(stats, 'positions') else None
    }

def main():
    # Fetch training data (first 3 quarters of 2024)
    train_data = fetch_data('2024-01-01', '2024-09-30')
    
    # Fetch test data (last quarter of 2024)
    test_data = fetch_data('2024-10-01', '2024-12-31')
    
    # Define optimization parameters
    opt_params = {
        'volume_sma_length': range(3, 40, 5),

        # Basic trend parameters
        'n_sma_fast': range(3, 30, 5),
        'n_sma_slow': range(10, 50, 5),

        # # RSI parameters
        # 'rsi_oversold': range(30, 45, 5),
        # 'rsi_overbought': range(55, 70, 5),
        # 'atr_multiplier': [1.5, 2.0, 2.5],
        ####

                
        # # RSI parameters
        'rsi_period': range(10, 21, 2),
        'rsi_oversold': range(20, 40, 5),
        'rsi_overbought': range(60, 80, 5),
        
        # ATR and volatility parameters
        'atr_period': range(3, 35, 2),
        'atr_multiplier': [1.5, 2.0, 2.5, 3.0],
        'tp_atr_multiplier': [1.5, 2.0, 2.5],
        
        # # Volume parameters

        
        # 'entry_threshold': [0.2, 0.4, 0.5, 0.6, 0.9],
        # # Feature toggles
        # 'use_volume_profile': [True, False],
        # 'use_market_structure': [True, False],
        # 'use_bollinger_bands': [True, False],
        # 'use_macd': [True, False],
        # 'use_rsi': [True, False],
        # 'use_volume': [True, False],
        
        # # Weights
        # 'weight_trend': [1.0, 1.5, 2.0],
        # 'weight_momentum': [0.5, 1.0, 1.5],
        # 'weight_volatility': [0.5, 1.0, 1.5],
        # 'weight_volume': [0.5, 1.0, 1.5],
        # 'weight_structure': [1.0, 1.5, 2.0],


        # #NEW

        'volume_surge_threshold': [1.1, 1.2, 1.3, 1.5, 2.0, 2.5],

        # 'volume_profile_period': range(15, 30, 5),
        # 'vwap_deviation': [1.0, 1.5, 2.0],
        
        # Entry threshold
        'entry_threshold': [0.2, 0.3, 0.4, 0.5, 0.6, 0.9],
        
        # Market structure parameters
        # 'swing_window': range(3, 8, 1),
        # 'trend_window': range(15, 30, 5),
        
        # Bollinger Bands parameters
        'bb_window': range(15, 30, 5),
        'bb_dev': [1.5, 2.0, 2.5],
        
        # MACD parameters
        'macd_fast': range(8, 16, 2),
        'macd_slow': range(20, 32, 3),
        'macd_signal_length': range(7, 12, 1),
        
        # Feature toggles
        'use_volume_profile': [ False],
        'use_market_structure': [False],
        'use_bollinger_bands': [True, False],
        'use_macd': [True, False],
        'use_rsi': [True, False],
        'use_volume': [True, False],
        
        # Scoring weights
        'weight_trend': [0.5,1.0, 1.5, 2.0, 2.5],
        'weight_momentum': [0.5, 1.0, 1.5, 2.5],
        'weight_volatility': [0.5, 1.0, 1.5, 2.5],
        'weight_volume': [0.5, 1.0, 1.5, 2.5],
        'weight_structure': [0.5, 1.0, 1.5, 2.0, 2.5],
        
        'use_max_position_duration': [True, False],
        'max_hold_period': [6, 12, 24, 48, 72, 96],  # Test different durations in hours
        
        'use_breakeven_stop': [True, False],
        'breakeven_threshold_atr': [0.5, 1.0, 1.5, 2.0],  # Move to breakeven after this many ATRs
        'breakeven_buffer_atr': [0.1, 0.2, 0.3, 0.5],  # Buffer size in ATR
        

        # Risk management
        'risk_per_trade': [0.01, 0.02, 0.03, 0.05],
        #####

        'use_multiple_tps': [True, False],
        'tp1_atr_multiplier': [1.0, 1.5, 2.0, 2.5],
        'tp2_atr_multiplier': [2.0, 2.5, 3.0, 3.5],
        'tp3_atr_multiplier': [3.0, 3.5, 4.0, 4.5],
        'tp_position_size_1': [0.3, 0.4, 0.5],  # First portion size
        'tp_position_size_2': [0.2, 0.3, 0.4],  # Second portion size
        'tp_position_size_3': [0.2, 0.3, 0.4],  # Third portion size
        
        # # Enhanced Support/Resistance
        # 'use_enhanced_sr': [True, False],
        # 'sr_lookback': range(50, 150, 25),
        # 'sr_min_touches': [2, 3, 4],
        # 'sr_price_threshold': [0.001, 0.002, 0.003],
        # 'sr_score_threshold': [0.6, 0.7, 0.8],

        # # Market Regime Detection
        # 'use_regime_detection': [True, False],
        # 'regime_lookback': range(30, 70, 10),
        # 'trend_strength_threshold': [0.5, 0.6, 0.7],
        # 'volatility_regime_threshold': [1.3, 1.5, 1.7],
        
        # # Order Flow Analysis
        # 'use_order_flow': [True, False],
        # 'flow_window': range(10, 30, 5),
        # 'flow_threshold': [0.5, 0.6, 0.7],
        # 'flow_volume_factor': [1.3, 1.5, 1.7],

        # # Smart Exit Management
        # 'use_smart_exits': [True, False],
        # 'trailing_stop_atr': [1.5, 2.0, 2.5],
        # 'trailing_stop_activation': [1.3, 1.5, 1.7],
        # 'time_based_exit_hours': [24, 48, 72],
        # 'profit_lock_threshold': [0.3, 0.5, 0.7],
    }


    # Create and run backtest on training data using the optimized strategy
    bt = Backtest(train_data, TradingStrategy, cash=100000, commission=.002)
    
    # Optimize strategy
    stats, heatmap = bt.optimize(
        **opt_params,
        maximize='Return (Ann.) [%]',
        #constraint=lambda param: param.n_sma_fast < param.n_sma_slow, #param.tp_position_size_1 + param.tp_position_size_2 + param.tp_position_size_3 <= 1,
        return_heatmap=True,
        method='skopt',
        max_tries=500
    )
    
    # Convert optimization results to DataFrame
    results_df = pd.DataFrame(heatmap)

    print(stats)

    # Analyze optimization results
    print("\n=== Optimization Results ===")
    optimization_results = analyze_backtest_results(stats)
    
    # Run test set with optimized parameters
    best_params = {
        # Basic trend parameters
        'n_sma_fast': stats._strategy.n_sma_fast,
        'n_sma_slow': stats._strategy.n_sma_slow,
        
        # RSI parameters
        'rsi_period': stats._strategy.rsi_period,
        'rsi_oversold': stats._strategy.rsi_oversold,
        'rsi_overbought': stats._strategy.rsi_overbought,
        
        # ATR and volatility parameters
        'atr_period': stats._strategy.atr_period,
        'atr_multiplier': stats._strategy.atr_multiplier,
        'tp_atr_multiplier': stats._strategy.tp_atr_multiplier,
        
        # Volume parameters
        'volume_sma_length': stats._strategy.volume_sma_length,
        'volume_surge_threshold': stats._strategy.volume_surge_threshold,
        'volume_profile_period': stats._strategy.volume_profile_period,
        'vwap_deviation': stats._strategy.vwap_deviation,
        
        # Entry threshold
        'entry_threshold': stats._strategy.entry_threshold,
        
        # Market structure parameters
        'swing_window': stats._strategy.swing_window,
        'trend_window': stats._strategy.trend_window,
        
        # Bollinger Bands parameters
        'bb_window': stats._strategy.bb_window,
        'bb_dev': stats._strategy.bb_dev,
        
        # MACD parameters
        'macd_fast': stats._strategy.macd_fast,
        'macd_slow': stats._strategy.macd_slow,
        'macd_signal_length': stats._strategy.macd_signal_length,
        
        # Feature toggles
        'use_volume_profile': stats._strategy.use_volume_profile,
        'use_market_structure': stats._strategy.use_market_structure,
        'use_bollinger_bands': stats._strategy.use_bollinger_bands,
        'use_macd': stats._strategy.use_macd,
        'use_rsi': stats._strategy.use_rsi,
        'use_volume': stats._strategy.use_volume,
        
        # Scoring weights
        'weight_trend': stats._strategy.weight_trend,
        'weight_momentum': stats._strategy.weight_momentum,
        'weight_volatility': stats._strategy.weight_volatility,
        'weight_volume': stats._strategy.weight_volume,
        'weight_structure': stats._strategy.weight_structure,
        
        # Risk management
        'risk_per_trade': stats._strategy.risk_per_trade,

        'use_max_position_duration': stats._strategy.use_max_position_duration,
        'max_hold_period': stats._strategy.max_hold_period,

        'use_breakeven_stop': stats._strategy.use_breakeven_stop,
        'breakeven_threshold_atr': stats._strategy.breakeven_threshold_atr,
        'breakeven_buffer_atr': stats._strategy.breakeven_buffer_atr,

        # Multiple Take Profit Levels
        # 'use_multiple_tps': stats._strategy.use_multiple_tps,
        # 'tp1_atr_multiplier': stats._strategy.tp1_atr_multiplier,
        # 'tp2_atr_multiplier': stats._strategy.tp2_atr_multiplier,
        # 'tp3_atr_multiplier': stats._strategy.tp3_atr_multiplier,
        # 'tp_position_size_1': stats._strategy.tp_position_size_1,
        # 'tp_position_size_2': stats._strategy.tp_position_size_2,
        # 'tp_position_size_3': stats._strategy.tp_position_size_3,

        # # Enhanced Support/Resistance
        # 'use_enhanced_sr': stats._strategy.use_enhanced_sr,
        # 'sr_lookback': stats._strategy.sr_lookback,
        # 'sr_min_touches': stats._strategy.sr_min_touches,
        # 'sr_price_threshold': stats._strategy.sr_price_threshold,
        # 'sr_score_threshold': stats._strategy.sr_score_threshold,

        # # Market Regime Detection
        # 'use_regime_detection': stats._strategy.use_regime_detection,
        # 'regime_lookback': stats._strategy.regime_lookback,
        # 'trend_strength_threshold': stats._strategy.trend_strength_threshold,
        # 'volatility_regime_threshold': stats._strategy.volatility_regime_threshold,

        # # Order Flow Analysis
        # 'use_order_flow': stats._strategy.use_order_flow,
        # 'flow_window': stats._strategy.flow_window,
        # 'flow_threshold': stats._strategy.flow_threshold,
        # 'flow_volume_factor': stats._strategy.flow_volume_factor,

        # # Smart Exit Management
        # 'use_smart_exits': stats._strategy.use_smart_exits,
        # 'trailing_stop_atr': stats._strategy.trailing_stop_atr,
        # 'trailing_stop_activation': stats._strategy.trailing_stop_activation,
        # 'time_based_exit_hours': stats._strategy.time_based_exit_hours,
        # 'profit_lock_threshold': stats._strategy.profit_lock_threshold,
        
        
    }
    
    print("\nBest Parameters:")
    print(best_params)
    
    # Test optimized strategy
    print("\nTesting optimized strategy...")
    bt_test = Backtest(test_data, TradingStrategy, cash=100000, commission=.002)
    stats_test = bt_test.run(**best_params)
    test_results = analyze_backtest_results(stats_test)
    
    print("\nTest Results:")
    print(stats_test)

    # Save results to file
    results = {
        'optimization': optimization_results,
        'test': test_results,
        'best_parameters': best_params,
        'heatmap': heatmap
    }
    
    
    # Create plots
    backtest_plot = plot_results(stats_test)
    optimization_plot = plot_optimization_grid(
        results_df,
        param_names=opt_params.keys(),
        metric='Return (Ann.) [%]'
    )

    # Save results to file
    with open('backtest_results.txt', 'w') as f:
        f.write("=== Optimization Results ===\n")
        f.write(str(optimization_results['overall_metrics']))
        f.write("\n\nBest Parameters:\n")
        f.write(str(best_params))
        f.write("\n\n=== Test Results ===\n")
        f.write(str(test_results['overall_metrics']))

        f.write("\n\n=== Test Results ===\n")
        f.write(str(results))
    
    # Combine plots and save
    final_layout = column(backtest_plot, optimization_plot)
    save(final_layout)

if __name__ == "__main__":
    main() 
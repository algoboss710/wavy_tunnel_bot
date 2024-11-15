import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import sys
from pathlib import Path
import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from itertools import product
import time
import json
from typing import Dict, List, Tuple, Optional, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Emergency stop file path
STOP_FILE = "stop_optimization.txt"

# Global configurations
SYMBOL = "XAUUSD"
OPTIMIZATION_TIMEFRAMES = ["M5", "M15", "M30", "H1", "H4", "D1"]

# Risk management configurations
RISK_CONFIG = {
    'max_position_size': 0.02,  # Maximum position size as percentage of account
    'max_daily_risk': 0.05,    # Maximum daily risk as percentage of account
    'max_correlation_risk': 0.7, # Maximum correlation between positions
    'position_scaling': {
        'excellent_condition': 1.0,  # Full position size
        'good_condition': 0.75,     # 75% of normal position size
        'moderate_condition': 0.5,   # 50% of normal position size
        'poor_condition': 0.25      # 25% of normal position size
    },
    'market_conditions': {
        'volatility_threshold': 1.5,  # Volatility multiplier threshold
        'trend_strength_threshold': 0.5, # Minimum trend strength required
        'volume_threshold': 1.2,     # Minimum volume multiplier
        'correlation_threshold': 0.7  # Maximum correlation allowed
    }
}

# Optimized entry parameters for XAUUSD
TIMEFRAME_ENTRY_PARAMS = {
    "M5": {
        'wavy_period': 25,
        'tunnel_period1': 280,
        'tunnel_period2': 300,
        'min_gap_second': 5.0,
        'max_zone_percentage': 0.45000000000000007
    },
    "M15": {
        'wavy_period': 30,
        'tunnel_period1': 180,
        'tunnel_period2': 320,
        'min_gap_second': 5.0,
        'max_zone_percentage': 0.45000000000000007
    },
    "M30": {
        'wavy_period': 20,
        'tunnel_period1': 300,
        'tunnel_period2': 320,
        'min_gap_second': 5.0,
        'max_zone_percentage': 0.30000000000000004
    },
    "H1": {
        'wavy_period': 20,
        'tunnel_period1': 180,
        'tunnel_period2': 280,
        'min_gap_second': 5.0,
        'max_zone_percentage': 0.45000000000000007
    },
    "H4": {
        'wavy_period': 50,
        'tunnel_period1': 160,
        'tunnel_period2': 300,
        'min_gap_second': 5.0,
        'max_zone_percentage': 0.20000000000000004
    },
    "D1": {
        'wavy_period': 50,
        'tunnel_period1': 160,
        'tunnel_period2': 160,
        'min_gap_second': 5.0,
        'max_zone_percentage': 0.1
    }
}
def test_mt5_connection():
    """Test MT5 connection and basic functionality"""
    print("\nTesting MT5 Connection...")
    mt5.shutdown()

    if not mt5.initialize():
        print("Failed to initialize MT5")
        return False
        
    print("MT5 Package Version:", mt5.__version__)
    print("Terminal Info:", mt5.terminal_info())
    
    # Test symbol info
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is not None:
        print(f"Symbol: {SYMBOL}")
        print(f"Points: {symbol_info.point}")
        print(f"Digits: {symbol_info.digits}")
        print(f"Trade Mode: {symbol_info.trade_mode}")
    else:
        print(f"Failed to get symbol info for {SYMBOL}")
        return False
    
    account_info = mt5.account_info()
    if account_info is not None:
        print("Connected to account:", account_info.login)
        print("Server:", account_info.server)
    else:
        print("Failed to get account info")
        return False
    
    return True

class OptimizationLogger:
    """Enhanced logging system for optimization process"""
    def __init__(self, base_path: Path, symbol: str, timeframe: str):
        self.base_path = base_path
        self.symbol = symbol
        self.timeframe = timeframe
        self.log_path = base_path / f"{symbol}_{timeframe}"
        self.log_path.mkdir(parents=True, exist_ok=True)

        # Setup loggers
        self.setup_loggers()

    def setup_loggers(self):
        # Main optimization logger
        self.main_logger = self._setup_logger('main', 'optimization.log')
        # Progress logger
        self.progress_logger = self._setup_logger('progress', 'progress.log')
        # Results logger
        self.results_logger = self._setup_logger('results', 'results.log')
        # Debug logger
        self.debug_logger = self._setup_logger('debug', 'debug.log')
        # Trades logger
        self.trades_logger = self._setup_logger('trades', 'trades.log')

    def _setup_logger(self, name: str, filename: str) -> logging.Logger:
        logger = logging.getLogger(f"{self.symbol}_{self.timeframe}_{name}")
        logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(self.log_path / filename)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Clear existing handlers
        logger.handlers = []
        
        # Add handlers
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger

    def warning(self, message: str):
        """Add warning method to logger"""
        self.main_logger.warning(message)
        self.debug_logger.warning(message)

    def log_params(self, params: Dict):
        self.main_logger.info(f"Starting optimization with parameters: {params}")

    def log_progress(self, current: int, total: int, elapsed_time: float, remaining_time: float,
                    best_result: Optional[Dict] = None):
        progress = (current / total) * 100
        msg = (f"Progress: {current}/{total} ({progress:.2f}%) - "
               f"Elapsed: {elapsed_time:.2f}s - "
               f"Remaining: {remaining_time:.2f}s")
        
        if best_result:
            msg += (f"\nBest so far - "
                   f"Trades: {best_result.get('total_trades', 0)}, "
                   f"Win Rate: {best_result.get('win_rate', 0)*100:.2f}%, "
                   f"Profit: {best_result.get('avg_profit', 0)*100:.2f}%")
        
        self.progress_logger.info(msg)

    def log_debug(self, message: str):
        self.debug_logger.debug(message)

    def log_results(self, results: Dict):
        self.results_logger.info(f"Optimization Results:\n{results}")

    def log_trade(self, trade_info: Dict):
        self.trades_logger.info(
            f"Trade: {trade_info.get('position', 'unknown')} - "
            f"Entry: {trade_info.get('entry_price', 0):.2f} - "
            f"Exit: {trade_info.get('exit_price', 0):.2f} - "
            f"Profit: {trade_info.get('profit', 0)*100:.2f}% - "
            f"Type: {trade_info.get('exit_reason', 'unknown')}"
        )

    def log_error(self, error_message: str, exception: Exception = None):
        error_log = f"Error: {error_message}"
        if exception:
            error_log += f"\nException: {str(exception)}"
        self.main_logger.error(error_log)
        self.debug_logger.error(error_log)

    def log_warning(self, message: str):
        self.main_logger.warning(message)
        self.debug_logger.warning(message)

    def log_optimization_complete(self, results: Dict):
        self.main_logger.info("Optimization completed")
        self.main_logger.info(f"Final Results:\n{results}")

class MarketAnalyzer:
    """Market analysis and condition detection"""
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        self.market_states = []
        self.current_regime = None
        
    def analyze_market_condition(self, data: pd.DataFrame) -> Dict:
        """Analyze current market conditions"""
        try:
            # Calculate volatility metrics
            returns = data['close'].pct_change()
            volatility = returns.rolling(20).std()
            avg_volatility = volatility.rolling(50).mean()
            
            # Calculate trend metrics
            sma_short = data['close'].rolling(20).mean()
            sma_long = data['close'].rolling(50).mean()
            trend_strength = ((sma_short - sma_long) / sma_long * 100)
            
            # Calculate volume metrics
            volume_sma = data['tick_volume'].rolling(20).mean()
            relative_volume = data['tick_volume'] / volume_sma
            
            # Determine market regime
            regime = self._determine_regime(volatility.iloc[-1], 
                                         avg_volatility.iloc[-1],
                                         trend_strength.iloc[-1],
                                         relative_volume.iloc[-1])
            
            return {
                'regime': regime,
                'volatility': volatility.iloc[-1],
                'avg_volatility': avg_volatility.iloc[-1],
                'trend_strength': trend_strength.iloc[-1],
                'relative_volume': relative_volume.iloc[-1]
            }
            
        except Exception as e:
            logging.error(f"Error in market analysis: {str(e)}")
            return None
            
    def _determine_regime(self, 
                         current_vol: float, 
                         avg_vol: float,
                         trend_str: float,
                         rel_volume: float) -> str:
        """Determine current market regime"""
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        
        if vol_ratio > 2.0:
            regime = 'high_volatility'
        elif vol_ratio < 0.5:
            regime = 'low_volatility'
        elif abs(trend_str) > 0.5 and rel_volume > 1.2:
            regime = 'trending'
        else:
            regime = 'ranging'
            
        self.current_regime = regime
        return regime
        
    def should_trade(self, market_condition: Dict) -> bool:
        """Determine if market conditions are suitable for trading"""
        if market_condition['regime'] == 'high_volatility':
            return False
            
        if market_condition['volatility'] > market_condition['avg_volatility'] * 2:
            return False
            
        if abs(market_condition['trend_strength']) < 0.2:
            return False
            
        return True


class PositionSizer:
    """Position sizing and risk management class"""
    def __init__(self, account_info: dict, risk_config: dict):
        self.account_info = account_info
        self.risk_config = risk_config
        self.daily_risk_used = 0
        self.open_positions = []
        self.market_analyzer = MarketAnalyzer()
        self.peak_balance = account_info['balance']
        
    def calculate_position_size(self, 
                              signal_type: str, 
                              entry_price: float, 
                              stop_loss: float,
                              market_condition: str,
                              volatility: float,
                              trend_strength: float,
                              trade_history: List[Dict] = None) -> float:
        """Calculate optimal position size based on multiple factors"""
        try:
            # Base position size calculation
            risk_amount = self.account_info['balance'] * self.risk_config['max_position_size']
            risk_per_pip = abs(entry_price - stop_loss)
            base_position_size = risk_amount / risk_per_pip
            
            # Recent performance adjustment
            if trade_history:
                recent_performance = self._calculate_recent_performance(trade_history)
                performance_multiplier = min(1.0, max(0.5, recent_performance))
            else:
                performance_multiplier = 1.0
            
            # Adjust for market conditions
            condition_multiplier = self.risk_config['position_scaling'][market_condition]
            
            # Adjust for signal type
            signal_multiplier = 1.0 if signal_type == 'primary' else 0.7
            
            # Adjust for volatility
            volatility_multiplier = min(1.0, 
                                     self.risk_config['market_conditions']['volatility_threshold'] / volatility)
            
            # Adjust for trend strength
            trend_multiplier = min(1.0, 
                                 abs(trend_strength) / self.risk_config['market_conditions']['trend_strength_threshold'])
            
            # Calculate drawdown adjustment
            current_drawdown = self._calculate_drawdown()
            drawdown_multiplier = max(0.5, 1 - current_drawdown)
            
            # Calculate final position size
            final_position_size = (base_position_size * 
                                 condition_multiplier * 
                                 signal_multiplier * 
                                 volatility_multiplier * 
                                 trend_multiplier *
                                 performance_multiplier *
                                 drawdown_multiplier)
            
            # Apply daily risk limit
            remaining_daily_risk = (self.risk_config['max_daily_risk'] * 
                                  self.account_info['balance'] - 
                                  self.daily_risk_used)
            
            max_position_risk = remaining_daily_risk / risk_per_pip
            final_position_size = min(final_position_size, max_position_risk)
            
            # Additional safety checks
            final_position_size = self._apply_safety_limits(final_position_size, entry_price)
            
            return final_position_size
            
        except Exception as e:
            logging.error(f"Error calculating position size: {str(e)}")
            return 0.0
            
    def _calculate_recent_performance(self, trade_history: List[Dict], lookback: int = 10) -> float:
        """Calculate recent trading performance"""
        if not trade_history:
            return 1.0
            
        recent_trades = trade_history[-lookback:]
        if not recent_trades:
            return 1.0
            
        wins = sum(1 for t in recent_trades if t['profit'] > 0)
        win_rate = wins / len(recent_trades)
        
        return min(1.0, max(0.5, win_rate))
        
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown"""
        current_equity = self.account_info['equity']
        self.peak_balance = max(self.peak_balance, current_equity)
        
        if self.peak_balance == 0:
            return 0.0
            
        drawdown = (self.peak_balance - current_equity) / self.peak_balance
        return drawdown
        
    def _apply_safety_limits(self, position_size: float, entry_price: float) -> float:
        """Apply additional safety limits to position size"""
        # Maximum position value limit
        max_position_value = self.account_info['balance'] * 0.1  # Max 10% of account in single trade
        max_size_by_value = max_position_value / entry_price
        position_size = min(position_size, max_size_by_value)
        
        # Ensure minimum free margin
        margin_required = self._estimate_margin_required(position_size, entry_price)
        if margin_required > self.account_info['margin_free'] * 0.8:  # Keep 20% margin buffer
            position_size *= 0.8 * self.account_info['margin_free'] / margin_required
            
        return position_size
        
    def _estimate_margin_required(self, position_size: float, price: float) -> float:
        """Estimate required margin for position"""
        leverage = 100  # Example leverage 1:100
        return (position_size * price) / leverage
        
    def update_daily_risk(self, risk_amount: float):
        """Update daily risk tracker"""
        self.daily_risk_used += risk_amount
        
    def reset_daily_risk(self):
        """Reset daily risk tracker"""
        self.daily_risk_used = 0.0
        
    def check_correlation_risk(self, new_position: dict) -> bool:
        """Check if new position violates correlation risk limits"""
        for position in self.open_positions:
            correlation = self._calculate_correlation(position, new_position)
            if correlation > self.risk_config['max_correlation_risk']:
                return False
        return True
        
    def _calculate_correlation(self, pos1: dict, pos2: dict) -> float:
        """Calculate correlation between two positions"""
        if pos1['position'] == pos2['position']:  # Same direction
            return 1.0
        elif pos1['position'] != pos2['position']:  # Opposite direction
            return -1.0
        return 0.0
        
    def get_position_status(self) -> Dict:
        """Get current position status and risk metrics"""
        return {
            'daily_risk_used': self.daily_risk_used,
            'daily_risk_remaining': (self.risk_config['max_daily_risk'] * 
                                   self.account_info['balance'] - 
                                   self.daily_risk_used),
            'open_positions': len(self.open_positions),
            'current_drawdown': self._calculate_drawdown()
        }
class RiskManager:
    """Advanced risk management system"""
    def __init__(self, risk_config: dict):
        self.risk_config = risk_config
        self.daily_stats = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'profit': 0.0,
            'max_drawdown': 0.0,
            'consecutive_losses': 0,
            'largest_loss': 0.0,
            'peak_balance': 0.0,
            'current_drawdown': 0.0
        }
        self.position_sizer = None
        self.market_conditions = {}
        self.trade_history = []
        self.risk_alerts = []
        self.stop_trading = False
        
    def initialize_position_sizer(self, account_info: dict):
        """Initialize position sizer with account information"""
        self.position_sizer = PositionSizer(account_info, self.risk_config)
        self.daily_stats['peak_balance'] = account_info['balance']
    
    def evaluate_market_condition(self, 
                                data: pd.DataFrame, 
                                current_idx: int) -> Tuple[str, dict]:
        """Evaluate current market conditions with enhanced metrics"""
        try:
            # Calculate volatility metrics
            volatility = self._calculate_volatility(data, current_idx)
            avg_volatility = volatility.rolling(50).mean().iloc[current_idx]
            rel_volatility = volatility.iloc[current_idx] / avg_volatility
            
            # Calculate trend metrics
            trend_strength = self._calculate_trend_strength(data, current_idx)
            
            # Calculate volume metrics
            volume_ratio = self._calculate_volume_ratio(data, current_idx)
            
            # Calculate momentum
            momentum = self._calculate_momentum(data, current_idx)
            
            # Determine market condition
            condition = self._determine_market_condition(
                rel_volatility,
                trend_strength,
                volume_ratio,
                momentum
            )
            
            metrics = {
                'volatility': volatility.iloc[current_idx],
                'avg_volatility': avg_volatility,
                'trend_strength': trend_strength,
                'volume_ratio': volume_ratio,
                'momentum': momentum,
                'rel_volatility': rel_volatility
            }
            
            # Update market conditions history
            self.market_conditions[data.index[current_idx]] = metrics
            
            return condition, metrics
            
        except Exception as e:
            logging.error(f"Error evaluating market condition: {str(e)}")
            return 'poor_condition', {}
            
    def _calculate_volatility(self, data: pd.DataFrame, idx: int) -> pd.Series:
        """Calculate enhanced volatility metrics"""
        returns = data['close'].pct_change()
        volatility = returns.rolling(20).std()
        return volatility
        
    def _calculate_trend_strength(self, data: pd.DataFrame, idx: int) -> float:
        """Calculate trend strength using multiple indicators"""
        # EMA-based trend
        short_ema = data['close'].ewm(span=20).mean()
        long_ema = data['close'].ewm(span=50).mean()
        ema_trend = ((short_ema - long_ema) / long_ema * 100).iloc[idx]
        
        # ADX-based trend strength
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        dx = (atr / data['close'] * 100).iloc[idx]
        
        # Combine metrics
        trend_strength = (ema_trend * 0.7 + dx * 0.3)
        return trend_strength
        
    def _calculate_volume_ratio(self, data: pd.DataFrame, idx: int) -> float:
        """Calculate volume strength ratio"""
        volume_sma = data['tick_volume'].rolling(20).mean()
        current_volume = data['tick_volume'].iloc[idx]
        return current_volume / volume_sma.iloc[idx] if volume_sma.iloc[idx] > 0 else 1.0
        
    def _calculate_momentum(self, data: pd.DataFrame, idx: int) -> float:
        """Calculate price momentum"""
        returns = data['close'].pct_change()
        momentum = returns.rolling(10).mean().iloc[idx]
        return momentum
        
    def _determine_market_condition(self,
                                  rel_volatility: float,
                                  trend_strength: float,
                                  volume_ratio: float,
                                  momentum: float) -> str:
        """Determine market condition based on multiple factors"""
        if rel_volatility > self.risk_config['market_conditions']['volatility_threshold']:
            return 'poor_condition'
            
        if (abs(trend_strength) > self.risk_config['market_conditions']['trend_strength_threshold'] and
            volume_ratio > self.risk_config['market_conditions']['volume_threshold'] and
            abs(momentum) > 0.001):
            return 'excellent_condition'
            
        if (abs(trend_strength) > self.risk_config['market_conditions']['trend_strength_threshold'] * 0.7 and
            volume_ratio > 1.0):
            return 'good_condition'
            
        if rel_volatility < 0.8:
            return 'moderate_condition'
            
        return 'poor_condition'
        
    def validate_trade(self, 
                      signal_type: str, 
                      entry_price: float, 
                      stop_loss: float,
                      market_metrics: dict) -> Tuple[bool, float]:
        """Validate trade and calculate position size with enhanced checks"""
        try:
            # Check if trading should be stopped
            if self.stop_trading:
                return False, 0.0
                
            # Check if daily risk limit is exceeded
            if self.position_sizer.daily_risk_used >= (self.risk_config['max_daily_risk'] * 
                                                      self.position_sizer.account_info['balance']):
                self.risk_alerts.append("Daily risk limit exceeded")
                return False, 0.0
            
            # Check market conditions
            condition = self._determine_market_condition(
                market_metrics['rel_volatility'],
                market_metrics['trend_strength'],
                market_metrics['volume_ratio'],
                market_metrics['momentum']
            )
            
            # Validate signal type and market condition combination
            if condition == 'poor_condition' and signal_type == 'secondary':
                self.risk_alerts.append("Poor market condition for secondary signal")
                return False, 0.0
            
            # Calculate position size
            position_size = self.position_sizer.calculate_position_size(
                signal_type=signal_type,
                entry_price=entry_price,
                stop_loss=stop_loss,
                market_condition=condition,
                volatility=market_metrics['volatility'],
                trend_strength=market_metrics['trend_strength'],
                trade_history=self.trade_history
            )
            
            if position_size <= 0:
                self.risk_alerts.append("Invalid position size calculated")
                return False, 0.0
            
            # Perform correlation check
            new_position = {
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'position_size': position_size,
                'signal_type': signal_type
            }
            
            if not self.position_sizer.check_correlation_risk(new_position):
                self.risk_alerts.append("Correlation risk limit exceeded")
                return False, 0.0
            
            return True, position_size
            
        except Exception as e:
            logging.error(f"Error validating trade: {str(e)}")
            return False, 0.0
    
    def update_trade_stats(self, trade_result: dict):
        """Update trading statistics with enhanced metrics"""
        self.trade_history.append(trade_result)
        self.daily_stats['trades'] += 1
        
        if trade_result['profit'] > 0:
            self.daily_stats['wins'] += 1
            self.daily_stats['consecutive_losses'] = 0
        else:
            self.daily_stats['losses'] += 1
            self.daily_stats['consecutive_losses'] += 1
            self.daily_stats['largest_loss'] = min(
                self.daily_stats['largest_loss'],
                trade_result['profit']
            )
        
        self.daily_stats['profit'] += trade_result['profit']
        
        # Update drawdown statistics
        current_equity = (self.position_sizer.account_info['balance'] + 
                         self.daily_stats['profit'])
        self.daily_stats['peak_balance'] = max(
            self.daily_stats['peak_balance'],
            current_equity
        )
        current_drawdown = ((self.daily_stats['peak_balance'] - current_equity) / 
                           self.daily_stats['peak_balance'])
        self.daily_stats['max_drawdown'] = max(
            self.daily_stats['max_drawdown'],
            current_drawdown
        )
        self.daily_stats['current_drawdown'] = current_drawdown
    
    def should_stop_trading(self) -> bool:
        """Determine if trading should be stopped based on risk metrics"""
        # Check daily loss limit
        if self.daily_stats['profit'] <= -self.risk_config['max_daily_risk']:
            self.risk_alerts.append("Daily loss limit reached")
            self.stop_trading = True
            return True
        
        # Check consecutive losses
        if self.daily_stats['consecutive_losses'] >= 3:
            self.risk_alerts.append("Maximum consecutive losses reached")
            self.stop_trading = True
            return True
        
        # Check drawdown limit
        if self.daily_stats['current_drawdown'] >= 0.1:  # 10% drawdown limit
            self.risk_alerts.append("Maximum drawdown limit reached")
            self.stop_trading = True
            return True
        
        return False

    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics and statistics"""
        win_rate = (self.daily_stats['wins'] / self.daily_stats['trades'] 
                   if self.daily_stats['trades'] > 0 else 0.0)
        
        return {
            'daily_stats': self.daily_stats,
            'win_rate': win_rate,
            'risk_alerts': self.risk_alerts,
            'market_conditions': self.market_conditions,
            'position_status': self.position_sizer.get_position_status()
        }
class WavyTunnelExitOptimizer:
    def __init__(self, symbol: str, timeframe: str, start_date: datetime, 
                end_date: datetime, entry_params: Dict, 
                base_path: str = "exit_optimization_results"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.entry_params = entry_params
        self.base_path = Path(base_path) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.logger = OptimizationLogger(self.base_path, symbol, timeframe)
        self.market_analyzer = MarketAnalyzer()

        # Get symbol info and store necessary properties
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise ValueError(f"Could not get symbol info for {symbol}")
    
        # Store symbol properties
        self.point = symbol_info.point
        self.digits = symbol_info.digits
        self.trade_mode = symbol_info.trade_mode

        # Get account info once and store it as class variable
        if not hasattr(WavyTunnelExitOptimizer, '_account_info'):
            account_info = mt5.account_info()
            if account_info is not None:
                WavyTunnelExitOptimizer._account_info = {
                    'balance': account_info.balance,
                    'equity': account_info.equity,
                    'margin_free': account_info.margin_free
                }
            else:
                WavyTunnelExitOptimizer._account_info = {
                    'balance': 10000.0,
                    'equity': 10000.0,
                    'margin_free': 10000.0
                }
                self.logger.warning("Using default account values as MT5 account info unavailable")
    
        self.account_info = WavyTunnelExitOptimizer._account_info
        self.risk_manager = RiskManager(RISK_CONFIG)
        self.risk_manager.initialize_position_sizer(self.account_info)

        # Parameter ranges for optimization
        self.param_ranges = self._setup_param_ranges()
    
        # Configuration parameters
        self.min_signals_required = 50
        self.max_signal_ratio = 3.0
        self.min_secondary_winrate = 0.5
        self.regime_period = 50
    
    # Performance metrics
        self.performance_metrics = {
            'signal_balance': 0.0,
            'avg_profit_primary': 0.0,
            'avg_profit_secondary': 0.0,
            'risk_reward_ratio': 0.0
        }

        # Optimization state tracking
        self.optimization_results = []
        self.best_params = None
        self.best_performance = None
    
    def _setup_param_ranges(self) -> Dict:
        """Define parameter ranges with significantly reduced combinations"""
        return {
            # Take profit distribution parameters
            'tp1_lot_percent': [40, 50, 60],
            'tp2_lot_percent': [20, 25, 30],
            'tp3_lot_percent': [10, 15],
            'tp4_lot_percent': [10, 15],
            
            # Take profit distances calibrated for XAUUSD
            'tp1_weight': np.array([0.001, 0.002, 0.003]),
            'tp2_weight': np.array([0.004, 0.005, 0.006]),
            'tp3_weight': np.array([0.007, 0.008, 0.009]),
            'tp4_weight': np.array([0.010, 0.011, 0.012]),
            
            # Stop loss parameters
            'wave_cross_buffer': np.array([0.001, 0.002]),
            'tunnel_touch_buffer': np.array([0.001, 0.002])
        }
    
    def _get_default_results(self) -> Dict:
        """Return default results dictionary for error cases"""
        return {
        'total_trades': 0,
        'profitable_trades': 0,
        'total_profit_pct': 0.0,
        'avg_profit_per_trade_pct': 0.0,
        'max_drawdown': 1.0,
        'win_rate': 0.0,
        'profit_factor': 0.0,
        'sharpe_ratio': 0.0,
        'avg_hold_time': 0.0,
        'primary_profit_per_trade': 0.0,
        'secondary_profit_per_trade': 0.0
    }
    def _generate_param_combinations(self) -> List[Dict]:
        """Generate parameter combinations with validation"""
        param_keys = list(self.param_ranges.keys())
        param_values = list(self.param_ranges.values())
    
        combinations = []
        total_attempted = 0
        valid_count = 0
    
        try:
            for values in product(*param_values):
                total_attempted += 1
                params = dict(zip(param_keys, values))
            
                if self._validate_params(params):
                    combinations.append(params)
                    valid_count += 1
        
            self.logger.debug_logger.info(
                f"Parameter generation: {valid_count} valid combinations "
                f"from {total_attempted} attempted"
            )
        
            return combinations

        except Exception as e:
            self.logger.main_logger.error(f"Error generating parameter combinations: {str(e)}")
            return []
    
    def _evaluate_params(self, params: Dict, data: pd.DataFrame) -> Dict:
        """Evaluate parameters with enhanced performance metrics"""
        try:
            # Generate signals
            long_signals, short_signals, is_primary = self.generate_signals(data)
            trades = []
            current_position = None
            entry_price = 0
            entry_time = None
            remaining_position = 0
        
            for i in range(len(data)-1):
                # Skip invalid data
                if pd.isna(data['open'].iloc[i]) or data['open'].iloc[i] <= 0:
                    continue
                
                # Get market condition and metrics
                market_condition, metrics = self.risk_manager.evaluate_market_condition(
                    data, i
                )

                # Entry logic
                if current_position is None:
                    entry_valid = False
                    position_size = 0.0
                
                    if long_signals.iloc[i]:
                        signal_type = 'primary' if is_primary.iloc[i] else 'secondary'
                        stop_loss = data['open'].iloc[i] * (1 - params['wave_cross_buffer'])
                    
                        entry_valid, position_size = self.risk_manager.validate_trade(
                            signal_type=signal_type,
                            entry_price=data['open'].iloc[i],
                            stop_loss=stop_loss,
                            market_metrics=metrics
                        )
                    
                        if entry_valid:
                            current_position = 'long'
                            entry_price = data['open'].iloc[i]
                            entry_time = data.index[i]
                            remaining_position = position_size
                        
                    elif short_signals.iloc[i]:
                        signal_type = 'primary' if is_primary.iloc[i] else 'secondary'
                        stop_loss = data['open'].iloc[i] * (1 + params['wave_cross_buffer'])
                    
                        entry_valid, position_size = self.risk_manager.validate_trade(
                            signal_type=signal_type,
                            entry_price=data['open'].iloc[i],
                            stop_loss=stop_loss,
                            market_metrics=metrics
                        )
                    
                        if entry_valid:
                            current_position = 'short'
                            entry_price = data['open'].iloc[i]
                            entry_time = data.index[i]
                            remaining_position = position_size

                # Exit logic for open positions
                elif remaining_position > 0:
                    if current_position == 'long':
                        # Process take profits
                        for j, (lot_percent, weight) in enumerate(zip(
                            [params['tp1_lot_percent'], params['tp2_lot_percent'],
                             params['tp3_lot_percent'], params['tp4_lot_percent']],
                            [params['tp1_weight'], params['tp2_weight'],
                            params['tp3_weight'], params['tp4_weight']]
                        )):
                            if remaining_position >= lot_percent:
                                tp_level = entry_price * (1 + weight)
                                if data['high'].iloc[i] >= tp_level:
                                    trade_result = {
                                        'entry_price': entry_price,
                                        'exit_price': tp_level,
                                        'position_size': lot_percent,
                                        'profit': (tp_level - entry_price) / entry_price,
                                        'hold_time': (data.index[i] - entry_time).total_seconds() / 3600,
                                        'exit_reason': f'tp{j+1}',
                                        'is_primary': is_primary.iloc[i],
                                        'position': current_position
                                    }
                                    trades.append(trade_result)
                                    remaining_position -= lot_percent
                    
                        # Process stop loss
                        if remaining_position > 0:
                            sl_level = entry_price * (1 - params['wave_cross_buffer'])
                            if data['low'].iloc[i] <= sl_level:
                                trade_result = {
                                    'entry_price': entry_price,
                                    'exit_price': data['close'].iloc[i],
                                    'position_size': remaining_position,
                                    'profit': (data['close'].iloc[i] - entry_price) / entry_price,
                                    'hold_time': (data.index[i] - entry_time).total_seconds() / 3600,
                                    'exit_reason': 'sl',
                                    'is_primary': is_primary.iloc[i],
                                    'position': current_position
                                }
                                trades.append(trade_result)
                                remaining_position = 0
                
                    elif current_position == 'short':
                        # Similar logic for short positions...
                        for j, (lot_percent, weight) in enumerate(zip(
                            [params['tp1_lot_percent'], params['tp2_lot_percent'],
                            params['tp3_lot_percent'], params['tp4_lot_percent']],
                            [params['tp1_weight'], params['tp2_weight'],
                            params['tp3_weight'], params['tp4_weight']]
                        )):
                            if remaining_position >= lot_percent:
                                tp_level = entry_price * (1 - weight)
                                if data['low'].iloc[i] <= tp_level:
                                    trade_result = {
                                        'entry_price': entry_price,
                                        'exit_price': tp_level,
                                        'position_size': lot_percent,
                                        'profit': (entry_price - tp_level) / entry_price,
                                        'hold_time': (data.index[i] - entry_time).total_seconds() / 3600,
                                        'exit_reason': f'tp{j+1}',
                                        'is_primary': is_primary.iloc[i],
                                        'position': current_position
                                    }
                                    trades.append(trade_result)
                                    remaining_position -= lot_percent
                    
                        if remaining_position > 0:
                            sl_level = entry_price * (1 + params['wave_cross_buffer'])
                            if data['high'].iloc[i] >= sl_level:
                                trade_result = {
                                    'entry_price': entry_price,
                                    'exit_price': data['close'].iloc[i],
                                    'position_size': remaining_position,
                                    'profit': (entry_price - data['close'].iloc[i]) / entry_price,
                                    'hold_time': (data.index[i] - entry_time).total_seconds() / 3600,
                                    'exit_reason': 'sl',
                                    'is_primary': is_primary.iloc[i],
                                    'position': current_position
                                }
                                trades.append(trade_result)
                                remaining_position = 0
                
                    # Reset position if fully closed
                    if remaining_position == 0:
                        current_position = None
                        entry_price = 0
                        entry_time = None

        # Calculate performance metrics
            if trades:
                df_trades = pd.DataFrame(trades)
            
                return {
                    'total_trades': len(df_trades),
                    'profitable_trades': len(df_trades[df_trades['profit'] > 0]),
                    'total_profit_pct': df_trades['profit'].sum() * 100,
                    'avg_profit_per_trade_pct': df_trades['profit'].mean() * 100,
                    'max_drawdown': self._calculate_max_drawdown(df_trades['profit']),
                    'win_rate': len(df_trades[df_trades['profit'] > 0]) / len(df_trades),
                    'profit_factor': self._calculate_profit_factor(df_trades['profit']),
                    'sharpe_ratio': self._calculate_sharpe_ratio(df_trades['profit']),
                    'avg_hold_time': df_trades['hold_time'].mean(),
                    'primary_profit_per_trade': df_trades[df_trades['is_primary']]['profit'].mean() * 100,
                    'secondary_profit_per_trade': df_trades[~df_trades['is_primary']]['profit'].mean() * 100
                }
            return self._get_default_results()

        except Exception as e:
            self.logger.main_logger.error(f"Error evaluating parameters: {str(e)}")
            return self._get_default_results()
    
    def _get_market_data(self) -> Optional[pd.DataFrame]:
        """Fetch and preprocess market data with enhanced error handling"""
        try:
            if not mt5.initialize():
                self.logger.main_logger.error("Failed to initialize MT5")
                return None

            timeframe = getattr(mt5, f"TIMEFRAME_{self.timeframe}")
            rates = mt5.copy_rates_range(self.symbol, timeframe,
                                       self.start_date, self.end_date)
            
            if rates is None or len(rates) == 0:
                self.logger.main_logger.error(
                    f"No data available for {self.symbol} {self.timeframe}"
                )
                return None

            # Convert to DataFrame and handle timezone
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Basic data validation
            required_cols = ['open', 'high', 'low', 'close', 'tick_volume']
            if not all(col in df.columns for col in required_cols):
                self.logger.main_logger.error("Missing required columns in data")
                return None
            
            return self._preprocess_data(df)
            
        except Exception as e:
            self.logger.main_logger.error(f"Error fetching data: {str(e)}")
            return None
        finally:
            mt5.shutdown()

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess market data with improved cleaning"""
        try:
            # Remove any zero or negative prices
            for col in ['open', 'high', 'low', 'close']:
                data = data[data[col] > 0]

            # Handle missing values
            data = data.fillna(method='ffill')
            
            # Remove outliers using rolling median
            window = 5
            for col in ['high', 'low', 'close', 'open']:
                median = data[col].rolling(window=window, center=True).median()
                std = data[col].rolling(window=window, center=True).std()
                
                # More conservative outlier threshold for XAUUSD
                threshold = 2.5
                data[col] = data[col].where(
                    abs(data[col] - median) <= threshold * std,
                    median
                )
            
            # Validate high/low relationships
            data['high'] = data[['high', 'open', 'close']].max(axis=1)
            data['low'] = data[['low', 'open', 'close']].min(axis=1)
            
            self.logger.debug_logger.info(
                f"Preprocessed data shape: {data.shape}, "
                f"Date range: {data.index[0]} to {data.index[-1]}"
            )
            
            return data

        except Exception as e:
            self.logger.main_logger.error(f"Error preprocessing data: {str(e)}")
            return data
        
    def generate_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Generate entry signals using fixed parameters with improved validation"""
        try:
        # Calculate EMAs using entry parameters
            wavy_h = data['high'].ewm(span=self.entry_params['wavy_period'], adjust=False).mean()
            wavy_c = data['close'].ewm(span=self.entry_params['wavy_period'], adjust=False).mean()
            wavy_l = data['low'].ewm(span=self.entry_params['wavy_period'], adjust=False).mean()
            tunnel1 = data['close'].ewm(span=self.entry_params['tunnel_period1'], adjust=False).mean()
            tunnel2 = data['close'].ewm(span=self.entry_params['tunnel_period2'], adjust=False).mean()

            wavy_max = pd.concat([wavy_h, wavy_c, wavy_l], axis=1).max(axis=1)
            wavy_min = pd.concat([wavy_h, wavy_c, wavy_l], axis=1).min(axis=1)
            tunnel_max = pd.concat([tunnel1, tunnel2], axis=1).max(axis=1)
            tunnel_min = pd.concat([tunnel1, tunnel2], axis=1).min(axis=1)
           
            # Store for later use in exit conditions
            self.wavy_max = wavy_max
            self.wavy_min = wavy_min
            self.tunnel_max = tunnel_max
            self.tunnel_min = tunnel_min

            # Calculate trend and volume conditions
            trend = self._calculate_trend(data)
            volume_condition = data['tick_volume'] > data['tick_volume'].rolling(20).mean()

                    # Generate signals
            primary_longs = (
                (data['open'] > wavy_max) & 
                (wavy_min > tunnel_max) & 
                (trend > 0) &
                volume_condition &
                (data['open'] > 0)
            )
        
            primary_shorts = (
                (data['open'] < wavy_min) & 
                (wavy_max < tunnel_min) & 
                (trend < 0) &
                volume_condition &
                (data['open'] > 0)
            )

            # Secondary signals with additional validation
            min_gap = self.entry_params['min_gap_second']
            max_zone = self.entry_params['max_zone_percentage']
        
            secondary_longs = (
                (data['close'].shift(1) <= wavy_max.shift(1)) & 
                (data['close'] > wavy_max) &
                (data['close'] < tunnel_min) &
                ((tunnel_min - data['close']) > min_gap) &
                ((data['close'] - wavy_max) / (tunnel_min - wavy_max) <= max_zone) &
                (trend > 0) &
                volume_condition &
                (data['close'] > 0)
            )
        
            secondary_shorts = (
                (data['close'].shift(1) >= wavy_min.shift(1)) & 
                (data['close'] < wavy_min) &
                (data['close'] > tunnel_max) &
                ((data['close'] - tunnel_max) > min_gap) &
                ((wavy_min - data['close']) / (wavy_min - tunnel_max) <= max_zone) &
                (trend < 0) &
                volume_condition &
                (data['close'] > 0)
            )

        # Log signal generation stats (only once)
            if not hasattr(self, '_signals_logged'):
                self.logger.debug_logger.info(
                    f"Generated signals:\n"
                    f"Primary Longs: {primary_longs.sum()}\n"
                    f"Primary Shorts: {primary_shorts.sum()}\n"
                    f"Secondary Longs: {secondary_longs.sum()}\n"
                    f"Secondary Shorts: {secondary_shorts.sum()}"
                )
                self._signals_logged = True

            # Validate signal balance (only once per data set)
            long_signals = primary_longs | secondary_longs
            short_signals = primary_shorts | secondary_shorts
            signal_ratio = (long_signals.sum() + 1) / (short_signals.sum() + 1)
        
            if not hasattr(self, '_imbalance_warned') and (signal_ratio > self.max_signal_ratio or signal_ratio < (1/self.max_signal_ratio)):
                self.logger.warning(f"Signal imbalance detected: {signal_ratio:.2f} ratio")
                self._imbalance_warned = True

            return (
                long_signals,
                short_signals,
                pd.Series(primary_longs | primary_shorts, index=data.index)
            )

        except Exception as e:
            self.logger.main_logger.error(f"Error generating signals: {str(e)}")
            return (
                pd.Series(False, index=data.index),
                pd.Series(False, index=data.index),
                pd.Series(False, index=data.index)
            )

    def _calculate_trend(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trend direction and strength"""
        try:
            # Multiple timeframe trend calculation
            short_ma = data['close'].ewm(span=20).mean()
            medium_ma = data['close'].ewm(span=50).mean()
            long_ma = data['close'].ewm(span=100).mean()
            
            # Trend strength indicators
            short_trend = (short_ma - medium_ma) / medium_ma
            long_trend = (medium_ma - long_ma) / long_ma
            
            # Combine trends with more weight on shorter timeframe
            combined_trend = short_trend * 0.6 + long_trend * 0.4
            
            return combined_trend

        except Exception as e:
            self.logger.debug_logger.error(f"Error calculating trend: {str(e)}")
            return pd.Series(0, index=data.index)

    def test_data_and_signals(self, start_date: datetime, end_date: datetime) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Test function to verify data fetching and signal generation"""
        print("\nRunning diagnostic test...")
        
        # Test data fetching
        print("1. Testing MT5 data fetch...")
        data = self._get_market_data()
        if data is not None:
            print(f"✓ Successfully fetched {len(data)} bars of data")
            print(f"Date range: {data.index[0]} to {data.index[-1]}")
            print("\nSample data:")
            print(data.head())
            self.logger.debug_logger.info(f"Data fetch successful. Shape: {data.shape}")
        else:
            print("✗ Failed to fetch data")
            self.logger.debug_logger.error("Data fetch failed")
            return None, None, None, None
        
        # Test signal generation
        print("\n2. Testing signal generation...")
        long_signals, short_signals, is_primary = self.generate_signals(data)
        
        total_signals = long_signals.sum() + short_signals.sum()
        min_required = self.min_signals_required
        
        if total_signals < min_required:
            print(f"✗ Insufficient signals: {total_signals} found, {min_required} required")
            return None, None, None, None
            
        print(f"Long signals found: {long_signals.sum()}")
        print(f"Short signals found: {short_signals.sum()}")
        print(f"Primary signals found: {is_primary.sum()}")
        
        self.logger.debug_logger.info(
            f"Signal generation test - Longs: {long_signals.sum()}, "
            f"Shorts: {short_signals.sum()}, Primary: {is_primary.sum()}"
        )
        
        if long_signals.sum() > 0 or short_signals.sum() > 0:
            print("\nSignal dates:")
            signal_dates = data.index[long_signals | short_signals]
            for date in signal_dates[:5]:  # Show first 5 signals
                print(f"Signal at: {date}")
        
        # Test parameter generation
        print("\n3. Testing parameter combinations...")
        params = self._generate_param_combinations()
        print(f"Generated {len(params)} valid parameter combinations")
        if params:
            print("\nSample parameter set:")
            print(params[0])
            self.logger.debug_logger.info(f"Parameter combinations generated: {len(params)}")
        
        # Test single evaluation
        print("\n4. Testing parameter evaluation...")
        if params:
            test_results = self._evaluate_params(params[0], data)
            print("Evaluation results:")
            print(test_results)
            self.logger.debug_logger.info(f"Test evaluation results: {test_results}")

        return data, long_signals, short_signals, is_primary

    def _validate_data_quality(self, data: pd.DataFrame) -> bool:
        """Validate data quality before optimization"""
        try:
            # Check for minimum data points
            if len(data) < 100:
                self.logger.main_logger.error("Insufficient data points")
                return False
        
            # Filter out rows with missing data instead of rejecting entirely
            data = data.dropna()
            if len(data) < 100:
                self.logger.main_logger.error("Insufficient valid data points after cleaning")
                return False
        
            # Check price validity
            if (data['high'] < data['low']).any():
                self.logger.main_logger.error("Invalid price data detected")
                return False
        
            # Check volume
            if (data['tick_volume'] <= 0).any():
                self.logger.main_logger.warning("Invalid volume data detected")
                return False
        
            return True
            
        except Exception as e:
            self.logger.main_logger.error(f"Error in data validation: {str(e)}")
            return False
    
    def optimize_parallel(self) -> Tuple[Dict, pd.DataFrame]:
        """Run parallel optimization process with enhanced monitoring"""
        try:
            # Get and validate market data
            data = self._get_market_data()
            if data is None or not self._validate_data_quality(data):
                return None, None

            # Generate parameter combinations
            param_combinations = self._generate_param_combinations()
            total_combinations = len(param_combinations)
            
            self.logger.main_logger.info(
                f"Starting optimization with {total_combinations} parameter combinations"
            )
            
            # Setup parallel processing
            num_cores = mp.cpu_count()
            chunk_size = max(1, min(1000, total_combinations // (num_cores * 4)))
            chunks = [param_combinations[i:i + chunk_size] 
                     for i in range(0, len(param_combinations), chunk_size)]
            
            start_time = time.time()
            results = []
            
            # Initialize progress tracking
            completed = 0
            best_result = None
            
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                futures = {
                    executor.submit(self._process_chunk, chunk, data.copy()): i 
                    for i, chunk in enumerate(chunks)
                }
                
                try:
                    for future in concurrent.futures.as_completed(futures):
                        # Check for emergency stop
                        if os.path.exists(STOP_FILE):
                            self.logger.main_logger.warning("Emergency stop triggered")
                            executor.shutdown(wait=False)
                            return None, None
                            
                        chunk_results = future.result(timeout=300)
                        if chunk_results:
                            results.extend(chunk_results)
                            completed += len(chunk_results)
                            
                            # Update progress and best result
                            elapsed_time = time.time() - start_time
                            self._update_optimization_progress(
                                completed, total_combinations, elapsed_time, chunk_results
                            )
                            
                            # Update best result if necessary
                            best_result = self._update_best_result(best_result, chunk_results)
                            
                except KeyboardInterrupt:
                    self.logger.main_logger.warning("Optimization interrupted by user")
                    executor.shutdown(wait=False)
                    return None, None
                    
                except Exception as e:
                    self.logger.main_logger.error(f"Error in optimization: {str(e)}")
                    executor.shutdown(wait=False)
                    return None, None

            # Process final results
            results_df = pd.DataFrame(results)
            if len(results_df) > 0:
                best_params = self._get_best_params(results_df)
                self._save_optimization_results(results_df, best_params)
                return best_params, results_df
            else:
                self.logger.main_logger.warning("No valid results found")
                return None, None

        except Exception as e:
            self.logger.main_logger.error(f"Optimization error: {str(e)}")
            return None, None

    def _process_chunk(self, chunk: List[Dict], data: pd.DataFrame) -> List[Dict]:
        """Process a chunk of parameter combinations"""
        chunk_results = []
        for params in chunk:
            try:
                if self._validate_params(params):
                    evaluation = self._evaluate_params(params, data)
                    if evaluation['total_trades'] > 0:
                        chunk_results.append({**params, **evaluation})
                        
            except Exception as e:
                self.logger.debug_logger.error(
                    f"Error processing parameters {params}: {str(e)}"
                )
                
        return chunk_results

    def _validate_params(self, params: Dict) -> bool:
        """Validate parameter combinations with enhanced rules"""
        try:
            # Check lot percentages sum to 100%
            lot_sum = (params['tp1_lot_percent'] + params['tp2_lot_percent'] + 
                      params['tp3_lot_percent'] + params['tp4_lot_percent'])
            if abs(lot_sum - 100) > 0.0001:
                return False

            # Validate take profit weights are properly ordered
            weights = [
                params['tp1_weight'],
                params['tp2_weight'],
                params['tp3_weight'],
                params['tp4_weight']
            ]
            
            # Check ascending order with minimum spacing
            if not all(weights[i] + 0.001 < weights[i+1] for i in range(len(weights)-1)):
                return False

            # Validate buffer parameters
            if (params['wave_cross_buffer'] <= 0 or 
                params['tunnel_touch_buffer'] <= 0):
                return False

            return True

        except Exception as e:
            self.logger.debug_logger.error(f"Parameter validation error: {str(e)}")
            return False

    def _get_best_params(self, results_df: pd.DataFrame) -> Dict:
        """Get best parameters based on multiple criteria"""
        # Calculate composite score
        results_df['composite_score'] = (
            results_df['win_rate'] * 0.3 +
            results_df['profit_factor'] * 0.3 +
            results_df['sharpe_ratio'] * 0.2 +
            (1 - results_df['max_drawdown']) * 0.2
        )
        
        # Get best parameters
        best_params = results_df.nlargest(1, 'composite_score').iloc[0].to_dict()
        
        # Log best parameters
        self.logger.main_logger.info(f"Best parameters found: {best_params}")
        
        return best_params

    def _update_optimization_progress(self, completed: int, total: int, 
                                   elapsed_time: float, chunk_results: List[Dict]):
        """Update optimization progress with enhanced metrics"""
        progress = (completed / total) * 100
        remaining_time = (elapsed_time / progress) * (100 - progress) if progress > 0 else 0
        
        # Calculate chunk statistics
        if chunk_results:
            chunk_df = pd.DataFrame(chunk_results)
            chunk_stats = {
                'avg_profit': chunk_df['total_profit_pct'].mean(),
                'max_profit': chunk_df['total_profit_pct'].max(),
                'avg_trades': chunk_df['total_trades'].mean(),
                'avg_win_rate': chunk_df['win_rate'].mean()
            }
        else:
            chunk_stats = {}
        
        self.logger.log_progress(completed, total, elapsed_time, remaining_time, chunk_stats)

    def _calculate_max_drawdown(self, profits: pd.Series) -> float:
        """Calculate maximum drawdown from a series of profits"""
        try:
            cumulative = (1 + profits).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = cumulative / running_max - 1
            return abs(drawdowns.min())
        except Exception as e:
            self.logger.debug_logger.error(f"Error calculating max drawdown: {str(e)}")
            return 1.0

    def _calculate_profit_factor(self, profits: pd.Series) -> float:
        """Calculate profit factor with safety checks"""
        try:
            wins = profits[profits > 0].sum()
            losses = abs(profits[profits < 0].sum())
            return wins / losses if losses != 0 else float('inf')
        except Exception as e:
            self.logger.debug_logger.error(f"Error calculating profit factor: {str(e)}")
            return 0.0

    def _calculate_sharpe_ratio(self, profits: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio with safety checks"""
        try:
            if len(profits) < 2:
                return 0.0
            excess_returns = profits - risk_free_rate
            return excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0.0
        except Exception as e:
            self.logger.debug_logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0

    def _calculate_sortino_ratio(self, profits: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio focusing on downside deviation"""
        try:
            excess_returns = profits - risk_free_rate
            downside_returns = excess_returns[excess_returns < 0]
            downside_std = np.sqrt(np.mean(downside_returns**2))
            return excess_returns.mean() / (downside_std + 1e-10)
        except Exception as e:
            self.logger.debug_logger.error(f"Error calculating Sortino ratio: {str(e)}")
            return 0.0
    def create_optimization_report(self, results_df: pd.DataFrame):
        """Create comprehensive optimization report with visualizations"""
        if results_df is None or len(results_df) == 0:
            self.logger.main_logger.warning("No results to create report")
            return

        report_path = self.base_path / f"{self.symbol}_{self.timeframe}_exit_report"
        report_path.mkdir(exist_ok=True)

        try:
            fig = make_subplots(
                rows=4, cols=2,
                subplot_titles=(
                    'Profit Distribution',
                    'Risk-Return Analysis',
                    'Take Profit Level Distribution',
                    'Hold Time vs Profit',
                    'Drawdown Analysis',
                    'Trade Count Distribution',
                    'Market Condition Performance',
                    'Signal Type Performance'
                ),
                vertical_spacing=0.12,
                horizontal_spacing=0.1,
                specs=[[{'type': 'histogram'}, {'type': 'scatter'}],
                      [{'type': 'box'}, {'type': 'scatter'}],
                      [{'type': 'scatter'}, {'type': 'bar'}],
                      [{'type': 'heatmap'}, {'type': 'bar'}]]
            )

            # 1. Profit Distribution
            fig.add_trace(
                go.Histogram(
                    x=results_df['avg_profit_per_trade_pct'],
                    name='Profit Distribution',
                    nbinsx=50,
                    marker_color='blue'
                ),
                row=1, col=1
            )

            # 2. Risk-Return Analysis
            fig.add_trace(
                go.Scatter(
                    x=results_df['sharpe_ratio'],
                    y=results_df['win_rate'].multiply(100),
                    mode='markers',
                    marker=dict(
                        color=results_df['profit_factor'],
                        colorscale='Viridis',
                        showscale=True,
                        size=8,
                        colorbar=dict(title='Profit Factor')
                    ),
                    name='Risk-Return'
                ),
                row=1, col=2
            )

            # 3. Take Profit Level Distribution
            tp_levels = pd.DataFrame({
                'TP1': results_df['tp1_weight'].multiply(100),
                'TP2': results_df['tp2_weight'].multiply(100),
                'TP3': results_df['tp3_weight'].multiply(100),
                'TP4': results_df['tp4_weight'].multiply(100)
            })
            
            fig.add_trace(
                go.Box(
                    x=tp_levels.values.flatten(),
                    name='TP Levels',
                    marker_color='green'
                ),
                row=2, col=1
            )

            # 4. Hold Time vs Profit Analysis
            fig.add_trace(
                go.Scatter(
                    x=results_df['avg_hold_time'],
                    y=results_df['avg_profit_per_trade_pct'],
                    mode='markers',
                    marker=dict(
                        color=results_df['total_trades'],
                        colorscale='Viridis',
                        showscale=True,
                        size=8,
                        colorbar=dict(title='Total Trades')
                    ),
                    name='Hold Time Impact'
                ),
                row=2, col=2
            )

            # 5. Drawdown Analysis
            fig.add_trace(
                go.Scatter(
                    x=results_df['max_drawdown'].multiply(100),
                    y=results_df['profit_factor'],
                    mode='markers',
                    marker=dict(
                        color=results_df['sharpe_ratio'],
                        colorscale='Viridis',
                        showscale=True,
                        size=8,
                        colorbar=dict(title='Sharpe Ratio')
                    ),
                    name='Drawdown vs Profit Factor'
                ),
                row=3, col=1
            )

            # 6. Trade Count Distribution
            fig.add_trace(
                go.Histogram(
                    x=results_df['total_trades'],
                    name='Trade Count',
                    nbinsx=30,
                    marker_color='purple'
                ),
                row=3, col=2
            )

            # 7. Market Condition Performance Heatmap
            market_condition_data = self._prepare_market_condition_heatmap(results_df)
            fig.add_trace(
                go.Heatmap(
                    z=market_condition_data['values'],
                    x=market_condition_data['columns'],
                    y=market_condition_data['rows'],
                    colorscale='RdYlGn',
                    name='Market Conditions'
                ),
                row=4, col=1
            )

            # 8. Signal Type Performance
            signal_performance = self._prepare_signal_performance_chart(results_df)
            fig.add_trace(
                go.Bar(
                    x=['Primary', 'Secondary'],
                    y=[signal_performance['primary_profit'], 
                       signal_performance['secondary_profit']],
                    name='Signal Performance',
                    marker_color=['blue', 'red']
                ),
                row=4, col=2
            )

            # Update layout
            fig.update_layout(
                height=1800,
                width=1200,
                showlegend=True,
                title_text=f"Exit Strategy Optimization Results - {self.symbol} {self.timeframe}"
            )

            # Update axes
            self._update_chart_axes(fig)
            
            # Save visualization and reports
            fig.write_html(report_path / "optimization_results.html")
            self._create_summary_report(results_df, report_path)
            self._analyze_parameter_distributions(results_df, report_path)
            self._create_trade_analysis_report(results_df, report_path)

        except Exception as e:
            self.logger.main_logger.error(f"Error creating optimization report: {str(e)}")

    def _prepare_market_condition_heatmap(self, results_df: pd.DataFrame) -> Dict:
        """Prepare data for market condition heatmap"""
        try:
            conditions = ['excellent_condition', 'good_condition', 
                         'moderate_condition', 'poor_condition']
            metrics = ['win_rate', 'avg_profit', 'trade_count']
            
            values = []
            for condition in conditions:
                condition_data = []
                condition_results = results_df[results_df['market_condition'] == condition]
                for metric in metrics:
                    if metric in condition_results.columns:
                        value = condition_results[metric].mean()
                        condition_data.append(value)
                values.append(condition_data)
            
            return {
                'values': values,
                'rows': conditions,
                'columns': metrics
            }
        except Exception as e:
            self.logger.main_logger.error(f"Error preparing market condition heatmap: {str(e)}")
            return {'values': [], 'rows': [], 'columns': []}

    def _prepare_signal_performance_chart(self, results_df: pd.DataFrame) -> Dict:
        """Prepare data for signal type performance chart"""
        try:
            primary_trades = results_df[results_df['is_primary']]
            secondary_trades = results_df[~results_df['is_primary']]
            
            return {
                'primary_profit': primary_trades['avg_profit_per_trade_pct'].mean(),
                'secondary_profit': secondary_trades['avg_profit_per_trade_pct'].mean()
            }
        except Exception as e:
            self.logger.main_logger.error(f"Error preparing signal performance chart: {str(e)}")
            return {'primary_profit': 0, 'secondary_profit': 0}

    def _update_chart_axes(self, fig: go.Figure):
        """Update chart axes labels"""
        try:
            fig.update_xaxes(title_text="Profit per Trade (%)", row=1, col=1)
            fig.update_xaxes(title_text="Sharpe Ratio", row=1, col=2)
            fig.update_xaxes(title_text="Take Profit Levels (%)", row=2, col=1)
            fig.update_xaxes(title_text="Average Hold Time (hours)", row=2, col=2)
            fig.update_xaxes(title_text="Maximum Drawdown (%)", row=3, col=1)
            fig.update_xaxes(title_text="Number of Trades", row=3, col=2)
            fig.update_xaxes(title_text="Performance Metrics", row=4, col=1)
            fig.update_xaxes(title_text="Signal Type", row=4, col=2)

            fig.update_yaxes(title_text="Frequency", row=1, col=1)
            fig.update_yaxes(title_text="Win Rate (%)", row=1, col=2)
            fig.update_yaxes(title_text="Distribution", row=2, col=1)
            fig.update_yaxes(title_text="Average Profit (%)", row=2, col=2)
            fig.update_yaxes(title_text="Profit Factor", row=3, col=1)
            fig.update_yaxes(title_text="Frequency", row=3, col=2)
            fig.update_yaxes(title_text="Market Conditions", row=4, col=1)
            fig.update_yaxes(title_text="Average Profit (%)", row=4, col=2)

        except Exception as e:
            self.logger.main_logger.error(f"Error updating chart axes: {str(e)}")

    def _analyze_parameter_distributions(self, results_df: pd.DataFrame, report_path: Path):
        """Analyze parameter distributions for successful trades"""
        try:
            # Filter for successful parameter combinations
            successful_params = results_df[
                (results_df['win_rate'] > 0.5) & 
                (results_df['profit_factor'] > 1.5)
            ]

            param_analysis = {}
            for param in self.param_ranges.keys():
                param_analysis[param] = {
                    'mean': successful_params[param].mean(),
                    'median': successful_params[param].median(),
                    'std': successful_params[param].std(),
                    'min': successful_params[param].min(),
                    'max': successful_params[param].max()
                }

            # Save analysis
            with open(report_path / "parameter_analysis.txt", "w") as f:
                f.write("Parameter Distribution Analysis\n")
                f.write("=" * 50 + "\n\n")
                
                for param, stats in param_analysis.items():
                    f.write(f"\n{param}:\n")
                    for metric, value in stats.items():
                        f.write(f"  {metric}: {value:.4f}\n")

        except Exception as e:
            self.logger.main_logger.error(f"Error analyzing parameter distributions: {str(e)}")

    def _create_trade_analysis_report(self, results_df: pd.DataFrame, report_path: Path):
        """Create detailed trade analysis report"""
        try:
            with open(report_path / "trade_analysis.txt", "w") as f:
                f.write("Trade Analysis Report\n")
                f.write("=" * 50 + "\n\n")
                
                # Overall statistics
                f.write("Overall Statistics:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total Combinations Tested: {len(results_df)}\n")
                f.write(f"Average Win Rate: {results_df['win_rate'].mean()*100:.2f}%\n")
                f.write(f"Average Profit Factor: {results_df['profit_factor'].mean():.2f}\n")
                f.write(f"Average Sharpe Ratio: {results_df['sharpe_ratio'].mean():.2f}\n")
                f.write(f"Average Maximum Drawdown: {results_df['max_drawdown'].mean()*100:.2f}%\n\n")
                
                # Market condition analysis
                f.write("Market Condition Analysis:\n")
                f.write("-" * 20 + "\n")
                for condition in ['excellent_condition', 'good_condition', 
                                'moderate_condition', 'poor_condition']:
                    condition_data = results_df[results_df['market_condition'] == condition]
                    if len(condition_data) > 0:
                        f.write(f"\n{condition}:\n")
                        f.write(f"  Trade Count: {len(condition_data)}\n")
                        f.write(f"  Win Rate: {condition_data['win_rate'].mean()*100:.2f}%\n")
                        f.write(f"  Average Profit: {condition_data['avg_profit_per_trade_pct'].mean():.2f}%\n")
                
                # Signal type analysis
                f.write("\nSignal Type Analysis:\n")
                f.write("-" * 20 + "\n")
                primary = results_df[results_df['is_primary']]
                secondary = results_df[~results_df['is_primary']]
                
                f.write("\nPrimary Signals:\n")
                f.write(f"  Count: {len(primary)}\n")
                f.write(f"  Win Rate: {primary['win_rate'].mean()*100:.2f}%\n")
                f.write(f"  Average Profit: {primary['avg_profit_per_trade_pct'].mean():.2f}%\n")
                
                f.write("\nSecondary Signals:\n")
                f.write(f"  Count: {len(secondary)}\n")
                f.write(f"  Win Rate: {secondary['win_rate'].mean()*100:.2f}%\n")
                f.write(f"  Average Profit: {secondary['avg_profit_per_trade_pct'].mean():.2f}%\n")

        except Exception as e:
            self.logger.main_logger.error(f"Error creating trade analysis report: {str(e)}")

def create_combined_summary_report(all_results: Dict, base_dir: Path, symbol: str):
    """Create a comprehensive summary report combining results from all timeframes"""
    try:
        summary_path = base_dir / "combined_summary.txt"
        
        with open(summary_path, "w") as f:
            f.write(f"Combined Summary Report for {symbol}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            for timeframe, results in all_results.items():
                f.write(f"\nTimeframe: {timeframe}\n")
                f.write("-" * 40 + "\n")
                
                # Entry Parameters
                f.write("Entry Parameters:\n")
                for param, value in results['entry_params'].items():
                    f.write(f"  {param}: {value}\n")
                
                # Exit Parameters
                f.write("\nOptimized Exit Parameters:\n")
                exit_params = results['exit_params']
                
                f.write("Take Profit Levels:\n")
                for i in range(1, 5):
                    f.write(f"  TP{i}: {exit_params[f'tp{i}_lot_percent']}% "
                           f"at {exit_params[f'tp{i}_weight']*100:.2f}%\n")
                
                f.write("\nStop Loss Parameters:\n")
                f.write(f"  Wave Cross Buffer: {exit_params['wave_cross_buffer']*100:.3f}%\n")
                f.write(f"  Tunnel Touch Buffer: {exit_params['tunnel_touch_buffer']*100:.3f}%\n")
                
                f.write("\nPerformance Metrics:\n")
                f.write(f"  Total Trades: {exit_params['total_trades']}\n")
                f.write(f"  Win Rate: {exit_params['win_rate']*100:.1f}%\n")
                f.write(f"  Average Profit: {exit_params['avg_profit_per_trade_pct']:.2f}%\n")
                f.write(f"  Sharpe Ratio: {exit_params['sharpe_ratio']:.2f}\n")
                f.write(f"  Profit Factor: {exit_params['profit_factor']:.2f}\n")
                f.write(f"  Max Drawdown: {exit_params['max_drawdown']*100:.2f}%\n")
                
                f.write("\n" + "=" * 80 + "\n")
                
    except Exception as e:
        logging.error(f"Error creating combined summary report: {str(e)}")

def create_comparative_visualization(all_results: Dict, output_dir: Path, symbol: str):
    """Create comparative visualization across timeframes"""
    try:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Win Rate Comparison',
                'Profit Factor Comparison',
                'Drawdown Comparison',
                'Risk-Adjusted Return Comparison'
            )
        )

        timeframes = list(all_results.keys())
        metrics = {
            'Win Rate': [all_results[tf]['exit_params']['win_rate'] * 100 for tf in timeframes],
            'Profit Factor': [all_results[tf]['exit_params']['profit_factor'] for tf in timeframes],
            'Max Drawdown': [all_results[tf]['exit_params']['max_drawdown'] * 100 for tf in timeframes],
            'Sharpe Ratio': [all_results[tf]['exit_params']['sharpe_ratio'] for tf in timeframes]
        }

        fig.add_trace(
            go.Bar(x=timeframes, y=metrics['Win Rate'], name='Win Rate'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=timeframes, y=metrics['Profit Factor'], name='Profit Factor'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=timeframes, y=metrics['Max Drawdown'], name='Max Drawdown'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=timeframes, y=metrics['Sharpe Ratio'], name='Sharpe Ratio'),
            row=2, col=2
        )

        fig.update_layout(
            height=800,
            width=1200,
            title_text=f"Comparative Analysis Across Timeframes - {symbol}",
            showlegend=False
        )

        fig.write_html(output_dir / "timeframe_comparison.html")

    except Exception as e:
        logging.error(f"Error creating comparative visualization: {str(e)}")

def main():
    """Main execution function with enhanced error handling and monitoring"""
    try:
        # Configuration
        symbol = SYMBOL
        start_date = datetime.now() - timedelta(days=5)
        end_date = datetime.now()
        
        print(f"\nRunning optimization for {symbol} from {start_date} to {end_date}")

        # Create results directory with timestamp
        base_results_dir = Path(f"exit_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        base_results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize monitoring and logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(base_results_dir / 'optimization.log'),
                logging.StreamHandler()
            ]
        )

        # Summary of all timeframe results
        all_results = {}
        
        # Test MT5 connection first
        if not test_mt5_connection():
            raise ConnectionError("Failed to establish MT5 connection")

        for timeframe in OPTIMIZATION_TIMEFRAMES:
            try:
                logging.info(f"\nOptimizing exit strategy for {symbol} on {timeframe}")
                entry_params = TIMEFRAME_ENTRY_PARAMS[timeframe]
                
                # Create timeframe-specific directory
                timeframe_dir = base_results_dir / timeframe
                timeframe_dir.mkdir(exist_ok=True)
                
                # Initialize optimizer
                optimizer = WavyTunnelExitOptimizer(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    entry_params=entry_params,
                    base_path=str(timeframe_dir)
                )

                # Run diagnostic tests
                logging.info("Running diagnostic tests...")
                data, long_signals, short_signals, is_primary = optimizer.test_data_and_signals(
                    start_date, end_date
                )

                if data is None:
                    logging.warning(f"No data available for {timeframe}")
                    continue

                if (long_signals.sum() + short_signals.sum() == 0):
                    logging.warning(f"No signals generated for {timeframe}")
                    continue

                logging.info(f"Signal generation test results for {timeframe}:")
                logging.info(f"Long signals: {long_signals.sum()}")
                logging.info(f"Short signals: {short_signals.sum()}")
                logging.info(f"Primary signals: {is_primary.sum()}")

                # Run optimization
                logging.info("Starting optimization process...")
                best_params, results_df = optimizer.optimize_parallel()
                
                if best_params is not None:
                    # Store results
                    all_results[timeframe] = {
                        'entry_params': entry_params,
                        'exit_params': best_params,
                        'performance_metrics': {
                            'total_trades': best_params['total_trades'],
                            'win_rate': best_params['win_rate'],
                            'profit_factor': best_params['profit_factor'],
                            'sharpe_ratio': best_params['sharpe_ratio'],
                            'max_drawdown': best_params['max_drawdown']
                        }
                    }
                    
                    # Create detailed reports
                    optimizer.create_optimization_report(results_df)
                    
                    # Log best parameters
                    logging.info(f"\nOptimization completed for {timeframe}")
                    logging.info("Best parameters found:")
                    for param, value in best_params.items():
                        logging.info(f"{param}: {value}")
                else:
                    logging.warning(f"No valid results found for {timeframe}")

            except Exception as e:
                logging.error(f"Error optimizing {timeframe}: {str(e)}")
                continue

        # Create combined analysis and reports if we have results
        if all_results:
            try:
                create_combined_summary_report(all_results, base_results_dir, symbol)
                create_comparative_visualization(all_results, base_results_dir, symbol)
                logging.info("Created combined analysis reports")
            except Exception as e:
                logging.error(f"Error creating combined reports: {str(e)}")
        else:
            logging.warning("No valid results found for any timeframe")

    except Exception as e:
        logging.error(f"Critical error during execution: {str(e)}")
    finally:
        # Ensure MT5 is properly shut down
        if mt5.initialize():
            mt5.shutdown()
        logging.info("\nExit strategy optimization completed")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\nOptimization interrupted by user")
        if os.path.exists(STOP_FILE):
            os.remove(STOP_FILE)
    except Exception as e:
        logging.error(f"Error during execution: {str(e)}")
    finally:
        if mt5.initialize():
            mt5.shutdown()
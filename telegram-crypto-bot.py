#!/usr/bin/env python3

import os
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
import asyncio
import time
import schedule
from datetime import datetime, timedelta
import threading
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, Bidirectional, GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import io
import traceback

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
TELEGRAM_BOT_TOKEN = "6256142340:AAFw-Jd29xzzgMb4sRoopnKxh0bH2_K0k_4"  # Replace with your bot token
AUTHORIZED_USERS = []  # Leave empty to allow all users, or add user IDs for restricted access
DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_TIMEFRAME = "1h"
DEFAULT_LIMIT = 300

# Top cryptocurrencies to monitor
TOP_COINS = [
    "BTC/USDT",  # Bitcoin
    "ETH/USDT",  # Ethereum
    "BNB/USDT",  # Binance Coin
    "XRP/USDT",  # Ripple
    "SOL/USDT",  # Solana
    "ADA/USDT",  # Cardano
    "DOGE/USDT", # Dogecoin
    "DOT/USDT",  # Polkadot
    "MATIC/USDT", # Polygon
    "LINK/USDT"  # Chainlink
]

# Store active signals and their status
active_signals = {
    'signals': {},  # Store active trading signals
    'entries': {},  # Store entry points
    'exits': {},    # Store exit points
    'status': {}    # Store current status of each signal
}

# Store active analysis jobs
active_jobs = {}

# Signal tracking configuration
signal_tracking_config = {
    'enabled': True,
    'check_interval': 5,  # minutes
    'price_update_interval': 1,  # minutes
    'notification_channels': set(),  # Store chat_ids for notifications
    'atr_multiplier_stop_loss': 1.5,
    'atr_multiplier_take_profit_1': 3.0,
    'atr_multiplier_take_profit_2': 5.0,
    'target_notification_threshold': 0.005  # 0.5% price proximity threshold for notifications
}
# Store user preferences
user_preferences = {}
# Store last signals to avoid duplicate alerts
last_signals = {}
# Store signal tracking status
signal_tracking = {
    'enabled': False,
    'last_check': None,
    'check_interval': 5  # minutes
}

#=========================================================================
# Smart Trade Analyzer Implementation
#=========================================================================

class SmartTradeAnalyzer:
    def __init__(self, symbol='BTC/USDT', timeframe='1h', limit=300):
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
        self.exchange = ccxt.binance()
        self.df = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.lstm_model = None
        self.rf_model = None
        self.last_signal = None
        self.signal_time = None
        self.wyckoff_phase = None
        self.elliott_wave_count = None
        self.wyckoff_score = 0
        self.elliott_score = 0
        self.price_action_score = 0
        self.support_resistance_levels = []
        self.swing_points = []
        
    def fetch_data(self):
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=self.limit)
            self.df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms')
            self.df.set_index('timestamp', inplace=True)
            print(f"Fetched {len(self.df)} candles for {self.symbol}")
            return self.df
        except Exception as e:
            print(f"Error fetching data: {e}")
            traceback.print_exc()
            return None
    
    def add_technical_features(self):
        try:
            # Original features
            self.df['sma_20'] = self.df['close'].rolling(window=20).mean()
            self.df['sma_50'] = self.df['close'].rolling(window=50).mean()
            self.df['sma_200'] = self.df['close'].rolling(window=200).mean()
            
            delta = self.df['close'].diff()
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss.replace(0, 0.001)
            self.df['rsi'] = 100 - (100 / (1 + rs))
            
            self.df['volatility'] = self.df['close'].pct_change().rolling(window=20).std() * 100
            
            self.df['norm_volume'] = (self.df['volume'] - self.df['volume'].rolling(window=20).min()) / \
                                   (self.df['volume'].rolling(window=20).max() - self.df['volume'].rolling(window=20).min() + 1e-10)
            
            self.df['body_size'] = abs(self.df['close'] - self.df['open'])
            self.df['direction'] = np.where(self.df['close'] > self.df['open'], 1, -1)
            self.df['avg_body'] = self.df['body_size'].rolling(20).mean()
            self.df['avg_body'] = self.df['avg_body'].fillna(self.df['body_size'].mean())
            
            self.df['large_body'] = self.df['body_size'] > (1.5 * self.df['avg_body'])
            self.df['prev_direction'] = self.df['direction'].shift(1).fillna(0)
            self.df['direction_change'] = (self.df['direction'] != self.df['prev_direction']).astype(int)
            self.df['potential_ob'] = (self.df['large_body'] & self.df['direction_change'].shift(-2)).astype(int)
            
            self.df['fvg_up'] = (self.df['low'] > self.df['high'].shift(1)).astype(int)
            self.df['fvg_down'] = (self.df['high'] < self.df['low'].shift(1)).astype(int)
            
            self.df['structure_change'] = 0
            
            threshold = 0.02
            for i in range(3, len(self.df)):
                if self.df['high'].iloc[i] > max(self.df['high'].iloc[i-3:i]) * (1 + threshold):
                    self.df.loc[self.df.index[i], 'structure_change'] = 1
                elif self.df['low'].iloc[i] < min(self.df['low'].iloc[i-3:i]) * (1 - threshold):
                    self.df.loc[self.df.index[i], 'structure_change'] = -1
            
            # Wyckoff Analysis
            self.df['volume_delta'] = self.df['volume'] * self.df['direction']
            self.df['volume_delta_ma'] = self.df['volume_delta'].rolling(window=20).mean()
            self.df['price_relative_to_ma'] = np.where(self.df['close'] > self.df['sma_50'], 1, -1)
            
            # Price Spread - useful for Wyckoff analysis
            self.df['price_spread'] = self.df['high'] - self.df['low']
            self.df['spread_ma'] = self.df['price_spread'].rolling(window=20).mean()
            self.df['relative_spread'] = self.df['price_spread'] / self.df['spread_ma']
            
            # Elliott Wave indicators
            self.df['trend_strength'] = abs(self.df['close'].pct_change(20))
            self.df['wave_pivots'] = 0
            
            # Find potential wave pivot points
            for i in range(5, len(self.df)-5):
                # Local peaks (potential wave 1, 3, 5)
                if (self.df['high'].iloc[i] > self.df['high'].iloc[i-1:i].max() and 
                    self.df['high'].iloc[i] > self.df['high'].iloc[i+1:i+6].max()):
                    self.df.loc[self.df.index[i], 'wave_pivots'] = 1
                # Local troughs (potential wave 2, 4)
                elif (self.df['low'].iloc[i] < self.df['low'].iloc[i-1:i].min() and 
                      self.df['low'].iloc[i] < self.df['low'].iloc[i+1:i+6].min()):
                    self.df.loc[self.df.index[i], 'wave_pivots'] = -1

            # =====================================================
            # ENHANCED PRICE ACTION ANALYSIS
            # =====================================================
            
            # 1. Candlestick Patterns
            # Doji Pattern
            self.df['doji'] = ((self.df['body_size'] / self.df['price_spread']) < 0.1).astype(int)
            
            # Hammer Pattern (bullish)
            self.df['hammer'] = 0
            for i in range(1, len(self.df)):
                if (self.df['direction'].iloc[i] == 1 or self.df['direction'].iloc[i] == -1) and \
                   (self.df['body_size'].iloc[i] < 0.5 * self.df['price_spread'].iloc[i]) and \
                   (self.df['high'].iloc[i] - max(self.df['open'].iloc[i], self.df['close'].iloc[i])) < \
                   (min(self.df['open'].iloc[i], self.df['close'].iloc[i]) - self.df['low'].iloc[i]) * 0.25:
                    self.df.loc[self.df.index[i], 'hammer'] = 1
            
            # Shooting Star Pattern (bearish)
            self.df['shooting_star'] = 0
            for i in range(1, len(self.df)):
                if (self.df['direction'].iloc[i] == 1 or self.df['direction'].iloc[i] == -1) and \
                   (self.df['body_size'].iloc[i] < 0.5 * self.df['price_spread'].iloc[i]) and \
                   (min(self.df['open'].iloc[i], self.df['close'].iloc[i]) - self.df['low'].iloc[i]) < \
                   (self.df['high'].iloc[i] - max(self.df['open'].iloc[i], self.df['close'].iloc[i])) * 0.25:
                    self.df.loc[self.df.index[i], 'shooting_star'] = 1
            
            # Engulfing Patterns
            self.df['bullish_engulfing'] = 0
            self.df['bearish_engulfing'] = 0
            
            for i in range(1, len(self.df)):
                # Bullish Engulfing
                if (self.df['direction'].iloc[i-1] == -1 and  # Previous candle is bearish
                    self.df['direction'].iloc[i] == 1 and     # Current candle is bullish
                    self.df['open'].iloc[i] <= self.df['close'].iloc[i-1] and  # Open below previous close
                    self.df['close'].iloc[i] >= self.df['open'].iloc[i-1]):    # Close above previous open
                    self.df.loc[self.df.index[i], 'bullish_engulfing'] = 1
                    
                # Bearish Engulfing
                if (self.df['direction'].iloc[i-1] == 1 and   # Previous candle is bullish
                    self.df['direction'].iloc[i] == -1 and    # Current candle is bearish
                    self.df['open'].iloc[i] >= self.df['close'].iloc[i-1] and  # Open above previous close
                    self.df['close'].iloc[i] <= self.df['open'].iloc[i-1]):    # Close below previous open
                    self.df.loc[self.df.index[i], 'bearish_engulfing'] = 1
            
            # 2. Swing High/Low Detection
            self.df['swing_high'] = 0
            self.df['swing_low'] = 0
            
            for i in range(3, len(self.df)-3):
                # Swing High (price peak with lower prices on both sides)
                if (self.df['high'].iloc[i] > self.df['high'].iloc[i-3:i].max() and 
                    self.df['high'].iloc[i] > self.df['high'].iloc[i+1:i+4].max()):
                    self.df.loc[self.df.index[i], 'swing_high'] = 1
                    self.swing_points.append((self.df.index[i], self.df['high'].iloc[i], 'high'))
                    
                # Swing Low (price valley with higher prices on both sides)
                if (self.df['low'].iloc[i] < self.df['low'].iloc[i-3:i].min() and 
                    self.df['low'].iloc[i] < self.df['low'].iloc[i+1:i+4].min()):
                    self.df.loc[self.df.index[i], 'swing_low'] = 1
                    self.swing_points.append((self.df.index[i], self.df['low'].iloc[i], 'low'))
            
            # 3. Support and Resistance Levels based on Swing points
            self.identify_support_resistance()
            
            # 4. Price Action Based Breakouts and Fakeouts
            self.df['breakout_up'] = 0
            self.df['breakout_down'] = 0
            self.df['fakeout_up'] = 0
            self.df['fakeout_down'] = 0
            
            for level in self.support_resistance_levels:
                level_price = level['price']
                level_type = level['type']
                level_strength = level['strength']
                
                if level_strength >= 2:  # Only consider strong levels
                    for i in range(5, len(self.df)):
                        # Bullish Breakout (breaking resistance)
                        if (level_type == 'resistance' and 
                            self.df['close'].iloc[i-1] < level_price and 
                            self.df['close'].iloc[i] > level_price and
                            self.df['close'].iloc[i] > self.df['open'].iloc[i]):
                            self.df.loc[self.df.index[i], 'breakout_up'] = 1
                            
                            # Check for fakeout in the next 3 candles
                            for j in range(i+1, min(i+4, len(self.df))):
                                if self.df['close'].iloc[j] < level_price:
                                    self.df.loc[self.df.index[i], 'fakeout_up'] = 1
                                    break
                        
                        # Bearish Breakout (breaking support)
                        if (level_type == 'support' and 
                            self.df['close'].iloc[i-1] > level_price and 
                            self.df['close'].iloc[i] < level_price and
                            self.df['close'].iloc[i] < self.df['open'].iloc[i]):
                            self.df.loc[self.df.index[i], 'breakout_down'] = 1
                            
                            # Check for fakeout in the next 3 candles
                            for j in range(i+1, min(i+4, len(self.df))):
                                if self.df['close'].iloc[j] > level_price:
                                    self.df.loc[self.df.index[i], 'fakeout_down'] = 1
                                    break
            
            # 5. Accumulation/Distribution Analysis
            self.df['accum_dist'] = 0
            window_size = 10
            
            for i in range(window_size, len(self.df)):
                window = self.df.iloc[i-window_size:i]
                price_range = window['high'].max() - window['low'].min()
                
                if price_range < window['high'].max() * 0.03:  # Narrow range (less than 3% of price)
                    if window['volume'].mean() > self.df['volume'].iloc[i-30:i].mean() * 1.5:
                        # High volume in narrow range
                        if window['close'].iloc[-1] > window['close'].mean():
                            # Accumulation (prices tending upward)
                            self.df.loc[self.df.index[i], 'accum_dist'] = 1
                        else:
                            # Distribution (prices tending downward)
                            self.df.loc[self.df.index[i], 'accum_dist'] = -1
            
            # 6. Harmonic Patterns (simplified)
            self.df['harmonic_bullish'] = 0
            self.df['harmonic_bearish'] = 0
            
            # Find potential XABCD patterns
            for i in range(20, len(self.df)):
                recent_swings = [point for point in self.swing_points if point[0] in self.df.index[i-20:i]]
                if len(recent_swings) >= 5:
                    # Check for Bat, Gartley, Butterfly, Crab patterns (simplified)
                    # This is a simplified version that just looks for the general shape
                    highlow_pattern = [point[2] for point in recent_swings[-5:]]
                    
                    # Check for alternating high-low points
                    if highlow_pattern in [['low', 'high', 'low', 'high', 'low']]:
                        # Potential bullish pattern
                        self.df.loc[self.df.index[i], 'harmonic_bullish'] = 1
                    
                    elif highlow_pattern in [['high', 'low', 'high', 'low', 'high']]:
                        # Potential bearish pattern
                        self.df.loc[self.df.index[i], 'harmonic_bearish'] = 1
            
            # Calculate overall Price Action score
            self.df['price_action_bull_score'] = (
                self.df['hammer'] * 1.0 + 
                self.df['bullish_engulfing'] * 1.5 + 
                self.df['breakout_up'] * (1 - self.df['fakeout_up']) * 2.0 +
                self.df['swing_low'] * 0.8 +
                self.df['harmonic_bullish'] * 2.0 +
                (self.df['accum_dist'] == 1) * 1.5
            )
            
            self.df['price_action_bear_score'] = (
                self.df['shooting_star'] * 1.0 + 
                self.df['bearish_engulfing'] * 1.5 + 
                self.df['breakout_down'] * (1 - self.df['fakeout_down']) * 2.0 +
                self.df['swing_high'] * 0.8 +
                self.df['harmonic_bearish'] * 2.0 +
                (self.df['accum_dist'] == -1) * 1.5
            )
            
            # Identify Wyckoff Phase
            self.identify_wyckoff_phase()
            
            # Identify Elliott Wave Count
            self.identify_elliott_wave()
            
            self.df = self.df.fillna(0)
            
            print("Added technical features, ICT, Wyckoff, Elliott Wave and enhanced Price Action patterns")
            return self.df
        except Exception as e:
            print(f"Error adding technical features: {e}")
            traceback.print_exc()
            return None
    
    def identify_support_resistance(self):
        """Identify support and resistance levels based on swing points"""
        if len(self.swing_points) < 5:
            return
            
        price_levels = {}
        
        # Group close swing points
        for date, price, swing_type in self.swing_points:
            # Round price to improve clustering of levels
            rounded_price = round(price, 1)
            key = (rounded_price, swing_type)
            
            if key not in price_levels:
                price_levels[key] = {
                    'price': price,
                    'count': 1,
                    'type': 'resistance' if swing_type == 'high' else 'support',
                    'dates': [date]
                }
            else:
                price_levels[key]['count'] += 1
                price_levels[key]['dates'].append(date)
                # Average the actual prices
                price_levels[key]['price'] = (price_levels[key]['price'] * (len(price_levels[key]['dates']) - 1) + price) / len(price_levels[key]['dates'])
        
        # Convert to list and calculate strength
        sr_levels = []
        for key, data in price_levels.items():
            if data['count'] >= 2:  # Only consider levels that appear at least twice
                level = {
                    'price': data['price'],
                    'type': data['type'],
                    'strength': data['count'],
                    'last_touch': max(data['dates'])
                }
                sr_levels.append(level)
        
        # Sort by strength (descending)
        sr_levels.sort(key=lambda x: x['strength'], reverse=True)
        
        # Take top levels
        self.support_resistance_levels = sr_levels[:8]
        
        print(f"Identified {len(self.support_resistance_levels)} support and resistance levels")
        
    def identify_wyckoff_phase(self):
        # A simplified Wyckoff phase identification
        if len(self.df) < 50:
            self.wyckoff_phase = "Insufficient data"
            return
            
        # Get the recent data
        recent_df = self.df.iloc[-50:]
        
        # Calculate key metrics for Wyckoff analysis
        price_trend = recent_df['close'].iloc[-1] > recent_df['close'].iloc[-20]
        volume_trend = recent_df['volume'].iloc[-10:].mean() > recent_df['volume'].iloc[-30:-10].mean()
        price_making_higher_highs = recent_df['high'].iloc[-5:].max() > recent_df['high'].iloc[-20:-5].max()
        price_making_lower_lows = recent_df['low'].iloc[-5:].min() < recent_df['low'].iloc[-20:-5].min()
        rsi_overbought = recent_df['rsi'].iloc[-1] > 70
        rsi_oversold = recent_df['rsi'].iloc[-1] < 30
        
        # Look for volume climax
        volume_climax = recent_df['volume'].iloc[-5:].max() > 2 * recent_df['volume'].iloc[-30:].mean()
        
        # Price range expansion/contraction
        recent_ranges = recent_df['price_spread'].iloc[-5:].mean()
        previous_ranges = recent_df['price_spread'].iloc[-20:-5].mean()
        range_expansion = recent_ranges > 1.5 * previous_ranges
        range_contraction = recent_ranges < 0.7 * previous_ranges
        
        # Determine the Wyckoff phase
        if price_trend and volume_trend and price_making_higher_highs:
            if rsi_overbought and volume_climax:
                self.wyckoff_phase = "Distribution"
                self.wyckoff_score = -0.7  # Strong sell signal
            elif range_expansion:
                self.wyckoff_phase = "Markup"
                self.wyckoff_score = 0.8  # Strong buy signal
            else:
                self.wyckoff_phase = "Late Markup"
                self.wyckoff_score = 0.3  # Moderate buy signal
        elif not price_trend and price_making_lower_lows:
            if rsi_oversold and volume_climax:
                self.wyckoff_phase = "Accumulation"
                self.wyckoff_score = 0.6  # Buy signal
            elif range_expansion:
                self.wyckoff_phase = "Markdown"
                self.wyckoff_score = -0.8  # Strong sell signal
            else:
                self.wyckoff_phase = "Late Markdown"
                self.wyckoff_score = -0.3  # Moderate sell signal
        elif range_contraction:
            self.wyckoff_phase = "Spring/Test"
            # Check if this looks like a spring (potential buy signal)
            if rsi_oversold:
                self.wyckoff_score = 0.5  # Moderate buy signal
            else:
                self.wyckoff_score = -0.2  # Slight sell bias
        else:
            self.wyckoff_phase = "Neutral"
            self.wyckoff_score = 0
            
    def identify_elliott_wave(self):
        if len(self.df) < 100:
            self.elliott_wave_count = "Insufficient data"
            return
            
        # Get pivot points
        pivots = self.df[self.df['wave_pivots'] != 0].iloc[-15:]
        
        if len(pivots) < 5:
            self.elliott_wave_count = "Unclear pattern"
            self.elliott_score = 0
            return
            
        # Check the pattern of pivots (alternating highs and lows)
        pivot_values = pivots['wave_pivots'].values
        
        # Simplified Elliott Wave count
        # The real Elliott Wave analysis is much more complex
        # This is a basic approximation
        
        # Check for 5-wave impulse structure
        if len(pivot_values) >= 9:
            recent_pivots = pivot_values[-9:]
            # A potential 5-wave pattern would have this signature:
            # [1, -1, 1, -1, 1] or [-1, 1, -1, 1, -1]
            
            if np.array_equal(recent_pivots[-5:], [1, -1, 1, -1, 1]):
                # Potential completion of 5 waves up - bearish signal
                self.elliott_wave_count = "Completing 5th wave up"
                self.elliott_score = -0.6
            elif np.array_equal(recent_pivots[-5:], [-1, 1, -1, 1, -1]):
                # Potential completion of 5 waves down - bullish signal
                self.elliott_wave_count = "Completing 5th wave down"
                self.elliott_score = 0.6
            elif np.array_equal(recent_pivots[-3:], [1, -1, 1]) and recent_pivots[-4] == -1:
                # Potential 3rd wave up - bullish signal
                self.elliott_wave_count = "In 3rd wave up"
                self.elliott_score = 0.8
            elif np.array_equal(recent_pivots[-3:], [-1, 1, -1]) and recent_pivots[-4] == 1:
                # Potential 3rd wave down - bearish signal
                self.elliott_wave_count = "In 3rd wave down"
                self.elliott_score = -0.8
            elif np.array_equal(recent_pivots[-7:], [1, -1, 1, -1, 1, -1, 1]):
                # Potential start of ABC correction after 5 waves up
                self.elliott_wave_count = "Start of correction after impulse up"
                self.elliott_score = -0.5
            elif np.array_equal(recent_pivots[-7:], [-1, 1, -1, 1, -1, 1, -1]):
                # Potential start of ABC correction after 5 waves down
                self.elliott_wave_count = "Start of correction after impulse down"
                self.elliott_score = 0.5
            else:
                self.elliott_wave_count = "Complex structure"
                # Analyze recent trend for bias
                recent_trend = self.df['close'].iloc[-20:].mean() > self.df['close'].iloc[-40:-20].mean()
                self.elliott_score = 0.3 if recent_trend else -0.3
        else:
            self.elliott_wave_count = "Building structure"
            # Default to trend analysis
            recent_trend = self.df['close'].iloc[-20:].mean() > self.df['close'].iloc[-40:-20].mean()
            self.elliott_score = 0.2 if recent_trend else -0.2
    
    def build_advanced_model(self):
        try:
            print("Building advanced hybrid model...")
            
            self.set_signal_context()
            
            data = self.df[['close', 'sma_20', 'sma_50', 'rsi', 'volatility', 'norm_volume']].values
            data_scaled = self.scaler.fit_transform(data)
            
            train_size = int(len(data_scaled) * 0.8)
            train_data = data_scaled[:train_size]
            
            def create_dataset(dataset, time_step=1):
                X, y = [], []
                for i in range(len(dataset) - time_step - 1):
                    X.append(dataset[i:(i + time_step), :])
                    next_price = dataset[i + time_step, 0]
                    current_price = dataset[i + time_step - 1, 0]
                    y.append(1 if next_price > current_price else 0)
                return np.array(X), np.array(y)
            
            time_step = 20
            X_train, y_train = create_dataset(train_data, time_step)
            
            input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
            
            lstm_branch = LSTM(64, return_sequences=True)(input_layer)
            lstm_branch = Dropout(0.3)(lstm_branch)
            lstm_branch = LSTM(32)(lstm_branch)
            
            gru_branch = GRU(64, return_sequences=True)(input_layer)
            gru_branch = Dropout(0.3)(gru_branch)
            gru_branch = GRU(32)(gru_branch)
            
            merged = tf.keras.layers.Concatenate()([lstm_branch, gru_branch])
            
            dense1 = Dense(32, activation='relu')(merged)
            dense1 = Dropout(0.2)(dense1)
            output = Dense(1, activation='sigmoid')(dense1)
            
            self.lstm_model = Model(inputs=input_layer, outputs=output)
            self.lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            print("Training hybrid LSTM/GRU model...")
            self.lstm_model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=1)
            
            print("Training Random Forest model for pattern recognition...")
            # Enhanced features for RF model including Price Action
            rf_features = self.df[[
                'body_size', 'direction', 'large_body', 'potential_ob', 
                'fvg_up', 'fvg_down', 'rsi', 'volatility', 'structure_change',
                'hammer', 'shooting_star', 'bullish_engulfing', 'bearish_engulfing',
                'swing_high', 'swing_low', 'breakout_up', 'breakout_down',
                'price_action_bull_score', 'price_action_bear_score'
            ]].values
            
            rf_target = np.zeros(len(self.df))
            
            for i in range(14, len(self.df)-10):
                if self.df['rsi'].iloc[i] < 30 and self.df['close'].iloc[i+5] > self.df['close'].iloc[i] * 1.02:
                    rf_target[i] = 1
                elif self.df['rsi'].iloc[i] > 70 and self.df['close'].iloc[i+5] < self.df['close'].iloc[i] * 0.98:
                    rf_target[i] = -1
            
            rf_features = rf_features[14:-10]
            rf_target = rf_target[14:-10]
            
            X_train, X_test, y_train, y_test = train_test_split(rf_features, rf_target, test_size=0.2, random_state=42)
            self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.rf_model.fit(X_train, y_train)
            
            print("Models trained successfully")
        except Exception as e:
            print(f"Error building advanced model: {e}")
            traceback.print_exc()
            
    def set_signal_context(self):
        self.df['market_regime'] = 0
        
        for i in range(50, len(self.df)):
            if self.df['sma_20'].iloc[i] > self.df['sma_50'].iloc[i] and self.df['close'].iloc[i] > self.df['sma_20'].iloc[i]:
                self.df.loc[self.df.index[i], 'market_regime'] = 1
            elif self.df['sma_20'].iloc[i] < self.df['sma_50'].iloc[i] and self.df['close'].iloc[i] < self.df['sma_20'].iloc[i]:
                self.df.loc[self.df.index[i], 'market_regime'] = -1
    
    def analyze_price_action(self):
        """Analyze recent price action patterns and calculate overall score"""
        
        # Get most recent candle data
        recent_pa_bull = self.df['price_action_bull_score'].iloc[-5:].sum()
        recent_pa_bear = self.df['price_action_bear_score'].iloc[-5:].sum()
        
        # Calculate net Price Action score (positive = bullish, negative = bearish)
        self.price_action_score = recent_pa_bull - recent_pa_bear
        
        # Analyze key price action patterns
        pa_signals = []
        
        # Check for recent strong patterns
        if self.df['bullish_engulfing'].iloc[-3:].sum() > 0:
            pa_signals.append("Bullish Engulfing detected")
        
        if self.df['bearish_engulfing'].iloc[-3:].sum() > 0:
            pa_signals.append("Bearish Engulfing detected")
            
        if self.df['hammer'].iloc[-3:].sum() > 0:
            pa_signals.append("Hammer pattern detected (bullish)")
            
        if self.df['shooting_star'].iloc[-3:].sum() > 0:
            pa_signals.append("Shooting Star pattern detected (bearish)")
        
        # Check for swing points
        if self.df['swing_high'].iloc[-3:].sum() > 0:
            pa_signals.append("Recent Swing High (bearish)")
            
        if self.df['swing_low'].iloc[-3:].sum() > 0:
            pa_signals.append("Recent Swing Low (bullish)")
        
        # Check for breakouts/fakeouts
        if self.df['breakout_up'].iloc[-3:].sum() > 0:
            if self.df['fakeout_up'].iloc[-3:].sum() > 0:
                pa_signals.append("Failed breakout above resistance (bearish)")
            else:
                pa_signals.append("Successful breakout above resistance (bullish)")
                
        if self.df['breakout_down'].iloc[-3:].sum() > 0:
            if self.df['fakeout_down'].iloc[-3:].sum() > 0:
                pa_signals.append("Failed breakdown below support (bullish)")
            else:
                pa_signals.append("Successful breakdown below support (bearish)")
        
        # Check accumulation/distribution
        recent_accum = self.df['accum_dist'].iloc[-5:].sum()
        if recent_accum >= 2:
            pa_signals.append("Recent price Accumulation detected (bullish)")
        elif recent_accum <= -2:
            pa_signals.append("Recent price Distribution detected (bearish)")
            
        # Check harmonic patterns
        if self.df['harmonic_bullish'].iloc[-5:].sum() > 0:
            pa_signals.append("Potential bullish harmonic pattern")
        
        if self.df['harmonic_bearish'].iloc[-5:].sum() > 0:
            pa_signals.append("Potential bearish harmonic pattern")
            
        # Support/Resistance context
        if len(self.support_resistance_levels) > 0:
            current_price = self.df['close'].iloc[-1]
            
            # Find nearest support and resistance
            supports = [level for level in self.support_resistance_levels if level['type'] == 'support' and level['price'] < current_price]
            resistances = [level for level in self.support_resistance_levels if level['type'] == 'resistance' and level['price'] > current_price]
            
            nearest_support = min(supports, key=lambda x: current_price - x['price']) if supports else None
            nearest_resistance = min(resistances, key=lambda x: x['price'] - current_price) if resistances else None
            
            if nearest_support:
                support_distance = (current_price - nearest_support['price']) / current_price * 100
                if support_distance < 2:
                    pa_signals.append(f"Price near strong support at {nearest_support['price']:.2f} (bullish)")
                    
            if nearest_resistance:
                resistance_distance = (nearest_resistance['price'] - current_price) / current_price * 100
                if resistance_distance < 2:
                    pa_signals.append(f"Price near strong resistance at {nearest_resistance['price']:.2f} (bearish)")
        
        print(f"Price Action Score: {self.price_action_score:.2f}")
        for signal in pa_signals:
            print(f"- {signal}")
            
        return {
            "score": self.price_action_score,
            "signals": pa_signals
        }
    
    def generate_smart_signal(self):
        print("Generating smart trading signal with Wyckoff, Elliott Wave and Price Action analysis...")
        
        # Analyze recent price action patterns
        price_action_analysis = self.analyze_price_action()
        
        current_price = self.df['close'].iloc[-1]
        current_rsi = self.df['rsi'].iloc[-1]
        current_regime = self.df['market_regime'].iloc[-1]
        current_structure = self.df['structure_change'].iloc[-1]
        
        recent_obs = self.df.iloc[-30:][self.df.iloc[-30:]['potential_ob'] == 1]
        last_ob_idx = recent_obs.index[-1] if not recent_obs.empty else None
        
        recent_fvg_up = self.df.iloc[-30:][self.df.iloc[-30:]['fvg_up'] == 1]
        recent_fvg_down = self.df.iloc[-30:][self.df.iloc[-30:]['fvg_down'] == 1]
        
        last_fvg_up_idx = recent_fvg_up.index[-1] if not recent_fvg_up.empty else None
        last_fvg_down_idx = recent_fvg_down.index[-1] if not recent_fvg_down.empty else None
        
        lstm_prediction = self.predict_with_lstm()
        lstm_signal = "BUY" if lstm_prediction > 0.6 else "SELL" if lstm_prediction < 0.4 else "NEUTRAL"
        
        signal = "NEUTRAL"
        entry_price = current_price
        stop_loss = 0
        take_profit = 0
        risk_reward = 0
        reason = []
        
        # Updated weights to include Price Action
        weights = {
            'ob': 0.15,          # Order Block
            'fvg': 0.10,         # Fair Value Gap
            'lstm': 0.15,        # LSTM prediction
            'structure': 0.10,   # Structure change
            'regime': 0.05,      # Market regime
            'wyckoff': 0.15,     # Wyckoff analysis
            'elliott': 0.15,     # Elliott Wave analysis
            'price_action': 0.15 # Price Action analysis
        }
        
        buy_score = 0
        sell_score = 0
        
        # ICT Analysis
        if last_ob_idx is not None:
            hours_since_ob = int((self.df.index[-1] - last_ob_idx).total_seconds() / 3600)
            if hours_since_ob < 48:
                ob_direction = self.df.loc[last_ob_idx, 'direction']
                if ob_direction > 0:
                    buy_score += weights['ob']
                    reason.append(f"Bullish Order Block detected {hours_since_ob}h ago")
                else:
                    sell_score += weights['ob']
                    reason.append(f"Bearish Order Block detected {hours_since_ob}h ago")
        
        if last_fvg_up_idx is not None:
            hours_since_fvg = int((self.df.index[-1] - last_fvg_up_idx).total_seconds() / 3600)
            if hours_since_fvg < 24:
                buy_score += weights['fvg']
                reason.append(f"Bullish FVG detected {hours_since_fvg}h ago")
        
        if last_fvg_down_idx is not None:
            hours_since_fvg = int((self.df.index[-1] - last_fvg_down_idx).total_seconds() / 3600)
            if hours_since_fvg < 24:
                sell_score += weights['fvg']
                reason.append(f"Bearish FVG detected {hours_since_fvg}h ago")
        
        # LSTM prediction
        if lstm_signal == "BUY":
            buy_score += weights['lstm']
            reason.append(f"LSTM model predicts upward movement ({lstm_prediction:.2f})")
        elif lstm_signal == "SELL":
            sell_score += weights['lstm']
            reason.append(f"LSTM model predicts downward movement ({lstm_prediction:.2f})")
        
        # Structure change analysis
        if current_structure == 1:
            buy_score += weights['structure']
            reason.append("Bullish structure change detected")
        elif current_structure == -1:
            sell_score += weights['structure']
            reason.append("Bearish structure change detected")
        
        # Market regime analysis
        if current_regime == 1:
            buy_score += weights['regime']
            reason.append("Bullish market regime")
        elif current_regime == -1:
            sell_score += weights['regime']
            reason.append("Bearish market regime")
        
        # Add Wyckoff analysis
        if self.wyckoff_phase:
            if self.wyckoff_score > 0:
                buy_score += weights['wyckoff'] * self.wyckoff_score
                reason.append(f"Wyckoff Analysis: {self.wyckoff_phase} (Bullish)")
            elif self.wyckoff_score < 0:
                sell_score += weights['wyckoff'] * abs(self.wyckoff_score)
                reason.append(f"Wyckoff Analysis: {self.wyckoff_phase} (Bearish)")
        
        # Add Elliott Wave analysis
        if self.elliott_wave_count and self.elliott_wave_count != "Insufficient data":
            if self.elliott_score > 0:
                buy_score += weights['elliott'] * self.elliott_score
                reason.append(f"Elliott Wave: {self.elliott_wave_count} (Bullish)")
            elif self.elliott_score < 0:
                sell_score += weights['elliott'] * abs(self.elliott_score)
                reason.append(f"Elliott Wave: {self.elliott_wave_count} (Bearish)")
        
        # Add Price Action analysis
        if self.price_action_score > 0:
            buy_score += weights['price_action'] * min(self.price_action_score / 5, 1.0)
            # Add top 3 price action signals to reason
            bullish_signals = [s for s in price_action_analysis['signals'] if "bullish" in s.lower()]
            for signal in bullish_signals[:2]:
                reason.append(f"Price Action: {signal}")
        elif self.price_action_score < 0:
            sell_score += weights['price_action'] * min(abs(self.price_action_score) / 5, 1.0)
            # Add top 3 price action signals to reason
            bearish_signals = [s for s in price_action_analysis['signals'] if "bearish" in s.lower()]
            for signal in bearish_signals[:2]:
                reason.append(f"Price Action: {signal}")
        
        # Lower threshold for more decisive signals
        threshold = 0.2
        
        # Always force a decision (no more NEUTRAL)
        if buy_score > sell_score:
            signal = "BUY"
        else:
            signal = "SELL"
        
        # Add confidence level based on score difference
        score_diff = abs(buy_score - sell_score)
        confidence = "HIGH" if score_diff > 0.4 else "MEDIUM" if score_diff > 0.2 else "LOW"
        
        # Calculate entry, stop loss and target based on confidence level
        if signal == "BUY":
            entry_price = current_price
            if confidence == "HIGH":
                stop_loss = entry_price * 0.97  # 3% stop loss
                take_profit = entry_price * 1.09  # 9% target
                risk_reward = 3.0
            elif confidence == "MEDIUM":
                stop_loss = entry_price * 0.98  # 2% stop loss
                take_profit = entry_price * 1.06  # 6% target
                risk_reward = 3.0
            else:
                stop_loss = entry_price * 0.985  # 1.5% stop loss
                take_profit = entry_price * 1.045  # 4.5% target
                risk_reward = 3.0
        else:  # SELL
            entry_price = current_price
            if confidence == "HIGH":
                stop_loss = entry_price * 1.03  # 3% stop loss
                take_profit = entry_price * 0.91  # 9% target
                risk_reward = 3.0
            elif confidence == "MEDIUM":
                stop_loss = entry_price * 1.02  # 2% stop loss
                take_profit = entry_price * 0.94  # 6% target
                risk_reward = 3.0
            else:
                stop_loss = entry_price * 1.015  # 1.5% stop loss
                take_profit = entry_price * 0.955  # 4.5% target
                risk_reward = 3.0
        
        self.last_signal = signal
        self.signal_time = self.df.index[-1]
        
        reason_str = " | ".join(reason)
        
        print(f"\n===== FINAL TRADING SIGNAL FOR {self.symbol} =====")
        print(f"Buy Score: {buy_score:.2f}, Sell Score: {sell_score:.2f}")
        print(f"Signal: {signal} (Confidence: {confidence})")
        print(f"Entry Price: {entry_price:.2f}")
        print(f"Stop Loss: {stop_loss:.2f}")
        print(f"Take Profit: {take_profit:.2f}")
        print(f"Risk/Reward Ratio: {risk_reward:.2f}")
        print(f"Wyckoff Phase: {self.wyckoff_phase}")
        print(f"Elliott Wave: {self.elliott_wave_count}")
        print(f"Price Action Score: {self.price_action_score:.2f}")
        print(f"Reason: {reason_str}")
        print("==========================================")
        
        return {
            "signal": signal,
            "entry": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward": risk_reward,
            "reason": reason_str,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "confidence": confidence,
            "wyckoff": self.wyckoff_phase,
            "elliott": self.elliott_wave_count,
            "price_action": price_action_analysis
        }
    
    def predict_with_lstm(self):
        try:
            data = self.df[['close', 'sma_20', 'sma_50', 'rsi', 'volatility', 'norm_volume']].values
            data_scaled = self.scaler.transform(data)
            
            time_step = 20
            x_input = data_scaled[-time_step:].reshape(1, time_step, 6)
            
            prediction = self.lstm_model.predict(x_input)[0][0]
            return prediction
        except Exception as e:
            print(f"Error in LSTM prediction: {e}")
            return 0.5  # Neutral prediction in case of error
    
    def create_analysis_chart(self, signal_info):
        print("Creating advanced analysis chart with Wyckoff, Elliott Wave and Price Action...")
        
        try:
            # Get all available data
            df_plot = self.df.copy()
            
            # Print debug information about the data
            print(f"Total candles available: {len(df_plot)}")
            print(f"Date range: {df_plot.index[0]} to {df_plot.index[-1]}")
            
            # Create the OHLC DataFrame for plotting
            plot_data = pd.DataFrame({
                'Open': df_plot['open'],
                'High': df_plot['high'],
                'Low': df_plot['low'],
                'Close': df_plot['close'],
                'Volume': df_plot['volume']
            })
            
            # Convert to proper datatypes
            plot_data = plot_data.astype(float)
            
            # Check if we have enough data
            if len(plot_data) < 3:
                print("Error: Not enough data points for a meaningful chart")
                return None
                
            # Create custom colors for market data
            mc = mpf.make_marketcolors(
                up='green', down='red',
                edge={'up': 'darkgreen', 'down': 'darkred'},
                wick={'up': 'green', 'down': 'red'},
                volume='blue'
            )
            
            # Create a custom style
            style = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='-',
                y_on_right=False,
                facecolor='white',
                edgecolor='black',
                figcolor='white',
                gridcolor='lightgray'
            )
            
            # Create additional plots for the moving averages if they exist
            additional_plots = []
            if 'sma_20' in df_plot.columns and 'sma_50' in df_plot.columns:
                additional_plots = [
                    mpf.make_addplot(df_plot['sma_20'], color='blue', width=0.7),
                    mpf.make_addplot(df_plot['sma_50'], color='red', width=0.7)
                ]
            
            # Use a buffer to save the image
            buf = io.BytesIO()
            
            # Plot the candlestick chart
            fig, axes = mpf.plot(
                plot_data,
                type='candle',
                volume=True,
                style=style,
                addplot=additional_plots,
                title=f'{self.symbol} Analysis with ICT, Wyckoff, Elliott Wave and Price Action',
                figsize=(14, 10),
                datetime_format='%Y-%m-%d %H:%M',
                returnfig=True
            )
            
            # Get the main price axis
            ax1 = axes[0]
            
            # Add Wyckoff and Elliott annotations
            wyckoff_text = f"Wyckoff: {self.wyckoff_phase}"
            elliott_text = f"Elliott: {self.elliott_wave_count}"
            price_action_text = f"Price Action Score: {self.price_action_score:.2f}"
            
            ax1.text(0.03, 0.05, wyckoff_text + "\n" + elliott_text + "\n" + price_action_text, 
                    transform=ax1.transAxes, 
                    fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
            
            # Display signal and buy/sell scores
            signal_info_text = f"SIGNAL: {signal_info['signal']} (Confidence: {signal_info['confidence']})\n"
            signal_info_text += f"Buy Score: {signal_info['buy_score']:.2f}\n"
            signal_info_text += f"Sell Score: {signal_info['sell_score']:.2f}"
            
            ax1.text(0.03, 0.95, signal_info_text, transform=ax1.transAxes, 
                    fontsize=10, fontweight='bold', 
                    bbox=dict(facecolor='white', alpha=0.7),
                    color='green' if signal_info['signal'] == 'BUY' else 'red')
            
            # Add text for entry, stop loss and target levels
            entry_text = f"Entry: {signal_info['entry']:.2f}\n"
            entry_text += f"SL: {signal_info['stop_loss']:.2f}\n"
            entry_text += f"TP: {signal_info['take_profit']:.2f}\n"
            entry_text += f"R/R: {signal_info['risk_reward']:.2f}"
            
            ax1.text(0.03, 0.85, entry_text, transform=ax1.transAxes, 
                    fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
            
            # Draw horizontal lines for entry, stop loss and target levels
            ax1.axhline(y=signal_info['entry'], color='black', linewidth=1, linestyle='-')
            ax1.axhline(y=signal_info['stop_loss'], color='red', linewidth=1, linestyle='-')
            ax1.axhline(y=signal_info['take_profit'], color='green', linewidth=1, linestyle='-')
            
            # Draw support and resistance levels
            for level in self.support_resistance_levels:
                color = 'green' if level['type'] == 'support' else 'red'
                linestyle = '-' if level['strength'] >= 3 else '--'
                ax1.axhline(y=level['price'], color=color, linewidth=1, linestyle=linestyle, alpha=0.7)
                
                # Annotate the level
                ax1.text(0.01, level['price'], f"{level['type'].upper()} ({level['strength']})", 
                         transform=ax1.get_yaxis_transform(), color=color, fontsize=8)
            
            # Mark Price Action patterns
            for i in range(max(0, len(df_plot)-30), len(df_plot)):
                # Mark Bullish Engulfing
                if df_plot['bullish_engulfing'].iloc[i] == 1:
                    ax1.plot(i, df_plot['low'].iloc[i] * 0.99, '^', color='green', markersize=10)
                
                # Mark Bearish Engulfing
                if df_plot['bearish_engulfing'].iloc[i] == 1:
                    ax1.plot(i, df_plot['high'].iloc[i] * 1.01, 'v', color='red', markersize=10)
                
                # Mark Hammers
                if df_plot['hammer'].iloc[i] == 1:
                    ax1.plot(i, df_plot['low'].iloc[i] * 0.99, '^', color='lime', markersize=8)
                
                # Mark Shooting Stars
                if df_plot['shooting_star'].iloc[i] == 1:
                    ax1.plot(i, df_plot['high'].iloc[i] * 1.01, 'v', color='orange', markersize=8)
                
                # Mark Breakouts
                if df_plot['breakout_up'].iloc[i] == 1:
                    if df_plot['fakeout_up'].iloc[i] == 1:
                        ax1.text(i, df_plot['high'].iloc[i] * 1.02, 'FB', fontsize=8, color='orange', 
                                ha='center', va='bottom')
                    else:
                        ax1.text(i, df_plot['high'].iloc[i] * 1.02, 'BO', fontsize=8, color='green', 
                                ha='center', va='bottom')
                
                if df_plot['breakout_down'].iloc[i] == 1:
                    if df_plot['fakeout_down'].iloc[i] == 1:
                        ax1.text(i, df_plot['low'].iloc[i] * 0.98, 'FB', fontsize=8, color='orange', 
                                ha='center', va='top')
                    else:
                        ax1.text(i, df_plot['low'].iloc[i] * 0.98, 'BD', fontsize=8, color='red', 
                                ha='center', va='top')
            
            # Add a legend for the Price Action marks
            legend_text = "PA Marks: ^ = Bullish pattern, v = Bearish pattern\n"
            legend_text += "BO = Breakout, BD = Breakdown, FB = Failed Break"
            
            ax1.text(0.03, 0.01, legend_text, transform=ax1.transAxes, 
                    fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
                    
            # Save the figure to the buffer
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            
            plt.close(fig)  # Close the figure to free memory
            
            return buf
            
        except Exception as e:
            print(f"Error creating candlestick chart: {e}")
            traceback.print_exc()
            return None
            
#=========================================================================
# Telegram Bot Implementation
#=========================================================================

async def run_analysis(user_id, chat_id):
    """Run the crypto analysis and send results to user"""
    try:
        # Get user preferences
        symbol = user_preferences[user_id]["symbol"]
        timeframe = user_preferences[user_id]["timeframe"]
        
        # Create analyzer instance
        analyzer = SmartTradeAnalyzer(symbol=symbol, timeframe=timeframe, limit=DEFAULT_LIMIT)
        
        # Send status updates to keep the user informed
        status_msg = await bot.send_message(
            chat_id=chat_id,
            text=f"🔍 تحلیل {symbol} با بازه زمانی {timeframe} آغاز شد...\n\n⏳ در حال دریافت داده‌ها..."
        )
        
        # Step 1: Fetch market data
        df = analyzer.fetch_data()
        if df is None:
            await status_msg.edit_text("❌ خطا در دریافت داده‌ها. لطفاً دوباره تلاش کنید.")
            return
            
        await status_msg.edit_text(
            f"🔍 تحلیل {symbol} با بازه زمانی {timeframe}...\n\n"
            f"✅ داده‌ها دریافت شد.\n"
            f"⏳ در حال محاسبه شاخص‌های تکنیکال و الگوهای پرایس اکشن..."
        )
        
        # Step 2: Add technical indicators
        df = analyzer.add_technical_features()
        if df is None:
            await status_msg.edit_text("❌ خطا در محاسبه شاخص‌های تکنیکال. لطفاً دوباره تلاش کنید.")
            return
            
        await status_msg.edit_text(
            f"🔍 تحلیل {symbol} با بازه زمانی {timeframe}...\n\n"
            f"✅ داده‌ها دریافت شد.\n"
            f"✅ شاخص‌های تکنیکال محاسبه شد.\n"
            f"⏳ در حال آموزش مدل‌های هوش مصنوعی..."
        )
        
        # Step 3: Build AI models
        analyzer.build_advanced_model()
        
        await status_msg.edit_text(
            f"🔍 تحلیل {symbol} با بازه زمانی {timeframe}...\n\n"
            f"✅ داده‌ها دریافت شد.\n"
            f"✅ شاخص‌های تکنیکال محاسبه شد.\n"
            f"✅ مدل‌های هوش مصنوعی آموزش داده شد.\n"
            f"⏳ در حال تولید سیگنال معاملاتی..."
        )
        
        # Step 4: Generate trading signal
        signal_info = analyzer.generate_smart_signal()
        
        # Save the last signal for this user and symbol
        key = f"{user_id}_{symbol}"
        last_signals[key] = {
            "signal": signal_info["signal"],
            "time": datetime.now(),
            "entry": signal_info["entry"]
        }
        
        await status_msg.edit_text(
            f"🔍 تحلیل {symbol} با بازه زمانی {timeframe}...\n\n"
            f"✅ داده‌ها دریافت شد.\n"
            f"✅ شاخص‌های تکنیکال محاسبه شد.\n"
            f"✅ مدل‌های هوش مصنوعی آموزش داده شد.\n"
            f"✅ سیگنال معاملاتی تولید شد.\n"
            f"⏳ در حال ایجاد نمودار تحلیلی..."
        )
        
        # Step 5: Create analysis chart
        chart_buffer = analyzer.create_analysis_chart(signal_info)
        
        # Create detailed analysis message
        current_price = signal_info["entry"]
        signal_emoji = "🟢 خرید" if signal_info["signal"] == "BUY" else "🔴 فروش"
        confidence_emoji = "⭐⭐⭐" if signal_info["confidence"] == "HIGH" else "⭐⭐" if signal_info["confidence"] == "MEDIUM" else "⭐"
        
        message = (
            f"<b>📊 تحلیل جامع {symbol} ({timeframe})</b>\n\n"
            f"<b>سیگنال: {signal_emoji} (اطمینان: {confidence_emoji})</b>\n"
            f"قیمت فعلی: {current_price:,.2f}\n"
            f"حد ضرر: {signal_info['stop_loss']:,.2f}\n"
            f"حد سود: {signal_info['take_profit']:,.2f}\n"
            f"نسبت ریسک/ریوارد: {signal_info['risk_reward']:.1f}\n\n"
            
            f"<b>💹 تحلیل مبتنی بر:</b>\n"
            f"• وایکوف: {signal_info['wyckoff']}\n"
            f"• امواج الیوت: {signal_info['elliott']}\n"
            f"• امتیاز پرایس اکشن: {signal_info['price_action']['score']:.2f}\n\n"
            
            f"<b>📝 دلایل سیگنال:</b>\n"
        )
        
        # Add reasons
        reasons = signal_info["reason"].split(" | ")
        for idx, reason in enumerate(reasons[:7]):  # Limit to 7 reasons to avoid too long messages
            message += f"• {reason}\n"
            
        # If there are more reasons, add a note
        if len(reasons) > 7:
            message += f"• و {len(reasons) - 7} دلیل دیگر...\n"
            
        # Add price action signals
        pa_signals = signal_info["price_action"]["signals"]
        if pa_signals:
            message += "\n<b>🔍 الگوهای پرایس اکشن اخیر:</b>\n"
            for idx, signal in enumerate(pa_signals[:5]):  # Limit to 5 signals
                message += f"• {signal}\n"
                
            # If there are more signals, add a note
            if len(pa_signals) > 5:
                message += f"• و {len(pa_signals) - 5} الگوی دیگر...\n"
        
        message += (
            f"\n<b>⏰ زمان تحلیل:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"<i>این سیگنال صرفاً جنبه آموزشی دارد و توصیه معاملاتی نیست.</i>"
        )
        
        # Send chart if available
        if chart_buffer:
            await bot.send_photo(
                chat_id=chat_id,
                photo=chart_buffer,
                caption=message,
                parse_mode='HTML'
            )
        else:
            await bot.send_message(
                chat_id=chat_id,
                text=message + "\n\n⚠️ متأسفانه ایجاد نمودار با مشکل مواجه شد.",
                parse_mode='HTML'
            )
        
        # Delete status message
        await status_msg.delete()
        
        # Add inline keyboard for follow-up actions
        keyboard = [
            [
                InlineKeyboardButton("تحلیل مجدد", callback_data="analyze"),
                InlineKeyboardButton("تغییر تنظیمات", callback_data="settings")
            ],
            [
                InlineKeyboardButton("تحلیل ارز دیگر", callback_data="symbols"),
                InlineKeyboardButton("بازگشت به منوی اصلی", callback_data="back_to_menu")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await bot.send_message(
            chat_id=chat_id,
            text="آیا اقدام دیگری مد نظر دارید؟",
            reply_markup=reply_markup
        )
        
    except Exception as e:
        print(f"Error in run_analysis: {e}")
        traceback.print_exc()
        await bot.send_message(
            chat_id=chat_id,
            text=f"❌ خطایی در تحلیل رخ داد: {str(e)}\n\nلطفاً دوباره تلاش کنید."
        )

async def calculate_atr(symbol: str, timeframe: str = '1h', limit: int = 14):
    """Calculate ATR (Average True Range) for the given symbol and timeframe."""
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        atr = df['true_range'].rolling(window=limit).mean().iloc[-1]
        return atr
    except Exception as e:
        print(f"Error calculating ATR for {symbol}: {e}")
        return None

async def monitor_signal(symbol: str, signal_info: dict, chat_id: int):
    """Monitor a trading signal for entry, stop loss, and multiple ATR-based target hits"""
    try:
        exchange = ccxt.binance()
        signal_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate ATR for dynamic target setting
        atr = await calculate_atr(symbol)
        if atr is None:
            atr = 0
        
        # Define ATR-based targets and stop loss
        stop_loss = signal_info['entry'] - signal_tracking_config['atr_multiplier_stop_loss'] * atr if signal_info['signal'] == 'BUY' else signal_info['entry'] + signal_tracking_config['atr_multiplier_stop_loss'] * atr
        take_profit_1 = signal_info['entry'] + signal_tracking_config['atr_multiplier_take_profit_1'] * atr if signal_info['signal'] == 'BUY' else signal_info['entry'] - signal_tracking_config['atr_multiplier_take_profit_1'] * atr
        take_profit_2 = signal_info['entry'] + signal_tracking_config['atr_multiplier_take_profit_2'] * atr if signal_info['signal'] == 'BUY' else signal_info['entry'] - signal_tracking_config['atr_multiplier_take_profit_2'] * atr
        
        # Store signal information with ATR targets
        active_signals['signals'][signal_id] = {
            'symbol': symbol,
            'entry': signal_info['entry'],
            'stop_loss': stop_loss,
            'take_profit_1': take_profit_1,
            'take_profit_2': take_profit_2,
            'direction': signal_info['signal'],  # 'BUY' or 'SELL'
            'status': 'PENDING',
            'start_time': datetime.now(),
            'last_price': signal_info['entry'],
            'notified_targets': set()
        }
        
        # Send initial signal notification with ATR targets
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=f"🎯 سیگنال جدید برای {symbol}\n\n"
                         f"نوع: {'خرید 🟢' if signal_info['signal'] == 'BUY' else 'فروش 🔴'}\n"
                         f"قیمت ورود: {signal_info['entry']:.8f}\n"
                         f"حد ضرر: {stop_loss:.8f}\n"
                         f"حد سود 1: {take_profit_1:.8f}\n"
                         f"حد سود 2: {take_profit_2:.8f}\n"
                         f"نسبت ریسک/ریوارد: {signal_info['risk_reward']:.2f}",
                    parse_mode='HTML'
                )
                break  # If successful, exit the retry loop
            except telegram.error.TimedOut:
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    print(f"Failed to send initial signal for {symbol} after {max_retries} attempts")
            except Exception as e:
                print(f"Error sending initial signal for {symbol}: {e}")
                break  # Exit on non-timeout errors
        
        while active_signals['signals'].get(signal_id) and active_signals['signals'][signal_id]['status'] != 'CLOSED':
            try:
                # Fetch current price
                ticker = exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                signal = active_signals['signals'][signal_id]
                
                # Update last price
                active_signals['signals'][signal_id]['last_price'] = current_price
                
                # Check for entry
                if signal['status'] == 'PENDING':
                    entry_threshold = 0.001  # 0.1% threshold for entry
                    entry_price = signal['entry']
                    
                    if signal['direction'] == 'BUY':
                        if current_price <= entry_price * (1 + entry_threshold):
                            active_signals['signals'][signal_id]['status'] = 'ACTIVE'
                            for attempt in range(max_retries):
                                try:
                                    await bot.send_message(
                                        chat_id=chat_id,
                                        text=f"✅ ورود به معامله {symbol}\n"
                                             f"قیمت ورود: {current_price:.8f}",
                                        parse_mode='HTML'
                                    )
                                    break
                                except telegram.error.TimedOut:
                                    if attempt < max_retries - 1:
                                        await asyncio.sleep(retry_delay)
                                        continue
                                    else:
                                        print(f"Failed to send entry signal for {symbol} after {max_retries} attempts")
                                except Exception as e:
                                    print(f"Error sending entry signal for {symbol}: {e}")
                                    break
                    else:  # SELL
                        if current_price >= entry_price * (1 - entry_threshold):
                            active_signals['signals'][signal_id]['status'] = 'ACTIVE'
                            for attempt in range(max_retries):
                                try:
                                    await bot.send_message(
                                        chat_id=chat_id,
                                        text=f"✅ ورود به معامله {symbol}\n"
                                             f"قیمت ورود: {current_price:.8f}",
                                        parse_mode='HTML'
                                    )
                                    break
                                except telegram.error.TimedOut:
                                    if attempt < max_retries - 1:
                                        await asyncio.sleep(retry_delay)
                                        continue
                                    else:
                                        print(f"Failed to send entry signal for {symbol} after {max_retries} attempts")
                                except Exception as e:
                                    print(f"Error sending entry signal for {symbol}: {e}")
                                    break
                
                # Check for stop loss or take profit if signal is active
                elif signal['status'] == 'ACTIVE':
                    # Check stop loss
                    if signal['direction'] == 'BUY':
                        if current_price <= signal['stop_loss']:
                            active_signals['signals'][signal_id]['status'] = 'CLOSED'
                            for attempt in range(max_retries):
                                try:
                                    await bot.send_message(
                                        chat_id=chat_id,
                                        text=f"🛑 حد ضرر {symbol}\n"
                                             f"قیمت خروج: {current_price:.8f}\n"
                                             f"ضرر: {((current_price - signal['entry']) / signal['entry'] * 100):.2f}%",
                                        parse_mode='HTML'
                                    )
                                    break
                                except telegram.error.TimedOut:
                                    if attempt < max_retries - 1:
                                        await asyncio.sleep(retry_delay)
                                        continue
                                    else:
                                        print(f"Failed to send stop loss signal for {symbol} after {max_retries} attempts")
                                except Exception as e:
                                    print(f"Error sending stop loss signal for {symbol}: {e}")
                                    break
                            continue
                    else:  # SELL
                        if current_price >= signal['stop_loss']:
                            active_signals['signals'][signal_id]['status'] = 'CLOSED'
                            for attempt in range(max_retries):
                                try:
                                    await bot.send_message(
                                        chat_id=chat_id,
                                        text=f"🛑 حد ضرر {symbol}\n"
                                             f"قیمت خروج: {current_price:.8f}\n"
                                             f"ضرر: {((signal['entry'] - current_price) / signal['entry'] * 100):.2f}%",
                                        parse_mode='HTML'
                                    )
                                    break
                                except telegram.error.TimedOut:
                                    if attempt < max_retries - 1:
                                        await asyncio.sleep(retry_delay)
                                        continue
                                    else:
                                        print(f"Failed to send stop loss signal for {symbol} after {max_retries} attempts")
                                except Exception as e:
                                    print(f"Error sending stop loss signal for {symbol}: {e}")
                                    break
                            continue
                    
                    # Check take profit 1
                    if signal['direction'] == 'BUY':
                        if current_price >= signal['take_profit_1'] and 'tp1' not in signal['notified_targets']:
                            signal['notified_targets'].add('tp1')
                            for attempt in range(max_retries):
                                try:
                                    await bot.send_message(
                                        chat_id=chat_id,
                                        text=f"🎯 حد سود 1 {symbol} رسید\n"
                                             f"قیمت: {current_price:.8f}",
                                        parse_mode='HTML'
                                    )
                                    break
                                except telegram.error.TimedOut:
                                    if attempt < max_retries - 1:
                                        await asyncio.sleep(retry_delay)
                                        continue
                                    else:
                                        print(f"Failed to send TP1 signal for {symbol} after {max_retries} attempts")
                                except Exception as e:
                                    print(f"Error sending TP1 signal for {symbol}: {e}")
                                    break
                    else:  # SELL
                        if current_price <= signal['take_profit_1'] and 'tp1' not in signal['notified_targets']:
                            signal['notified_targets'].add('tp1')
                            for attempt in range(max_retries):
                                try:
                                    await bot.send_message(
                                        chat_id=chat_id,
                                        text=f"🎯 حد سود 1 {symbol} رسید\n"
                                             f"قیمت: {current_price:.8f}",
                                        parse_mode='HTML'
                                    )
                                    break
                                except telegram.error.TimedOut:
                                    if attempt < max_retries - 1:
                                        await asyncio.sleep(retry_delay)
                                        continue
                                    else:
                                        print(f"Failed to send TP1 signal for {symbol} after {max_retries} attempts")
                                except Exception as e:
                                    print(f"Error sending TP1 signal for {symbol}: {e}")
                                    break
                    
                    # Check take profit 2
                    if signal['direction'] == 'BUY':
                        if current_price >= signal['take_profit_2'] and 'tp2' not in signal['notified_targets']:
                            signal['notified_targets'].add('tp2')
                            for attempt in range(max_retries):
                                try:
                                    await bot.send_message(
                                        chat_id=chat_id,
                                        text=f"🎯 حد سود 2 {symbol} رسید\n"
                                             f"قیمت: {current_price:.8f}",
                                        parse_mode='HTML'
                                    )
                                    break
                                except telegram.error.TimedOut:
                                    if attempt < max_retries - 1:
                                        await asyncio.sleep(retry_delay)
                                        continue
                                    else:
                                        print(f"Failed to send TP2 signal for {symbol} after {max_retries} attempts")
                                except Exception as e:
                                    print(f"Error sending TP2 signal for {symbol}: {e}")
                                    break
                    else:  # SELL
                        if current_price <= signal['take_profit_2'] and 'tp2' not in signal['notified_targets']:
                            signal['notified_targets'].add('tp2')
                            for attempt in range(max_retries):
                                try:
                                    await bot.send_message(
                                        chat_id=chat_id,
                                        text=f"🎯 حد سود 2 {symbol} رسید\n"
                                             f"قیمت: {current_price:.8f}",
                                        parse_mode='HTML'
                                    )
                                    break
                                except telegram.error.TimedOut:
                                    if attempt < max_retries - 1:
                                        await asyncio.sleep(retry_delay)
                                        continue
                                    else:
                                        print(f"Failed to send TP2 signal for {symbol} after {max_retries} attempts")
                                except Exception as e:
                                    print(f"Error sending TP2 signal for {symbol}: {e}")
                                    break
                
                # Sleep for the price update interval
                await asyncio.sleep(signal_tracking_config['price_update_interval'] * 60)
                
            except Exception as e:
                print(f"Error monitoring {symbol}: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying
                
    except Exception as e:
        print(f"Error in monitor_signal: {e}")
        traceback.print_exc()

async def analyze_all_top_coins(chat_id: int):
    """Analyze all top coins and generate signals"""
    try:
        for symbol in TOP_COINS:
            try:
                # Create analyzer instance
                analyzer = SmartTradeAnalyzer(symbol=symbol, timeframe='1h', limit=DEFAULT_LIMIT)
                
                # Fetch and analyze data
                df = analyzer.fetch_data()
                if df is None:
                    print(f"Warning: No data fetched for {symbol}, skipping.")
                    continue
                
                df = analyzer.add_technical_features()
                if df is None:
                    print(f"Warning: Failed to add technical features for {symbol}, skipping.")
                    continue
                
                analyzer.build_advanced_model()
                signal_info = analyzer.generate_smart_signal()
                
                # If signal confidence is high enough, start monitoring
                if signal_info['confidence'] in ['HIGH', 'MEDIUM']:
                    asyncio.create_task(monitor_signal(symbol, signal_info, chat_id))
                
                # Wait briefly between analyses to avoid rate limits
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue
                
    except Exception as e:
        print(f"Error in analyze_all_top_coins: {e}")
        traceback.print_exc()

async def schedule_alerts(user_id, chat_id):
    """Schedule regular alerts for a user"""
    try:
        while user_preferences[user_id]["alert_enabled"]:
            # Run analysis for all top coins
            await analyze_all_top_coins(chat_id)
            
            # Update last alert time
            user_preferences[user_id]["last_alert"] = datetime.now()
            
            # Get interval in minutes
            interval_minutes = int(user_preferences[user_id]["alert_interval"] * 60)
            
            # Wait for the interval
            for _ in range(interval_minutes):
                if not user_preferences[user_id]["alert_enabled"]:
                    break
                await asyncio.sleep(60)  # Check every minute if alerts still enabled
    except Exception as e:
        print(f"Error in schedule_alerts: {e}")
        traceback.print_exc()
        
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message when the command /start is issued."""
    user = update.effective_user
    
    # Initialize user preferences if not exists
    if user.id not in user_preferences:
        user_preferences[user.id] = {
            "symbol": DEFAULT_SYMBOL,
            "timeframe": DEFAULT_TIMEFRAME,
            "alert_enabled": False,
            "alert_interval": 4,
            "last_alert": None
        }
    
    welcome_text = (
        f"👋 سلام {user.first_name}!\n\n"
        "🤖 به ربات تحلیل ارز دیجیتال خوش آمدید.\n\n"
        "🔹 <b>قابلیت‌های ربات:</b>\n"
        "• تحلیل تکنیکال پیشرفته\n"
        "• تحلیل پرایس اکشن\n"
        "• تحلیل وایکوف و امواج الیوت\n"
        "• پیش‌بینی با هوش مصنوعی\n"
        "• نمودارهای تحلیلی\n"
        "• هشدارهای خودکار\n\n"
        "🔸 <b>دستورات اصلی:</b>\n"
        "/analyze - شروع تحلیل جدید\n"
        "/symbols - مشاهده جفت ارزها\n"
        "/timeframes - مشاهده بازه‌های زمانی\n"
        "/settings - تنظیمات\n"
        "/help - راهنما\n\n"
        "برای شروع، از دستور /analyze استفاده کنید."
    )
    
    keyboard = [
        [
            InlineKeyboardButton("شروع تحلیل", callback_data="analyze"),
            InlineKeyboardButton("انتخاب ارز", callback_data="symbols")
        ],
        [
            InlineKeyboardButton("تایم فریم", callback_data="timeframes"),
            InlineKeyboardButton("تنظیمات", callback_data="settings")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(welcome_text, reply_markup=reply_markup, parse_mode='HTML')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a help message when the command /help is issued."""
    help_text = (
        "🔹 <b>راهنمای دستورات</b> 🔹\n\n"
        "• /start - شروع مجدد ربات\n"
        "• /analyze - شروع تحلیل جدید\n"
        "• /symbols - مشاهده لیست جفت ارزها\n"
        "• /timeframes - مشاهده بازه‌های زمانی\n"
        "• /settings - تنظیمات ربات\n"
        "• /help - نمایش این راهنما\n\n"
        
        "🔸 <b>نکات مهم:</b>\n"
        "• سیگنال‌های تولید شده صرفاً جنبه آموزشی دارند\n"
        "• برای نتایج بهتر، از تایم‌فریم‌های بالاتر استفاده کنید\n"
        "• می‌توانید هشدارهای خودکار را در بخش تنظیمات فعال کنید\n"
        "• حتماً حد ضرر و حد سود پیشنهادی را رعایت کنید\n\n"
        
        "🔸 <b>پشتیبانی:</b>\n"
        "در صورت بروز مشکل یا نیاز به راهنمایی، با ادمین در تماس باشید."
    )
    
    keyboard = [
        [
            InlineKeyboardButton("شروع تحلیل", callback_data="analyze"),
            InlineKeyboardButton("تنظیمات", callback_data="settings")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(help_text, reply_markup=reply_markup, parse_mode='HTML')

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start the analysis process."""
    user = update.effective_user
    chat_id = update.effective_chat.id
    
    if user.id not in user_preferences:
        user_preferences[user.id] = {
            "symbol": DEFAULT_SYMBOL,
            "timeframe": DEFAULT_TIMEFRAME,
            "alert_enabled": False,
            "alert_interval": 4,
            "last_alert": None
        }
    
    # Create keyboard with analysis options
    keyboard = [
        [
            InlineKeyboardButton("تحلیل تک ارز", callback_data="analyze_single"),
            InlineKeyboardButton("تحلیل همه ارزها", callback_data="analyze_all")
        ],
        [
            InlineKeyboardButton("بازگشت به منو", callback_data="back_to_menu")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🔍 لطفاً نوع تحلیل را انتخاب کنید:\n\n"
        "• تحلیل تک ارز: تحلیل کامل یک ارز\n"
        "• تحلیل همه ارزها: تحلیل و پیگیری ده ارز برتر",
        reply_markup=reply_markup
    )

async def symbols_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show available trading pairs."""
    symbols_text = (
        "🔹 <b>جفت ارزهای قابل تحلیل</b> 🔹\n\n"
        "<b>ارزهای اصلی:</b>\n"
        "• BTC/USDT - بیت‌کوین\n"
        "• ETH/USDT - اتریوم\n"
        "• BNB/USDT - بایننس کوین\n"
        "• XRP/USDT - ریپل\n"
        "• ADA/USDT - کاردانو\n"
        "• SOL/USDT - سولانا\n"
        "• DOT/USDT - پولکادات\n"
        "• DOGE/USDT - دوج‌کوین\n\n"
        "<i>برای تحلیل سایر ارزها با ادمین در تماس باشید.</i>"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("BTC/USDT", callback_data="symbol_BTC/USDT"),
            InlineKeyboardButton("ETH/USDT", callback_data="symbol_ETH/USDT")
        ],
        [
            InlineKeyboardButton("BNB/USDT", callback_data="symbol_BNB/USDT"),
            InlineKeyboardButton("XRP/USDT", callback_data="symbol_XRP/USDT")
        ],
        [
            InlineKeyboardButton("بازگشت به منو", callback_data="back_to_menu")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(symbols_text, reply_markup=reply_markup, parse_mode='HTML')

async def timeframes_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show available timeframes."""
    timeframes_text = (
        "🔹 <b>بازه‌های زمانی تحلیل</b> 🔹\n\n"
        "<b>🔄 تایم‌فریم‌های اسکالپ:</b>\n"
        "• 1m - یک دقیقه (مناسب برای اسکالپ)\n"
        "• 3m - سه دقیقه (مناسب برای اسکالپ)\n"
        "• 5m - پنج دقیقه (مناسب برای نوسان‌گیری سریع)\n\n"
        "<b>📊 تایم‌فریم‌های نوسان‌گیری:</b>\n"
        "• 15m - پانزده دقیقه (نوسان‌گیری کوتاه مدت)\n"
        "• 30m - سی دقیقه (نوسان‌گیری میان مدت)\n"
        "• 1h - یک ساعت (تحلیل روند)\n"
        "• 2h - دو ساعت (تحلیل روند)\n\n"
        "<b>📈 تایم‌فریم‌های تحلیلی:</b>\n"
        "• 4h - چهار ساعت (تحلیل تکنیکال)\n"
        "• 6h - شش ساعت (تحلیل تکنیکال)\n"
        "• 8h - هشت ساعت (تحلیل روند اصلی)\n\n"
        "<b>📉 تایم‌فریم‌های بلندمدت:</b>\n"
        "• 12h - دوازده ساعت (روند اصلی)\n"
        "• 1d - روزانه (تحلیل کلی بازار)\n"
        "• 3d - سه روزه (روند بلندمدت)\n"
        "• 1w - هفتگی (تحلیل کلان)\n"
        "• 1M - ماهانه (دید بلندمدت)\n\n"
        "💡 <i>راهنمای انتخاب تایم‌فریم:</i>\n"
        "• برای اسکالپ: 1m تا 5m\n"
        "• برای نوسان‌گیری: 15m تا 1h\n"
        "• برای ترید میان‌مدت: 2h تا 6h\n"
        "• برای سرمایه‌گذاری: 1d تا 1M"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("1m", callback_data="timeframe_1m"),
            InlineKeyboardButton("3m", callback_data="timeframe_3m"),
            InlineKeyboardButton("5m", callback_data="timeframe_5m")
        ],
        [
            InlineKeyboardButton("15m", callback_data="timeframe_15m"),
            InlineKeyboardButton("30m", callback_data="timeframe_30m"),
            InlineKeyboardButton("1h", callback_data="timeframe_1h")
        ],
        [
            InlineKeyboardButton("2h", callback_data="timeframe_2h"),
            InlineKeyboardButton("4h", callback_data="timeframe_4h"),
            InlineKeyboardButton("6h", callback_data="timeframe_6h")
        ],
        [
            InlineKeyboardButton("1d", callback_data="timeframe_1d"),
            InlineKeyboardButton("3d", callback_data="timeframe_3d"),
            InlineKeyboardButton("1w", callback_data="timeframe_1w")
        ],
        [
            InlineKeyboardButton("بازگشت به منو", callback_data="back_to_menu")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(timeframes_text, reply_markup=reply_markup, parse_mode='HTML')

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show current settings."""
    # Handle both direct commands and callback queries
    if update.callback_query:
        user = update.callback_query.from_user
        message = update.callback_query.message
    else:
        user = update.effective_user
        message = update.message
    
    if user.id not in user_preferences:
        user_preferences[user.id] = {
            "symbol": DEFAULT_SYMBOL,
            "timeframe": DEFAULT_TIMEFRAME,
            "alert_enabled": False,
            "alert_interval": 4,
            "last_alert": None
        }
    
    settings_text = (
        f"🔹 <b>تنظیمات فعلی</b> 🔹\n\n"
        f"جفت ارز: {user_preferences[user.id]['symbol']}\n"
        f"بازه زمانی: {user_preferences[user.id]['timeframe']}\n"
        f"هشدار خودکار: {'فعال' if user_preferences[user.id]['alert_enabled'] else 'غیرفعال'}\n"
        f"فاصله هشدارها: {user_preferences[user.id]['alert_interval']} ساعت\n"
        f"آخرین هشدار: {user_preferences[user.id]['last_alert'].strftime('%Y-%m-%d %H:%M') if user_preferences[user.id]['last_alert'] else 'هنوز هشداری ارسال نشده'}"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("تغییر جفت ارز", callback_data="symbols"),
            InlineKeyboardButton("تغییر بازه زمانی", callback_data="timeframes")
        ],
        [
            InlineKeyboardButton("تنظیم فاصله هشدارها", callback_data="set_interval"),
            InlineKeyboardButton(
                f"{'غیرفعال کردن' if user_preferences[user.id]['alert_enabled'] else 'فعال کردن'} هشدار", 
                callback_data="toggle_alert"
            )
        ],
        [
            InlineKeyboardButton("بازگشت به منو اصلی", callback_data="back_to_menu")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if update.callback_query:
        await update.callback_query.edit_message_text(
            text=settings_text,
            reply_markup=reply_markup,
            parse_mode='HTML'
        )
    else:
        await message.reply_text(
            text=settings_text,
            reply_markup=reply_markup,
            parse_mode='HTML'
        )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button callbacks."""
    query = update.callback_query
    user = query.from_user
    chat_id = query.message.chat_id
    
    await query.answer()  # Acknowledge the button press
    
    if query.data == "analyze":
        # Show analysis options keyboard
        keyboard = [
            [
                InlineKeyboardButton("تحلیل تک ارز", callback_data="analyze_single"),
                InlineKeyboardButton("تحلیل همه ارزها", callback_data="analyze_all")
            ],
            [
                InlineKeyboardButton("بازگشت به منو", callback_data="back_to_menu")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "🔍 لطفاً نوع تحلیل را انتخاب کنید:\n\n"
            "• تحلیل تک ارز: تحلیل کامل یک ارز\n"
            "• تحلیل همه ارزها: تحلیل و پیگیری ده ارز برتر",
            reply_markup=reply_markup
        )
    
    elif query.data == "analyze_single":
        await run_analysis(user.id, chat_id)
    
    elif query.data == "analyze_all":
        status_msg = await bot.send_message(
            chat_id=chat_id,
            text="🔄 در حال شروع تحلیل ده ارز برتر...\n"
                 "این فرآیند ممکن است چند دقیقه طول بکشد."
        )
        
        # Start analyzing all top coins
        await analyze_all_top_coins(chat_id)
        
        await status_msg.edit_text(
            "✅ تحلیل ده ارز برتر آغاز شد.\n\n"
            "• سیگنال‌های معتبر به محض شناسایی ارسال خواهند شد\n"
            "• وضعیت معاملات (ورود، حد ضرر و حد سود) به صورت خودکار اطلاع‌رسانی می‌شود\n"
            "• برای توقف پیگیری سیگنال‌ها، از منوی تنظیمات استفاده کنید"
        )
    
    elif query.data == "settings":
        await settings_command(update, context)
    
    elif query.data == "back_to_menu":
        # Create a new update object for start command
        new_update = Update(update_id=update.update_id, message=query.message)
        await start(new_update, context)
    
    elif query.data == "symbols":
        # Create a new update object for symbols command
        new_update = Update(update_id=update.update_id, message=query.message)
        await symbols_command(new_update, context)
    
    elif query.data == "timeframes":
        # Create a new update object for timeframes command
        new_update = Update(update_id=update.update_id, message=query.message)
        await timeframes_command(new_update, context)
    
    elif query.data.startswith("symbol_"):
        symbol = query.data.split("_")[1]
        user_preferences[user.id]["symbol"] = symbol
        await query.edit_message_text(
            f"✅ جفت ارز به {symbol} تغییر کرد.\n\nبرای شروع تحلیل، دستور /analyze را ارسال کنید.",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("شروع تحلیل", callback_data="analyze")
            ]])
        )
    
    elif query.data.startswith("timeframe_"):
        timeframe = query.data.split("_")[1]
        user_preferences[user.id]["timeframe"] = timeframe
        await query.edit_message_text(
            f"✅ بازه زمانی به {timeframe} تغییر کرد.\n\nبرای شروع تحلیل، دستور /analyze را ارسال کنید.",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("شروع تحلیل", callback_data="analyze")
            ]])
        )
    
    elif query.data == "set_interval":
        intervals = [
            [
                InlineKeyboardButton("15 دقیقه", callback_data="interval_15"),
                InlineKeyboardButton("30 دقیقه", callback_data="interval_30")
            ],
            [
                InlineKeyboardButton("1 ساعت", callback_data="interval_60"),
                InlineKeyboardButton("2 ساعت", callback_data="interval_120")
            ],
            [
                InlineKeyboardButton("4 ساعت", callback_data="interval_240"),
                InlineKeyboardButton("8 ساعت", callback_data="interval_480")
            ],
            [
                InlineKeyboardButton("بازگشت به تنظیمات", callback_data="settings")
            ]
        ]
        await query.edit_message_text(
            "⚙️ لطفاً فاصله زمانی بین هشدارها را انتخاب کنید:",
            reply_markup=InlineKeyboardMarkup(intervals)
        )
    
    elif query.data.startswith("interval_"):
        minutes = int(query.data.split("_")[1])
        user_preferences[user.id]["alert_interval"] = minutes / 60  # Convert to hours
        await query.edit_message_text(
            f"✅ فاصله هشدارها به {minutes} دقیقه تغییر کرد.",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("بازگشت به تنظیمات", callback_data="settings")
            ]])
        )
    
    elif query.data == "toggle_alert":
        user_preferences[user.id]["alert_enabled"] = not user_preferences[user.id]["alert_enabled"]
        
        if user_preferences[user.id]["alert_enabled"]:
            # Start alert scheduling in background
            asyncio.create_task(schedule_alerts(user.id, chat_id))
            interval_minutes = int(user_preferences[user.id]["alert_interval"] * 60)
            await query.edit_message_text(
                "✅ هشدارهای خودکار فعال شد.\n"
                f"هر {interval_minutes} دقیقه تحلیل جدید دریافت خواهید کرد.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("غیرفعال کردن هشدار", callback_data="toggle_alert")],
                    [InlineKeyboardButton("تغییر فاصله هشدارها", callback_data="set_interval")],
                    [InlineKeyboardButton("بازگشت به تنظیمات", callback_data="settings")]
                ])
            )
        else:
            await query.edit_message_text(
                "✅ هشدارهای خودکار غیرفعال شد.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("فعال کردن هشدار", callback_data="toggle_alert")],
                    [InlineKeyboardButton("بازگشت به تنظیمات", callback_data="settings")]
                ])
            )

def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CommandHandler("symbols", symbols_command))
    application.add_handler(CommandHandler("timeframes", timeframes_command))
    application.add_handler(CommandHandler("settings", settings_command))
    
    # Button callback handler
    application.add_handler(CallbackQueryHandler(button_callback))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)

# Create bot instance for direct use
bot = Application.builder().token(TELEGRAM_BOT_TOKEN).build().bot

if __name__ == '__main__':
    main()
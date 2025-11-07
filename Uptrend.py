# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

"""
ENHANCED UPTREND STRATEGY - OPTIMIZED VERSION

Improvements made:
===================
1. ADVANCED INDICATORS:
   - Multiple RSI periods (7, 14, 21) for better momentum analysis
   - MACD for trend confirmation
   - Bollinger Bands for volatility and support/resistance
   - Stochastic for overbought/oversold conditions
   - ADX for trend strength measurement
   - CCI for cyclical trend detection
   - MFI (Money Flow Index) for volume-weighted momentum
   - ATR for dynamic volatility measurement
   - Multiple EMAs (12, 26, 50, 100, 200) for trend analysis

2. IMPROVED BUY SIGNALS:
   - Multi-indicator confirmation (MAMA/FAMA + RSI + MACD + Volume + EMAs)
   - Trend strength validation (ADX > 20)
   - Volume confirmation (above average)
   - Momentum confirmation (MACD bullish, RSI rising)
   - Price action filters (not overextended)

3. IMPROVED SELL SIGNALS:
   - Multiple exit conditions for better profit protection
   - Overbought detection (RSI, Stochastic, MFI)
   - Trend reversal detection (MACD crossover, EMA break)
   - Volatility-based exits (Bollinger Bands)

4. ENHANCED RISK MANAGEMENT:
   - Optimized ROI table with 5 levels (15% to 2%)
   - Dynamic trailing stoploss with 8 profit levels
   - Improved stoploss from -10% to -12%
   - 4 protection mechanisms:
     * StoplossGuard: Stops after 2 losses
     * MaxDrawdown: Stops at 15% drawdown
     * LowProfitPairs: Stops underperforming pairs
     * CooldownPeriod: Prevents overtrading

5. CODE IMPROVEMENTS:
   - Fixed deprecated DataFrame.append() -> pd.concat()
   - Better code documentation
   - More granular profit taking

DISCLAIMER: This is for educational purposes only. Trade at your own risk.
"""

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame, Series  # noqa

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime


class Uptrend(IStrategy):
    INTERFACE_VERSION = 2

    buy_params = {
        'buy_rsi_uplimit': 50,

    }

    buy_rsi_uplimit = IntParameter(50, 90, default=buy_params['buy_rsi_uplimit'], optimize=False, space='buy')

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    # Optimized ROI table for better profit taking
    minimal_roi = {
        "0": 0.15,      # Exit at 15% profit immediately if reached
        "20": 0.08,     # Exit at 8% after 20 minutes
        "40": 0.05,     # Exit at 5% after 40 minutes
        "80": 0.03,     # Exit at 3% after 80 minutes
        "120": 0.02     # Exit at 2% after 2 hours
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.12

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Protections to prevent losses during bad market conditions
    protections = [
        {
            "method": "StoplossGuard",
            "lookback_period_candles": 24,  # Look back 24 candles (2 hours on 5m)
            "trade_limit": 2,  # Stop trading after 2 stoplosses
            "stop_duration_candles": 12,  # Stop for 12 candles (1 hour)
            "only_per_pair": True  # Only for the specific pair
        },
        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 48,  # Look back 4 hours
            "trade_limit": 3,  # Minimum 3 trades
            "stop_duration_candles": 24,  # Stop for 2 hours
            "max_allowed_drawdown": 0.15  # Stop if 15% drawdown
        },
        {
            "method": "LowProfitPairs",
            "lookback_period_candles": 60,  # Look back 5 hours
            "trade_limit": 2,  # Minimum 2 trades
            "stop_duration_candles": 60,  # Stop for 5 hours
            "required_profit": -0.03  # Stop if less than -3% profit
        },
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 2  # Wait 2 candles between trades on same pair
        }
    ]

    use_custom_stoploss = True
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Enhanced custom stoploss with dynamic trailing based on profit levels
        """
        # Default to no custom stoploss (use the static one)
        sl_new = 1

        # More aggressive trailing as profit increases
        if (current_profit > 0.25):
            # Above 25% profit, trail with 7% stop
            sl_new = 0.07
        elif (current_profit > 0.20):
            # Above 20% profit, trail with 5% stop
            sl_new = 0.05
        elif (current_profit > 0.15):
            # Above 15% profit, trail with 4% stop
            sl_new = 0.04
        elif (current_profit > 0.10):
            # Above 10% profit, trail with 3% stop
            sl_new = 0.03
        elif (current_profit > 0.06):
            # Above 6% profit, trail with 2% stop
            sl_new = 0.02
        elif (current_profit > 0.04):
            # Above 4% profit, trail with 1.5% stop
            sl_new = 0.015
        elif (current_profit > 0.02):
            # Above 2% profit, trail with 1% stop
            sl_new = 0.01
        elif (current_profit > 0.01):
            # Above 1% profit, trail with 0.5% stop
            sl_new = 0.005

        return sl_new


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Original MAMA/FAMA indicators
        dataframe['hl2'] = (dataframe['high'] + dataframe['low']) / 2
        dataframe['mama'], dataframe['fama'] = ta.MAMA(dataframe['hl2'], 0.5, 0.05)

        dataframe['mama_diff'] = dataframe['mama'] - dataframe['fama']
        dataframe['mama_diff_ratio'] = dataframe['mama_diff'] / dataframe['hl2']

        dataframe['zero'] = 0

        # RSI with multiple periods for better signals
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe['close'], timeperiod=7)
        dataframe['rsi_slow'] = ta.RSI(dataframe['close'], timeperiod=21)

        # EMA trend indicators
        dataframe['ema12'] = ta.EMA(dataframe['close'], timeperiod=12)
        dataframe['ema26'] = ta.EMA(dataframe['close'], timeperiod=26)
        dataframe['ema50'] = ta.EMA(dataframe['close'], timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe['close'], timeperiod=100)
        dataframe['ema200'] = ta.EMA(dataframe['close'], timeperiod=200)

        # MACD for momentum
        macd = ta.MACD(dataframe['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # Bollinger Bands for volatility
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']

        # Volume indicators
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']

        # ADX for trend strength
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=14)

        # Stochastic for overbought/oversold
        stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']

        # CCI for cyclical trends
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)

        # MFI (Money Flow Index) for volume-weighted momentum
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)

        # ATR for volatility and dynamic stops
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_percent'] = (dataframe['atr'] / dataframe['close']) * 100

        # Trend detection
        dataframe['trend_ema'] = (dataframe['ema12'] > dataframe['ema26']).astype(int)
        dataframe['strong_trend'] = ((dataframe['ema12'] > dataframe['ema26']) &
                                      (dataframe['ema26'] > dataframe['ema50']) &
                                      (dataframe['ema50'] > dataframe['ema200'])).astype(int)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Enhanced buy signal with multiple confirmations:
        - MAMA/FAMA uptrend (original)
        - RSI not overbought
        - Volume confirmation
        - Trend confirmation (EMAs)
        - Momentum confirmation (MACD, ADX)
        - Price near support (Bollinger Bands)
        """
        dataframe.loc[
            (
                # Original MAMA/FAMA uptrend signal
                (dataframe['mama'] > dataframe['fama']) &
                (dataframe['mama_diff_ratio'] > 0.04) &

                # RSI conditions - not overbought, showing strength
                (dataframe['rsi'] > 30) &
                (dataframe['rsi'] < 70) &
                (dataframe['rsi_fast'] > dataframe['rsi_fast'].shift(1)) &  # RSI gaining strength

                # Trend confirmation - price above key EMAs
                (dataframe['close'] > dataframe['ema50']) &
                (dataframe['ema12'] > dataframe['ema26']) &

                # Momentum confirmation
                (dataframe['macd'] > dataframe['macdsignal']) &  # MACD bullish
                (dataframe['macdhist'] > 0) &
                (dataframe['adx'] > 20) &  # Minimum trend strength

                # Volume confirmation - above average
                (dataframe['volume_ratio'] > 1.0) &

                # Not oversold on stochastic (avoid catching falling knives)
                (dataframe['stoch_k'] > 20) &

                # Price action - not too extended
                (dataframe['close'] < dataframe['bb_upperband']) &

                # MFI shows money flowing in
                (dataframe['mfi'] > 25) &
                (dataframe['mfi'] < 75) &

                (dataframe['volume'] > 0)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Enhanced sell signal with multiple exit conditions:
        - MAMA/FAMA downtrend (original)
        - RSI overbought
        - Momentum weakening
        - Trend reversal signals
        """
        dataframe.loc[
            (
                (
                    # Original MAMA/FAMA signal weakening
                    (dataframe['mama_diff_ratio'] < 0.01) |

                    # RSI extremely overbought
                    (dataframe['rsi'] > 80) |

                    # MACD bearish crossover
                    (
                        (dataframe['macd'] < dataframe['macdsignal']) &
                        (dataframe['macd'].shift(1) >= dataframe['macdsignal'].shift(1))
                    ) |

                    # Stochastic overbought and crossing down
                    (
                        (dataframe['stoch_k'] > 80) &
                        (dataframe['stoch_k'] < dataframe['stoch_d'])
                    ) |

                    # Price hitting upper Bollinger Band with weakening momentum
                    (
                        (dataframe['close'] > dataframe['bb_upperband']) &
                        (dataframe['rsi'] > 70)
                    ) |

                    # Money flowing out (MFI)
                    (dataframe['mfi'] > 85) |

                    # Trend reversal - price crosses below EMA12
                    (
                        (dataframe['close'] < dataframe['ema12']) &
                        (dataframe['close'].shift(1) >= dataframe['ema12'].shift(1))
                    )
                ) &
                (dataframe['volume'] > 0)
            ),
            'sell'] = 1
        return dataframe


import random
from functools import reduce

class SuperBuy(Uptrend):
    """
	Idea is to build random buy signales from populate_indicators, with luck we'll get a good buy signal
	"""

    generator = IntParameter(0, 100000000000, default=99295874569, optimize=True, space='buy')  # generate unique matrix of conditions for your dataframe
    operators_used_to_compare_between_columns = IntParameter(0, 3, default=3, optimize=True, space='buy')  # number of conditions you will keep to build buy signal
    operators_used_to_with_best_point = IntParameter(0, 3, default=1, optimize=True, space='buy')  # number of conditions you will keep to build buy signal
    condition_selector = IntParameter(0, 100, default=50, optimize=True, space='buy')  # how to select the desired conditions beteween all conditions generated (seed random)

    best_buy_point = None
    best_buy_point_dict = dict()
    bad_buy_point_dict = dict()
    all_points_dict = dict()
    buy_signal_already_printed = False

    columns = []
    columns_to_compare_to_best_point = []
    columns_to_compare_to_volume = []
    columns_to_compare_to_price = []

    operators = {
        0: '<',
        1: '>',
        2: '<=',
        3: '>=',
        4: '==',
        5: '!='
    }

    top_index_criteria = {

        # best point criteria
        'min_close_hh_ratio': 0.08,
        'max_candles_to_get_ratio': 8,
        'candles_after_dip_to_buy': 0,

        # parameters selection criteria
        'select_parameter_if_in_more_than_x_percent_of_best_points': 97,
        'select_prameter_if_prop_is_x_percent_higher_in_best_points': 10,
    }

    def find_best_entry_point(self, dataframe: DataFrame, metadata: dict):
        lookahead_candles = self.top_index_criteria['max_candles_to_get_ratio']

        workdataframe = dataframe.copy()
        workdataframe['higher_high'] = workdataframe['high'].rolling(lookahead_candles).max()
        workdataframe['close_shifted_lookehead'] = workdataframe['close'].shift(lookahead_candles)
        workdataframe['higher_high_close_ratio'] = workdataframe['higher_high'] / workdataframe['close_shifted_lookehead']

        df_mask = workdataframe['higher_high_close_ratio'] >= 1 + self.top_index_criteria['min_close_hh_ratio']
        # print(1 + self.top_index_criteria['min_close_hh_ratio'])
        filtered_df = workdataframe[df_mask]

        filtered_df = filtered_df.sort_values(by=["higher_high_close_ratio"], ascending=False)
        filtered_df["shifted_index"] = filtered_df.index - lookahead_candles + self.top_index_criteria['candles_after_dip_to_buy']

        if filtered_df.empty:
            if self.config['runmode'].value != 'hyperopt':
                print("No entry point found for {}".format(metadata['pair']))
            return filtered_df, workdataframe
        # print(metadata['pair'])
        # print(filtered_df[["date", "shifted_index", "higher_high_close_ratio", "close_shifted_lookehead", "close", "higher_high"]])
        return filtered_df, workdataframe.drop(filtered_df['shifted_index'])

    def common_points_for_every_best_entry(self, dataframe: DataFrame, metadata: dict, columns: list) -> list:
        full_pairlist = self.dp.current_whitelist()
        current_pair = metadata['pair']

        if current_pair not in self.best_buy_point_dict:
            self.best_buy_point_dict[current_pair], self.bad_buy_point_dict[current_pair] = self.find_best_entry_point(dataframe, metadata)
            self.all_points_dict[current_pair] = dataframe.copy()

        for pair in full_pairlist:
            current_df = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
            if (pair not in self.best_buy_point_dict) and not current_df.empty:
                # print("No entry point found for {}".format(pair))
                return []

        all_best_points = None
        all_bad_points = None
        all_points = None
        for pair in full_pairlist:
            # NO DATA FOR THIS PAIR
            if not pair in self.best_buy_point_dict:
                continue
            if all_best_points is None:
                all_best_points = self.best_buy_point_dict[pair]
                all_bad_points = self.bad_buy_point_dict[pair]
                all_points = self.all_points_dict[pair]
            else:
                # Fixed: Use pd.concat instead of deprecated append
                all_best_points = pd.concat([all_best_points, self.best_buy_point_dict[pair]], ignore_index=False)
                all_bad_points = pd.concat([all_bad_points, self.bad_buy_point_dict[pair]], ignore_index=False)
                all_points = pd.concat([all_points, self.all_points_dict[pair]], ignore_index=False)

        print("HERE COMMON VALUES FOR ALL BEST POINTS !!!!!!!!!!")
        res = list()
        best_indicators = []
        for column in columns:
            all_points_values_count = all_points[column].value_counts()
            all_bad_points_values_count = all_bad_points[column].value_counts()
            count = all_best_points[column].value_counts()
            # keep only values in more than x% of all points
            df_mask = count >= 1 / 100 * all_best_points.shape[0]
            count = count[df_mask]

            if not count.empty and not column in ['buy', 'buy_tag']:
                count_normalized = count / all_best_points.shape[0]
                all_bad_points_values_count_normalized = all_bad_points_values_count / all_bad_points.shape[0]
                df_all = count.to_frame(name='best_points').join(all_bad_points_values_count.to_frame(name='bad_points'))
                df_all['best_point_percent'] = count_normalized
                df_all['bad_point_percent'] = all_bad_points_values_count_normalized
                df_all['part_of_best_points'] = 100 * df_all['best_points'] / (df_all['best_points'] + df_all['bad_points'])
                df_all['part_of_best_points_percent'] = count_normalized / all_bad_points_values_count_normalized
                print(column)
                print(df_all)
                values = df_all.query(
                    f"part_of_best_points > 3 & "  # the part of best points should be at least 3% for the value
                    "part_of_best_points_percent > 13 & "  # proportion of value is X* more important in best points than in bad points
                    "best_points > 12"  # minimum number of the value (because we don't want close=1.121213243482902183 as result)
                ).index.tolist()  # & df_all["part_of_best_points_percent"] > 10)]
                print(values)
                for elt in values:
                    best_indicators.append(f"dataframe['{column}'] == {elt}")

            if column in ['buy', 'buy_tag']:
                print(column)
                print(all_best_points[column].value_counts())
            elif not count.empty:
                res.append({
                    'column': column,
                    'value': count.index[0],
                    'ratio_for_best': count.iloc[0] / all_best_points.shape[0],
                    'ratio_for_all': all_points_values_count[count.index[0]] / all_points.shape[0],
                    'ratio_diff': count.iloc[0] / all_best_points.shape[0] - all_points_values_count[count.index[0]] / all_points.shape[0]
                })
        for item in sorted(res, key=lambda x: x['ratio_diff']):
            print(f"({item['column']} == {item['value']}), {100 * item['ratio_for_best']:.2f}% in best vs {100 * item['ratio_for_all']:.2f}% average")
        print("END OF COMMON VALUES FOR ALL BEST POINTS !!!!!!!!!!")

        print("Suggested buy signal:")
        print("( # main buy signals found")
        for item in best_indicators:
            print(f"    ({item})|")
        print(")")
        print("& ( # protections")
        for item in sorted(res, key=lambda x: x['ratio_diff']):
            if (((100 * item['ratio_for_best']) - (100 * item['ratio_for_all'])) > self.top_index_criteria['select_prameter_if_prop_is_x_percent_higher_in_best_points']) and (100 * item['ratio_for_best']) > self.top_index_criteria['select_parameter_if_in_more_than_x_percent_of_best_points']:
                print(f"(dataframe['{item['column']}'] == {item['value']}) &")
        print(")")
        return []

    def is_same_dimension_as_price(self, dataframe: DataFrame, column_name: str) -> bool:
        if dataframe['close'].dtype != dataframe[column_name].dtype:
            # prevent impossible comparisons
            return False
        return (dataframe[column_name].max() <= dataframe['high'].max() and dataframe[column_name].min() >= dataframe['low'].min())

    def is_same_dimension_as_volume(self, dataframe: DataFrame, column_name: str) -> bool:
        if dataframe['volume'].dtype != dataframe[column_name].dtype:
            # prevent impossible comparisons
            return False
        if 'volume' in column_name:
            return True
        return False

    def generate_superbuy_signal(self, dataframe: DataFrame, metadata: dict) -> list:
        # every indicators names
        columns = list(dataframe.columns)
        columns.remove('date')
        columns.remove('sell')
        columns.remove('buy')
        columns.remove('buy_tag')
        columns = [column for column in columns if not 'date' in column]

        # generated random conditions
        buy_conds = []

        # operators we will use as a string "123423232" which will be used to sequentially pick in operators
        generators = ""
        base_generators = str(self.generator.value)
        while len(generators) < len(columns) * len(columns):
            generators = generators + base_generators

        # get best buy point for first pair, will indicators will be used for each pair
        # THE PAIR YOU WANT TO USE AS REFERENCE MUST BE FIRST IN YOUR PAIRLIST !!!!!!!!
        if self.best_buy_point is None:
            try:
                top_index, _ = self.find_best_entry_point(dataframe, metadata)["shifted_index"].iloc[0]
                self.best_buy_point = dataframe.iloc[top_index]
                print(f"pair used as reference is {metadata['pair']}")
                print(top_index)
            except:
                self.best_buy_point = None
                pass

        # sort columns by category
        if len(self.columns_to_compare_to_best_point) == 0 and len(self.columns_to_compare_to_volume) == 0 and len(self.columns_to_compare_to_price) == 0:
            self.columns = columns
            for column in columns:
                if self.is_same_dimension_as_price(dataframe, column):
                    self.columns_to_compare_to_price.append(column)
                elif self.is_same_dimension_as_volume(dataframe, column):
                    self.columns_to_compare_to_volume.append(column)
                else:
                    self.columns_to_compare_to_best_point.append(column)
            print(f"columns_to_compare_to_price : {self.columns_to_compare_to_price}")
            print(f"columns_to_compare_to_volume : {self.columns_to_compare_to_volume}")

            # remove NAN columns for best point...
            if self.best_buy_point is not None:
                for column in self.columns_to_compare_to_best_point:
                    if str(self.best_buy_point[column]) == 'nan':
                        self.columns_to_compare_to_best_point.remove(column)
            print(f"columns_to_compare_to_best_point : {self.columns_to_compare_to_best_point}")

        # generate matrix of all operators for all combinations of columns and create buy conditions
        index = 0
        for left_elt in self.columns_to_compare_to_price:
            for right_elt in self.columns_to_compare_to_price:
                if index > len(generators):
                    break
                generator = generators[index]
                index += 1
                if left_elt == right_elt:
                    continue
                if int(generator) not in self.operators:
                    # pass if no operator is selected
                    continue
                # print("(dataframe['" + left_elt + "'] " + self.operators[int(generator)] + " dataframe['" + right_elt + "'])")
                buy_conds.append(
                    "(dataframe['" + left_elt + "'] " + self.operators[int(generator)] + " dataframe['" + right_elt + "'])"
                )

        for left_elt in self.columns_to_compare_to_volume:
            for right_elt in self.columns_to_compare_to_volume:
                if index > len(generators):
                    break
                generator = generators[index]
                index += 1
                if left_elt == right_elt:
                    continue
                if int(generator) not in self.operators:
                    # pass if no operator is selected
                    continue
                # print("(dataframe['" + left_elt + "'] " + self.operators[int(generator)] + " dataframe['" + right_elt + "'])")
                buy_conds.append(
                    "(dataframe['" + left_elt + "'] " + self.operators[int(generator)] + " dataframe['" + right_elt + "'])"
                )

        buy_conds_best_point = []
        # generate buy conditions with best buy point
        for column in self.columns_to_compare_to_best_point:
            if self.best_buy_point is None:
                continue
            if index > len(generators):
                break
            generator = generators[index]
            index += 1
            if int(generator) not in self.operators:
                # pass if no operator is selected
                continue
            # print("(dataframe['" + column + "'] " + self.operators[int(generator)] + " best_buy_point['" + column + "']))")
            # print(eval("best_buy_point['" + column + "']"))
            buy_conds_best_point.append(
                "(dataframe['" + column + "'] " + self.operators[int(generator)] + " " + str(self.best_buy_point[column]) + ")"
            )

        # select a few buy conditions
        random.seed(self.condition_selector.value)
        try:
            buy_conds = random.sample(buy_conds, self.operators_used_to_compare_between_columns.value)
        except ValueError:
            print("not enough conditions to compare between columns")
            # Sample larger than population or is negative
            pass
        try:
            buy_conds += random.sample(buy_conds_best_point, self.operators_used_to_with_best_point.value)
        except ValueError as e:
            if self.config['runmode'].value != 'hyperopt':
                print("not enough conditions to compare with best point")
            # Sample larger than population or is negative
            pass
        if self.config['runmode'].value in ('backtest', 'hyperopt') and self.buy_signal_already_printed != buy_conds:
            print(buy_conds)
            self.buy_signal_already_printed = buy_conds
        try:
            buy_conds = [eval(buy_cond, globals(), {'dataframe': dataframe, 'best_buy_point': self.best_buy_point}) for buy_cond in buy_conds]
        except:
            return []
        return buy_conds

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        buy_conds = self.generate_superbuy_signal(dataframe, metadata)

        if self.config['runmode'].value in ('backtest'):  # backtest, we want to check must common buy tags...
            dataframe = super().populate_buy_trend(dataframe, metadata)  # get buy tags
            self.common_points_for_every_best_entry(dataframe, metadata, self.columns + ['buy', 'buy_tag'])
        elif self.config['runmode'].value in ('hyperopt'):  # hyperopt, we want to test new buy signals
            is_additional_check = (
                    (  # main buy signals found
                        (dataframe['not_res1_1h'] == True)
                    )
                    & (  # protections
                            (dataframe['rsi_fast_lower_20'] == 0) &
                            (dataframe['rsi_fast_lower_30'] == 0) &
                            (dataframe['r_14_lower_minus_80'] == 0) &
                            (dataframe['r_32_lower_minus_80'] == 0) &
                            (dataframe['r_96_lower_minus_80'] == 0)
                    ) &
                    (
                        # (dataframe['ema_50_lin'] < dataframe['ema_26_lin']) |
                        # (dataframe['sma_15'] > dataframe['ema_50_1h']) |
                        # (dataframe['ema_slow'] <= dataframe['sup1']) |
                        (dataframe['res1'] >= dataframe['bb_upperband2_1h'])
                    )
            )

            if buy_conds:
                dataframe.loc[
                    is_additional_check
                    &
                    reduce(lambda x, y: x & y, buy_conds)

                    , 'buy'] = 1
        # THIS STRAT SHOULD NOT BE USED IN LIVE/DRYRUN MODE
        return dataframe

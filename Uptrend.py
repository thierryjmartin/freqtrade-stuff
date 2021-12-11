# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

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
    minimal_roi = {
        "60": 0.1,
        "30": 0.02,
        "0": 0.04
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.10

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

    use_custom_stoploss = True
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        sl_new = 1

        if (current_profit > 0.2):
            sl_new = 0.05
        elif (current_profit > 0.1):
            sl_new = 0.03
        elif (current_profit > 0.06):
            sl_new = 0.02
        elif (current_profit > 0.03):
            sl_new = 0.015
        elif (current_profit > 0.015):
            sl_new = 0.0075

        return sl_new


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['hl2'] = (dataframe['high'] + dataframe['low']) / 2
        dataframe['mama'], dataframe['fama'] = ta.MAMA(dataframe['hl2'], 0.5, 0.05)

        dataframe['mama_diff'] = dataframe['mama'] - dataframe['fama']
        dataframe['mama_diff_ratio'] = dataframe['mama_diff'] / dataframe['hl2']

        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)

        # EMA 50
        dataframe['ema50'] = ta.EMA(dataframe['close'], timeperiod=50)

        # EMA 200
        dataframe['ema200'] = ta.EMA(dataframe['close'], timeperiod=200)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < 80) &
                (dataframe['mama'] >  dataframe['fama']) & # uptrend
                (dataframe['mama_diff_ratio'] > 0.04) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['mama_diff_ratio'] < 0.01) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe

import random

class SuperBuy(Uptrend):
    """
    Idea is to build random buy signales from populate_indicators, with luck we'll get a good buy signal
    """

    generator = IntParameter(0, 100000000000, default=99295874569, optimize=True, space='buy') # generate unique matrix of conditions for your dataframe
    operators_used_to_compare_between_columns = IntParameter(1, 4, default=3, optimize=True, space='buy') # number of conditions you will keep to build buy signal
    operators_used_to_with_best_point = IntParameter(0, 2, default=1, optimize=True, space='buy') # number of conditions you will keep to build buy signal
    condition_selector = IntParameter(0, 100, default=50, optimize=True, space='buy') # how to select the desired conditions beteween all conditions generated (seed random)

    best_buy_point = None
    buy_signal_already_printed = False

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

    def find_best_entry_point(self, dataframe: DataFrame, metadata: dict, lookehead_candles :int = 10) -> list:
        workdataframe = dataframe.copy()
        workdataframe['higher_high'] = workdataframe['high'].rolling(lookehead_candles).max()
        workdataframe['close_shifted_lookehead'] = workdataframe['close'].shift(lookehead_candles)
        workdataframe['higher_high_close_ratio'] = workdataframe['higher_high'] / workdataframe['close_shifted_lookehead']
        top_index = workdataframe['higher_high_close_ratio'].idxmax() - lookehead_candles
        print(f"top index is : {top_index}, Date : {workdataframe['date'][top_index]}, HH : {workdataframe['higher_high'][top_index]}, Close : {workdataframe['close'][top_index]}")
        return top_index

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
            print(f"pair used as reference is {metadata['pair']}")
            top_index = self.find_best_entry_point(dataframe, metadata)
            self.best_buy_point = dataframe.iloc[top_index]

        # sort columns by category
        if len(self.columns_to_compare_to_best_point) == 0 and len(self.columns_to_compare_to_volume) == 0 and len(self.columns_to_compare_to_price) == 0:
            for column in columns:
                if self.is_same_dimension_as_price(dataframe, column):
                    self.columns_to_compare_to_price.append(column)
                elif self.is_same_dimension_as_volume(dataframe, column):
                    self.columns_to_compare_to_volume.append(column)
                else:
                    self.columns_to_compare_to_best_point.append(column)
            print(f"columns_to_compare_to_best_point : {self.columns_to_compare_to_best_point}")
            print(f"columns_to_compare_to_price : {self.columns_to_compare_to_price}")
            print(f"columns_to_compare_to_volume : {self.columns_to_compare_to_volume}")

            # remove NAN columns for best point...
            for column in self.columns_to_compare_to_best_point:
                if str(self.best_buy_point[column]) == 'nan':
                    self.columns_to_compare_to_best_point.remove(column)

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
            if index > len(generators):
                break
            generator = generators[index]
            index += 1
            if int(generator) not in self.operators:
                # pass if no operator is selected
                continue
            #print("(dataframe['" + column + "'] " + self.operators[int(generator)] + " best_buy_point['" + column + "']))")
            #print(eval("best_buy_point['" + column + "']"))
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
        #print(buy_conds)
        #buy_conds.append((dataframe['volume'] == 0))
        # TODO
        """
        is_additional_check = {
            *indicators we wanna use suitable for our wanted market situation*
        }

        if conditions:
            dataframe.loc[
                            is_additional_check
                            &
                            reduce(lambda x, y: x | y, conditions)

                        , 'buy' ] = 1"""

        dataframe['buy'] = 1
        for condition in buy_conds:
            dataframe.loc[~condition, 'buy'] = 0

        return dataframe
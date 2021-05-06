# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter

SMA = 'SMA'
EMA = 'EMA'

# Buy hyperspace params:
buy_params = {
	"base_nb_candles_buy": 30,
	"buy_trigger": SMA,
	"low_offset": 0.92,
}

# Sell hyperspace params:
sell_params = {
	"base_nb_candles_sell": 41,
	"high_offset": 1.026,
	"sell_trigger": SMA,
}

class SMAOffset(IStrategy):
    INTERFACE_VERSION = 2

    # ROI table:
    minimal_roi = {"0": 1}

    # Stoploss:
    stoploss = -0.5

    base_nb_candles_buy = IntParameter(5, 80, default=buy_params['base_nb_candles_buy'], space='buy')
    base_nb_candles_sell = IntParameter(5, 80, default=sell_params['base_nb_candles_sell'], space='sell')
    low_offset = DecimalParameter(0.9, 0.99, default=buy_params['low_offset'], space='buy')
    high_offset = DecimalParameter(0.99, 1.1, default=sell_params['high_offset'], space='sell')
    buy_trigger = CategoricalParameter([SMA, EMA], default=buy_params['buy_trigger'], space='buy')
    sell_trigger = CategoricalParameter([SMA, EMA], default=sell_params['sell_trigger'], space='sell')



    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.1
    trailing_stop_positive_offset = 0
    trailing_only_offset_is_reached = False

    # Optimal timeframe for the strategy
    timeframe = '5m'

    use_sell_signal = True
    sell_profit_only = False

    process_only_new_candles = True
    startup_candle_count = 30

    plot_config = {
        'main_plot': {
            'ma_offset_buy': {'color': 'orange'},
            'ma_offset_sell': {'color': 'orange'},
        },
    }

    use_custom_stoploss = False

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        if (current_profit < 0) & (current_time - timedelta(minutes=360) > trade.open_date_utc):
            return 0.02
        elif current_profit > 0.04:
            return stoploss_from_open(0.02, current_profit)
        elif current_profit > 0.06:
            return stoploss_from_open(0.03, current_profit)
        elif current_profit > 0.08:
            return stoploss_from_open(0.04, current_profit)
        elif current_profit > 0.10:
            return stoploss_from_open(0.05, current_profit)
        elif current_profit > 0.12:
            return stoploss_from_open(0.10, current_profit)
        elif current_profit > 0.14:
            return stoploss_from_open(0.11, current_profit)
        elif current_profit > 0.16:
            return stoploss_from_open(0.12, current_profit)
        return 1


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        #if self.buy_trigger.value == 'EMA':
        #    dataframe['ma_buy'] = ta.EMA(dataframe, timeperiod=self.base_nb_candles_buy.value)
        #else:
        #    dataframe['ma_buy'] = ta.SMA(dataframe, timeperiod=self.base_nb_candles_buy.value)

        #if self.sell_trigger.value == 'EMA':
        #    dataframe['ma_sell'] = ta.EMA(dataframe, timeperiod=self.base_nb_candles_sell.value)
        #else:
        #    dataframe['ma_sell'] = ta.SMA(dataframe, timeperiod=self.base_nb_candles_sell.value)

        #dataframe['ma_offset_buy'] = dataframe['ma_buy'] * self.low_offset.value
        #dataframe['ma_offset_sell'] = dataframe['ma_sell'] * self.high_offset.value

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.buy_trigger.value == 'EMA':
            dataframe['ma_buy'] = ta.EMA(dataframe, timeperiod=int(self.base_nb_candles_buy.value))
        else:
            dataframe['ma_buy'] = ta.SMA(dataframe, timeperiod=int(self.base_nb_candles_buy.value))

        dataframe['ma_offset_buy'] = dataframe['ma_buy'] * self.low_offset.value

        dataframe.loc[
            (
                (dataframe['close'] < dataframe['ma_offset_buy']) &
                (dataframe['volume'] > 0)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.sell_trigger.value == 'EMA':
            dataframe['ma_sell'] = ta.EMA(dataframe, timeperiod=int(self.base_nb_candles_sell.value))
        else:
            dataframe['ma_sell'] = ta.SMA(dataframe, timeperiod=int(self.base_nb_candles_sell.value))

        dataframe['ma_offset_sell'] = dataframe['ma_sell'] * self.high_offset.value

        dataframe.loc[
            (
                (dataframe['close'] > dataframe['ma_offset_sell']) &
                (dataframe['volume'] > 0)
            ),
            'sell'] = 1
        return dataframe

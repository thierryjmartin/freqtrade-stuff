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
from freqtrade.strategy import DecimalParameter, IntParameter, BooleanParameter

rangeUpper = 60
rangeLower = 5

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif

def valuewhen(dataframe, condition, source, occurrence):
    copy = dataframe.copy()
    copy['colFromIndex'] = copy.index
    copy = copy.sort_values(by=[condition, 'colFromIndex'], ascending=False).reset_index(drop=True)
    copy['valuewhen'] = np.where(copy[condition] > 0, copy[source].shift(-occurrence), copy[source])
    copy['barrsince'] = copy['colFromIndex'] - copy['colFromIndex'].shift(-occurrence)
    copy.loc[
        (
            (rangeLower <= copy['barrsince']) &
            (copy['barrsince']  <= rangeUpper)
        )
    , "in_range"] = 1
    copy['in_range'] = copy['in_range'].fillna(0)
    copy = copy.sort_values(by=['colFromIndex'], ascending=True).reset_index(drop=True)
    return copy['valuewhen'], copy['in_range']


class SOTCHDivergence(IStrategy):
    INTERFACE_VERSION = 2

    # Buy hyperspace params:
    buy_params = {
        'use_bull': True,
        'use_hidden_bull': False,
        "ewo_high": 5.835,
        "low_rsi_buy": 30,
        "high_rsi_buy": 60,
        "low_adx_buy": 30,
        "high_adx_buy": 30,
        "low_stoch_buy": 20,
        "high_stoch_buy": 80,
        "low_osc_buy": 80,
        "high_osc_buy": 80,
    }
    # Sell hyperspace params:
    sell_params = {
        'use_bear': True,
        'use_hidden_bear': True
    }

    # ROI table:
    minimal_roi = {
        "0": 0.05,
    }

    # Stoploss:
    stoploss = -0.08

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy
    timeframe = '5m'

    use_custom_stoploss = False

    use_bull = BooleanParameter(default=buy_params['use_bull'], space='buy', optimize=False)
    use_hidden_bull = BooleanParameter(default=buy_params['use_hidden_bull'], space='buy', optimize=False)
    use_bear = BooleanParameter(default=sell_params['use_bear'], space='sell', optimize=True)
    use_hidden_bear = BooleanParameter(default=sell_params['use_hidden_bear'], space='sell', optimize=True)
    # Protection
    fast_ewo = 50
    slow_ewo = 200
    ewo_high = DecimalParameter(0, 7.0, default=buy_params['ewo_high'], space='buy', optimize=False)
    low_rsi_buy = IntParameter(0, 100, default=buy_params['low_rsi_buy'], space='buy', optimize=True)
    high_rsi_buy = IntParameter(0, 100, default=buy_params['high_rsi_buy'], space='buy', optimize=True)
    low_adx_buy = IntParameter(0, 100, default=buy_params['low_adx_buy'], space='buy', optimize=True)
    high_adx_buy = IntParameter(0, 100, default=buy_params['high_adx_buy'], space='buy', optimize=True)
    low_stoch_buy = IntParameter(0, 100, default=buy_params['low_stoch_buy'], space='buy', optimize=True)
    high_stoch_buy = IntParameter(0, 100, default=buy_params['high_stoch_buy'], space='buy', optimize=True)
    low_osc_buy = IntParameter(0, 100, default=buy_params['low_osc_buy'], space='buy', optimize=True)
    high_osc_buy = IntParameter(0, 100, default=buy_params['high_osc_buy'], space='buy', optimize=True)

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    osc = 'slowd'
    len = 14
    src = 'close'
    lbL = 5
    lbR = 5

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        study(title="Divergence Indicator", format=format.price, resolution="")
        len = input(title="RSI Period", minval=1, defval=14)
        src = input(title="RSI Source", defval=close)
        lbR = input(title="Pivot Lookback Right", defval=5) # lookahead
        lbL = input(title="Pivot Lookback Left", defval=5)
        rangeUpper = input(title="Max of Lookback Range", defval=60)
        rangeLower = input(title="Min of Lookback Range", defval=5)
        plotBull = input(title="Plot Bullish", defval=true)
        plotHiddenBull = input(title="Plot Hidden Bullish", defval=false)
        plotBear = input(title="Plot Bearish", defval=true)
        plotHiddenBear = input(title="Plot Hidden Bearish", defval=false)
        bearColor = color.red
        bullColor = color.green
        hiddenBullColor = color.new(color.green, 80)
        hiddenBearColor = color.new(color.red, 80)
        textColor = color.white
        noneColor = color.new(color.white, 100)
        osc = rsi(src, len)
        """
        dataframe['RSI'] = ta.RSI(dataframe[self.src], self.len)
        dataframe['RSI'] = dataframe['RSI'].fillna(0)
        stoch = ta.STOCH(dataframe, fastk_period=10, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']
        dataframe['osc'] = dataframe[self.osc]

        # plFound = na(pivotlow(osc, lbL, lbR)) ? false : true
        dataframe['min'] = dataframe['osc'].rolling(self.lbL).min()
        dataframe['prevMin'] = np.where(dataframe['min'] > dataframe['min'].shift(), dataframe['min'].shift(), dataframe['min'])
        dataframe.loc[
            (
                (dataframe['osc'].shift(1) == dataframe['prevMin'].shift(1)) &
                (dataframe['osc'] != dataframe['prevMin'])
            )
        , 'plFound'] = 1

        # phFound = na(pivothigh(osc, lbL, lbR)) ? false : true
        dataframe['max'] = dataframe['osc'].rolling(self.lbL).max()
        dataframe['prevMax'] = np.where(dataframe['max'] < dataframe['max'].shift(), dataframe['max'].shift(), dataframe['max'])
        dataframe.loc[
            (
            (dataframe['osc'].shift(1) == dataframe['prevMax'].shift(1)) &
            (dataframe['osc'] != dataframe['prevMax'])
            )
        , 'phFound'] = 1


        #------------------------------------------------------------------------------
        # Regular Bullish
        # Osc: Higher Low
        # oscHL = osc[lbR] > valuewhen(plFound, osc[lbR], 1) and _inRange(plFound[1])
        dataframe['valuewhen_plFound_osc'], dataframe['inrange_plFound_osc'] = valuewhen(dataframe, 'plFound', 'osc', 1)
        dataframe.loc[
            (
                (dataframe['osc'] > dataframe['valuewhen_plFound_osc']) &
                (dataframe['inrange_plFound_osc'] == 1)
             )
        , 'oscHL'] = 1

        # Price: Lower Low
        # priceLL = low[lbR] < valuewhen(plFound, low[lbR], 1)
        dataframe['valuewhen_plFound_low'], dataframe['inrange_plFound_low'] = valuewhen(dataframe, 'plFound', 'low', 1)
        dataframe.loc[
            (dataframe['low'] < dataframe['valuewhen_plFound_low'])
            , 'priceLL'] = 1
        #bullCond = plotBull and priceLL and oscHL and plFound
        dataframe.loc[
            (
                (dataframe['priceLL'] == 1) &
                (dataframe['oscHL'] == 1) &
                (dataframe['plFound'] == 1)
            )
            , 'bullCond'] = 1

        # plot(
        #      plFound ? osc[lbR] : na,
        #      offset=-lbR,
        #      title="Regular Bullish",
        #      linewidth=2,
        #      color=(bullCond ? bullColor : noneColor)
        #      )
        #
        # plotshape(
        #      bullCond ? osc[lbR] : na,
        #      offset=-lbR,
        #      title="Regular Bullish Label",
        #      text=" Bull ",
        #      style=shape.labelup,
        #      location=location.absolute,
        #      color=bullColor,
        #      textcolor=textColor
        #      )

        # //------------------------------------------------------------------------------
        # // Hidden Bullish
        # // Osc: Lower Low
        #
        # oscLL = osc[lbR] < valuewhen(plFound, osc[lbR], 1) and _inRange(plFound[1])
        dataframe['valuewhen_plFound_osc'], dataframe['inrange_plFound_osc'] = valuewhen(dataframe, 'plFound', 'osc', 1)
        dataframe.loc[
            (
                (dataframe['osc'] < dataframe['valuewhen_plFound_osc']) &
                (dataframe['inrange_plFound_osc'] == 1)
             )
        , 'oscLL'] = 1
        #
        # // Price: Higher Low
        #
        # priceHL = low[lbR] > valuewhen(plFound, low[lbR], 1)
        dataframe['valuewhen_plFound_low'], dataframe['inrange_plFound_low'] = valuewhen(dataframe,'plFound', 'low', 1)
        dataframe.loc[
            (dataframe['low'] > dataframe['valuewhen_plFound_low'])
            , 'priceHL'] = 1
        # hiddenBullCond = plotHiddenBull and priceHL and oscLL and plFound
        dataframe.loc[
            (
                (dataframe['priceHL'] == 1) &
                (dataframe['oscLL'] == 1) &
                (dataframe['plFound'] == 1)
            )
            , 'hiddenBullCond'] = 1
        #
        # plot(
        #      plFound ? osc[lbR] : na,
        #      offset=-lbR,
        #      title="Hidden Bullish",
        #      linewidth=2,
        #      color=(hiddenBullCond ? hiddenBullColor : noneColor)
        #      )
        #
        # plotshape(
        #      hiddenBullCond ? osc[lbR] : na,
        #      offset=-lbR,
        #      title="Hidden Bullish Label",
        #      text=" H Bull ",
        #      style=shape.labelup,
        #      location=location.absolute,
        #      color=bullColor,
        #      textcolor=textColor
        #      )
        #
        # //------------------------------------------------------------------------------
        # // Regular Bearish
        # // Osc: Lower High
        #
        # oscLH = osc[lbR] < valuewhen(phFound, osc[lbR], 1) and _inRange(phFound[1])
        dataframe['valuewhen_phFound_osc'], dataframe['inrange_phFound_osc'] = valuewhen(dataframe, 'phFound', 'osc', 1)
        dataframe.loc[
            (
                (dataframe['osc'] < dataframe['valuewhen_phFound_osc']) &
                (dataframe['inrange_phFound_osc'] == 1)
             )
        , 'oscLH'] = 1
        #
        # // Price: Higher High
        #
        # priceHH = high[lbR] > valuewhen(phFound, high[lbR], 1)
        dataframe['valuewhen_phFound_high'], dataframe['inrange_phFound_high'] = valuewhen(dataframe, 'phFound', 'high', 1)
        dataframe.loc[
            (dataframe['high'] > dataframe['valuewhen_phFound_high'])
            , 'priceHH'] = 1
        #
        # bearCond = plotBear and priceHH and oscLH and phFound
        dataframe.loc[
            (
                (dataframe['priceHH'] == 1) &
                (dataframe['oscLH'] == 1) &
                (dataframe['phFound'] == 1)
            )
            , 'bearCond'] = 1
        #
        # plot(
        #      phFound ? osc[lbR] : na,
        #      offset=-lbR,
        #      title="Regular Bearish",
        #      linewidth=2,
        #      color=(bearCond ? bearColor : noneColor)
        #      )
        #
        # plotshape(
        #      bearCond ? osc[lbR] : na,
        #      offset=-lbR,
        #      title="Regular Bearish Label",
        #      text=" Bear ",
        #      style=shape.labeldown,
        #      location=location.absolute,
        #      color=bearColor,
        #      textcolor=textColor
        #      )
        #
        # //------------------------------------------------------------------------------
        # // Hidden Bearish
        # // Osc: Higher High
        #
        # oscHH = osc[lbR] > valuewhen(phFound, osc[lbR], 1) and _inRange(phFound[1])
        dataframe['valuewhen_phFound_osc'], dataframe['inrange_phFound_osc'] = valuewhen(dataframe, 'phFound', 'osc', 1)
        dataframe.loc[
            (
                (dataframe['osc'] > dataframe['valuewhen_phFound_osc']) &
                (dataframe['inrange_phFound_osc'] == 1)
             )
        , 'oscHH'] = 1
        #
        # // Price: Lower High
        #
        # priceLH = high[lbR] < valuewhen(phFound, high[lbR], 1)
        dataframe['valuewhen_phFound_high'], dataframe['inrange_phFound_high'] = valuewhen(dataframe, 'phFound', 'high', 1)
        dataframe.loc[
            (dataframe['high'] < dataframe['valuewhen_phFound_high'])
            , 'priceLH'] = 1
        #
        # hiddenBearCond = plotHiddenBear and priceLH and oscHH and phFound
        dataframe.loc[
            (
                (dataframe['priceLH'] == 1) &
                (dataframe['oscHH'] == 1) &
                (dataframe['phFound'] == 1)
            )
            , 'hiddenBearCond'] = 1
        #
        # plot(
        #      phFound ? osc[lbR] : na,
        #      offset=-lbR,
        #      title="Hidden Bearish",
        #      linewidth=2,
        #      color=(hiddenBearCond ? hiddenBearColor : noneColor)
        #      )
        #
        # plotshape(
        #      hiddenBearCond ? osc[lbR] : na,
        #      offset=-lbR,
        #      title="Hidden Bearish Label",
        #      text=" H Bear ",
        #      style=shape.labeldown,
        #      location=location.absolute,
        #      color=bearColor,
        #      textcolor=textColor
        #  )"""

        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        dataframe['ADX'] = ta.ADX(dataframe, timeperiod=14)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        if self.use_bull.value:
            conditions.append(
                    (
                        (dataframe['bullCond'] > 0) &
                        (dataframe['valuewhen_plFound_osc'] > self.low_osc_buy.value) &
                        (dataframe['valuewhen_plFound_osc'] < self.high_osc_buy.value) &
                        #(dataframe['EWO'] > self.ewo_high.value) &
                        (dataframe['RSI'] < self.high_rsi_buy.value) &
                        (dataframe['RSI'] > self.low_rsi_buy.value) &
                        (dataframe['ADX'] > self.low_adx_buy.value) &
                        (dataframe['ADX'] < self.high_adx_buy.value) &
                        (dataframe['slowk'] < self.high_stoch_buy.value) &
                        (dataframe['slowk'] > self.low_stoch_buy.value) &
                        (dataframe['volume'] > 0)
                    )
                )

        if self.use_hidden_bull.value:
            conditions.append(
                (
                    (dataframe['hiddenBullCond'] > 0) &
                    (dataframe['valuewhen_plFound_osc'] > self.low_osc_buy.value) &
                    (dataframe['valuewhen_plFound_osc'] < self.high_osc_buy.value) &
                    (dataframe['RSI'] < self.high_rsi_buy.value) &
                    (dataframe['RSI'] > self.low_rsi_buy.value) &
                    (dataframe['slowk'] < self.high_stoch_buy.value) &
                    (dataframe['slowk'] > self.low_stoch_buy.value) &
                    (dataframe['ADX'] > self.low_adx_buy.value) &
                    (dataframe['ADX'] < self.high_adx_buy.value) &
                    (dataframe['volume'] > 0)
                )
            )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'buy'
            ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        if self.use_bear.value:
            conditions.append(
                (
                    (dataframe['bearCond'] > 0) &
                    (dataframe['volume'] > 0)
                )
            )

        if self.use_hidden_bear.value:
            conditions.append(
                (
                    (dataframe['hiddenBearCond'] > 0) &
                    (dataframe['volume'] > 0)
                )
            )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ] = 1

        dataframe.to_csv('user_data/csvs/%s_%s.csv' % (self.__class__.__name__, metadata["pair"].replace("/", "_")))

        return dataframe

class RSIDivergence(SOTCHDivergence):

    # Buy hyperspace params:
    buy_params = {
        "high_adx_buy": 68,
        "high_osc_buy": 74,
        "high_rsi_buy": 53,
        "high_stoch_buy": 64,
        "low_adx_buy": 41,
        "low_osc_buy": 15,
        "low_rsi_buy": 9,
        "low_stoch_buy": 13,
        "ewo_high": 5.835,  # value loaded from strategy
        "use_bull": True,  # value loaded from strategy
        "use_hidden_bull": False,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "use_bear": False,  # value loaded from strategy
        "use_hidden_bear": False,  # value loaded from strategy
    }

    # ROI table:  # value loaded from strategy
    minimal_roi = {
        "0": 0.131,
        "13": 0.073,
        "56": 0.022,
        "133": 0
    }

    # Stoploss:
    stoploss = -0.2  # value loaded from strategy

    # Trailing stop:
    trailing_stop = True  # value loaded from strategy
    trailing_stop_positive = 0.005  # value loaded from strategy
    trailing_stop_positive_offset = 0.02  # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy


    osc = 'RSI'
    len = 14
    src = 'close'
    lbL = 40

class RSIDivergenceNPOriginal(RSIDivergence):
    # Buy hyperspace params:
    buy_params = {
        "adx_buy": 36,
        "high_rsi_buy": 60,
        "high_stoch_buy": 92,
        "low_rsi_buy": 21,
        "low_stoch_buy": 6,
        "use_bull": True,
        "use_hidden_bull": False,
        "ewo_high": 5.835,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "use_bear": False,
        "use_hidden_bear": False,
    }

    # ROI table:  # value loaded from strategy
    minimal_roi = {
        "0": 0.05
    }

    # Stoploss:
    stoploss = -0.2  # value loaded from strategy

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    osc = 'RSI'
    len = 14
    src = 'close'
    lbL = 40

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        if self.use_bull.value:
            conditions.append(
                    (
                        (dataframe['bullCond'] > 0) &
                        (dataframe['volume'] > 0)
                    )
                )

        if self.use_hidden_bull.value:
            conditions.append(
                (
                    (dataframe['hiddenBullCond'] > 0) &
                    (dataframe['volume'] > 0)
                )
            )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'buy'
            ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        if self.use_bear.value:
            conditions.append(
                (
                    (dataframe['bearCond'] > 0) &
                    (dataframe['volume'] > 0)
                )
            )

        if self.use_hidden_bear.value:
            conditions.append(
                (
                    (dataframe['hiddenBearCond'] > 0) &
                    (dataframe['volume'] > 0)
                )
            )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ] = 1

        dataframe.to_csv('user_data/csvs/%s_%s.csv' % (self.__class__.__name__, metadata["pair"].replace("/", "_")))

        return dataframe

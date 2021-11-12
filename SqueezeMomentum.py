# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
from freqtrade.strategy import DecimalParameter, IntParameter, BooleanParameter, CategoricalParameter



# --------------------------------
def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif


class SqueezeMomentum(IStrategy):
    INTERFACE_VERSION = 2

    # Buy hyperspace params:
    buy_params = {
        'BB_length': 20,
        'BB_multifactor': 2.0,
        'KC_length': 20,
        'KC_multifactor': 1.5,
        'use_true_range': True
    }

    # Sell hyperspace params:
    sell_params = {
    }

    # ROI table:  # value loaded from strategy
    minimal_roi = {
        "0": 0.1
    }

    # Stoploss:
    stoploss = -0.10  # value loaded from strategy

    # Trailing stop:
    trailing_stop = True  # value loaded from strategy
    trailing_stop_positive = 0.005  # value loaded from strategy
    trailing_stop_positive_offset = 0.025  # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False
    process_only_new_candles = True
    startup_candle_count = 30

    # Parameters
    BB_length = IntParameter(10, 30, default=buy_params['BB_length'], space='buy', optimize=True)
    BB_multifactor = CategoricalParameter([0.5, 1, 1.5, 2, 2.5, 3, 3.5], default=buy_params['BB_multifactor'], space='buy', optimize=False) # pas utilisé
    KC_length = IntParameter(10, 30, default=buy_params['KC_length'], space='buy', optimize=True)
    KC_multifactor = CategoricalParameter([0.5, 1, 1.5, 2, 2.5, 3, 3.5], default=buy_params['KC_multifactor'], space='buy', optimize=True)
    use_true_range = BooleanParameter(default=buy_params['use_true_range'], space='buy', optimize=True)


    # Optimal timeframe for the strategy
    timeframe = '4h'


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        //
        // @author LazyBear
        // List of all my indicators: https://www.tradingview.com/v/4IneGo8h/
        //

        // Calculate BB
        source = close
        basis = sma(source, length)
        dev = multKC * stdev(source, length)
        upperBB = basis + dev
        lowerBB = basis - dev

        // Calculate KC
        ma = sma(source, lengthKC)
        range = useTrueRange ? tr : (high - low)
        rangema = sma(range, lengthKC)
        upperKC = ma + rangema * multKC
        lowerKC = ma - rangema * multKC

        sqzOn  = (lowerBB > lowerKC) and (upperBB < upperKC)
        sqzOff = (lowerBB < lowerKC) and (upperBB > upperKC)
        noSqz  = (sqzOn == false) and (sqzOff == false)

        val = linreg(source  -  avg(avg(highest(high, lengthKC), lowest(low, lengthKC)),sma(close,lengthKC)),
                    lengthKC,0)

        bcolor = iff( val > 0,
                    iff( val > nz(val[1]), lime, green),
                    iff( val < nz(val[1]), red, maroon))
        scolor = noSqz ? blue : sqzOn ? black : gray
        plot(val, color=bcolor, style=histogram, linewidth=4)
        plot(0, color=scolor, style=cross, linewidth=2)
        """

        if self.use_true_range.value:
            dataframe[f'range'] = ta.TRANGE(dataframe)
        else:
            dataframe[f'range'] = dataframe['high'] - dataframe['low']

        for val in self.BB_length.range:
            # BB
            dataframe[f'ma_{val}'] = ta.SMA(dataframe, val)
            dataframe[f'stdev_{val}'] = ta.STDDEV(dataframe, val)
            # KC
            dataframe[f'rangema_{val}'] = ta.SMA(dataframe[f'range'], val)

            # Linreg
            dataframe[f'hh_close_{val}'] = ta.MAX(dataframe['high'], val)
            dataframe[f'll_close_{val}'] = ta.MIN(dataframe['low'], val)
            dataframe[f'avg_hh_ll_{val}'] = (dataframe[f'hh_close_{val}'] + dataframe[f'll_close_{val}']) / 2
            dataframe[f'avg_close_{val}'] = ta.SMA(dataframe['close'], val)
            dataframe[f'avg_{val}'] = (dataframe[f'avg_hh_ll_{val}'] + dataframe[f'avg_close_{val}']) / 2
            dataframe[f'val_{val}'] = ta.LINEARREG(dataframe['close'] - dataframe[f'avg_{val}'], val, 0)

            for kc in self.KC_multifactor.range:
                # BB
                dataframe[f'upperBB_{val}_{kc}'] = dataframe[f'ma_{val}'] + dataframe[f'stdev_{val}'] * kc
                dataframe[f'lowerBB_{val}_{kc}'] = dataframe[f'ma_{val}'] - dataframe[f'stdev_{val}'] * kc

                # KC
                dataframe[f'upperKC_{val}_{kc}'] = dataframe[f'ma_{val}'] + dataframe[f'rangema_{val}'] * kc
                dataframe[f'lowerKC_{val}_{kc}'] = dataframe[f'ma_{val}'] - dataframe[f'rangema_{val}'] * kc

                # SQZ
                dataframe.loc[
                    (
                        (dataframe[f'lowerBB_{val}_{kc}'] > dataframe[f'lowerKC_{val}_{kc}']) &
                        (dataframe[f'upperBB_{val}_{kc}'] < dataframe[f'upperKC_{val}_{kc}'])
                    )
                , f'sqzOn_{val}_{kc}'] = 1
                dataframe.loc[
                    (
                        (dataframe[f'lowerBB_{val}_{kc}'] < dataframe[f'lowerKC_{val}_{kc}']) &
                        (dataframe[f'upperBB_{val}_{kc}'] > dataframe[f'upperKC_{val}_{kc}'])
                    )
                , f'sqzOff_{val}_{kc}'] = 1
                dataframe.loc[
                    ((dataframe[f'sqzOn_{val}_{kc}'] == False) & (dataframe[f'sqzOff_{val}_{kc}'] == False))
                , f'noSqz_{val}_{kc}'] = 1

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe[f'sqzOn_{self.KC_length.value}_{self.KC_multifactor.value}'] == 1) &
                (dataframe[f'val_{self.BB_length.value}'] > 0) &
                (dataframe[f'val_{self.BB_length.value}'].shift(1) < 0) &
                (dataframe['volume'] > 0)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (

                (dataframe['volume'] < 0)

            ),
            'sell'] = 1
        #dataframe.to_csv('user_data/csvs/%s_%s.csv' % (self.__class__.__name__, metadata["pair"].replace("/", "_")))

        return dataframe



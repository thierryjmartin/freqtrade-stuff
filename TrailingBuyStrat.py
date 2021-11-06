# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, Series
# --------------------------------

import logging
import pandas as pd
import numpy as np
import datetime
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class YourStrat(IStrategy):
    # replace this by your strategy
    pass

class TrailingBuyStrat(YourStrat):
    # Orignal idea by @MukavaValkku, code by @tirail
    #
    # This class is designed to inherit from yours and starts trailing buy with your buy signals
    # Trailing buy starts at any buy signal
    # Trailing buy stops  with BUY if : price decreases and rises again more than trailing_buy_offset
    # Trailing buy stops with NO BUY : current price is > initial price * (1 +  trailing_buy_max) OR custom_sell tag
    # IT IS NOT COMPATIBLE WITH BACKTEST/HYPEROPT
    #
    # if process_only_new_candles = True, then you need to use 1m timeframe (and normal strategy timeframe as informative)
    # if process_only_new_candles = False, it will use ticker data and you won't need to change anything

    process_only_new_candles = False

    custom_info = dict()

    # Trailing buy parameters
    trailing_buy_order_enabled = True
    debug_mode = True
    trailing_buy_max = 0.1  # stop trailing buy if current_price > starting_price * (1+trailing_buy_max)

    init_trailing_dict = {
        'trailing_buy_order_started': False,
        'trailing_buy_order_uplimit': 0,
        'start_trailing_price': 0,
        'buy_tag': None,
        'start_trailing_time': None,
        'offset': 0,
    }

    def trailing_buy(self, pair, reinit=False):
        # returns trailing buy info for pair (init if necessary)
        if not pair in self.custom_info:
            self.custom_info[pair] = dict()
        if reinit or not 'trailing_buy' in self.custom_info[pair]:
            self.custom_info[pair]['trailing_buy'] = self.init_trailing_dict
        return self.custom_info[pair]['trailing_buy']

    def trailing_buy_info(self, pair: str, current_price: float):
        # current_time live, dry run
        current_time = datetime.datetime.now(datetime.timezone.utc)
        if not self.debug_mode:
            return
        trailing_buy = self.trailing_buy(pair)
        logger.info(
            f"pair: {pair} : "
            f"start: {trailing_buy['start_trailing_price']:.4f}, "
           f"duration: {current_time - trailing_buy['start_trailing_time']}, "
           f"current: {current_price:.4f}, "
           f"uplimit: {trailing_buy['trailing_buy_order_uplimit']:.4f}, "
           f"profit: {self.current_trailing_profit_ratio(pair, current_price)*100:.2f}%, "
           f"offset: {trailing_buy['offset']}")

    def current_trailing_profit_ratio(self, pair: str, current_price: float) -> float:
        trailing_buy = self.trailing_buy(pair)
        if trailing_buy['trailing_buy_order_started']:
            return (trailing_buy['start_trailing_price'] - current_price) / trailing_buy['start_trailing_price']
        else:
            return 0

    def buy(self, dataframe, pair: str, current_price: float, buy_tag: str):
        dataframe.iloc[-1, dataframe.columns.get_loc('buy')] = 1
        ratio = "%.2f" % ((self.current_trailing_profit_ratio(pair, current_price)) * 100)
        if 'buy_tag' in dataframe.columns:
            dataframe.iloc[-1, dataframe.columns.get_loc('buy_tag')] = f"{buy_tag} ({ratio} %)"
        self.trailing_buy_info(pair, current_price)
        logger.info(f"price OK for {pair} ({ratio} %, {current_price}), order may not be triggered if all slots are full")

    def trailing_buy_offset(self, dataframe, pair: str, current_price: float):
        # return rebound limit before a buy in % of initial price, function of current price
        # return None to stop trailing buy (will start again at next buy signal)
        # return 'forcebuy' to force immediate buy
        # (example with 0.5%. initial price : 100 (uplimit is 100.5), 2nd price : 99 (no buy, uplimit updated to 99.5), 3price 98 (no buy uplimit updated to 98.5), 4th price 99 -> BUY
        current_trailing_profit_ratio = self.current_trailing_profit_ratio(pair, current_price)
        default_offset = 0.005

        trailing_buy = self.trailing_buy(pair)
        if not trailing_buy['trailing_buy_order_started']:
            return default_offset

        # example with duration and indicators
        # dry run, live only
        last_candle = dataframe.iloc[-1]
        current_time = datetime.datetime.now(datetime.timezone.utc)
        trailing_duration = current_time - trailing_buy['start_trailing_time']
        if trailing_duration.total_seconds() > 3600:
            if current_trailing_profit_ratio > 0 and last_candle['pre_buy'] == 1:
                # more than 1h, price under first signal, buy signal still active -> buy
                return 'forcebuy'
            else:
                # wait for next signal
                return None
        elif trailing_duration.total_seconds() < 90 and current_trailing_profit_ratio > 0.02:
            # less than 90s and price is rising, buy
            return 'forcebuy'

        if current_trailing_profit_ratio < 0:
            # current price is higher than initial price
            return default_offset

        trailing_buy_offset = {
            0.06: 0.02,
            0.03: 0.01,
            0: default_offset,
        }

        for key in trailing_buy_offset:
            if current_trailing_profit_ratio > key:
                return trailing_buy_offset[key]

        return default_offset

    # end of trailing buy parameters
    # -----------------------------------------------------

    def custom_sell(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        tag = super().custom_sell(pair, trade, current_time, current_rate, current_profit, **kwargs)
        if tag:
            self.trailing_buy_info(pair, current_rate)
            self.trailing_buy(pair, reinit=True)
            logger.info(f'STOP trailing buy for {pair} because of {tag}')
        return tag

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        self.trailing_buy(metadata['pair'])
        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        val = super().confirm_trade_exit(pair, trade, order_type, amount, rate, time_in_force, sell_reason, **kwargs)
        self.trailing_buy(pair, reinit=True)
        return val

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        val = super().confirm_trade_entry(pair, order_type, amount, rate, time_in_force, **kwargs)
        # stop trailing when buy signal ! prevent from buying much higher price when slot is free
        self.trailing_buy_info(pair, rate)
        self.trailing_buy(pair, reinit=True)
        logger.info(f'STOP trailing buy for {pair} because I buy it')
        return val

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        def get_local_min(x):
            win = dataframe.loc[:, 'barssince_last_buy'].iloc[x.shape[0] - 1].astype('int')
            win = max(win, 0)
            return pd.Series(x).rolling(window=win).min().iloc[-1]

        dataframe = super().populate_buy_trend(dataframe, metadata)
        dataframe = dataframe.rename(columns={"buy": "pre_buy"})

        if self.trailing_buy_order_enabled and self.config['runmode'].value in ('live', 'dry_run'):  # trailing live dry ticker, 1m
            last_candle = dataframe.iloc[-1].squeeze()
            if not self.process_only_new_candles:
                current_price = self.get_current_price(metadata["pair"])
            else:
                current_price = last_candle['close']
            dataframe['buy'] = 0
            trailing_buy = self.trailing_buy(metadata['pair'])
            trailing_buy_offset = self.trailing_buy_offset(dataframe, metadata['pair'], current_price)

            if not trailing_buy['trailing_buy_order_started'] and last_candle['pre_buy'] == 1:
                open_trades = Trade.get_trades([Trade.pair == metadata['pair'], Trade.is_open.is_(True), ]).all()
                if not open_trades:
                    # start trailing buy
                    self.custom_info[metadata["pair"]]['trailing_buy'] = {
                        'trailing_buy_order_started': True,
                        'trailing_buy_order_uplimit': last_candle['close'],
                        'start_trailing_price': datetime.datetime.now(datetime.timezone.utc),
                        'buy_tag': last_candle['buy_tag'] if 'buy_tag' in last_candle else 'buy signal',
                        'start_trailing_time': last_candle['date'],
                        'offset': 0,
                    }
                    self.trailing_buy_info(metadata["pair"], current_price)
                    logger.info(f'start trailing buy for {metadata["pair"]} at {last_candle["close"]}')

            elif trailing_buy['trailing_buy_order_started']:
                if trailing_buy_offset == 'forcebuy':
                    # buy in custom conditions
                    self.buy(dataframe, metadata['pair'], current_price, trailing_buy['buy_tag'])
                elif trailing_buy_offset is None:
                    # stop trailing buy custom conditions
                    self.trailing_buy(metadata['pair'], reinit=True)
                    logger.info(f'STOP trailing buy for {metadata["pair"]} because "trailing buy offset" returned None')
                elif current_price < trailing_buy['trailing_buy_order_uplimit']:
                    # update uplimit
                    old_uplimit = trailing_buy["trailing_buy_order_uplimit"]
                    self.custom_info[metadata["pair"]]['trailing_buy']['trailing_buy_order_uplimit'] = min(current_price * (1 + trailing_buy_offset), self.custom_info[metadata["pair"]]['trailing_buy']['trailing_buy_order_uplimit'])
                    self.custom_info[metadata["pair"]]['trailing_buy']['offset'] = trailing_buy_offset
                    self.trailing_buy_info(metadata["pair"], current_price)
                    logger.info(f'update trailing buy for {metadata["pair"]} at {old_uplimit} -> {self.custom_info[metadata["pair"]]["trailing_buy"]["trailing_buy_order_uplimit"]}')
                elif current_price < trailing_buy['start_trailing_price']:
                    # buy ! current price > uplimit && lower thant starting price
                    self.buy(dataframe, metadata['pair'], current_price, trailing_buy['buy_tag'])
                elif current_price > (trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max)):
                    # stop trailing buy because price is too high
                    self.trailing_buy(metadata['pair'], reinit=True)
                    self.trailing_buy_info(metadata["pair"], current_price)
                    logger.info(f'STOP trailing buy for {metadata["pair"]} because of the price is higher than starting price * {1 + self.trailing_buy_max}')
                else:
                    # uplimit > current_price > max_price, continue trailing and wait for the price to go down
                    self.trailing_buy_info(metadata["pair"], current_price)
                    logger.info(f'price too high for {metadata["pair"]} !')
        elif self.trailing_buy_order_enabled:
            # FOR BACKTEST
            # NOT WORKING
            dataframe.loc[
                (dataframe['pre_buy'] == 1) &
                (dataframe['pre_buy'].shift() == 0)
                , 'pre_buy_switch'] = 1
            dataframe['pre_buy_switch'] = dataframe['pre_buy_switch'].fillna(0)

            dataframe['barssince_last_buy'] = dataframe['pre_buy_switch'].groupby(dataframe['pre_buy_switch'].cumsum()).cumcount()

            # Create integer positions of each row
            idx_positions = np.arange(len(dataframe))
            # "shift" those integer positions by the amount in shift col
            shifted_idx_positions = idx_positions - dataframe["barssince_last_buy"]
            # get the label based index from our DatetimeIndex
            shifted_loc_index = dataframe.index[shifted_idx_positions]
            # Retrieve the "shifted" values and assign them as a new column
            dataframe["close_5m_last_buy"] = dataframe.loc[shifted_loc_index, "close_5m"].values

            dataframe.loc[:, 'close_lower'] = dataframe.loc[:, 'close'].expanding().apply(get_local_min)
            dataframe['close_lower'] = np.where(dataframe['close_lower'].isna() == True, dataframe['close'], dataframe['close_lower'])
            dataframe['close_lower_offset'] = dataframe['close_lower'] * (1 + self.trailing_buy_offset)
            dataframe['trailing_buy_order_uplimit'] = np.where(dataframe['barssince_last_buy'] < 20, pd.DataFrame([dataframe['close_5m_last_buy'], dataframe['close_lower_offset']]).min(), np.nan)

            dataframe.loc[
                (dataframe['barssince_last_buy'] < 20) &  # must buy within last 20 candles after signal
                (dataframe['close'] > dataframe['trailing_buy_order_uplimit'])
                , 'trailing_buy'] = 1

            dataframe['trailing_buy_count'] = dataframe['trailing_buy'].rolling(20).sum()

            dataframe.log[
                (dataframe['trailing_buy'] == 1) &
                (dataframe['trailing_buy_count'] == 1)
                , 'buy'] = 1
        else:  # No buy trailing
            dataframe.loc[
                (dataframe['pre_buy'] == 1)
                , 'buy'] = 1
        return dataframe

    def get_current_price(self, pair: str) -> float:
        ticker = self.dp.ticker(pair)
        current_price = ticker['last']
        return current_price

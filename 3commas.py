# https://github.com/bogdanteodoru/py3cw

def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, **kwargs) -> bool:

        coin, currency = pair.split('/')

        p3cw = Py3CW(
            key='........',
            secret='............',
        )

        p3cw.request(
            entity='bots',
            action='start_new_deal',
            action_id='12312313',
            payload={
                "bot_id": 12312313,
                "pair": f"{currency}_{coin}",
            },
        )
        PairLocks.lock_pair(
            pair=pair,
            until=datetime.now(timezone.utc) + timedelta(minutes=5),
            reason="Send 3c buy order"
        )

        return False  # we don't want to keep the trade in freqtrade db

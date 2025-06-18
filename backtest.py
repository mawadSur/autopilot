import backtrader as bt
import pandas as pd

class SignalStrategy(bt.Strategy):
    params = (
        ('take_profit', 0.03),    # 5% take profit
        ('stop_loss', 0.020),     # 2.5% stop loss
        ('max_hold', 12),         # 24 bars max hold (~24 hours for hourly data)
    )

    def __init__(self):
        self.order = None
        self.entry_price = None
        self.entry_bar = None

        self.total_trades = 0
        self.total_wins = 0
        self.total_losses = 0
        self.log_list = []

    def log(self, txt):
        dt = self.datas[0].datetime.datetime(0)
        print(f'{dt.isoformat()} - {txt}')

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None

    def next(self):
        if self.order:
            return  # Wait for order to complete

        if not self.position:
            if self.datas[0].signal[0] == 1:
                self.order = self.buy()
                self.entry_price = self.data.close[0]
                self.entry_bar = len(self)
                self.log(f'BUY at {self.entry_price}')
        else:
            current_price = self.data.close[0]
            tp_price = self.entry_price * (1 + self.params.take_profit)
            sl_price = self.entry_price * (1 - self.params.stop_loss)
            held_bars = len(self) - self.entry_bar

            should_exit = (
                current_price >= tp_price or
                current_price <= sl_price or
                held_bars >= self.params.max_hold
            )

            if should_exit:
                self.order = self.sell()
                pnl = current_price - self.entry_price
                result = "WIN" if pnl > 0 else "LOSS"
                self.total_trades += 1
                if pnl > 0:
                    self.total_wins += 1
                else:
                    self.total_losses += 1
                self.log(f'SELL at {current_price} | PnL: {pnl:.2f} → {result}')
                self.log_list.append({
                    "entry_bar": self.entry_bar,
                    "exit_bar": len(self),
                    "entry_price": self.entry_price,
                    "exit_price": current_price,
                    "held_hours": held_bars,
                    "pnl": pnl,
                    "result": result
                })


class SignalData(bt.feeds.PandasData):
    lines = ('signal',)
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1),
        ('signal', -1),
    )

def run_backtest(csv_path='eth_signals.csv'):
    df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
    if 'signal' not in df.columns:
        raise ValueError("Your CSV must include a 'signal' column.")
    df.dropna(inplace=True)

    cerebro = bt.Cerebro()
    data = SignalData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(SignalStrategy)
    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.001)
    # cerebro.addsizer(bt.sizers.FixedSize, stake=2)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)  # risk 10% of portfolio per trade


    print("\n=== Starting Backtest ===")
    starting_value = cerebro.broker.getvalue()
    print("Start Portfolio Value:", starting_value)

    strat = cerebro.run()[0]

    ending_value = cerebro.broker.getvalue()
    print("Final Portfolio Value:", ending_value)

    print("\n=== Trade Log ===")
    for trade in strat.log_list:
        print(trade)

    print("\n=== Metrics ===")
    print(f"Total Trades: {strat.total_trades}")
    print(f"Wins: {strat.total_wins}")
    print(f"Losses: {strat.total_losses}")
    if strat.total_trades > 0:
        win_rate = strat.total_wins / strat.total_trades * 100
        print(f"Win Rate: {win_rate:.2f}%")
    else:
        print("No trades were made.")
    print(f"Total Return: {(ending_value - starting_value):.2f} USD")

    cerebro.plot()

if __name__ == "__main__":
    run_backtest()

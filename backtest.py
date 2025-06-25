import backtrader as bt
import pandas as pd
import json
import matplotlib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os


class SignalData(bt.feeds.PandasData):
    lines = ('signal', 'confidence',)
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1),
        ('signal', -1),
        ('confidence', -1),
    )

class SignalStrategy(bt.Strategy):
    params = (
        ('atr_period', 14),
        ('atr_tp_mult', 1.5),
        ('atr_sl_mult', 1.0),
        ('min_volume_ratio', 0.5),
        ('cooldown_bars', 5),
        ('max_hold', 24),
    )

    def __init__(self):
        self.order = None
        self.entry_price = None
        self.entry_bar = None
        self.last_exit_bar = -999

        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.pbar = tqdm(desc="Backtesting Progress", ncols=100)


        self.total_trades = 0
        self.total_wins = 0
        self.total_losses = 0
        self.log_list = []

        with open("model_meta.json", "r") as f:
            self.conf_threshold = json.load(f).get("best_threshold", 0.6)

    def log(self, txt):
        dt = self.datas[0].datetime.datetime(0)
        print(f'{dt.isoformat()} - {txt}')

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None

    def next(self):
        self.pbar.update(1)  # update progress bar  
        if self.order:
            return

        if not self.position:
            if self.data.signal[0] == 1 and self.data.confidence[0] >= self.conf_threshold:
                if self.data.volume[0] < self.data.volume[-1] * self.p.min_volume_ratio:
                    return
                if len(self) - self.last_exit_bar < self.p.cooldown_bars:
                    return

                self.order = self.buy()
                self.entry_price = self.data.close[0]
                self.entry_bar = len(self)
                self.tp_price = self.entry_price + self.atr[0] * self.p.atr_tp_mult
                self.sl_price = self.entry_price - self.atr[0] * self.p.atr_sl_mult
                self.log(f'BUY at {self.entry_price:.2f} | TP: {self.tp_price:.2f}, SL: {self.sl_price:.2f}')
        else:
            current_price = self.data.close[0]
            held_bars = len(self) - self.entry_bar

            should_exit = (
                current_price >= self.tp_price or
                current_price <= self.sl_price or
                held_bars >= self.p.max_hold
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
                self.last_exit_bar = len(self)
                self.log(f'SELL at {current_price:.2f} | PnL: {pnl:.2f} → {result}')
                self.log_list.append({
                    "entry_bar": self.entry_bar,
                    "exit_bar": len(self),
                    "entry_price": self.entry_price,
                    "exit_price": current_price,
                    "held_hours": held_bars,
                    "pnl": pnl,
                    "result": result
                })

    def stop(self):
        self.pbar.close()

CHUNK_SIZE = 1_000_000  # Tune based on memory capacity
PROGRESS_FILE = 'chunk_progress.txt'


def run_backtest_in_chunks(csv_path='eth_signals.csv'):
    chunk_iter = pd.read_csv(csv_path, parse_dates=['date'], index_col='date', chunksize=CHUNK_SIZE)

    # Load resume index if exists
    start_chunk = 0
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            start_chunk = int(f.read().strip())
        print(f"⏩ Resuming from chunk {start_chunk}")

    for i, chunk in enumerate(tqdm(chunk_iter, desc="Backtesting Chunks")):
        if i < start_chunk:
            continue  # skip previously processed chunks

        if 'signal' not in chunk.columns or 'confidence' not in chunk.columns:
            print(f"❌ Skipping chunk {i} due to missing signal/confidence")
            continue

        chunk.dropna(inplace=True)

        cerebro = bt.Cerebro()
        cerebro.adddata(SignalData(dataname=chunk))
        cerebro.addstrategy(SignalStrategy)
        cerebro.broker.setcash(10000)
        cerebro.broker.setcommission(commission=0.001)
        cerebro.addsizer(bt.sizers.PercentSizer, percents=95)

        cerebro.run()  # run single chunk
        print(f"✅ Completed chunk {i}")

        # Save progress
        with open(PROGRESS_FILE, 'w') as f:
            f.write(str(i + 1))

        # Optionally save results after each chunk
        # e.g. plot or log metrics

    print("✅ All chunks processed.")

    print("\n=== Starting Backtest ===")
    start_val = cerebro.broker.getvalue()
    print(f"Start Portfolio Value: {start_val}")

    strat = cerebro.run()[0]
    end_val = cerebro.broker.getvalue()
    print(f"Final Portfolio Value: {end_val}")

    print("\n=== Trade Log ===")
    for trade in strat.log_list:
        print(trade)

    print("\n=== Metrics ===")
    print(f"Total Trades: {strat.total_trades}")
    print(f"Wins: {strat.total_wins}")
    print(f"Losses: {strat.total_losses}")
    if strat.total_trades > 0:
        print(f"Win Rate: {strat.total_wins / strat.total_trades * 100:.2f}%")
    print(f"Total Return: {end_val - start_val:.2f} USD")

    print("\n=== Analyzers ===")
    print("Sharpe Ratio:", strat.analyzers.sharpe.get_analysis())
    print("Max Drawdown:", strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 'N/A'), "%")

    cerebro.plot(style='candlestick')[0][0].savefig("backtest_plot.png")
    print("📈 Plot saved to backtest_plot.png")
    cerebro.plot(style='candlestick')

if __name__ == "__main__":
    run_backtest_in_chunks()

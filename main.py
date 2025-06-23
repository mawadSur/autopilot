import argparse
import train_model
import label
import backtest
import history
import paper_trade
import generate_signal
import trade

def main():
    parser = argparse.ArgumentParser(description="Run AI trading pipeline")
    parser.add_argument("--step", choices=["history", "train", "label", "backtest", "trade", "paper"], required=True,
                        help="Which step to run")
    args = parser.parse_args()
    
    if args.step == "history":
        print("Get previous data...")
        history.fetch_eth_ohlc_vs_currency()

    elif args.step == "label":
        print("🏷️ Generating labeled data...")
        label.label_data()  # call your label.py logic here

    elif args.step == "train":
        print("▶️ Training model...")
        train_model.train_model()

    elif args.step == "generate":
        print("▶️ Generating Signal...")
        generate_signal.main()

    elif args.step == "backtest":
        print("📉 Running backtest...")
        backtest.run_backtest()

    elif args.step == "paper":
        print("🧪 Running paper trade...")
        paper_trade.run_paper_trade()

    elif args.step == "trade":
        print("💹 Executing live trade...")
        trade.live_trade()  # use real Binance API

if __name__ == "__main__":
    main()

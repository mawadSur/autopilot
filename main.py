import argparse
import train_model
import label
import backtest
import history
import paper_trade
import trade
# I'll assume you will create a generate_signals.py file as suggested below
# import generate_signals 

def main():
    parser = argparse.ArgumentParser(description="Run AI trading pipeline")
    # Added 'generate' to choices
    parser.add_argument("--step", choices=["history", "label", "train", "generate", "backtest", "paper", "trade"], required=True,
                        help="Which step to run")
    args = parser.parse_args()
    
    if args.step == "history":
        print("🚚 Fetching historical data...")
        # Corrected function name
        history.fetch_historical_data()

    elif args.step == "label":
        print("🏷️ Generating labeled data...")
        # Assumes you wrap the logic in label.py in a function like this
        label.generate_labels() 

    elif args.step == "train":
        print("▶️ Training model...")
        train_model.train_model()

    # This step is for backtesting. You will need to create this script.
    # I will provide a template for it below.
    # elif args.step == "generate":
    #     print("▶️ Generating Signals for Backtest...")
    #     generate_signals.run_signal_generation()

    elif args.step == "backtest":
        print("📉 Running backtest...")
        # Corrected function name
        backtest.run_backtest_in_chunks()

    elif args.step == "paper":
        print("🧪 Running paper trade...")
        # Corrected function name
        paper_trade.run_multiple_trades()

    elif args.step == "trade":
        print("💹 Executing live trade...")
        # Corrected function name
        trade.trade_live()

if __name__ == "__main__":
    main()
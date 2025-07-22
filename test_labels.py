import pandas as pd
import numpy as np
import os

# --- 🔧 STEP 1: CONFIGURE YOUR TEST PARAMETERS HERE 🔧 ---
#
# Adjust these values to see how they affect the class balance.
#

# This is a percentage (e.g., 0.5 means a 0.5% profit target).
PROFIT_THRESHOLD_PERCENT = 0.5

# This is the risk/reward ratio (e.g., 2.0 means profit must be 2x the risk).
RISK_REWARD_RATIO = 2.0

# This is the number of future minutes to look at for profit/loss.
LOOKAHEAD_PERIOD = 10

# The folder where you've placed one or more sample .csv files.
DATA_FOLDER = "eth_1m_data_sample"
# ---

def find_first_csv(folder):
    """Finds the first .csv file in a given folder."""
    try:
        for file in os.listdir(folder):
            if file.endswith(".csv"):
                return os.path.join(folder, file)
    except FileNotFoundError:
        return None
    return None

def analyze_and_tune_labels(df, profit_threshold_pct, risk_reward_ratio, lookahead_period):
    """
    This function incorporates Step 2 (Analyze) and Step 3 (Tune).
    It applies the labeling logic from your training script and returns
    a debug dataframe for easy inspection.
    """
    print(f"\n🔬 Analyzing data with RR_Ratio={risk_reward_ratio} and Profit_Threshold={profit_threshold_pct}%...")

    # --- Step 2: Isolate and Analyze the Labeling Logic ---
    future_highs = df['high'].rolling(window=lookahead_period).max().shift(-lookahead_period)
    future_lows = df['low'].rolling(window=lookahead_period).min().shift(-lookahead_period)

    potential_profit = future_highs - df['close']
    potential_loss = df['close'] - future_lows

    # --- Step 3: Tune the Labeling Parameters ---
    # Convert the percentage to a decimal for calculation (e.g., 0.5 -> 0.005)
    profit_threshold_decimal = profit_threshold_pct / 100
    # Make the threshold dynamic based on the closing price
    profit_threshold_dynamic = df['close'] * profit_threshold_decimal

    # Apply the logic to create the final label
    df['label'] = (
        (potential_profit > profit_threshold_dynamic) &
        (potential_loss > 0) &  # <-- ADD THIS CRUCIAL FILTER
        (potential_profit > risk_reward_ratio * potential_loss)
    ).astype(int)

    # Create a detailed debug DataFrame to see the calculation results
    debug_df = pd.DataFrame({
        'close': df['close'],
        'profit_threshold': profit_threshold_dynamic.round(4),
        'potential_profit': potential_profit.round(4),
        'potential_loss': potential_loss.round(4),
        'profit_check_pass': potential_profit > profit_threshold_dynamic,
        'rr_check_pass': potential_profit > risk_reward_ratio * potential_loss,
        'final_label': df['label']
    })

    # Get the class balance statistics
    class_balance = df['label'].value_counts()

    return debug_df, class_balance

def main():
    """
    Main script execution function.
    """
    print("--- Labeling Logic Investigation Script ---")

    sample_file = find_first_csv(DATA_FOLDER)
    if not sample_file:
        print(f"\n[ERROR] No .csv files found in the '{DATA_FOLDER}' folder.")
        print("Please create this folder and add at least one sample .csv file to it.")
        return

    print(f"Loading sample data from: {sample_file}")
    try:
        df = pd.read_csv(sample_file, header=None,
                        names=['date', 'open', 'high', 'low', 'close', 'volume'])
        
        # ✅ ADD THIS BLOCK TO FIX THE ERROR
        # Convert price columns to numbers, turning any errors into Not-a-Number (NaN)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows that have missing data after conversion
        df.dropna(inplace=True)
            
    except Exception as e:
        print(f"[ERROR] Could not load the CSV file: {e}")
        return

    # Run the analysis with the configured parameters
    debug_df, class_balance = analyze_and_tune_labels(
        df.copy(), # Use a copy to avoid modifying the original df
        PROFIT_THRESHOLD_PERCENT,
        RISK_REWARD_RATIO,
        LOOKAHEAD_PERIOD
    )

    # --- 📊 Display the results ---
    print("\n--- Analysis Results ---")
    print("Top 20 rows of the debug dataframe:")
    print(debug_df.head(20))

    print("\nClass Balance (0 = Hold, 1 = Buy):")
    print(class_balance)

    neg_count = class_balance.get(0, 0)
    pos_count = class_balance.get(1, 0)

    if pos_count > 0:
        ratio = neg_count / pos_count
        print(f"\nRatio of 'Hold' to 'Buy' signals: {ratio:.2f} : 1")
    else:
        print("\nNo 'Buy' signals were generated with these parameters.")

if __name__ == "__main__":
    main()
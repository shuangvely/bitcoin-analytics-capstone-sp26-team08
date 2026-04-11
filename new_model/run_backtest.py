import logging
import pandas as pd
from pathlib import Path

# Import template components
from template.prelude_template import load_data
from template.backtest_template import run_full_analysis

# Import new model
from new_model.model_development_new_model import precompute_features, compute_window_weights

# Global variable to store precomputed features
_FEATURES_DF = None

def compute_weights_wrapper(df_window: pd.DataFrame) -> pd.Series:
    """Wrapper for Example 1 compute_window_weights.
    
    Adapts the specific Example 1 model function to the interface expected 
    by the template backtest engine.
    """
    global _FEATURES_DF
    
    if _FEATURES_DF is None:
        raise ValueError("Features not precomputed. Call precompute_features() first.")
        
    if df_window.empty:
        return pd.Series(dtype=float)

    start_date = df_window.index.min()
    #try a more recent period
    #start_date = pd.Timestamp("2023-01-01")
    end_date = df_window.index.max()
    
    # For backtesting, current_date = end_date (all dates are in the past)
    current_date = end_date
    
    return compute_window_weights(_FEATURES_DF, start_date, end_date, current_date)


def main():
    global _FEATURES_DF
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logging.info("Starting Bitcoin DCA Strategy Analysis - New Model (Polymarket)")
    
    # 1. Load Data
    btc_df = load_data()
    
    # 2. Precompute Features (using Example 1 logic)
    logging.info("Precomputing features (MVRV + Churn + Macro + Whale + Risk)...")
    _FEATURES_DF = precompute_features(btc_df)
    
    # 3. Define Output Directory
    base_dir = Path(__file__).parent
    output_dir = base_dir / "output"
    
    # 4. Run Analysis (reusing Template engine)
    run_full_analysis(
        btc_df=btc_df,
        features_df=_FEATURES_DF,
        compute_weights_fn=compute_weights_wrapper,
        output_dir=output_dir,
        strategy_label="New Model (Polymarket)",
    )


if __name__ == "__main__":
    main()

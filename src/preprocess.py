import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_preprocessing():
    config = load_config('config/config.yaml')
    
    # Setup logging
    os.makedirs(os.path.dirname(config['paths']['log_path']), exist_ok=True)
    logging.basicConfig(
        filename=config['paths']['log_path'],
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("Preprocess")
    logger.info("Starting preprocessing with data split...")

    # Load raw data
    raw_df = pd.read_csv(config['paths']['raw_data'])
    target = config['preprocessing']['target_column']
    
    # Split data before any processing to prevent leakage
    train_df, test_df = train_test_split(
        raw_df, 
        test_size=config['preprocessing']['test_size'], 
        random_state=42
    )
    logger.info("Data split into train and test sets (test_size=%s)", config['preprocessing']['test_size'])

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # Handle missing values
    imputer = SimpleImputer(strategy=config['preprocessing']['impute_strategy'])
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # Scaling
    if config['preprocessing']['scaling']:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test.columns)
    else:
        X_train_scaled = X_train_imputed
        X_test_scaled = X_test_imputed

    # Save processed data
    processed_dir = config['paths']['processed_dir']
    os.makedirs(processed_dir, exist_ok=True)
    
    X_train_scaled.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)
    
    logger.info("Preprocessing complete. Artifacts saved.")
    print("Preprocessing complete. Train/Test split and processed data saved.")

if __name__ == "__main__":
    run_preprocessing()

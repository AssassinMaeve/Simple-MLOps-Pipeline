import pandas as pd
import yaml
import os
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_training():
    config = load_config('config/config.yaml')
    
    # Setup logging
    logging.basicConfig(
        filename=config['paths']['log_path'],
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("Train")
    logger.info("Starting model training...")

    # Load processed data
    processed_dir = config['paths']['processed_dir']
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv"))

    # Model training
    model_type = config['model']['type']
    params = config['model']['params']
    
    if model_type == "RandomForestClassifier":
        model = RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train, y_train.values.ravel())
    logger.info("Model training complete.")

    # Save model
    model_path = config['paths']['model_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    
    logger.info("Model saved to %s", model_path)
    print("Training complete. Model saved to", model_path)

if __name__ == "__main__":
    run_training()

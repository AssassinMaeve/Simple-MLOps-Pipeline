import pandas as pd
import yaml
import os
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier

# Helper: Load instructions from config
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

    # Step 1: Load the "Study Material" we prepared in the preprocessing step
    processed_dir = config['paths']['processed_dir']
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv"))

    # Step 2: Initialize the Detective (the Model)
    # A Random Forest is like a team of detectives voting on the result.
    model_type = config['model']['type']
    params = config['model']['params']
    
    if model_type == "RandomForestClassifier":
        # We use the parameters from config.yaml to tell the Detective how to think
        model = RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Step 3: Training (The 'Learning' Phase)
    # The Detective looks at thousands of examples to find the 'clues' that matter.
    model.fit(X_train, y_train.values.ravel())
    logger.info("Model training complete.")

    # Step 4: Save the Detective's Brain (Persistence)
    # We save the trained model to a file so we can reuse it later without retraining.
    model_path = config['paths']['model_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    
    logger.info("Model saved to %s", model_path)
    print("Training complete. The Detective's brain has been saved to", model_path)

if __name__ == "__main__":
    run_training()

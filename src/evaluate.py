import pandas as pd
import yaml
import os
import joblib
import json
import logging
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_evaluation():
    config = load_config('config/config.yaml')
    
    # Setup logging
    logging.basicConfig(
        filename=config['paths']['log_path'],
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("Evaluate")
    logger.info("Starting model evaluation and visualization...")

    # Load test data and model
    processed_dir = config['paths']['processed_dir']
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv"))
    
    model_path = config['paths']['model_path']
    model = joblib.load(model_path)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "classification_report": report
    }

    # Save metrics
    metrics_path = config['paths']['metrics_path']
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Visualization: Confusion Matrix
    plots_dir = config['paths']['plots_dir']
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 7))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Visualization: Feature Importance
    plt.figure(figsize=(12, 8))
    importances = model.feature_importances_
    feature_names = X_test.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    
    sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
    plt.title('Model Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
    plt.close()

    logger.info("Evaluation and visualization complete. Artifacts saved.")
    print(f"Evaluation complete. Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"Visualizations saved to {plots_dir}")

if __name__ == "__main__":
    run_evaluation()

import subprocess
import os
import sys
import logging
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_step(script_path, description):
    print(f"--- Running {description} ---")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error in {description}:")
        print(result.stderr)
        return False
    print(result.stdout)
    return True

def main():
    config_path = 'config/config.yaml'
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    config = load_config(config_path)
    
    # Setup logging
    os.makedirs(os.path.dirname(config['paths']['log_path']), exist_ok=True)
    logging.basicConfig(
        filename=config['paths']['log_path'],
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("Pipeline")
    logger.info("Starting MLOps Pipeline Orchestrator")

    steps = [
        ('src/preprocess.py', 'Preprocessing'),
        ('src/train.py', 'Training'),
        ('src/evaluate.py', 'Evaluation')
    ]

    for script, desc in steps:
        if not run_step(script, desc):
            logger.error(f"Pipeline failed at step: {desc}")
            print(f"Pipeline failed at step: {desc}")
            return

    logger.info("Pipeline completed successfully.")
    print("\n==================================")
    print("Pipeline Execution Successful!")
    print("==================================")

if __name__ == "__main__":
    main()

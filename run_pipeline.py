import subprocess
import os
import sys
import logging
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_step(step_name, command):
    print(f"\n--- Running {step_name} ---")
    # The Space Detective executes the command and watches for any anomalies
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ Error in {step_name}. Factory line halted.")
        return False
    return True

def main():
    config_path = 'config/config.yaml'
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    config = load_config(config_path)
    
    # Setup central logging for the whole factory line
    os.makedirs(os.path.dirname(config['paths']['log_path']), exist_ok=True)
    logging.basicConfig(
        filename=config['paths']['log_path'],
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("Pipeline")
    logger.info("Starting MLOps Pipeline Orchestrator")

    # Step 2: Run each stage of the factory line in order
    # 1. Cleanup & Preparation
    if not run_step("Preprocessing", "python src/preprocess.py"):
        logger.error("Pipeline failed at step: Preprocessing")
        return

    # 2. Learning/Training
    if not run_step("Training", "python src/train.py"):
        logger.error("Pipeline failed at step: Training")
        return

    # 3. Exam/Evaluation
    if not run_step("Evaluation", "python src/evaluate.py"):
        logger.error("Pipeline failed at step: Evaluation")
        return

    # Success! The Space Detective confirms the mission is complete.
    logger.info("Pipeline completed successfully.")
    print("\n==================================")
    print("✨ Factory Line Successful!")
    print("The Space Detective is ready.")
    print("==================================\n")

if __name__ == "__main__":
    main()

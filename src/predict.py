import pandas as pd
import joblib
import yaml
import os
import random

# Helper: Load instructions
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_demo():
    config = load_config('config/config.yaml')
    
    # Step 1: Load the Detective (the Model) and the Archives (the Data)
    model = joblib.load(config['paths']['model_path'])
    X_test = pd.read_csv(os.path.join(config['paths']['processed_dir'], "X_test.csv"))
    y_test = pd.read_csv(os.path.join(config['paths']['processed_dir'], "y_test.csv"))

    print("\n" + "="*40)
    print("🛸 SPACE DETECTIVE: LIVE INVESTIGATION")
    print("="*40)
    
    # Step 2: Let the user (the Junior Detective) choose a star to investigate
    print("Select a star from the Star Chart to investigate:")
    print(" A. Star Sirius (Classic Pulsar)")
    print(" B. Star Vega   (Quiet Signal)")
    print(" C. Star Orion  (Mystery Signal)")
    print(" D. Pick a Random Star")
    print(" E. Enter a Custom Star ID")
    
    # Mapping names to data indices
    star_map = {
        'A': 5,  # Verified Pulsar in dataset
        'B': 0,  # Verified Space Junk in dataset
        'C': 11, # Verified Pulsar in dataset
    }
    
    choice = input("\nEnter your selection (A-E): ").upper()
    
    # Step 3: Identify the star index based on the choice
    if choice in star_map:
        idx = star_map[choice]
    elif choice == 'D':
        idx = random.randint(0, len(X_test) - 1)
    elif choice == 'E':
        try:
            idx = int(input(f"Enter Star ID (0-{len(X_test)-1}): "))
            if idx < 0 or idx >= len(X_test):
                print("❌ Invalid ID! Picking a random star instead.")
                idx = random.randint(0, len(X_test) - 1)
        except ValueError:
            print("❌ Invalid input! Picking a random star instead.")
            idx = random.randint(0, len(X_test) - 1)
    else:
        print("⚠️ Unknown selection. The Detective picked a random star for you!")
        idx = random.randint(0, len(X_test) - 1)

    # Step 4: Get the specific star signal measurements
    sample = X_test.iloc[[idx]]
    actual_label = y_test.iloc[idx].values[0]

    # Step 5: Ask the Detective for their opinion
    prediction = model.predict(sample)[0]
    
    # Step 6: Reveal the Truth!
    print("\n" + "-"*40)
    name_label = choice if choice in star_map else f"ID: {idx}"
    print(f"Investigating Star: {name_label}")
    print("\nScanning star signal measurements (The Clues)...")
    for col in X_test.columns:
        print(f" - {col}: {sample[col].values[0]:.4f}")
    
    print("\n" + "-"*40)
    print(f"🕵️  Detective says: This looks like {'a REAL PULSAR' if prediction == 1 else 'SPACE JUNK (Noise)'}")
    print(f"📜 Ancient Archive (Truth): This is {'a REAL PULSAR' if actual_label == 1 else 'SPACE JUNK (Noise)'}")
    
    # Step 7: Check if the Detective was correct
    if prediction == actual_label:
        print("\n✅ THE DETECTIVE WAS RIGHT!")
    else:
        print("\n❌ OOPS! THE DETECTIVE WAS FOOLED.")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_demo()

# 🌌 Simple MLOps Pipeline Workshop
Created by **Maeve Kirk D'cruz E Fernandes** and **Vellon Moraes**

Welcome! This project is a simple, step-by-step guide to building a "Machine Learning Pipeline." Think of it as a factory line that takes raw data and turns it into a smart "Space Detective" that can find real stars in space!

---

## 🛠️ How it Works (The "Space Detective" Journey)

1.  **🧼 Preprocessing (The Cleanup):** Raw data from space can be messy. We clean it up, fill in missing spots, and make sure all the numbers are "scaled" so the detective can understand them.
2.  **🧠 Training (The Learning):** We give the data to our model (the Detective). It learns the patterns that separate a **Real Pulsar Star** from **Space Junk**.
3.  **📊 Evaluation (The Exam):** We give the Detective a final test with data it hasn't seen before to see how many it gets right!

---

## 📂 Project Folders

- `config/`: The "Brain Settings" (where we change how the detective thinks).
- `data/`: The raw space signals and the cleaned-up data.
- `src/`: The actual Python code (The factory line).
- `models/`: Where the trained "Detective" is saved.
- `reports/plots/`: The visual report cards (Confusion Matrix & Feature Importance).

---

## 🚀 Getting Started

### 1. Install the Tools
Run this in your terminal to install the necessary libraries:
```bash
pip install -r requirements.txt
```
or 
```bash
uv add -r requirements.txt
```

### 2. Run the Factory Line
To start the entire process, just run:
```bash
python run_pipeline.py
```

### 3. See the Space Detective in Action!
Once the factory line is finished, you can run a live investigation:
```bash
python src/predict.py
```
This script picks a random star and asks the "Detective" to identify it!

---

## 🎨 Understanding the Results (Analogy Time!)

### 1. The Sorting Chart (Confusion Matrix)
*Located at: `reports/plots/confusion_matrix.png`*
Imagine sorting apples from oranges. Did we put apples in the orange box? This chart shows exactly where the Detective got the answer right or made a mistake.

### 2. The "What Did You Look At?" Chart (Feature Importance)
*Located at: `reports/plots/feature_importance.png`*
When deciding if a fruit is an apple, did you look at the **color**, the **weight**, or the **smell**? This chart tells us which "clues" (like how skinny or bumpy the star signal is) the Detective used to make its decision.

---

**Good luck with your first MLOps pipeline! 🚀**

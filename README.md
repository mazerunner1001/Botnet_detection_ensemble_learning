# 🔍 Botnet Detection Using Ensemble and Deep Learning

## 📖 Overview
Botnets are one of the most dangerous threats in IoT networks. They hijack connected devices and use them for malicious activities such as DDoS, spam, and data theft. Traditional detection systems often fail to keep up with evolving attack patterns.  

This project explores **ensemble learning and deep learning** methods to build a robust botnet detection system. The goal is to compare classical ML baselines, deep neural architectures (CNN, LSTM), and tuned models to evaluate how well they identify botnet traffic.

---

## 🎯 Objectives
- Preprocess IoT traffic data into a usable format for ML/DL models.  
- Train **classical ML baselines** (Logistic Regression, Random Forest) as references.  
- Design **deep learning models** (CNNs, LSTMs) for sequential/temporal pattern detection.  
- Experiment with **hyperparameter tuning** to optimize learning rate and architecture.  
- Compare models on standard metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.  
- Show the advantage of ensemble/deep learning approaches over simple baselines.

---

## 🧩 Methodology

1. **Data Preprocessing**  
   - Load raw IoT traffic dataset(s).  
   - Scale and encode features for ML/DL readiness.  
   - Split into training/testing sets.  

2. **Baseline Models**  
   - Train simple classifiers like **Logistic Regression**.  
   - Evaluate using ROC-AUC, Accuracy, F1, Precision, Recall.  
   - Establish a baseline performance benchmark.  

3. **Deep Learning Models**  
   - Implement **LSTMs** (to capture sequential temporal dependencies).  
   - Implement **1D CNNs** (to capture local spatial/temporal patterns).  
   - Train models on preprocessed traffic data.  

4. **Hyperparameter Tuning**  
   - Vary learning rates and optimizers (Adam).  
   - Use callbacks like **EarlyStopping** to avoid overfitting.  
   - Compare accuracy vs learning rate using plots.  

5. **Model Comparison**  
   - Compare classical vs. deep vs. tuned models.  
   - Analyze confusion matrices, ROC curves, and metric tradeoffs.  
   - Highlight advantages of ensemble/deep models for IoT botnet detection.  

---

## 📂 Repository Structure
```
bot_net_detection/
├── 01_deep_learning_and_classical_models.ipynb
├── 02_hyperparameter_tuning_and_training.ipynb
├── 03_classical_ml_baseline_evaluation.ipynb
```

## 📓 Notebook Details

### `01_deep_learning_and_classical_models.ipynb`
- Implements both **deep learning** (CNN, LSTM) and **classical ML** (Random Forest).  
- Demonstrates how different architectures behave on the dataset.  
- Purpose: Build the first set of models and compare traditional vs deep learning approaches.  

---

### `02_hyperparameter_tuning_and_training.ipynb`
- Focused on **optimizing the deep learning models**.  
- Experiments with different learning rates, architectures, optimizers.  
- Uses **EarlyStopping** for efficient training.  
- Includes **accuracy vs. learning rate plots** to visualize performance trends.  
- Purpose: Find optimal configurations for DL models to maximize detection accuracy.  

---

### `03_classical_ml_baseline_evaluation.ipynb`
- Implements baseline classical ML models like **Logistic Regression**.  
- Preprocesses data using **StandardScaler**.  
- Evaluates with metrics: ROC-AUC, Accuracy, F1, Precision, Recall, Confusion Matrix.  
- Purpose: Provide a **reference benchmark** against which to measure deep/ensemble models.  

---

## 📊 Evaluation Metrics
Models are compared using:
- **Accuracy** – overall correctness.  
- **Precision & Recall** – balance between false positives/negatives.  
- **F1-score** – harmonic mean of precision & recall.  
- **ROC-AUC** – ability to distinguish between classes.  
- **Confusion Matrix** – detailed breakdown of predictions.  

---

## ⚙️ Installation & Usage

### 1. Clone the repository:
```bash
git clone https://github.com/mazerunner1001/Botnet_detection_ensemble_learning.git
cd Botnet_detection_ensemble_learning/bot_net_detection
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

If not provided, install common ones manually:
```bash
pip install numpy pandas scikit-learn tensorflow keras matplotlib seaborn
```

### 3. Run Jupyter:
```bash
jupyter notebook
```

Open and execute the notebooks in order.

## 🏷️ Keywords
botnet, IoT security, ensemble learning, deep learning, LSTM, CNN, anomaly detection, intrusion detection, machine learning, cybersecurity

## 📌 Future Work

- Extend ensemble strategies (stacking/blending multiple models).
- Deploy trained models for real-time botnet detection.
- Integrate explainability (SHAP, LIME) to interpret model predictions.
- xplore lightweight DL architectures for IoT/edge deployment.

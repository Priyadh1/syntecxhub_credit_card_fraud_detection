# 💳 Credit Card Fraud Detection using Machine Learning & Agentic AI

An end-to-end **Credit Card Fraud Detection** project developed during the **Syntecxhub Machine Learning Internship – Project 2**.

The core system focuses on detecting fraudulent transactions from a highly **imbalanced dataset** using **SMOTE, Random Forest, and XGBoost**, followed by model evaluation and business-oriented threshold analysis.

As an additional enhancement, an **Agentic AI workflow** was integrated into the application to provide multi-agent transaction analysis, risk assessment, policy evaluation, and automated case recommendations.

## 🚀 Live Application

🔗 **Streamlit App:**
https://priyadharshappcreditcardfrauddetection-6wkprbtssterfayupnbsjn.streamlit.app/

## 🎯 Project Aim

To develop a machine learning-based Credit Card Fraud Detection system that handles highly imbalanced transaction data using **SMOTE**, trains and compares **Random Forest and XGBoost** models, evaluates their performance using **Precision, Recall, F1-Score, and ROC-AUC**, and analyzes classification thresholds based on business costs.

An additional **Agentic AI layer** is integrated to orchestrate multiple specialized agents for transaction risk assessment, behavioural analysis, similarity checking, policy evaluation, and final case management.

## 📊 Dataset

The project uses the **Credit Card Fraud Detection dataset** from the ULB Machine Learning Group / Kaggle.

A sampled dataset is used:

* **25,492 transactions**
* **492 fraud transactions**
* **25,000 normal transactions**
* Approximately **1.9% fraud**
* **31 columns**
* `Class = 0` → Normal transaction
* `Class = 1` → Fraudulent transaction

Features include:

* `Time`
* `V1 – V28` anonymized PCA features
* `Amount`
* `Class`

## 🔄 Machine Learning Workflow

```text
Credit Card Transaction Dataset
              ↓
       Data Exploration
              ↓
      Imbalance Analysis
              ↓
    Stratified Train/Test Split
              ↓
       Feature Scaling
              ↓
      SMOTE on Training Data
              ↓
     ┌───────────────────┐
     │  Random Forest    │
     │  XGBoost          │
     └───────────────────┘
              ↓
     Model Evaluation
              ↓
 Precision / Recall / F1 / ROC-AUC
              ↓
 Threshold & Business Cost Analysis
```

### Key Steps

1. Load and inspect the dataset.
2. Perform basic EDA and identify missing values.
3. Visualize the class imbalance.
4. Perform an **80/20 stratified train-test split**.
5. Scale `Time` and `Amount` using training data only.
6. Establish a **Random Forest baseline** without sampling.
7. Apply **SMOTE only to the training data**.
8. Train Random Forest and XGBoost using the SMOTE-balanced training data.
9. Compare models using Precision, Recall, F1-Score and ROC-AUC.
10. Analyze different probability thresholds based on business costs.

## 🤖 Agentic AI Enhancement

The traditional ML pipeline provides the fraud prediction, while the additional **Agentic AI layer** orchestrates multiple specialized agents to analyze each transaction and generate a final recommendation.

### Agent Workflow

```text
                Transaction
                     ↓
              Intake Agent
                     ↓
           Risk Scoring Agent
            (XGBoost Model)
                     ↓
        ┌────────────┴────────────┐
        ↓                         ↓
Behavioural Baseline       Similarity Agent
     Agent                 (Past Cases Search)
        └────────────┬────────────┘
                     ↓
                Policy Agent
                     ↓
             Case Manager Agent
                     ↓
          Final Recommendation
```

### Specialized Agents

| Agent                          | Responsibility                                                  |
| ------------------------------ | --------------------------------------------------------------- |
| **Intake Agent**               | Receives and prepares the transaction case                      |
| **Risk Scoring Agent**         | Uses the trained XGBoost model to generate a fraud-risk score   |
| **Behavioural Baseline Agent** | Compares transaction behaviour against the consumer baseline    |
| **Similarity Agent**           | Searches for similar historical transaction/case patterns       |
| **Policy Agent**               | Applies configured risk thresholds and business rules           |
| **Case Manager Agent**         | Combines the available evidence and determines the final action |

The workflow also supports **conditional agent execution**. For example, when the risk score is sufficiently low, certain analysis agents can skip execution instead of unnecessarily processing the case.

### Possible Final Actions

* ⚡ **Autonomously Blocked**
* ⏳ **Escalated to Human**
* ✅ **Auto-approved**

> The ML model performs the fraud-risk prediction, while the Agentic AI layer provides orchestration, contextual analysis, policy evaluation, and case-level decision support.

## 📈 Model Results

Performance on the fraud class:

| Metric    | RF Baseline | RF + SMOTE | XGBoost + SMOTE |
| --------- | ----------: | ---------: | --------------: |
| Precision |       1.000 |      0.955 |           0.905 |
| Recall    |       0.837 |      0.867 |       **0.878** |
| F1-Score  |   **0.911** |      0.909 |           0.891 |
| ROC-AUC   |       0.969 |      0.969 |       **0.973** |

The results show that **SMOTE increased fraud recall**, helping the models identify more fraudulent transactions while introducing some additional false positives.

## 💰 Business Threshold Analysis

Fraud detection is not only about maximizing model accuracy. The cost of a **missed fraud** can be much higher than the cost of a false alarm.

Based on the project's business-cost assumption:

* If a missed fraud costs approximately **10×** a false alarm → threshold **0.3** provides the lowest estimated total cost.
* If false positives and false negatives have equal cost → threshold **0.7** is preferred.

This demonstrates the **precision-recall tradeoff** and how model thresholds can be selected according to business requirements.

## 🖥️ Application Features

The Streamlit application provides an interactive fraud investigation interface with:

* Case ID and transaction information
* Colour-coded risk level
* Transaction amount and case status
* Case queue and search-by-ID functionality
* Manual transaction entry
* Configurable risk thresholds
* Agent on/off controls
* Transaction profile
* Consumer behavioural baseline
* Device telemetry information
* Six-agent workflow timeline
* Expandable agent analysis sections
* Final recommendation
* Model performance dashboard
* Precision, Recall, F1 and ROC-AUC comparison
* Confusion-matrix analysis
* Precision vs Recall threshold analysis
* Agent capability overview

## 📊 Model Performance Dashboard

The application includes a dedicated performance section containing:

* Grouped comparison of Precision, Recall, F1 and ROC-AUC
* Confusion-matrix counts
* Precision vs Recall across different thresholds
* Model comparison between Random Forest and XGBoost approaches

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **XGBoost**
* **Imbalanced-learn / SMOTE**
* **Matplotlib**
* **Streamlit**
* **Machine Learning**
* **Agentic AI / Multi-Agent Workflow**
* **Nearest-Neighbour Similarity Search**

## 📁 Project Structure

```text
syntecxhub_credit_card_fraud_detection/
│
├── credit_card.ipynb
├── creditcard_small.csv
├── fraud_xgb_smote_model.pkl
├── scaler.pkl
└── README.md
```

## ⚠️ Limitations

* The test set contains only **98 fraud transactions**, so small differences between models should be interpreted carefully.
* The sampled dataset has a higher fraud rate than the original real-world dataset, so precision may be more optimistic.
* The system is a **fraud-risk detection and decision-support system**, not a replacement for human financial investigation.
* Business thresholds should be recalibrated using real transaction costs and production data.

## 🌟 Key Takeaways

* Handled severe **class imbalance** using SMOTE.
* Compared **Random Forest and XGBoost** for fraud detection.
* Focused on **Precision, Recall, F1 and ROC-AUC** rather than accuracy alone.
* Analyzed **classification thresholds and business costs**.
* Built an interactive **Streamlit fraud investigation application**.
* Extended the ML pipeline with a **multi-agent / Agentic AI decision workflow**.
* Implemented conditional agent execution and automated case recommendations.

## 👩‍💻 Author

**Priyadharshini Murugan**

Machine Learning | Data Science | AI/ML

### Internship

**Syntecxhub – Machine Learning Internship**
**Project 2: Credit Card Fraud Detection**

---

⭐ If you find this project useful, feel free to explore the repository and try the live application.

**Built by Priyadharshini Murugan**

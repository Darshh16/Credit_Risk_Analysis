Credit Risk Analysis & Predictive Modeling

An end-to-end machine learning and statistical analysis project for
understanding creditability, identifying risk-related factors, and
building a model to distinguish Good Credit from Bad Credit
applicants.

📌 Project Highlights

Analyzed 1,000 historical credit applications across 21
columns

Performed data-quality checks: 0 missing values, 0 duplicate
rows

Correctly treated integer-coded categorical variables as categorical
features

Compared Logistic Regression, Decision Tree, and Random Forest

Used an 80/20 stratified train-test split

Used 5-fold stratified cross-validation

Evaluated Accuracy, Precision, Recall, F1 Score, and ROC-AUC

Performed RandomizedSearchCV hyperparameter tuning

Investigated classification-threshold trade-offs

Increased bad-credit recall from 53.3% → 66.7% using a threshold
of 0.44

Aggregated one-hot encoded feature importance back to business-level
variables

Connected statistical evidence + predictive evidence + business
interpretation

🎯 Problem Statement

Credit-risk analysis requires more than simply predicting whether an
applicant is likely to be classified as good or bad credit.

The objectives were to:

Understand characteristics associated with creditability.

Identify categorical variables with statistically meaningful
relationships with credit risk.

Build and compare multiple classification models.

Determine which model provides the strongest overall discrimination.

Investigate whether the default probability threshold is appropriate
for a risk-sensitive setting.

Translate analytical findings into practical business insights.

📊 Dataset

Property                        Value

Records                         1,000
Columns                            21
Original Predictors                20
Target                  Creditability
Good Credit                 700 (70%)
Bad Credit                  300 (30%)
Missing Values                      0
Duplicate Rows                      0



The target is moderately imbalanced, so accuracy alone is not
sufficient for evaluating the model.

🧹 Data Preparation

Three variables were treated as numerical:

Credit Duration

Credit Amount

Age

The remaining 17 predictors were treated as categorical codes.

Preprocessing

StandardScaler for numerical variables

OneHotEncoder for categorical variables

ColumnTransformer

Pipeline

EDA helper columns were excluded from modeling by explicitly selecting
the original 20 predictors.

Key lesson: integer encoding does not automatically make a
variable continuous.

🔎 Exploratory Data Analysis

Numerical comparison

Variable            Good Credit Mean   Bad Credit Mean

Credit Duration         19.21 months      24.86 months
Credit Amount               2,985.44          3,938.13
Age                      36.22 years       33.96 years



Bad-credit applicants had higher average credit amounts and longer
average credit durations, while their average age was lower.

These are observational differences and do not establish causality.

📈 Risk-Rate Analysis

Bad-credit rates were examined within categories rather than relying
only on raw counts.

Notable observed ranges included:

Checking-account categories: 49.27% → 11.68% bad credit

Credit-history categories: 62.50% → 17.06% bad credit

Property categories: 21.28% → 43.51% bad credit

This helped identify potentially important risk segments before model
training.

🧪 Statistical Association Testing

Categorical predictors were tested against creditability using the
Chi-Square test of independence at:

α = 0.05

Cramér's V was used to quantify association strength.

Strongest associations

Feature                         Cramér's V

Checking account status     0.3517
Credit history                      0.2484
Savings                             0.1900
Purpose                             0.1826
Property                            0.1540
Type of apartment                   0.1367
Employment duration                 0.1355



Standout finding

Checking account status was the strongest categorical association
with creditability.

It later also became the most important Random Forest feature, creating
agreement between statistical analysis and predictive modeling.

🤖 Machine Learning Approach

The target was encoded as:

0 → Good Credit
1 → Bad Credit

This makes recall directly represent the model's ability to detect
bad-credit applicants.

Train-Test Split

80% training

20% testing

Stratified

random_state = 42

Models

Logistic Regression

Decision Tree

Random Forest

Metrics

Accuracy

Precision

Recall

F1 Score

ROC-AUC

📊 Baseline Model Results

Model            Accuracy    Precision       Recall           F1      ROC-AUC

Random            0.735        0.561    0.533    0.547    0.767
Forest

Logistic        0.740    0.591        0.433        0.500        0.742
Regression





Model Selection

Random Forest was retained as the preferred baseline because it achieved
the strongest recall, F1 score, and ROC-AUC.

Logistic Regression had slightly higher accuracy and precision, but
detected fewer bad-credit applicants.

🔄 Cross-Validation

Five-fold stratified cross-validation was performed on the baseline
Random Forest.

Metric              Mean   Std Dev

Accuracy          0.7480    0.0319
Precision         0.5849    0.0586
Recall            0.5633    0.0748
F1 Score          0.5717    0.0567
ROC-AUC       0.7852    0.0447

Mean CV ROC-AUC: 0.7852 ± 0.0447

⚙️ Hyperparameter Tuning

RandomizedSearchCV evaluated 40 configurations across five folds.

Best configuration

n_estimators = 300
min_samples_split = 15
min_samples_leaf = 4
max_features = log2
max_depth = None
class_weight = balanced_subsample

Best cross-validation ROC-AUC:

0.7958

However:

Baseline test ROC-AUC: 0.7670

Tuned test ROC-AUC: 0.7542

Therefore, the baseline Random Forest was retained.

Important lesson: hyperparameter tuning does not guarantee better
held-out performance.

🎚️ Threshold Optimization

The default probability threshold of 0.50 was investigated because a
risk-sensitive workflow may prioritize detecting more potentially
bad-credit applicants.

A threshold of 0.44 was selected using cross-validated training
predictions and then evaluated on the untouched test set.

Metric                Threshold 0.50   Threshold 0.44

Accuracy                       73.5%            68.5%
Precision                      56.1%            48.2%
Bad-Credit Recall              53.3%        66.7%
F1 Score                       54.7%        55.9%



At threshold 0.44:

40 of 60 bad-credit applicants were detected

At 0.50, 32 of 60 were detected

8 additional bad-credit applicants were detected

18 additional false positives were produced

A classification threshold is a business decision, not simply a
machine-learning default.

🧮 Confusion Matrix

Threshold 0.50

[[115, 25],
 [ 28, 32]]

Threshold 0.44

[[97, 43],
 [20, 40]]



The lower threshold catches more bad-credit applicants but requires more
applicants to be flagged for additional review.

🌲 Feature Importance

Random Forest feature importance was aggregated back from one-hot
encoded columns to the original business-level variables.

Feature                         Aggregated Importance

Checking account status              0.178504
Credit Amount                                0.086125
Purpose                                      0.073441
Credit History                               0.072626
Credit Duration                              0.071242
Savings                                      0.066684
Age                                          0.064685
Property                                     0.047533
Employment duration                          0.044982
Installment percent                          0.038465



Checking account status was the dominant predictive factor.

Feature importance represents predictive contribution, not causality.

💡 Key Insights

1. Checking account status is the strongest signal

It was strongest in both Cramér's V analysis and Random Forest feature
importance.

2. Credit history is consistently important

Credit history ranked highly in both statistical association and
predictive modeling.

3. Credit amount and duration matter

Bad-credit applicants had higher average credit amounts and longer
average durations.

4. Statistical significance ≠ predictive importance

A variable can show a statistical association without necessarily being
one of the strongest predictive variables.

5. Threshold selection changes model behavior

Moving from 0.50 to 0.44 substantially increased bad-credit recall, but
at the cost of precision and accuracy.

6. Tuning is not automatically better

The tuned model had better CV ROC-AUC but worse held-out test ROC-AUC.

⭐ What Makes This Project Different?

Many credit-risk projects stop at:

EDA → Train Model → Report Accuracy

This project follows a more decision-oriented workflow:

EDA
 ↓
Risk-rate analysis
 ↓
Chi-Square + Cramér's V
 ↓
Machine Learning
 ↓
Cross-validation
 ↓
Feature Importance
 ↓
Threshold Optimization
 ↓
Business Trade-off

Statistical + Predictive + Business Triangulation

The same risk factors were examined from multiple analytical
perspectives.

The project also demonstrates a realistic modeling principle:

The best model is not necessarily the model with the highest single
metric.

The final decision considered generalization, recall, false-positive
trade-offs, and business interpretation instead of blindly selecting the
most complex or most highly tuned model.

🏦 Business Recommendations

Give appropriate analytical attention to checking-account status
and credit history.

Consider credit amount and repayment duration together.

Use statistically informative categorical variables as supporting
signals rather than isolated decision rules.

If catching more potentially risky applicants is the priority,
consider a lower threshold with additional review.

Track precision, recall, F1, and ROC-AUC alongside accuracy.

Select the final threshold using actual business costs for false
positives and false negatives.

🛠️ Tech Stack

Language - Python

Data Analysis - Pandas - NumPy

Visualization - Matplotlib - Seaborn

Statistics - SciPy

Machine Learning - Scikit-learn - Logistic Regression - Decision
Tree - Random Forest - RandomizedSearchCV - Cross-validation -
Pipeline - ColumnTransformer - StandardScaler - OneHotEncoder

📁 Project Structure

credit-risk-analysis/
│
├── data/
│   └── credit_data.csv
│
├── notebooks/
│   └── credit_risk_analysis.ipynb
│
├── assets/
│   ├── 01_target_distribution.png
│   ├── 02_numerical_means.png
│   ├── 03_cramers_v.png
│   ├── 04_model_roc_auc.png
│   ├── 05_model_metrics.png
│   ├── 06_threshold_tradeoff.png
│   ├── 07_feature_importance.png
│   └── 08_confusion_matrices.png
│
├── documentation/
│   ├── Credit_Risk_Analysis_Documentation.pdf
│   └── Credit_Risk_Analysis_Documentation.docx
│
└── README.md

⚠️ Limitations

The dataset contains only 1,000 historical applications.

Generalization to another population or current lending environment
is uncertain.

Integer-coded categorical variables require careful interpretation.

Feature importance does not establish causality.

The held-out Random Forest ROC-AUC of 0.767 represents useful
but imperfect discrimination.

Threshold selection did not incorporate real financial cost
estimates.

A production credit-risk system would require representative current
data, calibration, monitoring, fairness assessment, governance, and
additional validation.

📚 Key Lessons Learned

Check data quality before modeling.

Do not automatically treat integer-coded categorical variables as
continuous.

Compare within-category risk rates, not only raw counts.

Use statistical significance and effect size together.

Accuracy alone is insufficient for imbalanced classification.

Cross-validation provides a better view of model stability.

Hyperparameter tuning does not guarantee better held-out
performance.

Classification thresholds should reflect business objectives.

Translate model feature importance back into business-level
variables.

Strong analytics connects statistical evidence, model behavior,
and actionable decisions.

📈 Final Results

Result                                     Value

Preferred baseline model       Random Forest
Held-out test ROC-AUC                  0.767
5-fold CV ROC-AUC            0.7852 ± 0.0447
Risk-sensitive threshold                0.44
Bad-credit recall @ 0.44               66.7%

The completed project demonstrates an end-to-end credit-risk analytics
workflow with emphasis on model evaluation, risk sensitivity,
statistical validation, and business decision-making.

📄 Detailed Documentation

For the complete analysis, graphs, statistical results, model
evaluation, threshold analysis, feature importance, recommendations,
limitations, and lessons learned:

documentation/Credit_Risk_Analysis_Documentation.pdf

documentation/Credit_Risk_Analysis_Documentation.docx

👤 Author

Darsh Jilka

Machine Learning & Data Analytics Project
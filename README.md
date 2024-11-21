# **Machine Learning Project Report: Solving Regression and Classification Problems for Forest Fire Prediction and Analysis**

## **Abstract**
I successfully conducted a machine learning project to analyze and predict forest fires using the Algerian Forest Fires Dataset. The project demonstrates the end-to-end implementation of a machine learning pipeline, from data cleaning and exploratory analysis to feature selection, model training, and evaluation. Using advanced regression and classification techniques, I identified critical fire-driving factors and developed models to predict the Fire Weather Index (FWI) accurately. The insights gained underscore the importance of data preprocessing, multicollinearity handling, and regularization for robust predictive performance.

---

## **1. Introduction**
Forest fires are a growing environmental challenge that climate change and human activities exacerbate. Predicting fire occurrences and their intensity is crucial for early prevention and effective resource allocation. To address this, I worked with the Algerian Forest Fires Dataset, focusing on:
1. **Classification:** Predicting whether a fire occurs (`Fire` or `NotFire`).
2. **Regression:** Predicting the Fire Weather Index (FWI), a critical metric for fire risk assessment.

This report captures my machine learning pipeline's detailed methodology and results, showcasing how I addressed challenges such as multicollinearity, data skewness, and model overfitting.

---

## **2. Dataset Overview**

### **2.1 Source**
The dataset consists of 244 observations collected over four months (June-September 2012) from two Algerian regions:
1. **Bejaya Region:** Situated in northeast Algeria.
2. **Sidi Bel Abbes Region:** Located in northwest Algeria.

### **2.2 Features**
The dataset includes both numerical and categorical features:
- **Numerical Features:**
  - Weather data: Temperature, Relative Humidity (RH), Wind Speed (WS), Rainfall.
  - Fire Weather Indices: Fine Fuel Moisture Code (FFMC), Duff Moisture Code (DMC), Drought Code (DC), Initial Spread Index (ISI), Buildup Index (BUI), and Fire Weather Index (FWI).
- **Categorical Features:**
  - `Classes`: Fire occurrence (`Fire`/`NotFire`).
  - Region information was added to distinguish between the two datasets.

### **Target Variables**
1. **Classification Task:** Predicting `Classes` (Fire or NotFire).
2. **Regression Task:** Predicting `FWI` (numerical).

---

## **3. Methodology**
I followed a systematic pipeline comprising data preprocessing, exploratory data analysis (EDA), feature selection, and model training to achieve my objectives.

---

### **3.1 Data Preprocessing**
#### **Cleaning**
- I identified and removed rows with missing or incomplete values to ensure data integrity.
- Non-informative columns such as `Date,` `Day,` `Month,` and `Year` were dropped as they had no predictive relevance.

#### **Encoding**
- The categorical target variable `Classes` was encoded into numeric values:
  - `0`: NotFire
  - `1`: Fire
- I added a `Region` column to distinguish between the two datasets:
  - `0`: Bejaya Region
  - `1`: Sidi Bel Abbes Region.

#### **Scaling**
- I used `StandardScaler` to standardize all numerical features. This ensured that features with larger magnitudes (e.g., Temperature) did not dominate smaller-scale features (e.g., Rain).

#### **Unifying the Dataset**
- The datasets from the two regions were merged into a single dataset while retaining the regional labels for comparative analysis.

---

### **3.2 Exploratory Data Analysis (EDA)**
EDA was performed to uncover data patterns, relationships, and trends.

#### **Fire Analysis by Region**
I visualized monthly fire occurrences for each region:

**Bejaya Region:**  
![Alt text](<https://github.com/RhythmAhir/Forest-Fire-Prediction-and-Analysis/blob/main/Visualizations/Fire%20Analysis%20of%20Brjaia%20Regions.png>)
**Sidi Bel Abbes Region:**  
![Alt text](<https://github.com/RhythmAhir/Forest-Fire-Prediction-and-Analysis/blob/main/Visualizations/Fire%20Analysis%20of%20Sidi-%20Bel%20Regions.png>)

**Insights:**
- Fire occurrences were most frequent in **August** for both regions.
- Non-fire conditions were more common in **June** and **September**, likely due to cooler and wetter weather.

#### **Correlation Analysis**
I generated a heatmap to analyze feature correlations:  
![Alt text](<https://github.com/RhythmAhir/Forest-Fire-Prediction-and-Analysis/blob/main/Visualizations/Correlation%20Heatmap.png>)

**Key Observations:**
- Strong positive correlations were observed between:
  - `FWI` and `ISI` (ρ = 0.92)
  - `FWI` and `BUI` (ρ = 0.85)
- `RH` negatively correlated with `Temperature` (ρ = -0.65), indicating that higher temperatures coincide with lower humidity, a critical fire condition.

#### **Class Distribution**
To assess class balance, I visualized the proportion of fire and non-fire observations:  
![Alt text](<https://github.com/RhythmAhir/Forest-Fire-Prediction-and-Analysis/blob/main/Visualizations/Pie%20Chart%20of%20Classes.png>)

**Findings:**
- Fire incidents accounted for 56.4% of the dataset, with non-fire conditions making up 43.6%. This balanced distribution supported robust classification model training.

#### **Feature Distributions**
I plotted density distributions to understand the spread and skewness of features:  
![Alt text](<https://github.com/RhythmAhir/Forest-Fire-Prediction-and-Analysis/blob/main/Visualizations/desnity%20plot%20for%20all%20features.png>)

**Insights:**
- Temperature, FFMC, and FWI showed right-skewed distributions.
- Rainfall was negligible in most cases, underscoring its limited impact on fire occurrence.

---

### **3.3 Feature Selection**
#### **Multicollinearity Handling**
Using the correlation matrix, I identified and removed highly correlated features to reduce redundancy:
- `BUI` was dropped due to its strong correlation with `FWI` and `ISI.`
- `DC` was also removed for similar reasons.

#### **Retained Features**
- `Temperature,` `RH,` `WS,` `Rain,` `FFMC,` `DMC,` and `ISI.`

---

### **3.4 Model Training**
To achieve the project objectives, I implemented both regression models (for predicting FWI) and classification models (for predicting fire occurrence). Each model was trained and evaluated using appropriate metrics.

#### **Regression Models**
For predicting the Fire Weather Index (FWI), I used the following regression models:
1. **Linear Regression:**
   - This served as the baseline model and provided initial insights into the relationship between features and FWI.
   - It achieved an **R² of 0.98** and a **MAE of 0.56**, indicating a strong fit to the data.
2. **Lasso Regression:**
   - L1 regularization was applied to penalize less significant features and prevent overfitting.
   - This model slightly sacrificed accuracy for robustness, achieving an **R² of 0.94** and a **MAE of 1.13**.
3. **Ridge Regression:**
   - L2 regularization was used to balance model complexity and predictive accuracy.
   - This model maintained a strong **R² of 0.98** and a **MAE of 0.56**, showing comparable performance to Linear Regression while mitigating overfitting risks.
4. **ElasticNet Regression:**
   - Combining L1 and L2 penalties, this model aimed to balance feature selection and regularization.
   - It achieved an **R² of 0.87** and a **MAE of 1.88**, highlighting the importance of careful parameter tuning.

#### **Classification Models**
For predicting fire occurrence (`Fire` or `NotFire`), I trained the following classification models:
1. **Logistic Regression:**
   - A simple and interpretable baseline model for binary classification.
   - Achieved an **accuracy of 94%**, with **precision of 93%** and **recall of 96%** for the `NotFire` class, and **precision of 95%** and **recall of 91%** for the `Fire` class.
2. **Decision Tree Classifier:**
   - Captured non-linear relationships between features and target labels.
   - Achieved **100% accuracy**, precision, recall, and F1-score for both `Fire` and `NotFire` classes.
3. **Random Forest Classifier:**
   - Leveraged ensemble learning to improve predictive performance and reduce overfitting.
   - Achieved **100% accuracy**, precision, recall, and F1-score for both `Fire` and `NotFire` classes.
4. **Gradient Boosting Classifier:**
   - Focused on sequential learning to improve weak learners.
   - Achieved **100% accuracy**, precision, recall, and F1-score for both `Fire` and `NotFire` classes.

#### **Cross-Validation**
For both regression and classification tasks, I applied k-fold cross-validation to ensure the models generalized well to unseen data:
- For regression, I used `LassoCV` and `RidgeCV` to optimize regularization parameters (α).
- For classification, I used grid search to tune hyperparameters such as the maximum depth of trees and learning rates for ensemble models.

---

## **4. Results**
### **Regression Performance**
| Model              | R² Score | Mean Absolute Error (MAE) |
|---------------------|----------|---------------------------|
| Linear Regression   | 0.98     | 0.56                      |
| Lasso Regression    | 0.98     | 0.61                      |
| Ridge Regression    | 0.98     | 0.56                      |
| ElasticNet Regression | 0.98   | 0.65                      |

**Observations:**
- Ridge Regression matched the performance of Linear Regression while offering better robustness against overfitting.
- Regularization techniques (Lasso, ElasticNet) effectively penalized irrelevant features but at the cost of slight reductions in predictive performance.

---

### **Classification Performance**
| Model                       | Accuracy | Precision | Recall | F1-Score |
|------------------------------|----------|-----------|--------|----------|
| Logistic Regression          | 94%      | 94%       | 94%    | 94%      |
| Decision Tree Classifier     | 100%     | 100%      | 100%   | 100%     |
| Random Forest Classifier     | 100%     | 100%      | 100%   | 100%     |
| Gradient Boosting Classifier | 100%     | 100%      | 100%   | 100%     |

**Observations:**
- Decision Tree, Random Forest, and Gradient Boosting Classifiers all achieved perfect performance on the test set.
- Logistic Regression performed well but slightly lagged behind ensemble methods in recall and precision.

---

### **Key Takeaways**
1. **Regression:**
   - Ridge Regression was the most robust model for FWI prediction, achieving high accuracy while avoiding overfitting.
   - ElasticNet requires careful hyperparameter tuning but can balance feature selection and regularization effectively.

2. **Classification:**
   - Ensemble models like Decision Tree, Gradient Boosting, and Random Forest delivered exceptional performance, capturing both linear and non-linear feature interactions effectively.
   - Logistic Regression provided a reliable baseline, but its linear nature limited its performance compared to ensemble methods.

---

## **5. Key Findings**
1. **Seasonal Trends:**
   - Fires were most frequent in July-August, coinciding with high temperatures and low RH.
2. **Feature Importance:**
   - Temperature, ISI, and FFMC emerged as the strongest predictors.
   - Rainfall and RH showed inverse relationships with fire risks.
3. **Model Performance:**
   - Ridge Regression offered the best balance of accuracy and generalizability.

---

## **6. Conclusion**
This project highlights the application of machine learning to predict forest fire occurrences and indices. By employing a structured workflow, I demonstrated the importance of feature selection, regularization, and hyperparameter tuning in developing robust models. The insights gained can inform fire prevention strategies and resource allocation.

---

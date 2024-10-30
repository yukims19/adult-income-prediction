# Modeling and Predicting Income from Socioeconomic Data

## Overview

This project focuses on predicting individual income levels based on demographic and socioeconomic factors using the Adult dataset from 1996. By building and evaluating several machine learning models, the goal is to identify the factors most strongly correlated with income and create a robust predictive system.

[Explore the web app here!](https://income-prediction-app.vercel.app/)

![Screen Recording 2024-10-30 at 9 32 08 AM](https://github.com/user-attachments/assets/9481183e-a3ff-4d48-a74a-b2545078dfad)

## Problem Statement

The objective is to develop a model that predicts whether an individual's income exceeds $50,000 using features such as age, education, occupation, and other relevant factors.

Dataset source: [Adult dataset details](https://www.cs.toronto.edu/~delve/data/adult/adultDetail.html)

## Notable Findings and Insights

### 1. Initial Exploration: Uncovering Patterns in the Data

Our **exploratory data analysis (EDA)** revealed several key patterns:

- **Age, education, and hours worked per week** show strong correlations with income levels.
  ![Correlation between numeric features and income](<images/Screenshot 2024-10-19 at 12.25.53 PM.png>)
  _This heatmap shows the correlation between numeric features. We observe that age, hours-per-week, and educational-num have moderate positive correlations with income, indicating that these factors are good predictors of higher income. This justifies their inclusion in the model._

  &nbsp;<br>

  ![Age distribution across income levels](<images/Screenshot 2024-10-19 at 12.18.02 PM.png>)
  _This violin plot highlights the distribution of ages across income groups. It shows that higher-income individuals tend to be older, particularly between the ages of 35 and 50. This further supports the insight that age is a significant predictor of income._

  &nbsp;<br>

- Education, workclass & occupation, and marital status also appear to influence income levels:
  ![Comparision of income by workclass and occupation](<images/Screenshot 2024-10-19 at 12.31.14 PM.png>)
  _The bar plot shows the income distribution across workclass and occupation. We see that managerial and professional occupations are associated with higher income levels. This insight highlights the importance of occupation in determining income, as higher-skilled professions tend to lead to higher earnings._

  &nbsp;<br>

  ![Analysis of relationship status and income levels](<images/Screenshot 2024-10-19 at 12.35.13 PM.png>)
  _This analysis of relationship status versus income reveals that married individuals, especially those in dual-income households, are more likely to be high earners. This provides additional evidence that marital status is a significant predictor of income._

  &nbsp;<br>

These insights guided us in selecting the most important features for model development.

### 2. Identifying Key Factors: What Influences Income the Most?

Through statistical tests, we found significant differences between individuals earning above and below $50,000:

- Marital status: Married individuals, especially in dual-income households, were more likely to be high earners.
- Education: Higher educational attainment was associated with higher income.
- Age: Older individuals were more likely to earn more.
- Occupation and work hours: Managerial roles and longer work hours correlated strongly with higher income.

### 3. Building Predictive Models: From Insights to Action

With these insights, we built several predictive models to classify whether an individual's income exceeds $50,000:

- **Logistic Regression** served as our initial benchmark, offering interpretability but lacking the ability to capture complex, non-linear relationships.
- **Decision Trees** and **Random Forests** captured intricate interactions between factors like marital status, education, and work hours.
- **Gradient Boosting** emerged as the top performer, delivering the highest accuracy and precision.

### 4. Model Performance: Measuring Success

Through **cross-validation** and **hyperparameter tuning**, we optimized the models for better performance. Both **Random Forest** and **Gradient Boosting** models achieved high accuracy and precision, outperforming the Logistic Regression model. The tree-based models were able to capture more complex relationships between the variables.

#### Feature Importance Analysis

The following chart highlights the top 10 most important features for predicting income levels using the Decision Tree model:

![Top 10 feature importance](<images/Screenshot 2024-10-19 at 1.03.37 PM.png>)

As seen in the chart, features such as marital status, education, and age are the strongest predictors of income, which align with our initial findings in the exploratory analysis.

### 5. Conclusion: Insights Beyond Prediction

This project successfully developed a reliable model for income classification, but it also deepened our understanding of the factors driving income inequality. **Education**, **occupation**, and **marital status** consistently emerged as the most impactful variables, both for prediction and understanding socioeconomic disparities.

Future directions for this project could include:

- Exploring interaction effects between demographic variables.
- Extending the model to predict income across different industries or regions.
- Using more advanced models, such as neural networks, to capture additional complexities in the data.

## Tools and Libraries

- **Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, SciPy
- **Machine Learning Models**: Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, Support Vector Machines (SVM), K-Nearest Neighbors, Naive Bayes

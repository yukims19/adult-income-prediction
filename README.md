# Modeling and Predicting Income from Socioeconomic Data

## Overview

This project aims to predict individual income levels based on demographic and socioeconomic factors using the well-known Adult dataset from 1996. By building and evaluating several machine learning models, we seek to identify the factors most strongly correlated with income and create a robust predictive system.

## Problem Statement

The primary objective is to analyze and develop a model to predict whether an individual's income exceeds $50,000, using features such as age, education, occupation, and more.

Dataset source: [Adult dataset details](https://www.cs.toronto.edu/~delve/data/adult/adultDetail.html)

## Notable Findings and Insights

### 1. Initial Exploration: Uncovering Patterns in the Data

Our **exploratory data analysis (EDA)** revealed several key patterns:

- **Age, education, and hours worked per week** show strong correlations with income levels.
- Individuals with higher educational attainment and those working longer hours tend to have higher incomes.

These insights guided us in selecting the most important features for model development.

### 2. Identifying Key Factors: What Influences Income the Most?

Using statistical tests, we identified several significant differences between high-income and low-income individuals:

- **Marital status** proved to be a strong indicator, with married individuals, especially in dual-income households, more likely to be high earners.
- **Occupation** and **work hours** also played critical roles, with managerial roles being particularly associated with higher income levels.

### 3. Building Predictive Models: From Insights to Action

With these insights, we built several predictive models to classify whether an individual's income exceeds $50,000:

- **Logistic Regression** served as our initial benchmark, offering interpretability but lacking the ability to capture complex, non-linear relationships.
- **Decision Trees** and **Random Forests** captured intricate interactions between factors like marital status, education, and work hours.
- **Gradient Boosting** emerged as the top performer, delivering the highest accuracy and precision.

### 4. Model Performance: Measuring Success

Through **cross-validation** and **hyperparameter tuning**, we optimized our models for better performance. The **Random Forest** and **Gradient Boosting** models both achieved exceptional accuracy and precision. While **Logistic Regression** was useful for initial insights, its performance was surpassed by tree-based methods.

### 5. Conclusion: Insights Beyond Prediction

This project not only developed a reliable model for income classification but also deepened our understanding of the factors driving income inequality. **Education, occupation, and marital status** consistently emerged as the most impactful variables, shaping both the model’s success and our understanding of the data.

Future work could explore more complex relationships, such as interaction effects between demographic variables, or expand the model to other domains like predicting income in different industries or regions.

## Tools and Libraries

- **Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, SciPy
- **Machine Learning Models**: Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, Support Vector Machines (SVM), K-Nearest Neighbors, Naive Bayes

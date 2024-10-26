# Statistical-Control-Project
# Import necessary libraries
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm

# Step 1: Load the data from 'moviedataReplicationSet.csv'
df = pd.read_csv('moviedataReplicationSet.csv')

# Check the first few rows to confirm data is loaded correctly
print(df.head())

# Step 2: Correlate education and income
correlation, p_value = pearsonr(df['education'], df['income'])
print(f"Correlation between education and income: {correlation:.4f} (p-value: {p_value:.4f})")

# Step 3: Compute the partial correlation between education and income, controlling for SES
def compute_partial_corr(df, x, y, control):
    """Compute partial correlation between x and y controlling for control variable."""
    model1 = sm.OLS(df[x], sm.add_constant(df[control])).fit()
    model2 = sm.OLS(df[y], sm.add_constant(df[control])).fit()
    residuals_x = model1.resid
    residuals_y = model2.resid
    partial_corr_result, p_val = pearsonr(residuals_x, residuals_y)
    return partial_corr_result, p_val

# Calculate partial correlation
partial_corr_value, partial_p_value = compute_partial_corr(df, 'education', 'income', 'SES')
print(f"Partial correlation (controlling for SES): {partial_corr_value:.4f} (p-value: {partial_p_value:.4f})")

# Step 4: Build a multiple regression model that predicts income from education and SES
X = df[['education', 'SES']]  # Predictors
X = sm.add_constant(X)  # Add a constant (intercept)
y = df['income']  # Response variable

# Fit the multiple regression model
model = sm.OLS(y, X).fit()

# Print the summary of the regression model
print(model.summary())


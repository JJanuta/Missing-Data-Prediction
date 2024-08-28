# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:17:45 2024

@author: jenni
"""

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
df_dirty = pd.read_csv('C:/Users/jenni/Desktop/Python_Projects/dirty_sales_data.csv')

# Display the first few rows to understand the structure of the data
print(df_dirty.head())

# Check for missing data
print(df_dirty.isnull().sum())

# Visualize missing data using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df_dirty.isnull(), cbar=False, cmap="viridis")
plt.show()

### Predict Missing Prices using Linear Regression ###
######################################################

# Impute missing 'Quantity' values with the mean
imputer = SimpleImputer(strategy='mean')
df_dirty['Quantity'] = imputer.fit_transform(df_dirty[['Quantity']])

# One-hot encode categorical variables ('Customer' and 'Product')
df_encoded = pd.get_dummies(df_dirty, columns=['Customer', 'Product'])

# Separate data into subsets for price prediction
df_with_prices = df_encoded.dropna(subset=['Price'])
df_missing_prices = df_encoded[df_encoded['Price'].isnull()]

# Prepare feature matrix (X) and target vector (y) for price prediction
X = df_with_prices[['Quantity'] + list(df_encoded.columns[df_encoded.columns.str.startswith("Product_")])]
y = df_with_prices['Price']

# Train the linear regression model to predict prices
model = LinearRegression()
model.fit(X, y)

# Predict missing 'Price' values
X_missing = df_missing_prices[['Quantity'] + list(df_encoded.columns[df_encoded.columns.str.startswith("Product_")])]
df_dirty.loc[df_dirty['Price'].isnull(), 'Price'] = model.predict(X_missing)

# Review the 'Price' column after filling missing values
print(df_dirty['Price'].head(45))
print(df_dirty["Price"].describe()) 

### Predict Missing Quantities using Linear Regression ###
##########################################################

# Prepare data for quantity prediction
df_with_quantities = df_encoded.dropna(subset=['Quantity'])
X = df_with_quantities.drop(columns=['Quantity', 'OrderID', 'OrderDate', 'Price', 'ShippingCost', 'DiscountCode'])
y = df_with_quantities['Quantity']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model to predict quantities
model = LinearRegression()
model.fit(X_train, y_train)

# Predict missing 'Quantity' values
df_missing_quantities = df_encoded[df_encoded['Quantity'].isnull()]
X_missing = df_missing_quantities.drop(columns=['Quantity', 'OrderID', 'OrderDate', 'Price', 'ShippingCost', 'DiscountCode'])
df_dirty.loc[df_dirty['Quantity'].isnull(), 'Quantity'] = model.predict(X_missing)

# Review the 'Quantity' column after filling missing values
print(df_dirty['Quantity'].describe())
print(df_dirty[["Customer", "Quantity"]])

### Impute Missing 'ShippingCost' and 'DiscountCode' ###
########################################################

# Fill missing 'ShippingCost' values with the mean
df_dirty["ShippingCost"].fillna(df_dirty["ShippingCost"].mean(), inplace=True)

# Fill missing 'DiscountCode' values with 'None'
df_dirty["DiscountCode"].fillna("None", inplace=True)

# Save the cleaned data to a new CSV file
df_dirty.to_csv('C:/Users/jenni/Desktop/Python_Projects/Cleaned_Data.csv', index=False)


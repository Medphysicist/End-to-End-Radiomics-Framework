# analysis.py
"""
This module contains all functions for downstream analysis and clinical correlation.
It includes methods for univariate and multivariate feature selection and
visualization tools like correlation heatmaps.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV

def run_univariate_analysis(merged_df, feature_cols, target_variable):
    """
    Performs univariate correlation analysis to find features with the
    strongest one-to-one correlation with a clinical outcome.

    Args:
        merged_df (pd.DataFrame): DataFrame containing both features and clinical data.
        feature_cols (list): A list of column names for the radiomic features.
        target_variable (str): The column name of the clinical outcome.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A DataFrame of the top 15 correlated features.
            - matplotlib.figure.Figure: A bar plot visualizing the correlations.
    """
    if target_variable not in merged_df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in the DataFrame.")

    # Calculate correlation of each feature with the target variable
    correlations = merged_df[feature_cols].corrwith(merged_df[target_variable])

    # Remove any NaN values that might result from columns with no variance
    correlations.dropna(inplace=True)

    # Get the absolute correlation and sort to find the strongest relationships
    top_features = correlations.abs().sort_values(ascending=False).head(15)

    # Prepare DataFrame for display
    top_features_df = top_features.reset_index()
    top_features_df.columns = ['Feature', 'Absolute Correlation']

    # Create the visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=top_features.values, y=top_features.index, ax=ax, palette="viridis")
    ax.set_title(f"Top 15 Features Correlated with '{target_variable}'", fontsize=16)
    ax.set_xlabel("Absolute Correlation", fontsize=12)
    ax.set_ylabel("Radiomic Feature", fontsize=12)
    plt.tight_layout()

    return top_features_df, fig


def run_lasso_selection(merged_df, feature_cols, target_variable):
    """
    Uses a LassoCV machine learning model to perform multivariate feature selection.
    This method identifies a group of features that are jointly predictive of the outcome.

    Args:
        merged_df (pd.DataFrame): DataFrame containing both features and clinical data.
        feature_cols (list): A list of column names for the radiomic features.
        target_variable (str): The column name of the clinical outcome.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A DataFrame of features selected by the model.
            - matplotlib.figure.Figure or None: A bar plot of feature importances, or None if no features were selected.
    """
    # 1. Prepare data (X: features, y: target)
    X = merged_df[feature_cols].copy()
    y = merged_df[target_variable].copy()

    # 2. Pre-processing: Handle missing values and scale data
    # Impute missing values (a common necessity for real-world data)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Scale features - essential for regularized models like Lasso
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

    # 3. Train the LassoCV model
    # LassoCV automatically finds the best regularization strength (alpha) via cross-validation
    lasso = LassoCV(cv=5, random_state=42, max_iter=5000, n_jobs=-1).fit(X_scaled, y)

    # 4. Extract and rank the selected features
    coefficients = pd.Series(lasso.coef_, index=X_scaled.columns)
    
    # Select features with non-zero coefficients
    selected_features = coefficients[coefficients != 0].abs().sort_values(ascending=False)

    if selected_features.empty:
        return pd.DataFrame(), None  # Return empty if no features were selected

    # Prepare DataFrame for display
    selected_features_df = selected_features.reset_index()
    selected_features_df.columns = ['Feature', 'Coefficient Magnitude']

    # 5. Create the visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=selected_features.values, y=selected_features.index, ax=ax, palette="mako")
    ax.set_title(f"Features Selected by Lasso for '{target_variable}'", fontsize=16)
    ax.set_xlabel("Coefficient Magnitude (Importance)", fontsize=12)
    ax.set_ylabel("Radiomic Feature", fontsize=12)
    plt.tight_layout()

    return selected_features_df, fig


def generate_correlation_heatmap(merged_df, columns_to_include):
    """
    Generates a correlation heatmap for a selected list of features and clinical variables.

    Args:
        merged_df (pd.DataFrame): DataFrame containing the data.
        columns_to_include (list): A list of column names to include in the heatmap.

    Returns:
        matplotlib.figure.Figure: The generated heatmap figure.
    """
    if len(columns_to_include) < 2:
        raise ValueError("Please select at least two variables for the heatmap.")

    # Select the relevant data and compute the correlation matrix
    heatmap_df = merged_df[columns_to_include]
    corr_matrix = heatmap_df.corr()

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr_matrix,
        annot=True,          # Show the correlation values on the map
        cmap='coolwarm',     # Use a diverging colormap
        fmt=".2f",           # Format annotations to two decimal places
        linewidths=.5,
        ax=ax
    )
    ax.set_title("Correlation Heatmap", fontsize=18)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    return fig

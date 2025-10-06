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
    # Validate input
    if target_variable not in merged_df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in the DataFrame.")

    # Filter out non-numeric features and target variable
    numeric_feature_cols = []
    for col in feature_cols:
        if col in merged_df.columns and pd.api.types.is_numeric_dtype(merged_df[col]):
            numeric_feature_cols.append(col)

    if not numeric_feature_cols:
        raise ValueError("No numeric features found for correlation analysis.")

    # Check if target variable is numeric
    if not pd.api.types.is_numeric_dtype(merged_df[target_variable]):
        # Try to convert categorical target to numeric if possible
        if merged_df[target_variable].nunique() <= 10:  # Likely categorical
            try:
                merged_df = merged_df.copy()
                merged_df[target_variable] = pd.factorize(merged_df[target_variable])[0]
            except:
                raise ValueError(f"Target variable '{target_variable}' is not numeric and cannot be converted.")
        else:
            raise ValueError(f"Target variable '{target_variable}' is not numeric.")

    # Calculate correlation of each feature with the target variable
    try:
        correlations = merged_df[numeric_feature_cols].corrwith(merged_df[target_variable])
    except Exception as e:
        raise ValueError(f"Error calculating correlations: {str(e)}")

    # Remove any NaN values that might result from columns with no variance
    correlations.dropna(inplace=True)

    if correlations.empty:
        raise ValueError("No valid correlations could be calculated.")

    # Get the absolute correlation and sort to find the strongest relationships
    top_features = correlations.abs().sort_values(ascending=False).head(15)

    # Prepare DataFrame for display
    top_features_df = top_features.reset_index()
    top_features_df.columns = ['Feature', 'Absolute Correlation']

    # Create the visualization with updated seaborn syntax
    fig, ax = plt.subplots(figsize=(10, 8))

    # Convert to DataFrame for proper plotting
    plot_df = top_features_df.copy()
    plot_df['Feature'] = plot_df['Feature'].astype(str)  # Ensure string type

    # Create barplot with updated syntax to avoid deprecation warnings
    sns.barplot(
        data=plot_df,
        x='Absolute Correlation',
        y='Feature',
        palette="viridis",
        hue='Feature',  # Required to avoid deprecation warning
        legend=False    # Hide legend since we don't need it
    )

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
            - dict: A dictionary of features and their coefficients.
            - matplotlib.figure.Figure or None: A bar plot of feature importances, or None if no features were selected.
    """
    # Validate target variable
    if target_variable not in merged_df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in the DataFrame.")

    # Filter to only numeric features
    numeric_feature_cols = []
    for col in feature_cols:
        if col in merged_df.columns and pd.api.types.is_numeric_dtype(merged_df[col]):
            numeric_feature_cols.append(col)

    if not numeric_feature_cols:
        raise ValueError("No numeric features found for Lasso selection.")

    # Check if target variable is numeric
    if not pd.api.types.is_numeric_dtype(merged_df[target_variable]):
        # Try to convert categorical target to numeric if possible
        if merged_df[target_variable].nunique() <= 10:  # Likely categorical
            try:
                merged_df = merged_df.copy()
                merged_df[target_variable] = pd.factorize(merged_df[target_variable])[0]
            except:
                raise ValueError(f"Target variable '{target_variable}' is not numeric and cannot be converted.")
        else:
            raise ValueError(f"Target variable '{target_variable}' is not numeric.")

    # Prepare data (X: features, y: target)
    X = merged_df[numeric_feature_cols].copy()
    y = merged_df[target_variable].copy()

    # Check for and remove columns with all NaN values
    X = X.dropna(axis=1, how='all')

    # Check if we have any features left
    if X.empty:
        raise ValueError("No valid features remaining after filtering.")

    # Pre-processing: Handle missing values and scale data
    try:
        # Impute missing values (a common necessity for real-world data)
        imputer = SimpleImputer(strategy='mean')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Scale features - essential for regularized models like Lasso
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

        # Train the LassoCV model
        # LassoCV automatically finds the best regularization strength (alpha) via cross-validation
        lasso = LassoCV(cv=5, random_state=42, max_iter=5000, n_jobs=-1).fit(X_scaled, y)

        # Extract and rank the selected features
        coefficients = pd.Series(lasso.coef_, index=X_scaled.columns)
        selected_features = coefficients[coefficients != 0]

        if selected_features.empty:
            return {}, None  # Return empty if no features were selected

        # Create the visualization with updated seaborn syntax
        fig, ax = plt.subplots(figsize=(10, 8))

        # Convert to DataFrame for proper plotting
        plot_df = selected_features.abs().sort_values(ascending=False).reset_index()
        plot_df.columns = ['Feature', 'Coefficient Magnitude']
        plot_df['Feature'] = plot_df['Feature'].astype(str)  # Ensure string type

        # Create barplot with updated syntax
        sns.barplot(
            data=plot_df,
            x='Coefficient Magnitude',
            y='Feature',
            palette="mako",
            hue='Feature',  # Required to avoid deprecation warning
            legend=False    # Hide legend since we don't need it
        )

        ax.set_title(f"Features Selected by Lasso for '{target_variable}'", fontsize=16)
        ax.set_xlabel("Coefficient Magnitude (Importance)", fontsize=12)
        ax.set_ylabel("Radiomic Feature", fontsize=12)
        plt.tight_layout()

        # Return coefficients as dictionary instead of DataFrame for more flexibility
        return selected_features.to_dict(), fig

    except Exception as e:
        raise ValueError(f"Error during Lasso selection: {str(e)}")

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

    # Generate heatmap with mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        annot=True,          # Show the correlation values on the map
        cmap='coolwarm',     # Use a diverging colormap
        fmt=".2f",           # Format annotations to two decimal places
        linewidths=.5,
        ax=ax,
        mask=mask,           # Mask upper triangle for cleaner visualization
        vmin=-1,             # Set color scale limits
        vmax=1
    )

    ax.set_title("Correlation Heatmap", fontsize=18)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    return fig

def generate_feature_importance_plot(feature_importances, title="Feature Importance"):
    """
    Generates a bar plot of feature importances with proper seaborn syntax.

    Args:
        feature_importances (dict or pd.Series): Feature names and their importance scores.
        title (str): Title for the plot.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    # Convert to DataFrame if it's not already
    if isinstance(feature_importances, dict):
        importances_df = pd.DataFrame.from_dict(
            feature_importances,
            orient='index',
            columns=['Importance']
        ).reset_index()
        importances_df.columns = ['Feature', 'Importance']
    else:
        importances_df = feature_importances.reset_index()
        importances_df.columns = ['Feature', 'Importance']

    # Sort by importance
    importances_df = importances_df.sort_values('Importance', key=abs, ascending=False)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use updated seaborn syntax
    sns.barplot(
        data=importances_df,
        x='Importance',
        y='Feature',
        palette="viridis",
        hue='Feature',  # Required to avoid deprecation warning
        legend=False    # Hide legend since we don't need it
    )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    plt.tight_layout()

    return fig

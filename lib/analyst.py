import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plotter(plot, columns, figsize=(12, 8)):
    """
    Plot subplots to show the distribution, trends and relations between data

    @params:
        plot (lambda): A lambda function that plots the subplots
        figsize (tuple): The size of the plots
        columns (list): The features to be plotted

    @returns:
        null
    """

    # Set the figsize for the plots
    plt.figure(figsize=figsize)

    # Plot each column as a subplot
    for i, col in enumerate(columns):
        plot((i, col))

    plt.tight_layout()
    plt.show()
    

def remove_outliers(df: pd.DataFrame, columns: str, n = 1.5) -> pd.DataFrame:
    """
    Remove outliers from a DataFrame based on the IQR method.

    @param: df (pd.DataFrame): The DataFrame to process.
    @param: column (str): The column name to check for outliers.

    @returns: pd.DataFrame: The DataFrame with outliers removed.
    """

    # Copy the dataframe
    df = df.copy()

    for column in columns:
        # Get the first quartile
        Q1 = df[column].quantile(0.25)

        # Get the third quartile
        Q3 = df[column].quantile(0.75)

        # Get the IQR
        IQR = Q3 - Q1

        # Set the lower and upper bounds
        lower_bound = Q1 - n * IQR
        upper_bound = Q3 + n * IQR

        # Remove the outliers in the feature
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return df


def get_cashflow(df: pd.DataFrame, group):
    """
        Get the cashflow with respect to the provided group

        @params group: the feature to group 
    """
    
    # Find groups with the highest volumn of transacionts
    cash_flow = df.groupby([group, 'type'])['amount'].sum().sort_values(ascending=False).to_frame().reset_index()

    # Get the netflow for each group
    cash_flow_pivot = cash_flow.pivot_table(index=group, columns='type', values='amount', aggfunc='sum').reset_index()
    cash_flow_pivot['NETFLOW'] = cash_flow_pivot['CREDIT'] - cash_flow_pivot['DEBIT']

    # Add the netflow for each group
    for netflow in cash_flow_pivot[[group, 'NETFLOW']].itertuples(index=False):
        cash_flow = pd.concat([cash_flow, pd.DataFrame({group: [netflow[0]], 'type': 'NETFLOW', 'amount': [netflow.NETFLOW]})], ignore_index=True)

    return cash_flow


def plot_cashflow(df: pd.DataFrame, columns):
    """
        Plot the cashflow with respect to the provided groups

        @params columns: the features to group 
    """

    plt.figure(figsize=(20, 20))

    for i, feature in enumerate(columns):
        cash_flow = get_cashflow(df, feature)
        plt.subplot(len(columns), 2, i+1)
        
        # Plot the cashflow by state of transaction using bar plot
        sns.barplot(data=cash_flow, x=feature, y='amount', hue='type')
        plt.title(f'Cashflow by {feature.capitalize()}')
        plt.xlabel(feature.capitalize())
        plt.ylabel('Amount')
        plt.tick_params(axis='x', rotation=90)
        
    plt.tight_layout()
    plt.show()


def bounded_correlation(corr_matrix, threshold=(.2, .8)):
    """
        Extract the correlations that are not too low or too high both positively or negatively

        @params threshold: The boundary allowed for correlation
    """

    # Initialize our dictionary for storing correlation of features
    data = {}

    # Get the valid correlation of each feature
    for column in corr_matrix:
        # Get correlation from the matrix
        corr = corr_matrix[column]

        # Get the correlations within the boundary
        in_boundary = corr[((corr >= threshold[0]) & (corr <= threshold[1])) | (corr <= -threshold[0]) & (corr >= -threshold[1])]

        # Add correlations that are within boundary
        if len(in_boundary.index):
            data.update({column: in_boundary})
    
    return data


def plot_bounded_correlation(data, threshold=(.2, .8)):
    """
        Get the correlations for the feature above or below the threshold (+-0.2 and +-0.8) as defaults

        @params threshold: The boundary allowed for correlation
    """

    columns = data.keys()
    for i, feature in enumerate(columns):
        plt.subplot(int(np.ceil(len(columns))/2), 4, i + 1)
        sns.barplot(data=data[feature])
        plt.tick_params(axis='x', rotation=90)
        plt.title(f'Correlation of {feature} with Features')

    plt.tight_layout()
    plt.show()


def plot_distribution(df: pd.DataFrame, columns, feature):
    # Plot the distribution of the specified columns with the feature
    for i, column in enumerate(columns):
        plt.subplot(len(columns), 2, i+1)
        sns.histplot(data=df, x=column, hue=feature)
        plt.title(f'Distribution of {column.capitalize()} by {feature.capitalize()}')
        plt.xlabel(column.capitalize())
        plt.ylabel('Frequency')
        plt.tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()


def plot_scatter(df: pd.DataFrame, feature, columns, color, limit):
    # Plot the scatter relationship of the specified columns with the feature
    for i, column in enumerate(columns):
        plt.subplot(len(columns), 2, i+1)
        sns.scatterplot(data=df, y=feature, x=column, hue=color)
        plt.title(f'Spread of {column} and {feature} along {color}')
        plt.ylabel(feature.capitalize())
        plt.xlabel(column.capitalize())
        plt.yscale(value='log')
        plt.axhline(y=limit, color='red', linestyle='--')
        plt.tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()    


def plot_trend(df: pd.DataFrame, features):
    # Plot the trend of the specified features
    plt.figure(figsize=(20, 20))

    for i, feature in enumerate(features):
        plt.subplot(len(features), 1, i+1)
        trend = df.groupby(feature).size().reset_index(name='count')
        
        sns.lineplot(trend, x=feature, y='count', marker='o')
        plt.title(f"{feature} Transaction Trend")
        plt.xlabel(feature.capitalize())
        plt.ylabel("Count")
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
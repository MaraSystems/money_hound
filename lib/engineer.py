import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, RobustScaler

from .tracker import hound, distance
from lib import analyst


def encoder(df, features):
    """
    Encode categorical features using Label Encoding.

    @param: df (pd.DataFrame): The DataFrame to process.
    @param: features (list): The list of features to encode.
    """
    
    feature_df = pd.DataFrame(index=df.index)
    for col in features:
        feature_df[col] = LabelEncoder().fit_transform(df[col])

    return feature_df


def count_related(df: pd.DataFrame, transaction, target, feature):
    return len(df[(df[target] == transaction[target]) & (df[feature] == transaction[feature])])


def bound_relation(df: pd.DataFrame, transaction, target, feature):
    name = feature['name']
    value = df[name]
    (low, high) = feature['bound'](value)
    tx_value = transaction[name]

    return len(df[(df[target] == transaction[target]) & (low <= tx_value) & (tx_value <= high)])


def count_occurrence(df: pd.DataFrame, transaction, target, feature, value):
    return len(df[(df[target] == transaction[target]) & (df[feature] == value)])


def get_occurrence_count(df: pd.DataFrame, target, features):
    df = df.sort_values(by='time').copy()
    data = df[[target, *[x['name'] for x in features]]]
    
    counts_dict = {feat['name']: {} for feat in features}
    results = {f"{target}_{feat['name']}_{feat['value']}_occurance": [] for feat in features}
    
    for idx, row in data.iterrows():
        for feat in features:
            feat_name = feat['name']
            feat_value = feat['value']
            row_value = row[feat_name]
            key = (row[target], row[feat_name])

            # Check if in bound
            if row_value == feat_value:
                # Get current count
                count = counts_dict[feat_name].get(key, 0)
                results[f"{target}_{feat_name}_{feat_value}_occurance"].append(count)
                # Update count
                counts_dict[feat_name][key] = count + 1
            else:
                # Out of bounds → count 0
                results[f"{target}_{feat_name}_{feat_value}_occurance"].append(0)

    for col, vals in results.items():
        df[col] = vals

    return df


def duration_from_last(df: pd.DataFrame, transaction, target):
    transaction_list = df[df[target] == transaction[target]]
    if transaction_list.empty:
        return 0
    
    last_transaction = transaction_list.tail(1).iloc[0]
    duration = (transaction['time'] - last_transaction['time']).total_seconds() / 3600
    return duration


def distance_from_last(df: pd.DataFrame, transaction, target):
    transaction_list = df[df[target] == transaction[target]]
    if transaction_list.empty:
        return 0
    
    last_transaction = transaction_list.iloc[-1]
    last_geo = (last_transaction['latitude'], last_transaction['longitude'])
    current_geo = (transaction['latitude'], transaction['longitude'])

    return distance(*last_geo, *current_geo)


def get_bound_relations_frequency(df: pd.DataFrame, target, features):
    df = df.sort_values(by='time').copy()
    data = df[[target, *[x['name'] for x in features]]]
    
    # Prepare dictionaries
    counts_dict = {feat['name']: {} for feat in features}
    results = {f"{target}_{feat['name']}_bound_frequency": [] for feat in features}

    # Precompute bounds per feature
    bounds = {feat['name']: feat['bound'](data[feat['name']]) for feat in features}
    # Iterate over rows
    for idx, row in data.iterrows():
        for feat in features:
            feat_name = feat['name']
            key = (row[target], row[feat_name])
            row_value = row[feat_name]
            low, high = bounds[feat_name]

            # Check if in bound
            if low[idx] <= row_value <= high[idx]:
                # Get current count
                count = counts_dict[feat_name].get(key, 0)
                results[f"{target}_{feat_name}_bound_frequency"].append(count)
                # Update count
                counts_dict[feat_name][key] = count + 1
            else:
                # Out of bounds → count 0
                results[f"{target}_{feat_name}_bound_frequency"].append(0)

    # Add results as columns
    for col, vals in results.items():
        df[col] = vals

    return df


def get_count_relations_frequency(df: pd.DataFrame, target, features):
    df = df.sort_values(by='time').copy()
    data = df[[target, *features]]
    
    counts_dict = {feat: {} for feat in features}
    results = {f"{target}_{feat}_count_frequency": [] for feat in features}
    
    for idx, row in data.iterrows():
        for feat in features:
            key = (row[target], row[feat])
            count = counts_dict[feat].get(key, 0)
            results[f"{target}_{feat}_count_frequency"].append(count)
            counts_dict[feat][key] = count + 1

    for col, vals in results.items():
        df[col] = vals

    return df


def anomalize(df: pd.DataFrame, name):
    df = df.copy()
    model_IF = IsolationForest(random_state=42)
    model_IF.fit(df)

    scores = model_IF.decision_function(df)
    flags = model_IF.predict(df)
    
    anomaly = [True if x == -1 else False for x in flags]
    anomaly_scores = (scores.max() - scores) / (scores.max() - scores.min())

    df[f'{name}_score'] = anomaly_scores
    df[name] = anomaly
    return df
    

def check_anomaly(df: pd.DataFrame, name, step=1):
    df = df.copy()
    columns = df.columns

    for i in range(step):
        print(len(columns))
        scores = anomalize(df, name)
        df[scores.columns] = scores

        corr_matrix = df.corr()
        bound_matrix = analyst.bounded_correlation(corr_matrix, (.2, .8))
        
        score_columns = bound_matrix.get(f'{name}_score').keys()
        class_columns = bound_matrix.get(name).keys()
        columns = list(set([*class_columns, *score_columns]))
        
        df = df[columns]
        print(len(columns))

    df[scores.columns] = scores
    return df

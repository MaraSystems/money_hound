import numpy as np 
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV, train_test_split, RandomizedSearchCV


def train_score_models(models, X_train, y_train, X_test, y_test, seed=42):
    """
    Fit and Score models
    Parameters:
        models (dict): A dictionary containing the models to train and score
    Returns:
        scores (dict): The scores of the models
    """

    # Set the random seed
    np.random.seed(seed)

    # Initialize the score store
    scores = {}

    for name, model in models.items():
        # Fit the model
        model.fit(X_train, y_train)

        # Store the score of the model
        score = model.score(X_test, y_test)
        print(f"{name} scored: {score}")


def crossval_models(models, X, y, scoring='accuracy', seed=42):
    """
    Cross Validate models
    Parameters:
        models (dict): A dictionary containing the models to train and score
    Returns:
        scores (dict): The scores of the models
    """

    # Set the random seed
    np.random.seed(seed)
    tscv = TimeSeriesSplit(n_splits=5)

    # Initialize the score store
    scores = {}
    
    for name, model in models.items():
        # Cross validate each model in 5 folds
        val = cross_val_score(model, X, y, cv=tscv, scoring=scoring)

        print(f"{name} mean: {float(val.mean())}, std: {float(val.std())}")


def random_search(models, X, y, n_iter=10, scoring='r2', seed=42):
    """
    Perform a Randomized Search on the models
    Parameters:
        models (dict): A dictionary containing the models to train and score
        n_iter (int): The number of iterations to perform
    Returns:
        scores (dict): The scores of the models
    """
    
    # Set random seed
    np.random.seed(seed)
    tscv = TimeSeriesSplit(n_splits=5)
    result = []

    for item in models:
        # Extract the model name, model and grid
        name, model, grid = item

        # Perform a Randomized Search on the model
        search = RandomizedSearchCV(model, param_distributions=grid, n_iter=n_iter, cv=tscv, scoring=scoring)
        search.fit(X, y)

        result.append({'Best Score': search.best_score_, 'Best Params': search.best_params_, 'Best Estimator': search.best_estimator_})
        print(f"{name} best score: {search.best_score_} scored by {scoring}")

    return result, scoring


def grid_search(models, X, y, n_iter=10, scoring='r2', seed=42):
    """
    Perform a Randomized Search on the models
    Parameters:
        models (dict): A dictionary containing the models to train and score
        n_iter (int): The number of iterations to perform
    Returns:
        scores (dict): The scores of the models
    """
    
    # Set random seed
    np.random.seed(seed)
    tscv = TimeSeriesSplit(n_splits=5)
    result = []

    for item in models:
        # Extract the model name, model and grid
        name, model, grid = item

        # Perform a Randomized Search on the model
        search = GridSearchCV(model, param_grid=grid, cv=tscv, scoring=scoring)
        search.fit(X, y)

        result.append({'Best Score': search.best_score_, 'Best Params': search.best_params_, 'Best Estimator': search.best_estimator_})
        print(f"{name} best score: {search.best_score_} scored by {scoring}")

    return result
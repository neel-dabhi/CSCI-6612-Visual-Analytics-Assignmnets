from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from assignments.assignment1.a_load_file import read_dataset
from assignments.assignment1.c_data_cleaning import fix_numeric_wrong_values, normalize_column
from assignments.assignment1.d_data_encoding import fix_outliers, fix_nans, normalize_column, \
    generate_one_hot_encoder, replace_with_one_hot_encoder, generate_label_encoder, replace_with_label_encoder, \
    get_numeric_columns
from assignments.assignment1.e_experimentation import process_iris_dataset, process_amazon_video_game_dataset, \
    process_iris_dataset_again, process_life_expectancy_dataset

pd.set_option('display.max_columns', 100)

"""
Regression is a supervised form of machine learning. It uses labeled data, which is data with an expected
result available, and uses it to train a machine learning model to predict the said result. Regression
focuses in results of the numerical type.
"""


##############################################
# Example(s). Read the comments in the following method(s)
##############################################
def simple_random_forest_regressor(X: pd.DataFrame, y: pd.Series) -> Dict:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # If necessary, change the n_estimators, max_depth and max_leaf_nodes in the below method to accelerate the model training,
    # but don't forget to comment why you did and any consequences of setting them!
    model = RandomForestRegressor()  # Now I am doing a regression!
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)  # Use this line to get the prediction from the model

    # In regression, there is no accuracy, but other types of score. See the following link for one example (R^2)
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor.score
    score = model.score(X_test, y_test)
    return dict(model=model, score=score, test_prediction=y_predict)


def simple_random_forest_on_iris() -> Dict:
    """
    Here I will run a regression on the iris dataset with a random-forest regressor
    Notice that my logic has changed. I am not predicting the species anymore, but
    am predicting the sepal_length. I am also removing the species column, and will handle
    it in the next example.
    """
    df = pd.read_csv(Path('..', '..', 'iris.csv'))
    X, y = df.iloc[:, 1:4], df.iloc[:, 0]
    return simple_random_forest_regressor(X, y)


def reusing_code_random_forest_on_iris() -> Dict:
    """
    Again I will run a regression on the iris dataset, but reusing
    the existing code from assignment1. I am also including the species column as a one_hot_encoded
    value for the prediction. Use this to check how different the results are (score and
    predictions).
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    for c in list(df.columns):
        df = fix_outliers(df, c)
        df = fix_nans(df, c)
        df[c] = normalize_column(df[c])

    ohe = generate_one_hot_encoder(df['species'])
    df = replace_with_one_hot_encoder(df, 'species', ohe, list(ohe.get_feature_names()))

    X, y = df.iloc[:, 1:], df.iloc[:, 0]
    return simple_random_forest_regressor(X, y)


##############################################
# Implement all the below methods
# Don't install any other python package other than provided by python or in requirements.txt
##############################################
def random_forest_iris_dataset_again() -> Dict:
    """
    Run the result of the process iris again task of e_experimentation and discuss (1 sentence)
    the differences from the above results. Also, as above, use sepal_length as the label column and
    the one_hot_encoder to transform the categorical column into a usable format.
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """

    df = process_iris_dataset_again()
    df.drop(columns=['large_sepal_lenght'], inplace=True)
    ohe = generate_one_hot_encoder(df['species'])
    df = replace_with_one_hot_encoder(df, 'species', ohe, list(ohe.get_feature_names()))

    X, y = df.iloc[:, 1:], df.iloc[:, 0]

    return simple_random_forest_regressor(X, y)


def decision_tree_regressor(X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Reimplement the method "simple_random_forest_regressor" but using
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.DecisionTreeRegressor.html
    Optional: also optimise the parameters of the model to maximise the R^2 score
    :param X: Input dataframe
    :param y: Label data
    :return: model, score and prediction of the test set
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    score = model.score(X_test, y_test)
    return dict(model=model, score=score, test_prediction=y_predict)


def train_iris_dataset_again() -> Dict:
    """
    Run the result of the process iris again task of e_experimentation, but now using the
    decision tree regressor AND random_forest regressor. Return the one with highest R^2.
    Use the same label column and one hot encoding logic as before.
    Discuss (1 sentence) what you found different between the results.
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """
    df = process_iris_dataset_again()

    ohe = generate_one_hot_encoder(df['species'])
    df = replace_with_one_hot_encoder(df, 'species', ohe, list(ohe.get_feature_names()))

    X, y = df.iloc[:, 1:], df.iloc[:, 0]

    dtr = decision_tree_regressor(X, y)
    rfr = simple_random_forest_regressor(X, y)

    if dtr['score'] > rfr['score']:
        return dtr
    else:
        return rfr


def train_amazon_video_game() -> Dict:
    """
    Run the result of the amazon dataset task of e_experimentation using the
    decision tree regressor AND random_forest regressor. Return the one with highest R^2.
    The Label column is the count column
    Discuss (1 sentence) what you found different between the results.
    In one sentence, why is the score different (or the same) compared to the iris score?
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """

    df = process_amazon_video_game_dataset()
    df = df.drop(columns=['time'])

    # Not feeding the asin to the model, as they all are unique.
    # X: number of product reviewed by user
    # y: avg. of all the reviews
    X, y = df.iloc[:, [2]], df.iloc[:, 1]

    dtr = decision_tree_regressor(X, y)
    rfr = simple_random_forest_regressor(X, y)

    """
    Here I am getting negative R^2 which means the data is poorly fitted
    Also, it implies that the model does not follow any trend with the data,
    and the regression line of our model is worse then the mean line. 
    """

    if dtr['score'] > rfr['score']:
        return dtr
    else:
        return rfr


def train_life_expectancy() -> Dict:
    """
    Do the same as the previous task with the result of the life expectancy task of e_experimentation.
    The label column is the value column. Remember to convert drop columns you think are useless for
    the machine learning (say why you think so) and convert the remaining categorical columns with one_hot_encoding.
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """
    df = process_life_expectancy_dataset()

    # Dropping regions, and Latitude as it is irrelevant to life expectancy value.
    # we can see a trend over year
    df = df[['country', 'value', 'year']]

    # Keeping every third row because it might be possible that life expectancy is not
    # changed significantly in the interval of one year, keeping interval to 3 can saw significant increase in value.
    df = df[df.index % 3 == 0]

    ohe = generate_one_hot_encoder(df['country'])
    df = replace_with_one_hot_encoder(df, 'country', ohe, list(ohe.get_feature_names()))

    X, y = df.iloc[:, 1:], df.iloc[:, 0]

    dtr = decision_tree_regressor(X, y)
    rfr = simple_random_forest_regressor(X, y)

    if dtr['score'] > rfr['score']:
        print(dtr)
        return dtr
    else:
        print(rfr)
        return rfr


def your_choice() -> Dict:
    """
    Now choose one of the datasets included in the assignment1 (the raw one, before anything done to them)
    and decide for yourself a set of instructions to be done (similar to the e_experimentation tasks).
    Specify your goal (e.g. analyse the reviews of the amazon dataset), say what you did to try to achieve the goal
    and use one (or both) of the models above to help you answer that. Remember that these models are regression
    models, therefore it is useful only for numerical labels.
    We will not grade your result itself, but your decision-making and suppositions given the goal you decided.
    Use this as a small exercise of what you will do in the project.
    """

    """
    Goal: check if converting numeric values into categorical data impacts the accuracy of the model
          For this problem first I have min-max normalized the sepal_length, sepal_width, petal_length, and petal_width
          to the categories such as low, medium and high. converting 
    """

    df = read_dataset(Path('..', '..', 'iris.csv'))
    numeric_columns = get_numeric_columns(df)

    # Min Max normalizing each numeric col
    for col in numeric_columns:
        df[col] = normalize_column(df[col])

    # converting all the numeric values into categorical
    for col in numeric_columns:
        df[col] = np.where(df[col] < 0.33, 'low', np.where(df[col] > 0.66, 'high', 'medium'))

    for col in numeric_columns:
        le = generate_label_encoder(df[col])
        df = replace_with_label_encoder(df, col, le)

    # one hot encode the species col
    ohe = generate_one_hot_encoder(df['species'])
    df = replace_with_one_hot_encoder(df, 'species', ohe, list(ohe.get_feature_names()))

    X, y = df.iloc[:, :4], df.iloc[:, 4:]

    dtr = decision_tree_regressor(X, y)
    rfr = simple_random_forest_regressor(X, y)

    """
    here the accuracy increases for the iris data set compared to train_iris_dataset_again()
    even generalizing a data bit.
    """

    if dtr['score'] > rfr['score']:
        return dtr
    else:
        return rfr

    pass


if __name__ == "__main__":
    assert simple_random_forest_on_iris() is not None
    assert reusing_code_random_forest_on_iris() is not None
    assert random_forest_iris_dataset_again() is not None
    assert train_iris_dataset_again() is not None
    assert train_amazon_video_game() is not None
    assert train_life_expectancy() is not None
    assert your_choice() is not None

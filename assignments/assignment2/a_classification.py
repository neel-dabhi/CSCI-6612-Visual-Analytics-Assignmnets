from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from assignments.assignment1.a_load_file import read_dataset
from assignments.assignment1.b_data_profile import get_numeric_columns, get_text_categorical_columns
from assignments.assignment1.d_data_encoding import generate_label_encoder, replace_with_label_encoder, fix_outliers, \
    fix_nans, normalize_column, generate_one_hot_encoder, replace_with_one_hot_encoder
from assignments.assignment1.e_experimentation import process_iris_dataset, process_amazon_video_game_dataset_again, \
    process_iris_dataset_again, process_life_expectancy_dataset

"""
Classification is a supervised form of machine learning. It uses labeled data, which is data with an expected
result available, and uses it to train a machine learning model to predict the said result. Classification
focuses in results of the categorical type.
"""
pd.set_option('display.max_columns', 100)


##############################################
# Example(s). Read the comments in the following method(s)
##############################################
def simple_random_forest_classifier(X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Simple method to create and train a random forest classifier
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

    # If necessary, change the n_estimators, max_depth and max_leaf_nodes in the below method to accelerate the model training,
    # but don't forget to comment why you did and any consequences of setting them!
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)  # Use this line to get the prediction from the model
    accuracy = model.score(X_test, y_test)
    return dict(model=model, accuracy=accuracy, test_prediction=y_predict)


def simple_random_forest_on_iris() -> Dict:
    """
    Here I will run a classification on the iris dataset with random forest
    """
    df = pd.read_csv(Path('..', '..', 'iris.csv'))
    X, y = df.iloc[:, :4], df.iloc[:, 4]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return simple_random_forest_classifier(X, y_encoded)


def reusing_code_random_forest_on_iris() -> Dict:
    """
    Again I will run a classification on the iris dataset, but reusing
    the existing code from assignment1. Use this to check how different the results are (score and
    predictions).
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    for c in list(df.columns):
        # Notice that I am now passing though all columns.
        # If your code does not handle normalizing categorical columns, do so now (just return the unchanged column)
        df = fix_outliers(df, c)
        df = fix_nans(df, c)
        df[c] = normalize_column(df[c])

    X, y = df.iloc[:, :4], df.iloc[:, 4]
    le = generate_label_encoder(y)

    # Be careful to return a copy of the input with the changes, instead of changing inplace the inputs here!
    y_encoded = replace_with_label_encoder(y.to_frame(), column='species', le=le)
    return simple_random_forest_classifier(X, y_encoded['species'])


##############################################
# Implement all the below methods
# Don't install any other python package other than provided by python or in requirements.txt
##############################################
def random_forest_iris_dataset_again() -> Dict:
    """
    Run the result of the process iris again task of e_experimentation and discuss (1 sentence)
    the differences from the above results. Use the same random forest method.
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """
    df = process_iris_dataset_again()
    X, y = df.iloc[:, :4], df.iloc[:, 4]

    return simple_random_forest_classifier(X, y)


def decision_tree_classifier(X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Reimplement the method "simple_random_forest_classifier" but using the technique we saw in class: decision trees
    (you can use sklearn to help you).
    Optional: also optimise the parameters of the model to maximise accuracy
    :param X: Input dataframe
    :param y: Label data
    :return: model, accuracy and prediction of the test set
    """

    # setting the random_state to keep the output of every run same
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    test_prediction = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

    return dict(model=model, accuracy=accuracy, test_prediction=test_prediction)


def train_iris_dataset_again() -> Dict:
    """
    Run the result of the iris dataset again task of e_experimentation using the
    decision_tree classifier AND random_forest classifier. Return the one with highest score.
    Discuss (1 sentence) what you found different between the two models and scores.
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """

    df = process_iris_dataset_again()
    X, y = df.iloc[:, :4], df.iloc[:, 4]

    dt = decision_tree_classifier(X, y)
    rf = simple_random_forest_classifier(X, y)

    if dt['accuracy'] > rf['accuracy']:
        return dt
    else:
        return rf


def train_amazon_video_game_again() -> Dict:
    """
    Run the result of the amazon dataset again task of e_experimentation using the
    decision tree classifier AND random_forest classifier. Return the one with highest score.
    The Label column is the user column. Choose what you wish to do with the time column (drop, convert, etc)
    Discuss (1 sentence) what you found different between the results.
    In one sentence, why is the score worse than the iris score (or why is it not worse) in your opinion?
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """
    df = process_amazon_video_game_dataset_again()
    # dropping time col
    df = df.drop(columns=['time'])
    # removing all the products which are rated only once.
    df = df[df.duplicated(subset=['asin'], keep=False)]
    # removing all the users who just rated one product.
    df = df[df.duplicated(subset=['user'], keep=False)]
    # Removing Neutral reviews
    df = df[df['review'] != 3]

    le = generate_label_encoder(df['asin'])
    df = replace_with_label_encoder(df, 'asin', le)

    # Taking first 10,000 records into consideration to avoid memory overflow
    # we can use partial fit, but it is not supported
    X, y = df.iloc[:10000, 1:3], df.iloc[:10000, 0]

    dt = decision_tree_classifier(X, y)
    rf = simple_random_forest_classifier(X, y)

    """
    The accuracy of model will be low because of the imbalance in the features
    for example we have got many positive reviews compared to negative reviews.
    Other thing is feature selection it is very hard to predict the user based on the productID and the rating, 
    it would make more sense the other way around.
    """

    if dt['accuracy'] > rf['accuracy']:
        return dt
    else:
        return rf


def train_life_expectancy() -> Dict:
    """
    Do the same as the previous task with the result of the life expectancy task of e_experimentation.
    The label column is the column which has north/south. Remember to convert drop columns you think are useless for
    the machine learning (say why you think so) and convert the remaining categorical columns with one_hot_encoding.
    (check the c_regression examples to see example on how to do this one hot encoding)
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """

    df = process_life_expectancy_dataset()
    df = df[
        ['country', 'value', 'Latitude', 'x0_asia_west', 'x1_europe_east', 'x2_africa_north', 'x3_africa_sub_saharan',
         'x4_america_north', 'x5_america_south', 'x6_east_asia_pacific', 'x7_europe_west']]

    df['value'] = df['value'].apply(pd.to_numeric)

    # group by country and taking average of life expectancy review over all the years
    df = df.groupby(by='country', as_index=False).agg({
        'value': np.mean, 'Latitude': np.mean, 'x0_asia_west': np.mean, 'x1_europe_east': np.mean,
        'x2_africa_north': np.mean,
        'x3_africa_sub_saharan': np.mean, 'x4_america_north': np.mean, 'x5_america_south': np.mean,
        'x6_east_asia_pacific': np.mean, 'x7_europe_west': np.mean
    })

    # using the location of country and it's life expectancy rate to predict if it is in North/South
    X, y = df.iloc[:, [1, 3, 4, 5, 6, 7, 8, 9, 10]], df.iloc[:, 2]

    dt = decision_tree_classifier(X, y)
    rf = simple_random_forest_classifier(X, y)

    if dt['accuracy'] > rf['accuracy']:
        return dt
    else:
        return rf


def your_choice() -> Dict:
    """
    Now choose one of the datasets included in the assignment1 (the raw one, before anything done to them)
    and decide for yourself a set of instructions to be done (similar to the e_experimentation tasks).
    Specify your goal (e.g. analyse the reviews of the amazon dataset), say what you did to try to achieve the goal
    and use one (or both) of the models above to help you answer that. Remember that these models are classification
    models, therefore it is useful only for categorical labels.
    We will not grade your result itself, but your decision-making and suppositions given the goal you decided.
    Use this as a small exercise of what you will do in the project.
    """

    """
    Goal: given the increase/decrease in life expectancy rate of country after becoming the UN member and 
          the how long the country is the member of UN, Classify the region it resides in.
          
    Rational: As a member of UN a country gets some protections/perks in fields such as Defence, Health and Human 
              Rights which can lead to increase in life expectancy. Also, I wanted to know if their geography and becoming
              UN member has any correlation.
    """

    df = read_dataset(Path('..', '..', 'life_expectancy_years.csv'))
    df = df.T

    df_geography = read_dataset(Path('..', '..', 'geography.csv'))
    df_geography = df_geography.rename(columns={'name': 'country'})

    # Removing countries with more than 50% of missing data
    df = df.loc[:, df.isnull().mean() < .5]
    df.reset_index(level=0, inplace=True)

    # considering zeroth row as header after transposing
    new_header = df.iloc[0]
    df = df[1:]
    df.reset_index(level=0, inplace=True)
    x = ['index']
    x.extend(new_header)
    df.columns = x

    # Fix outliers and missing values
    numeric_columns = get_numeric_columns(df)
    for nc in numeric_columns:
        df = fix_outliers(df, nc)

    text_categorical_columns = get_text_categorical_columns(df_geography)
    for tcc in text_categorical_columns:
        df_geography = fix_outliers(df_geography, tcc)
        df_geography = fix_nans(df_geography, tcc)

    # melting life expectancy data
    df = df.drop(['index'], axis=1)
    df1 = pd.melt(df, id_vars=['country'])
    df1 = df1.rename(columns={'country': 'year', 'variable': 'country'})

    # merging two dataframes : df_geo(with geographic data) and df(life_expectancy_data)
    df_merged = pd.merge(left=df1, right=df_geography, left_on='country', right_on='country')

    # Dropping all columns except country, continent, year, value and latitude
    # eight_regions as continent because it gives more accurate position of country on the globe
    df_merged = df_merged[['country', 'UN member since', 'value', 'eight_regions', 'Latitude']]
    df_merged = df_merged.rename(columns={'eight_regions': 'continent'})

    # Take UN join date count days till today, and replace in UN member since
    df_merged['UN member since'] = pd.Timestamp.now().normalize() - pd.to_datetime(df_merged['UN member since'])
    df_merged['UN member since'] = (df_merged['UN member since'].dt.days.div(365).round().astype(int))

    # Get LE rating differance before and after being a UN member
    df_temp = df.T
    # considering zeroth row as header after transposing
    new_header = df_temp.iloc[0]
    df_temp = df_temp[1:]
    df_temp.reset_index(level=0, inplace=True)
    x = ['index']
    x.extend(new_header)
    df_temp.columns = x

    # Getting a difference of life expectancy in 2018 and when country became UN member
    diff = 2018 - df_merged['UN member since']
    diff.drop_duplicates(keep='first')
    df_temp = df_temp['2018'] - df_temp[diff[0].astype(str)]

    # label encoding of continent
    le = generate_label_encoder(df_merged['continent'])
    df_le_encoded = replace_with_label_encoder(df_merged, 'continent', le)

    df_le_encoded.drop(columns=['Latitude'], axis='columns', inplace=True)
    df_le_encoded['value'] = pd.to_numeric(df_le_encoded['value'])
    df_le_encoded = df_le_encoded.groupby(by='country', as_index=False).agg(
        {'continent': 'first', 'UN member since': np.mean})
    df_le_encoded['diff'] = df_temp

    # given the membership years and life expectancy classify region

    X, y = df_le_encoded.iloc[:, [2, 3]], df_le_encoded.iloc[:, 1]

    dt = decision_tree_classifier(X, y)
    rf = simple_random_forest_classifier(X, y)

    """
    Accuracy: Accuracy of the model is low because there seem to be no correlation between these two features,
              Years in UN and place on the globe.
    """
    if dt['accuracy'] > rf['accuracy']:
        return dt
    else:
        return rf


if __name__ == "__main__":
    assert simple_random_forest_on_iris() is not None
    assert reusing_code_random_forest_on_iris() is not None
    assert random_forest_iris_dataset_again() is not None
    assert train_iris_dataset_again() is not None
    assert train_amazon_video_game_again() is not None
    assert train_life_expectancy() is not None
    assert your_choice() is not None

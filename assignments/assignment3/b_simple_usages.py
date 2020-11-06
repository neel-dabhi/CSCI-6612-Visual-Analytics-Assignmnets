from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

##############################################
# In this file, we will use data and methods of previous assignments with visualization.
# But before you continue on, take some time to look on the internet about the many existing visualization types and their usages, for example:
# https://extremepresentation.typepad.com/blog/2006/09/choosing_a_good.html
# https://datavizcatalogue.com/
# https://plotly.com/python/
# https://www.tableau.com/learn/whitepapers/which-chart-or-graph-is-right-for-you
# Or just google "which visualization to use", and you'll find a near-infinite number of resources
#
# You may want to create a new visualization in the future, and for that I suggest using JavaScript and D3.js, but for the course, we will only
# use python and already available visualizations
##############################################
# Implement all the below methods
# Use the documentation of the libraries in case you need help to understand the required methods
# For ALL methods return the fig and ax of matplotlib or fig from plotly!
##############################################
from assignments.assignment1.a_load_file import read_dataset
from assignments.assignment1.b_data_profile import get_numeric_columns, get_text_categorical_columns, get_binary_columns
from assignments.assignment1.d_data_encoding import generate_one_hot_encoder, replace_with_one_hot_encoder, \
    fix_outliers, fix_nans, replace_one_hot_encoder_with_original_column
from assignments.assignment1.e_experimentation import process_life_expectancy_dataset
from assignments.assignment2.a_classification import your_choice as your_choice_a
from assignments.assignment2.c_clustering import cluster_iris_dataset_again
from assignments.assignment3.a_libraries import matplotlib_bar_chart as matplotlib_bar_chart_a, \
    matplotlib_pie_chart as matplotlib_pie_chart_a, \
    matplotlib_heatmap_chart as matplotlib_heatmap_chart_a, plotly_polar_chart as plotly_polar_chart_a

pd.set_option('display.max_columns', 100)


def matplotlib_bar_chart() -> Tuple:
    """
    Create a bar chart with a1/b_data_profile's get column max.
    Show the max of each numeric column from iris dataset as the bars
    """
    df = read_dataset('../../iris.csv')
    df = df[get_numeric_columns(df)]
    df = df.max()
    fig, ax = matplotlib_bar_chart_a(df)
    return fig, ax


def matplotlib_pie_chart() -> Tuple:
    """
    Create a pie chart where each piece of the chart has the number of columns which are numeric/categorical/binary
    from the output of a1/e_/process_life_expectancy_dataset
    """
    df = process_life_expectancy_dataset()
    data = [len(get_numeric_columns(df)), len(get_binary_columns(df)), len(get_text_categorical_columns(df))]
    return matplotlib_pie_chart_a(np.array(data))


def matplotlib_histogram() -> Tuple:
    """
    Build 4 histograms as subplots in one figure with the numeric values of the iris dataset
    """
    df = read_dataset('../../iris.csv')
    df = df[get_numeric_columns(df)]

    fig, ax = plt.subplots(2, 2)
    ax = ax.ravel()

    ax[0].hist(df['petal_length'])
    ax[0].set_title('petal_length')
    ax[1].hist(df['petal_width'])
    ax[1].set_title('petal_width')
    ax[2].hist(df['sepal_length'])
    ax[2].set_title('sepal_length')
    ax[3].hist(df['sepal_width'])
    ax[3].set_title('sepal_width')
    plt.tight_layout()

    return fig, ax


def matplotlib_heatmap_chart() -> Tuple:
    """
    Remember a1/b_/pandas_profile? There is a heat map over there to analyse the correlation among columns.
    Use the pearson correlation (e.g. https://docs.scipy.org/doc/scipy-1.5.3/reference/generated/scipy.stats.pearsonr.html)
    to calculate the correlation between two numeric columns and show that as a heat map. Use the iris dataset.
    """
    df = read_dataset('../../iris.csv')
    return matplotlib_heatmap_chart_a(df.corr(method='pearson'))


# There are many other possibilities. Please, do check the documentation and examples so you
# may have a good breadth of tools for future work (in assignments, projects, and your own career)
###################################
# Once again, for ALL methods return the fig and ax of matplotlib or fig from plotly!


def plotly_scatter_plot_chart():
    """
    Use the result of a2/c_clustering/cluster_iris_dataset_again() as the color of a scatterplot made from the original (unprocessed)
    iris dataset. Choose among the numeric values to be the x and y coordinates.
    """
    clustering_iris = cluster_iris_dataset_again()
    df = read_dataset('../../iris.csv')
    fig = px.scatter(df, x="sepal_width", y="sepal_length", color=clustering_iris['clusters'])

    return fig


def plotly_bar_plot_chart():
    """
    Use the result of a2/c_clustering/cluster_iris_dataset_again() and use x as 3 groups of bars (one for each iris species)
    and each group has multiple bars, one for each cluster, with y as the count of instances in the specific cluster/species combination.
    The grouped bar chart is like https://plotly.com
    /python/bar-charts/#grouped-bar-chart (search for the grouped bar chart visualization)
    """
    clustering_iris = cluster_iris_dataset_again()
    df = read_dataset('../../iris.csv')
    df['clusters'] = pd.Series(clustering_iris['clusters'])
    df = df.loc[df['clusters'] > -1]

    species = ['setosa', 'versicolor', 'virginica']
    fig = go.Figure(data=[
        go.Bar(name='cluster 0', x=species, y=[df.where(df['clusters'] == 0)['species'].str.count('setosa').sum(),
                                               df.where(df['clusters'] == 0)['species'].str.count('versicolor').sum(),
                                               df.where(df['clusters'] == 0)['species'].str.count('virginica').sum()]),

        go.Bar(name='cluster 1', x=species, y=[df.where(df['clusters'] == 1)['species'].str.count('setosa').sum(),
                                               df.where(df['clusters'] == 1)['species'].str.count('versicolor').sum(),
                                               df.where(df['clusters'] == 1)['species'].str.count('virginica').sum()]),

        go.Bar(name='cluster 2', x=species, y=[df.where(df['clusters'] == 2)['species'].str.count('setosa').sum(),
                                               df.where(df['clusters'] == 2)['species'].str.count('versicolor').sum(),
                                               df.where(df['clusters'] == 2)['species'].str.count('virginica').sum()])
    ])
    fig.update_layout(barmode='group')
    return fig


def plotly_polar_scatterplot_chart():
    """
    Do something similar to a1/e_/process_life_expectancy_dataset, but don't drop the latitude and longitude.
    Use these two values to figure out the theta to plot values as a compass (example: https://plotly.com/python/polar-chart/).
    Each point should be one country and the radius should be thd value from the dataset (add up all years and feel free to ignore everything else)
    """
    df = read_dataset('../../life_expectancy_years.csv')
    df = df.T
    # removing the country if more than 50% of it's data is nan, it is better to just remove the data then replacing
    # it with mean or any other value.
    df = df.loc[:, df.isnull().mean() < .5]
    df.reset_index(level=0, inplace=True)

    # considering zeroth row as header after transposing
    new_header = df.iloc[0]
    df = df[1:]
    df.reset_index(level=0, inplace=True)
    x = ['index']
    x.extend(new_header)
    df.columns = x

    # handling outliers before moving further
    numeric_columns = get_numeric_columns(df)
    for nc in numeric_columns:
        df = fix_outliers(df, nc)

    df_geo = pd.read_csv('../../geography.csv', encoding='utf-8')
    df_geo = df_geo.rename(columns={'name': 'country'})

    # Handling outliers and nans before joining Geo Data with life expectancy
    text_categorical_columns = get_text_categorical_columns(df_geo)
    for tcc in text_categorical_columns:
        df_geo = fix_outliers(df_geo, tcc)
        df_geo = fix_nans(df_geo, tcc)

    # melting life expectancy data
    df = df.drop(['index'], axis=1)
    df1 = pd.melt(df, id_vars=['country'])
    df1 = df1.rename(columns={'country': 'year', 'variable': 'country'})

    # merging two dataframes : df_geo(with geographic data) and df(life_expectancy_data)
    df_merged = pd.merge(left=df1, right=df_geo, left_on='country', right_on='country')

    df_merged = df_merged[['value', 'country', 'year', 'Latitude', 'Longitude']]
    df_merged['year'] = pd.to_numeric(df_merged['year'])
    df_merged['value'] = pd.to_numeric(df_merged['value'])

    df_merged = df_merged.groupby(by='country', as_index=False) \
        .agg({'year': np.sum,
              'Latitude': 'first', 'Longitude': 'first', 'value': np.mean}).fillna(0)

    # converting lat long to theta

    X = np.cos(df_merged['Latitude']) * np.sin(df_merged['Longitude'])
    Y = np.cos(0) * np.sin(df_merged['Latitude']) - np.sin(0) * np.cos(df_merged['Latitude']) * np.cos(
        df_merged['Longitude'])

    df_merged['bearing'] = abs(np.degrees((np.arctan2(X, Y)) + 360) % 360)

    fig = px.scatter_polar(df_merged, r='value', theta='bearing',)
    return fig


def plotly_table():
    """
    Show the data from a2/a_classification/your_choice() as a table
    See https://plotly.com/python/table/ for documentation
    """
    results = your_choice_a()
    fig = go.Figure(data=[go.Table(header=dict(values=['test_prediction']),
                                   cells=dict(values=[results['test_prediction']]))])
    return fig


def plotly_composite_line_bar():
    """
    Use the data from a1/e_/process_life_expectancy_dataset and show in a single graph on year on x and value on y where
    there are 5 line charts of 5 countries (you choose which) and one bar chart on the background with the total value of all 5
    countries added up.
    """
    df = process_life_expectancy_dataset()

    countries = ['Afghanistan', 'Albania', 'Algeria', 'Angola', 'Argentina']
    df = df.loc[df['country'].isin(countries)]

    df_total = df.groupby(by='year', as_index=False) \
        .agg({'value': np.sum}).fillna(0)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=df.loc[df['country'] == countries[0]]['year'], y=df.loc[df['country'] == countries[0]]['value'],
                   name=countries[0]))
    fig.add_trace(
        go.Scatter(x=df.loc[df['country'] == countries[1]]['year'], y=df.loc[df['country'] == countries[1]]['value'],
                   name=countries[1]))
    fig.add_trace(
        go.Scatter(x=df.loc[df['country'] == countries[2]]['year'], y=df.loc[df['country'] == countries[2]]['value'],
                   name=countries[2]))
    fig.add_trace(
        go.Scatter(x=df.loc[df['country'] == countries[3]]['year'], y=df.loc[df['country'] == countries[3]]['value'],
                   name=countries[3]))
    fig.add_trace(
        go.Scatter(x=df.loc[df['country'] == countries[4]]['year'], y=df.loc[df['country'] == countries[4]]['value'],
                   name=countries[4]))
    fig.add_trace(go.Bar(x=df_total['year'], y=df_total['value'], name='Total value'))
    fig.update_layout(
        title="Life expectancy value of Afghanistan, Albania, Algeria, Angola, and Argentina over the years")

    return fig


def plotly_map():
    """
    Use the data from a1/e_/process_life_expectancy_dataset on a plotly map (anyone will do)
    Examples: https://plotly.com/python/maps/, https://plotly.com/python/choropleth-maps/#using-builtin-country-and-state-geometries
    Use the value from the dataset of a specific year (e.g. 1900) to show as the color in the map
    """
    df = process_life_expectancy_dataset()
    df = df.loc[df['year'] == '1947']
    df.reset_index(drop=True, inplace=True)
    df['country'] = pd.Series(df['country'], dtype="string")
    df = df[['country', 'value']]

    fig = px.choropleth(df, locations="country", locationmode='country names',
                        color="value",
                        hover_name="country",
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title="Life Expectancy value in 1947")

    return fig


def plotly_tree_map():
    """
    Use plotly's treemap to plot any data returned from any of a1/e_experimentation or a2 tasks
    Documentation: https://plotly.com/python/treemaps/
    """
    df = process_life_expectancy_dataset()
    years = ['1800', '1850', '1900', '1950', '2000', '2018']
    df = df.loc[df['year'].isin(years)]
    df.reset_index(drop=True, inplace=True)
    fig = px.treemap(df, path=['year', 'country'], values='value',
                     color='value', hover_data=['country'],
                     maxdepth=2,
                     color_continuous_scale='RdBu',
                     color_continuous_midpoint=np.average(df['value'], weights=df['value']))

    fig.update_layout(title='Life Expectancy values of countries in 50 year intervals ')

    return fig


if __name__ == "__main__":
    # Here are examples of how to run each method
    # Although you may remove or comment them out for testing, make sure they are present
    # and work when submitting, since when we run your code, all methods will be run like this.
    # If these lines below returns errors when running, your file will be considered to not
    # run, and graded accordingly.
    fig_m_bc, _ = matplotlib_bar_chart()
    fig_m_pc, _ = matplotlib_pie_chart()
    fig_m_h, _ = matplotlib_histogram()
    fig_m_hc, _ = matplotlib_heatmap_chart()

    fig_p_s = plotly_scatter_plot_chart()
    fig_p_bpc = plotly_bar_plot_chart()
    fig_p_psc = plotly_polar_scatterplot_chart()
    fig_p_t = plotly_table()
    fig_p_clb = plotly_composite_line_bar()
    fig_p_map = plotly_map()
    fig_p_treemap = plotly_tree_map()

    # Uncomment the below lines to test your code
    # When submitting, leave the code below commented!!!
    # fig_m_bc.show()
    # fig_m_pc.show()
    # fig_m_h.show()
    # fig_m_hc.show()

    # fig_p_s.show()
    # fig_p_bpc.show()
    # fig_p_psc.show()
    # fig_p_t.show()
    # fig_p_clb.show()
    # fig_p_map.show()
    # fig_p_treemap.show()

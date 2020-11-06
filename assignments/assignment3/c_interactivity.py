from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.widgets import Button, Slider

from assignments.assignment2.c_clustering import simple_k_means
from assignments.assignment3.b_simple_usages import *


###############
# Interactivity in visualizations is challenging due to limitations and clunkiness of libraries.
# For example, some libraries works well in Jupyter Notebooks, but then the code makes barely any sense and becomes hard to change/update,
# defeating the purpose of using Jupyter notebooks in the first place, and other libraries provide a window of their own, but
# they are very tied to the running code, and far from the experience you'd expect from a proper visual analytics webapp
#
# We will try out some approaches to exercise in this file, but the next file will give you the proper tooling to make a
# well-rounded and efficient code for visual interactions.
##############################################
# Example(s). Read the comments in the following method(s)
##############################################


def matplotlib_simple_example():
    """
    Using the same logic from before, we can add sliders or buttons to select options for interactivity.
    Matplotlib is limited to only a few options, but do take a look, since they are useful for fast prototyping and analysis

    In case you are using PyCharm, I suggest you to uncheck the 'Show plots in tool window'
    to be able to see and interact with the buttons defined below.
    This example comes from https://matplotlib.org/3.1.1/gallery/widgets/buttons.html
    """
    freqs = np.arange(2, 20, 3)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    t = np.arange(0.0, 1.0, 0.001)
    s = np.sin(2 * np.pi * freqs[0] * t)
    l, = plt.plot(t, s, lw=2)

    class Index(object):
        ind = 0

        def next(self, event):
            self.ind += 1
            i = self.ind % len(freqs)
            ydata = np.sin(2 * np.pi * freqs[i] * t)
            l.set_ydata(ydata)
            plt.draw()

        def prev(self, event):
            self.ind -= 1
            i = self.ind % len(freqs)
            ydata = np.sin(2 * np.pi * freqs[i] * t)
            l.set_ydata(ydata)
            plt.draw()

    callback = Index()
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    ax_bar = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(ax_bar, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    return fig, ax


def matplotlib_simple_example2():
    """
    Here is another example, which also has a slider and simplifies a bit the callbacks
    """
    data = np.random.rand(10, 5)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax.bar(np.arange(10).astype(str).tolist(), data[:, 0])

    class Index(object):
        ind = 0
        multiplier = 1

        def change_data(self, event, i):
            self.ind = np.clip(self.ind + i, 0, data.shape[1] - 1)
            ax.clear()
            ax.bar(np.arange(10).astype(str).tolist(), data[:, self.ind] * self.multiplier)
            plt.draw()

        def change_multiplier(self, value):
            self.multiplier = value
            ax.clear()
            ax.bar(np.arange(10).astype(str).tolist(), data[:, self.ind] * self.multiplier)
            plt.draw()

    callback = Index()
    axprev = plt.axes([0.1, 0.05, 0.12, 0.075])
    ax_bar = plt.axes([0.23, 0.05, 0.12, 0.075])
    axslider = plt.axes([0.55, 0.1, 0.35, 0.03])
    bnext = Button(ax_bar, 'Next')
    bnext.on_clicked(lambda event: callback.change_data(event, 1))
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(lambda event: callback.change_data(event, -1))
    slider = Slider(axslider, 'multiplier', 1, 10, 1)
    slider.on_changed(callback.change_multiplier)

    return fig, ax


def plotly_slider_example():
    """
    Here is a simple example from https://plotly.com/python/sliders/ of how to include a slider in plotly
    Notice how similar it already is to GapMinder!
    """
    df = px.data.gapminder()
    fig = px.scatter(df, x="gdpPercap", y="lifeExp",
                     animation_frame="year",  # set which column makes the animation though a slider
                     size="pop",
                     color="continent",
                     hover_name="country",
                     log_x=True,
                     size_max=55,
                     range_x=[100, 100000],
                     range_y=[25, 90])

    fig["layout"].pop("updatemenus")  # optional, drop animation buttons
    return fig


def plotly_button_example():
    """
    To have buttons, plotly requires us to use go (and not px) to generate the graph.
    The button options are very restrictive, since all they can do is change a parameter from the go graph.
    In the example below, it changes the "mode" value of the graph (between lines and scatter)
    The code is a modified code example taken from https://plotly.com/python/custom-buttons/
    """
    x = np.random.rand(50) * np.random.randint(-10, 10)
    y = np.random.rand(50) * np.random.randint(-10, 10)

    fig = go.Figure()

    # Add surface trace
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers'))

    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(type="buttons",
                 direction="left",
                 buttons=[
                     dict(
                         label="line",  # just the name of the button
                         method="update",
                         # This is the method of update (check https://plotly.com/python/custom-buttons/)
                         args=[{"mode": "markers"}],  # This is the value being updated in the visualization
                     ), dict(
                         label="scatter",  # just the name of the button
                         method="update",
                         # This is the method of update (check https://plotly.com/python/custom-buttons/)
                         args=[{"mode": "line"}],  # This is the value being updated in the visualization
                     )
                 ],
                 pad={"r": 10, "t": 10}, showactive=True, x=0.11, xanchor="left", y=1.1, yanchor="top"
                 # Layout-related values
                 ),
        ]
    )

    fig.show()
    return fig


##############################################
# Implement all the below methods
# Use the documentation of the libraries in case you need help to understand the required methods
##############################################
def matplotlib_interactivity():
    """
    Do an interactive matplotlib plot where I can select which visualization I want.
    Make either a slider, a dropdown or several buttons and make so each option gives me a different visualization from
    the matplotlib figures of b_simple_usages. Return just the resulting fig as is done in plotly_slider_example.
    """

    data = np.random.rand(10, 5)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    df = read_dataset('../../iris.csv')
    df_bar = df[get_numeric_columns(df)]
    df_bar = df_bar.max()
    data_pie = [len(get_numeric_columns(df)), len(get_binary_columns(df)), len(get_text_categorical_columns(df))]

    ax.bar(range(df_bar.shape[0]), df_bar)

    class Index(object):
        ind = 0

        def bar(self, event, i):
            self.ind = np.clip(self.ind + i, 0, data.shape[1] - 1)
            ax.clear()
            ax.bar(range(df_bar.shape[0]), df_bar)
            plt.draw()

        def pie(self, event, i):
            self.ind = np.clip(self.ind + i, 0, data.shape[1] - 1)
            ax.clear()
            ax.pie(np.array(data_pie), labels=range(0, len(data_pie)))
            plt.draw()

        def hm(self, event, i):
            self.ind = np.clip(self.ind + i, 0, data.shape[1] - 1)
            ax.clear()
            ax.imshow(df.corr(method='pearson'))
            plt.draw()

    callback = Index()
    ax_bar = plt.axes([0.1, 0.05, 0.12, 0.075])
    ax_pie = plt.axes([0.23, 0.05, 0.12, 0.075])
    ax_hm = plt.axes([0.36, 0.05, 0.12, 0.075])

    button_bar = Button(ax_bar, 'Bar')
    button_bar.on_clicked(lambda event: callback.bar(event, 1))

    button_pie = Button(ax_pie, "Pie")
    button_pie.on_clicked(lambda event: callback.pie(event, 1))

    button_hm = Button(ax_hm, "Heat map")
    button_hm.on_clicked(lambda event: callback.hm(event, 1))

    return fig


def matplotlib_cluster_interactivity():
    """
    Do an interactive matplotlib plot where I can select how many clusters I want to train from.
    Use iris dataset (just numeric columns) and k-means (feel free to reuse as/c_clustering if you wish).
    The slider (or dropdown) should range from 2 to 10. Return just the resulting fig.
    """

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    df = read_dataset('../../iris.csv')
    df = df[['petal_length', 'petal_width']]

    result = simple_k_means(df, 2)
    df['results'] = result["clusters"]

    for cluster_number in df['results'].unique():
        ax.scatter(df[df['results'] == cluster_number]['petal_length'],
                   df[df['results'] == cluster_number]['petal_width'])
        ax.scatter(df[df['results'] == cluster_number]['petal_length'].mean(),
                   df[df['results'] == cluster_number]['petal_width'].mean(), color='black')

    ax.set_xlabel('petal_length')
    ax.set_ylabel('petal_width')
    ax.set_title('Clustering based on petal_length and petal_width')

    class Index(object):
        ind = 0

        def change_clusters(self, value):
            self.multiplier = round(value)
            ax.clear()
            result = simple_k_means(df, self.multiplier)
            df['results'] = result["clusters"]

            for cluster_number in df['results'].unique():
                ax.scatter(df[df['results'] == cluster_number]['petal_length'], df[df['results'] == cluster_number]['petal_width'])
                ax.scatter(df[df['results'] == cluster_number]['petal_length'].mean(), df[df['results'] == cluster_number]['petal_width'].mean(), color='black')

            ax.set_xlabel('petal_length')
            ax.set_ylabel('petal_width')
            ax.set_title('Clustering based on petal_length and petal_width')
            plt.draw()

    callback = Index()

    axslider = plt.axes([0.55, 0.05, 0.35, 0.03])

    slider = Slider(axslider, 'Number of clusters', 2, 10, 1)
    slider.on_changed(callback.change_clusters)

    return fig


def plotly_interactivity():
    """
    Do a plotly graph with all plotly 6 figs from b_simple_usages, and make 6 buttons (one for each fig).
    Change the displayed graph depending on which button I click. Return just the resulting fig.
    """

    return None


if __name__ == "__main__":
    # Here are examples of how to run each method
    # Although you may remove or comment them out for testing, make sure they are present
    # and work when submitting, since when we run your code, all methods will be run like this.
    # If these lines below returns errors when running, your file will be considered to not
    # run, and graded accordingly.
    fig_m_i = matplotlib_interactivity()
    fig_m_ci = matplotlib_cluster_interactivity()
    fig_p = plotly_interactivity()

    # Uncomment the below lines to test your code
    # When submitting, leave the code below commented!!!
    # matplotlib_simple_example()[0].show()
    # matplotlib_simple_example2()[0].show()
    # plotly_slider_example().show()
    # plotly_button_example().show()
    # fig_m_i.show()
    # fig_m_ci.show()
    # fig_p.show()

from typing import Tuple
import json
import dash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Output, Input, State
from matplotlib.widgets import Button, Slider
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

##############################################
# Now let's use dash, a library built on top of flask (a backend framework for python) and plotly
# Check the documentation at https://dash.plotly.com/
# For a complete example, check https://dash-bootstrap-components.opensource.faculty.ai/examples/iris/
# Example(s). Read the comments in the following method(s)
##############################################
from assignments.assignment1.a_load_file import read_dataset
from assignments.assignment3.b_simple_usages import plotly_map, plotly_tree_map, plotly_polar_scatterplot_chart


def dash_simple_example():
    """
    Here is a simple example from https://dash.plotly.com/layout
    The focus is to create a fig from plotly and add it to dash, but differently from using just plotly, now we can use html elements,
    such as H1 for headers, Div for dividers, and all interations (buttons, sliders, etc).
    Check dash documentation for html and core components.
    """
    app = dash.Dash(__name__)

    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    # You create a fig just as you did in a_
    fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

    app.layout = html.Div(children=[
        html.H1(children='Hello Dash'),
        html.Div(children='Dash: A web application framework for Python.'),
        dcc.Graph(
            id='example-graph',
            figure=fig  # and include the fig here as a dcc.Graph
        )
    ])

    # To run the app, use app.run_server(debug=True)
    # However, do return the app instead so that the file can be run as in the end of this file
    return app


def dash_with_bootstrap_example():
    """
    Here is a simple example from https://dash.plotly.com/layout
    See examples of components from the bootstrap library at https://dash-bootstrap-components.opensource.faculty.ai/docs/components/alert/
    """
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    fig1 = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
    fig2 = px.line(df, x="Fruit", y="Amount", color="City")

    app.layout = dbc.Container([
        html.H1(children='Hello Dash'),
        html.Hr(),
        dbc.Row([
            dbc.Col(html.Div(children='Dash: A web application framework for Python.'), md=4),
            dbc.Col(dbc.Button('Example Button', color='primary', style={'margin-bottom': '1em'}, block=True), md=8)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='example-graph1', figure=fig1)),
            dbc.Col(dcc.Graph(id='example-graph2', figure=fig2))
        ])
    ])

    # To run the app, use app.run_server(debug=True)
    # However, do return the app instead so that the file can be run as in the end of this file
    return app


def dash_callback_example():
    """
    Here is a more complex example that uses callbacks. With this example, I believe you will suddenly perceive why dash (and webapps)
    are so much better for visual analysis.
    """
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    app.layout = dbc.Container([
        html.H1(children='Hello Dash'),
        html.Div(children='Dash: A web application framework for Python.'),
        html.Hr(),
        dbc.FormGroup([
            dbc.Label("Choose Dataset"),
            dcc.Dropdown(id="dropdown", value=1,
                         options=[{"label": "First Data", "value": 1}, {"label": "Second Data", "value": 2}]),
        ]),
        dbc.FormGroup([
            dbc.Label(id='slider-value'),
            dcc.Slider(id="slider", min=1, max=10, step=0.5, value=1),
        ]),
        dbc.Button('Run Callback', id='example-button', color='primary', style={'margin-bottom': '1em'}, block=True),
        dbc.Row([
            dbc.Col(dcc.Graph(id='example-graph')),
            # Not including fig here because it will be generated with the callback
        ])
    ])

    @app.callback(  # See documentation or tutorial to see how to use this
        Output('example-graph', 'figure'),
        # Outputs is what you wish to update with the callback, which in this case is the figure
        [Input('example-button', 'n_clicks')],
        # Use inputs to define when this callback is called, and read from the values in the inputs as parameters in the method
        [State('dropdown', 'value'),
         # Use states to read values from the interface, but values only in states will not trigger the callback when changed
         State('slider',
               'value')])  # For example, here if you change the slider, this method will not be called, it will only be called when you click the button
    def update_figure(n_clicks, dropdown_value, slider_value):
        df2 = df[:]
        df2.Amount = df2.Amount * slider_value
        if dropdown_value == 1:
            return px.bar(df2, x="Fruit", y="Amount", color="City", barmode="group")
        else:
            return px.line(df2, x="City", y="Amount", color="Fruit")

    @app.callback(Output('slider-value', 'children'), [Input('slider', 'value')])
    def update_slider_value(slider):
        return f'Multiplier: {slider}'

    #  You can also use app.callback to get selection from any of the plotly graphs, including tables and maps, and update anything you wish.
    #  See some examples at https://dash-gallery.plotly.host/Portal/

    return app


##############################################
# Implement all the below methods
# Use the documentation of the libraries in case you need help to understand the required methods
##############################################
def dash_task():
    """
    There is only only one task to do, a web app with:
    1. Some nice title
    2. One visualization placeholder for dataset visualization
        a. A dropdown to allow me to select which dataset I want to see (iris, video_game and life_expectancy)
        b. Two other dropdowns for me to choose what column to put in x and what column to put in y of the visualization
        c. Another dropdown for me to choose what type of graph I want (see examples in file a_) (at least 3 choices of graphs)
        d. Feel free to change the structure of the dataset if you prefer (e.g. change life_expectancy so that
            there is one column of "year", one for "country" and one for "value")
    4. A https://dash-bootstrap-components.opensource.faculty.ai/docs/components/card/ with the number of rows being showed on the above graph
    5. Another visualization with:
        a. It will containing the figure created in the tasks in a_, b_ or c_ related to plotly's figures
        b. Add a dropdown for me to choose among 3 (or more if you wish) different grap hs from a_, b_ or c_ (choose the ones you like)
        c. In this visualization, if I select data in the visualization, update some text in the page (can be a new bootstrap card with text inside)
            with the number of values selected. (see https://dash.plotly.com/interactive-graphing for examples)
    """

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    all_options = {
        'iris': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'],
        'video_game': ['user', 'asin', 'review'],
        'life_expectancy': ['value', 'country', 'ns']
    }

    app.layout = dbc.Container([

        dbc.NavbarSimple(
            brand="SunDash",
            brand_href="#",
            color="dark",
            dark=True,
            style={"width": "100%"}
        ),
        html.Br(),
        html.Div([
            html.Div([
                html.H3('Dataset Visualization'),
                html.Div([
                    html.Div([
                        dbc.FormGroup([
                            dbc.Label("Choose Dataset"),
                            dcc.Dropdown(
                                id='dataset-dropdown',
                                options=[{'label': k, 'value': k} for k in all_options.keys()],
                                value='iris',
                                multi=False,
                                style={'width': "100%"}
                            ), ]
                        ),
                    ], className='col-md-2', ),

                    html.Div([
                        dbc.FormGroup([
                            dbc.Label("Choose Graph"),
                            dcc.Dropdown(id='graph-dropdown',
                                         options=[
                                             {'label': 'Line', 'value': 'line'},
                                             {'label': 'Scatter', 'value': 'scatter'},
                                             {'label': 'Bar', 'value': 'bar'},
                                         ],
                                         value='scatter',
                                         style={'width': "100%"}), ]
                        ),
                    ], className='col-md-2', ),

                ], className='row', style={'width': "200%"}),

                html.Div([
                    html.Div([
                        dbc.FormGroup([
                            dbc.Label("Choose Column X"),
                            dcc.Dropdown(id='colx-dropdown'), ]
                        ),
                    ], className='col-md-2', ),

                    html.Div([
                        dbc.FormGroup([
                            dbc.Label("Choose Column Y"),
                            dcc.Dropdown(id='coly-dropdown', ), ]
                        ),
                    ], className='col-md-2', ),

                ], className='row', style={'width': "200%"}),
                html.Div(
                    dbc.Alert(id='display-row-count', color="primary"),
                    style={"font-weight": "normal", 'width': '70%'}),
                html.Br(),

                dcc.Graph(id='first-visualization', figure={}),
            ], className="two columns", style={'width': '50%'}),

            html.Div([
                html.H3('Clicked Data'),
                dbc.FormGroup([
                    dbc.Label("Choose Visualization"),
                    dcc.Dropdown(
                        id='vis-dropdown',
                        options=[
                            {'label': 'Plotly Map', 'value': 'map'},
                            {'label': 'Plotly Tree Map', 'value': 'tree-map'},
                            {'label': 'Plotly Polar Scatter', 'value': 'polar-scatter'},
                        ],
                        value='map',
                        multi=False,
                        style={'width': "60%"}
                    ), ]
                ),
                html.Div([
                    dbc.Alert('Nothing Selected', id='click-data'),
                ], className='three columns'),
                dcc.Graph(id='second-visualization', figure={})
            ], className="six columns", style={'width': '50%'}),
        ], className="row", style={'width': '100%'})

    ], style={'padding': 0})

    @app.callback(
        Output('second-visualization', 'figure'),
        [Input('vis-dropdown', 'value')])
    def set_cities_value(vis_value):
        if vis_value == 'map':
            return plotly_map()

        if vis_value == 'tree-map':
            return plotly_tree_map()

        if vis_value == 'polar-scatter':
            return plotly_polar_scatterplot_chart()

    @app.callback(
        Output('click-data', 'children'),
        [Input('second-visualization', 'clickData'), Input('vis-dropdown', 'value')])
    def display_click_data(clickData, vis_value):
        if vis_value == 'map':
            return "Country: " + str(clickData['points'][0]['location'])

        if vis_value == 'tree-map':
            return "Level: " + str(clickData['points'][0]['label'])

        if vis_value == 'polar-scatter':
            return "Theta: " + str(clickData['points'][0]['theta']) + " and Value: " + str(
                clickData['points'][0]['r'])

    @app.callback(
        Output('colx-dropdown', 'options'),
        [Input('dataset-dropdown', 'value')])
    def set_cities_options(selected_dataset):
        return [{'label': i, 'value': i} for i in all_options[selected_dataset]]

    @app.callback(
        Output('coly-dropdown', 'options'),
        [Input('dataset-dropdown', 'value')])
    def set_cities_options(selected_dataset):
        return [{'label': i, 'value': i} for i in all_options[selected_dataset]]

    @app.callback(
        Output('colx-dropdown', 'value'),
        [Input('colx-dropdown', 'options')])
    def set_cities_value(available_options):
        return available_options[0]['value']

    @app.callback(
        Output('coly-dropdown', 'value'),
        [Input('coly-dropdown', 'options')])
    def set_cities_value(available_options):
        return available_options[0]['value']

    @app.callback(
        Output('display-row-count', 'children'),
        [Input('dataset-dropdown', 'value')])
    def set_display_children(dataset):
        if dataset == 'iris':
            df = read_dataset('../../iris.csv')
            row = df.shape[0]

        if dataset == 'video_game':
            df = read_dataset('../../ratings_Video_Games.csv')
            df = df[:3000]
            row = df.shape[0]

        if dataset == 'life_expectancy':
            df = read_dataset('processed_le.csv')
            row = df.shape[0]

        return u'# Row displayed: {} '.format(row)

    @app.callback(
        Output('first-visualization', 'figure'),
        [Input('dataset-dropdown', 'value'),
         Input('colx-dropdown', 'value'),
         Input('coly-dropdown', 'value'),
         Input('graph-dropdown', 'value')])
    def update_graph(dataset, colx, coly, graph):

        if dataset == 'iris':
            df = read_dataset('../../iris.csv')

        if dataset == 'video_game':
            df = read_dataset('../../ratings_Video_Games.csv')
            df = df[:3000]

        if dataset == 'life_expectancy':
            df = read_dataset('processed_le.csv')

        if graph == 'scatter':
            fig = go.Figure(data=go.Scatter(x=df[colx], y=df[coly], mode='markers'))

        if graph == 'line':
            fig = go.Figure(data=go.Scatter(x=df[colx], y=df[coly], mode='lines'))

        if graph == 'bar':
            fig = go.Figure(data=go.Bar(x=df[colx], y=df[coly]))

        fig.update_layout(
            xaxis_title=colx,
            yaxis_title=coly,
        )

        return fig

    app.css.append_css({
        'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
    })

    return app


if __name__ == "__main__":
    # Here are examples of how to run each method
    # Although you may remove or comment them out for testing, make sure they are present
    # and work when submitting, since when we run your code, all methods will be run like this.
    # If these lines below returns errors when running, your file will be considered to not
    # run, and graded accordingly.
    app_ce = dash_callback_example()
    app_b = dash_with_bootstrap_example()
    app_c = dash_callback_example()
    app_t = dash_task()

    # Uncomment the below lines to test your code
    # When submitting, leave the code below commented!!!
    # app_ce.run_server(debug=True)
    # app_b.run_server(debug=True)
    # app_c.run_server(debug=True)
    # app_t.run_server(debug=True)

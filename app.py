import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from mlfromscratch.models import (k_means, k_nearest_neighbors, linear_regression, logistic_regression, naive_bayes,
                                  principal_component_analysis)
from mlfromscratch.utils.load import cls_data, reg_data

app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"
    ],
)


app.layout = html.Div(
    className="container",
    children=[
        html.Div(
            className="row p-5 text-center",
            children=[html.H4("Machine Learning From Scratch"), ],
        ),
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="col-4 jumbotron",
                    children=[
                        dbc.FormGroup(
                            [
                                dbc.Label("Dropdown", html_for="dropdown"),
                                dbc.RadioItems(
                                    id="data-selector",
                                    className='m-2 p-2',
                                    options=[
                                        {"label": "Regression", "value": "reg"},
                                        {"label": "Classification", "value": "cls"},
                                    ],
                                    value="reg"
                                ),
                            ],
                        ),
                        dbc.FormGroup(
                            [
                                dbc.Label("Dropdown", html_for="dropdown"),
                                dcc.Slider(
                                    id="slider",
                                    className='m-2 p-2',
                                    min=50,
                                    max=500,
                                    step=50,
                                    value=100,
                                    marks={
                                        50: {'label': '50', 'style': {'color': '#77b0b1'}},
                                        200: {'label': '200'},
                                        350: {'label': '350'},
                                        500: {'label': '500', 'style': {'color': '#f50'}}
                                    }
                                ),
                            ],
                        ),
                        dbc.FormGroup(
                            [
                                dbc.Label("Dropdown", html_for="dropdown"),
                                dcc.Dropdown(
                                    id="algo-selector",
                                    className='m-2 p-2',
                                    options=[
                                        {"label": "K Means", "value": "kmeans"},
                                        {"label": "K Nearest Neighbors", "value": "knn"},
                                        {"label": "Linear Regression", "value": "lin-reg"},
                                        {"label": "Logistic Regression", "value": "log-reg"},
                                        {"label": "Naive Bayes", "value": "naive-bayes"},
                                        {"label": "PCA", "value": "pca"},
                                    ],
                                    value="reg",
                                ),
                            ]
                        ),
                    ]
                ),
                html.Div(
                    className="col",
                    children=[
                        html.Div(
                            dcc.Graph(
                                id='graph',
                                # figure=None
                            )
                        )
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("graph", "figure"), [
        Input("data-selector", "value"), Input("slider", "value")]
)
def display_data(dropdown_val, slider_val):
    # make custom dataload function that accepts 'reg' or 'clf' as input
    # and generates data accordingly
    if dropdown_val == "reg":
        X, y = reg_data(n_samples=slider_val)
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=X.flatten(),
                    y=y,
                    mode='markers',
                    marker=dict(color=y)
                )
            ]
        )
    else:
        X, y = cls_data(n_samples=slider_val)
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=X[:, 0],
                    y=X[:, 1],
                    mode='markers',
                    marker=dict(color=y)
                )
            ]
        )
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)

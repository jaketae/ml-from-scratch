import dash
import dash_core_components as dcc
import dash_html_components as html
from mlfromscratch.models import (k_means, k_nearest_neighbors, linear_regression, logistic_regression, naive_bayes,
                                  principal_component_analysis)
from mlfromscratch.utils.load import clf_data, cls_data, reg_data

app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"
    ],
)


app.layout = html.Div(
    className='container',
    children=[
        html.H1(children="Hello Dash"),
        html.Div(
            className='row',
            children=[
                html.Div(
                    className='col',
                    children=[
                        dcc.Graph(
                            id="example-graph",
                            figure={
                                "data": [
                                    {"x": [1, 2, 3], "y": [4, 1, 2],
                                     "type": "bar", "name": "SF"},
                                    {
                                        "x": [1, 2, 3],
                                        "y": [2, 4, 5],
                                        "type": "bar",
                                        "name": u"Montréal",
                                    },
                                ],
                                "layout": {"title": "Dash Data Visualization"},
                            },
                        ),
                    ]
                ),
                html.Div(
                    className='col',
                    children=[
                        dcc.Graph(
                            id="example-graph2",
                            figure={
                                "data": [
                                    {"x": [1, 2, 3], "y": [4, 1, 2],
                                     "type": "bar", "name": "SF"},
                                    {
                                        "x": [1, 2, 3],
                                        "y": [2, 4, 5],
                                        "type": "bar",
                                        "name": u"Montréal",
                                    },
                                ],
                                "layout": {"title": "Dash Data Visualization"},
                            },
                        ),
                    ]
                )
            ]
        )
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)

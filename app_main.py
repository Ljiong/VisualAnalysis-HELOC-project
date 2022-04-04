import base64
import time
import os

import dash_table
import joblib
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import shap
from dash.dependencies import Input, Output
from matplotlib import pyplot as plt

import util_s.dash_reusable_components as drc
import util_s.create_figures as c_figs

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
app.title = "HELOC explaining dashboard"
server = app.server

features = pd.read_csv('./data/heloc_dataset_v1.csv')
X = features.drop('RiskPerformance', axis = 1)
id = X.index.values
Bad_F = features[features['RiskPerformance']=='Bad']
Bad_F = Bad_F[~Bad_F.ExternalRiskEstimate.isin([-9])]
Good_F = features[features['RiskPerformance']=='Good']
Good_F = Good_F[~Good_F.ExternalRiskEstimate.isin([-9])]

# the columns that stores the labels
labelDimension = "RiskPerformance"

# load trained model
dirs = 'SAVE_MODEL'
if not os.path.exists(dirs):
    os.makedirs(dirs)
model = joblib.load('./SAVE_MODEL/trained_model.pkl')

#SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values_sum = explainer.shap_values(X)
shap.summary_plot(shap_values_sum, X, show=False)
plt.savefig("full_importance_shap")

y_importance = model.feature_importances_

def description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H3("HELOC Analytics"),
            html.H5("Welcome to the HELOC Analytics Dashboard"),
            html.Div(
                id="intro",
                children="Explore Home Equity Line of Credit(HELOC) model, explain the prediction model. Select the figure to get more information about the model and how it works.",
            ),
        ],
    )


def generate_control_card():
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        style={"width": "100%"},
        children=[
            html.P("Select Feature"),
            dcc.Dropdown(
                id="dropdown-select-features",
                options=[
                    {"label": "ExternalRiskEstimate", "value": "ExternalRiskEstimate"},
                    {
                        "label": "NetFractionRevolvingBurden",
                        "value": "NetFractionRevolvingBurden",
                    },
                    {
                        "label": "AverageMInFile",
                        "value": "AverageMInFile",
                    },
                    {
                        "label": "MSinceOldestTradeOpen",
                        "value": "MSinceOldestTradeOpen",
                    },
                    {
                        "label": "PercentTradesWBalance",
                        "value": "PercentTradesWBalance",
                    },
                ],
                value="ExternalRiskEstimate",
            ),
            html.Br(),
            html.P("Select Global analysis"),
            dcc.Dropdown(
                id="dropdown-select-graph",
                options=[
                    {"label": "Confusion Matrix", "value": "conf"},
                    {"label": "Feature Importance", "value": "FI"},
                    {"label": "Correlation Heatmap", "value": "CH"},
                    {"label": "Feature influence", "value": "infl"},
                ],
                value="conf",
            ),
            html.Br(),
            html.P("Select ID"),
            dcc.Dropdown(
                id="input_id",
                options=[{"label": i, "value": i} for i in id],
                value=id[0],
            ),
            html.Br(),
            html.P("HELOC Data"),
            html.Div(
                id='datatable-interactivity-container',
                style={'display': 'flex', 'justify-content': 'center'},
                children=[
                    dash_table.DataTable(
                        id='datatable-interactivity',
                        columns=[
                            {"name": i, "id": i, "deletable": True, "selectable": True} for i in features.columns
                        ],
                        style_table={
                            'width': '100%',
                            'maxWidth': '50ex',
                            'overflowX': 'scroll',
                        },
                        data=features.to_dict('records'),
                        editable=True,
                        filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                        column_selectable="single",
                        row_selectable="multi",
                        row_deletable=True,
                        selected_columns=[],
                        selected_rows=[],
                        page_action="native",
                        page_current=0,
                        page_size=10,
                    ),
                ]
            ),
        ],
    )

app.layout = html.Div(
    children=[
        # .container class is fixed, .container.scalable is scalable
        # html.Div(
        #     className="banner",
        #     children=[
        #         html.H2(
        #             id="banner-title",
        #             children=[
        #                 html.A(
        #                     "HELOC explaining dashboard",
        #                     # href="https://github.com/plotly/dash-svm",
        #                     style={
        #                         "text-decoration": "none",
        #                         "color": "inherit",
        #                     },
        #                 )
        #             ],
        #         ),
        #     ],
        # ),

        # left-column
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[html.Img(src=app.get_asset_url("dash-logo-new.png"))],
        ),
        # html.Div(
        #     id="banner",
        #     className="banner",
        #     children=[html.H3("HELOC Analytics"),],
        # ),
        html.Div(
            id="left-column",
            className="four columns",
            children=[description_card(), generate_control_card()],
        ),
        # html.Div(
        #     id="app-container",
        #     # className="row",
        #     children=[
        #         html.H4("Global Analysis"),
        #         drc.Card(
        #             drc.NamedDropdown(
        #                 name="Select figs",
        #                 id="dropdown-select-graph",
        #                 options=[
        #                     {"label": "Confusion Matrix", "value": "conf"},
        #                     {
        #                         "label": "Feature Importance",
        #                         "value": "FI",
        #                     },
        #                     {
        #                         "label": "Correlation Heatmap",
        #                         "value": "CH",
        #                     },
        #                 ],
        #                 clearable=False,
        #                 searchable=False,
        #                 value="conf",
        #             ),
        #         ),
        #
        #     ]
        # ),

        # html.Div(
        #     id = "features-select-dropdown",
        #     children = [
        #         drc.Card(
        #             drc.NamedDropdown(
        #                         name="Select features",
        #                         id="dropdown-select-features",
        #                         options=[
        #                             {"label": "ExternalRiskEstimate", "value": "ExternalRiskEstimate"},
        #                             {
        #                                 "label": "NetFractionRevolvingBurden",
        #                                 "value": "NetFractionRevolvingBurden",
        #                             },
        #                             {
        #                                 "label": "AverageMInFile",
        #                                 "value": "AverageMInFile",
        #                             },
        #                             {
        #                                 "label": "MSinceOldestTradeOpen",
        #                                 "value": "MSinceOldestTradeOpen",
        #                             },
        #                             {
        #                                 "label": "PercentTradesWBalance",
        #                                 "value": "PercentTradesWBalance",
        #                             },
        #                         ],
        #                         clearable=False,
        #                         searchable=False,
        #                         value="ExternalRiskEstimate",
        #                     ),
        #         ),
        #     ]
        # ),

        html.Div(
            id="right-column",
            className="eight columns",
            children=[
                    # feature distribution
                    html.Div(
                        id="feature_distribution",
                        children=[
                            html.B("Feature distribution"),
                            html.Hr(),
                            dcc.Graph(id="feature_distribution_fig"),
                            #html.Div(id="wait_time_table", children=initialize_table()),
                        ],
                    ),

                    # Global analysis
                    html.Div(
                        id="global_graphs",
                        children=[
                            html.B("Global analysis"),
                            html.Hr(),
                            html.Div(
                                id="div_graphs",
                                children=dcc.Graph(
                                    id="graph-global",
                                    figure=dict(
                                        layout=dict(
                                            plot_bgcolor="#282b38", paper_bgcolor="#282b38"
                                        )
                                    ),
                                ),
                             ),
                        ]
                    ),
                    html.Div(
                        id="Local_graphs",
                        children=[
                            html.B("Local analysis"),
                            html.Hr(),
                            html.Div(
                                #Local Analyasis
                                html.Iframe(
                                    id="individual_feature_importance",
                                    srcDoc=None,
                                    style={"height": "100%", "width": "100%"},
                                ),
                            )
                        ]
                    )

                # html.Div(
                #     id="div-graphs",
                #     children=[
                #         html.B("Global analysis"),
                #         html.Hr(),
                #         dcc.Graph(
                #             id="graph-global",
                #             figure=dict(
                #                 layout=dict(
                #                     plot_bgcolor="#282b38", paper_bgcolor="#282b38"
                #                 )
                #             ),
                #         ),
                #     ]
                # ),
            ]
        ),

    ]
)

# app.clientside_callback(
#     ClientsideFunction(namespace="clientside", function_name="resize"),
#     Output("output-clientside", "children"),
#     [Input("global_graphs", "children")],
# )
#
#
# @app.callback(
#     Output("wait_time_table", "children"),
#     [
#         Input("dropdown-select-graph", "value"),
#         Input("dropdown-select-features", "value"),
#         Input("reset-btn", "n_clicks"),
#         Input("input_id", "value"),
#     ]
#
# )

# @app.callback(
#     Output('datatable-interactivity', 'style_data_conditional'),
#     Input('datatable-interactivity', 'selected_columns')
# )
# def update_styles(selected_columns):
#     return [{
#         'if': { 'column_id': i },
#         'background_color': '#D2F3FF'
#     } for i in selected_columns]


@app.callback(
    Output("feature_distribution_fig", "figure"),
    Input("dropdown-select-features", "value"),
)
def update_feature_distribution(feature_column):
    return c_figs.distribution_fig(column=feature_column, Bad_F=Bad_F, Good_F=Good_F)


@app.callback(
        Output("div_graphs", "children"),
        Input("dropdown-select-graph", "value"),
)
def update_global_graph(graphselect):
    # t_start = time.time()
    # h = 0.3  # step size in the mesh
    '''
    global figure
    '''
    confusion_figure = c_figs.CM(
        model=model, features=features,labelDimension=labelDimension
    )

    feature_import = c_figs.feature_imp(
        y_importance=y_importance, features=features,labelDimension=labelDimension
    )

    if graphselect == "conf":
        return html.Div(
                    id="CM-graph-container",
                    children=dcc.Loading(
                        className="graph-wrapper",
                        children=dcc.Graph(
                            id="graph-pie-confusion-matrix", figure=confusion_figure
                            ),
                    ),
                )

    elif graphselect == "CH":
        return html.Div(
                    id="CH-graph-container",
                    children=dcc.Loading(
                        className="graph-wrapper",
                        children=[html.Img(src=app.get_asset_url('figureCorrelation.jpg'), height="640" , width="800")],
                        style={"display": "none"},
                    ),
                )

    elif graphselect == "FI":
        return html.Div(
                    id="FI-graph-container",
                    children=dcc.Loading(
                        className="graph-wrapper",
                        children=dcc.Graph(
                            id="graph-bar-feature-importance", figure=feature_import
                            ),
                    ),
                )
    elif graphselect == "infl":
        image_filename = 'full_importance_shap.png'
        encoded_image = base64.b64encode(open(image_filename, 'rb').read())
        return html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))


@app.callback(
    Output("individual_feature_importance", "srcDoc"),
    Input("input_id", "value"),
    prevent_initial_call=True,
)
def update_output_div(input_id):
    if input_id == None:
        choosen_instance = X.loc[[0]]
    else:
        choosen_instance = X.loc[[input_id]]
    shap_v1 = explainer.shap_values(choosen_instance)
    force_plot = shap.force_plot(explainer.expected_value[0], shap_v1[1], choosen_instance)

    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"

    return shap_html


# Running the server
if __name__ == "__main__":
    app.run_server(debug=False,host='127.1.1.2')

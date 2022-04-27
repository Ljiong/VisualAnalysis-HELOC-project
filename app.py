import base64
import os

import dash_table
import joblib
import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import shap
from dash.dependencies import Input, Output, ClientsideFunction
from matplotlib import pyplot as plt
import plotly.graph_objs as go
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score


app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
app.title = "HELOC explaining dashboard"
server = app.server
plt.clf()

filePath = './data/heloc_dataset_v1.csv'

features = pd.read_csv('./data/heloc_dataset_v1.csv')
X = features.drop('RiskPerformance', axis=1)
y = np.array(features['RiskPerformance'])

train_X = pd.read_csv('trained_X.csv', index_col=0)
train_y = pd.read_csv('trained_y.csv', index_col=0)
test_X = pd.read_csv('tested_X.csv', index_col=0)
test_y = pd.read_csv('tested_y.csv', index_col=0)


id = X.index.values
Bad_F = features[features['RiskPerformance'] == 'Bad']
Bad_F = Bad_F[~Bad_F.ExternalRiskEstimate.isin([-9])]
Good_F = features[features['RiskPerformance'] == 'Good']
Good_F = Good_F[~Good_F.ExternalRiskEstimate.isin([-9])]
max_external_bad = max(Bad_F.ExternalRiskEstimate)
min_external_bad = min(Bad_F.ExternalRiskEstimate)
max_external_good = max(Good_F.ExternalRiskEstimate)
min_external_good = min(Good_F.ExternalRiskEstimate)
# print(max_external,min_external)
# the columns that stores the labels
labelDimension = "RiskPerformance"

# load trained model
dirs = 'SAVE_MODEL'
if not os.path.exists(dirs):
    os.makedirs(dirs)
model = joblib.load('./SAVE_MODEL/trained_model.pkl')
# print(test_X)
# test_X_array = np.array(test_X)

pred = model.predict(test_X)
acc = cross_val_score(model, X, y, scoring='accuracy', n_jobs=-1, error_score='raise')
precise = precision_score(test_y, pred, pos_label='Good')
rec = recall_score(test_y, pred, pos_label='Good')
# SHAP explainer
explainer = shap.TreeExplainer(model)

y_importance = model.feature_importances_

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
)
app.title = "HELOC Analysis"
server = app.server


# Create app layout
app.layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div(
            [
                # banner
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("dash-logo.png"),
                            id="plotly-image",
                            style={
                                "height": "60px",
                                "width": "auto",
                                "margin-bottom": "25px",
                            },
                        )
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "HELOC Analytics",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    "Welcome to the HELOC Analytics Dashboard. Get more information for HELOC model.",
                                    style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
                html.Div(
                    [
                        # html.A(
                        #     html.Button("Learn More", id="learn-more-button"),
                        #     href="https://plot.ly/dash/pricing/",
                        # )
                    ],
                    className="one-third column",
                    id="button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),

        # Global Part
        # First row
        html.Div(
            [
                html.Div(
                    [
                        html.H6("Global Analysis"),
                        html.P(
                            "The aim of Global Analysis is to interpret the model from two aspects: the construction and evaluation.",
                            className="control_label",
                        ),
                        html.P(
                            "There are Feature Distribution, Feature Correlation, scores of evaluation(Accuracy, Recall...) \
                            and Feature Importance to show more information about the HELOC model.",
                            className="control_label",
                        ),
                        html.P(
                            "The scores of evaluation shows the precise of the model. The feature correlation shows \
                            the relationship between the features.\
                            In Feature distribution figure, you can select specific group of applicants\
                            to check the feature importance for this group.",
                            className="control_label",
                        ),
                        html.P("Select Distribution Feature",
                               className="control_label",
                               ),
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
                                    "label": "MSinceMostRecentInqexcl7days",
                                    "value": "MSinceMostRecentInqexcl7days",
                                },
                                {
                                    "label": "MSinceMostRecentDelq",
                                    "value": "MSinceMostRecentDelq",
                                },
                            ],
                            value="ExternalRiskEstimate",
                        ),
                        html.P(
                            "Select feature with 'Good' result for distribution ",
                            className="control_label",
                        ),
                        dcc.RangeSlider(
                            id="slider_good",
                            min=min_external_good,
                            max=max_external_good,
                            value=[min_external_good, max_external_good],
                            className="dcc_control",
                        ),
                        html.P(
                            "Select feature with 'Bad' result for distribution ",
                            className="control_label",
                        ),
                        dcc.RangeSlider(
                            id="slider_bad",
                            min=min_external_bad,
                            max=max_external_bad,
                            value=[min_external_bad, max_external_bad],
                            className="dcc_control",
                        ),
                    ],
                    className="pretty_container four columns",
                    id="global-filter",
                ),
                html.Div(
                    [
                        # evaluation result section
                        html.Div(
                            [
                                html.Div(
                                    [html.H6(id="num_trained_applicants_text"),
                                     html.P("Number of Applicants for train")],
                                    id="train_applicants",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="num_test_applicants_text"), html.P("Number of Applicants for test")],
                                    id="test_applicants",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="accuracy_text"), html.P("accuracy")],
                                    id="accuracy",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="precise_text"), html.P("precise")],
                                    id="precise",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="recall_text"), html.P("recall")],
                                    id="recall",
                                    className="mini_container",
                                ),
                            ],
                            id="info-container",
                            className="row container-display",
                        ),
                        # correlation figure
                        html.Div([
                            html.Div(
                                id="CH-graph-container",
                                children=dcc.Loading(
                                    className="graph-wrapper",
                                    children=[
                                        html.P("Feature Correlation"),
                                        html.Img(src=app.get_asset_url('figureCorrelation.jpg'), height="100%",
                                                 width="100%")],
                                    style={"display": "none"},
                                ),
                            )],
                            className="pretty_container",
                        ),
                    ],
                    id="global-right-column",
                    className="eight columns",
                ),
            ],
            className="row flex-display",
        ),
        # Second row
        html.Div(
            [
                # distribution to select(history graph)
                html.Div(
                    id="feature_distribution",
                    children=[
                        html.B("Feature distribution"),
                        dcc.Graph(id="feature_distribution_fig_good"),
                        dcc.Graph(id="feature_distribution_fig_bad"),
                    ],
                    className="pretty_container five column",
                ),
                # feature importance
                html.Div(
                    id="feature_importance",
                    children=dcc.Loading(
                        className="graph-wrapper",
                        children=[
                            html.B("Feature Importance"),
                            html.Div(id="feature_importance_fig"),
                            html.B("Feature Importance in 'Good' result"),
                            html.Div(id="good_feature_importance_fig"),
                            html.B("Feature Importance in 'Bad' result"),
                            html.Div(id="bad_feature_importance_fig"),
                        ],
                        style={"display": "none"},
                    ),
                    className="pretty_container seven column",
                ),
            ],
            className="row flex-display",
        ),

        # Local Part
        html.Div(
            [
                # Control Panel for Local
                html.Div(
                    [
                        html.H6("Local Analysis"),
                        html.P(
                            "There are feature historical figure and individual feature importance figure for local analysis. \
                            The historical figure is to compare the features between the test data and the most similar data. \
                            You can enter the new data below. \
                            And for feature importance, it's to get a knowledge about the contribution of the features \
                            to the result for specific data point. The individual feature importance figure is \
                             updated by select id below.",
                            className="control_label",
                        ),
                        html.P("Select ID for local analysis", className="control_label", ),
                        dcc.Dropdown(
                            id="input_id",
                            options=[{"label": i, "value": i} for i in id],
                            value=id[0],
                            className="dcc_control",
                        ),
                        html.Br(),
                        html.P("Enter The New Data", className="control_label", ),
                        html.Br(),
                        html.P("ExternalRiskEstimate: "),
                        dcc.Input(
                            id="my-input1",
                            type='number',
                            value=10,
                        ),
                        html.Br(),
                        html.P("NetFractionRevolvingBurden: "),
                        dcc.Input(
                            id="my-input2",
                            value=10,
                            type='number',
                        ),
                        html.Br(),
                        html.P("AverageMinFile: "),
                        dcc.Input(
                            id="my-input3",
                            value=10,
                            type='number',
                        ),
                        html.Br(),
                        html.P("MSinceMostRecentinqexcl7days: "),
                        dcc.Input(
                            id="my-input4",
                            value=10,
                            type='number',
                        ),

                    ],
                    className="pretty_container four columns",
                    id="local-filter",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.B("Local analysis"),
                                html.Br(),
                                # Local Analyasis
                                html.Iframe(
                                    id="shap_iframe_id",
                                    srcDoc=None,
                                    style={"scrolling": "no"},
                                    className="iframe_container",
                                ),
                            ],
                            className="pretty_container",
                        ),
                        html.Div(
                            [
                                html.B("Similar data analysis"),
                                html.Hr(),
                                dcc.Graph(id="ouput_barchart")],
                            className="pretty_container",
                        ),
                    ],
                    id="local-right-column",
                    className="eight columns",
                ),
            ],
            className="row flex-display",
        ),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)

# Create callbacks
app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [
        Input("feature_distribution_fig_good", "figure"),
        Input("feature_distribution_fig_bad", "figure"),
    ],
)


@app.callback(
    [
        Output("feature_importance_fig", "children"),
        Output("good_feature_importance_fig", "children"),
        Output("bad_feature_importance_fig", "children"),
    ],
    [
        Input("slider_good", "value"),
        Input("slider_bad", "value"),
        Input("dropdown-select-features", "value"),
    ]
)
def update_feature_importance(select_good, select_bad, select_feature):
    # default feature importance figures
    good_image_filename = 'good_importance_shap.png'
    good_encoded_image = base64.b64encode(open(good_image_filename, 'rb').read())
    good_importance = html.Img(src='data:image/png;base64,{}'.format(good_encoded_image.decode()), height="25%",
                               width="100%")
    bad_image_filename = 'bad_importance_shap.png'
    bad_encoded_image = base64.b64encode(open(bad_image_filename, 'rb').read())
    bad_importance = html.Img(src='data:image/png;base64,{}'.format(bad_encoded_image.decode()), height="25%",
                              width="100%")
    full_image_filename = 'full_importance_shap.png'
    full_encoded_image = base64.b64encode(open(full_image_filename, 'rb').read())
    full_importance = html.Img(src='data:image/png;base64,{}'.format(full_encoded_image.decode()), height="25%",
                               width="100%")
    l_good = select_good[0]
    u_good = select_good[1]
    l_bad = select_bad[0]
    u_bad = select_bad[1]

    # if select_good is not None:
    if l_good != min(Good_F[select_feature]) and u_good != max(Good_F[select_feature]):
        print(f"select_good:{select_good}")
        good_filter = Good_F[Good_F[select_feature].isin(range(l_good, u_good))].drop('RiskPerformance', axis=1)
        shap_values_good_filter = explainer.shap_values(good_filter)
        shap.summary_plot(shap_values_good_filter[1], good_filter, show=False)
        plt.savefig("filter_good_importance_shap1", dpi=50, bbox_inches='tight')
        plt.clf()
        good_image_filename = 'filter_good_importance_shap1.png'
        good_encoded_image = base64.b64encode(open(good_image_filename, 'rb').read())
        good_importance = html.Img(src='data:image/png;base64,{}'.format(good_encoded_image.decode()), height="25%",
                                   width="100%")

    if l_bad != min(Bad_F[select_feature]) and u_bad != max(Bad_F[select_feature]):
        bad_filter = Bad_F[Bad_F[select_feature].isin(range(l_bad, u_bad))].drop('RiskPerformance', axis=1)
        shap_values_bad_filter = explainer.shap_values(bad_filter)
        shap.summary_plot(shap_values_bad_filter[0], bad_filter, show=False)
        plt.savefig("filter_bad_importance_shap1", dpi=50, bbox_inches='tight')
        plt.clf()
        bad_image_filename = 'filter_bad_importance_shap1.png'
        bad_encoded_image = base64.b64encode(open(bad_image_filename, 'rb').read())
        bad_importance = html.Img(src='data:image/png;base64,{}'.format(bad_encoded_image.decode()), height="25%",
                                  width="100%")
    return full_importance, good_importance, bad_importance


# Slider_good <- distribution graph
@app.callback(
    [
        Output("slider_good", "value"),
        Output("slider_good", "min"),
        Output("slider_good", "max"),
    ],
    [
        Input("feature_distribution_fig_good", "selectedData"),
        Input("dropdown-select-features", "value")
    ]
)
def update_year_slider(count_graph_selected, feature_column):
    l_bound = min(Good_F[feature_column])
    u_bound = max(Good_F[feature_column])
    if count_graph_selected is None:
        return [l_bound, u_bound],l_bound,u_bound
    nums = [point["x"] for point in count_graph_selected["points"]]
    return [min(nums) + l_bound, max(nums) + l_bound + 1]


@app.callback(
    Output("feature_distribution_fig_good", "figure"),
    [
        Input("dropdown-select-features", "value"),
        Input("slider_good", "value"),
    ]
)
def update_feature_distribution(feature_column, selection):
    df_G = Good_F.copy()

    colors_good = []
    feature_list = df_G[feature_column].sort_values()
    feature_list.drop_duplicates('first', False)
    feature_list = feature_list.tolist()
    #print(feature_list)
    for i in feature_list:
        if i >= selection[0] and i <= selection[1]:
            colors_good.append("rgb(255, 188, 46)")
        else:
            colors_good.append("rgba(255, 188, 46, 0.2)")

    trace_select = go.Histogram(x=df_G[feature_column], opacity=0.75, marker=dict(color=colors_good), name='GOOD')
    data = [trace_select]
    layout = go.Layout(
        title="Histogram of " + str(feature_column) + "in 'Good' result",
        plot_bgcolor="#F9F9F9",
        paper_bgcolor="#F9F9F9",
        # paper_bgcolor = "#FFFFFF",
        font={"color": "#a5b1cd"},
        xaxis_title_text='Value',
        yaxis_title_text='Count',
        # dragmode='select',
    )

    figure = go.Figure(layout=layout)
    figure.add_traces(data=data)
    figure.update_layout(bargap=0.2)
    # figure.add_traces(data=data)
    return figure


# Slider_bad <- distribution graph
@app.callback(
    [
        Output("slider_bad", "value"),
        Output("slider_bad", "min"),
        Output("slider_bad", "max"),

    ],
    [
        Input("feature_distribution_fig_bad", "selectedData"),
        Input("dropdown-select-features", "value"),
    ]
)
def update_year_slider(count_graph_selected, feature_column):
    l_bound = min(Bad_F[feature_column])
    u_bound = max(Bad_F[feature_column])
    if count_graph_selected is None:
        return [l_bound, u_bound],l_bound,u_bound
    nums = [point["x"] for point in count_graph_selected["points"]]
    return [min(nums) + l_bound, max(nums) + l_bound + 1]


@app.callback(
    Output("feature_distribution_fig_bad", "figure"),
    [
        Input("dropdown-select-features", "value"),
        Input("slider_bad", "value"),
    ]
)
def update_feature_distribution(feature_column, selection):
    df = Bad_F.copy()
    print(selection)

    colors_bad = []
    feature_list = df[feature_column].sort_values()
    feature_list.drop_duplicates('first', False)
    feature_list = feature_list.tolist()
    print(feature_list)
    for i in feature_list:
        if i >= selection[0] and i <= selection[1]:
            colors_bad.append("rgb(123, 199, 255)")
        else:
            colors_bad.append("rgba(123, 199, 255, 0.2)")

    trace = go.Histogram(x=df[feature_column], opacity=0.75, marker={"color": colors_bad}, name='Bad')
    data = [trace]
    layout = go.Layout(
        title="Histogram of " + str(feature_column) + "in 'Bad' result",
        plot_bgcolor="#F9F9F9",
        paper_bgcolor="#F9F9F9",
        # paper_bgcolor = "#FFFFFF",
        font={"color": "#a5b1cd"},
        xaxis_title_text='Value',
        yaxis_title_text='Count',
        # dragmode='select',
    )

    figure = go.Figure(layout=layout)
    figure.add_traces(data=data)
    figure.update_layout(bargap=0.2)
    return figure


# individual feature importance
@app.callback(
    Output("shap_iframe_id", "srcDoc"),
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

    shap_html = f"<head>{shap.getjs()}</head><body scroll='no' style='overflow: hidden'>{force_plot.html()}</body>"

    return shap_html


@app.callback(
    [
        Output("num_trained_applicants_text", "children"),
        Output("num_test_applicants_text", "children"),
        Output("accuracy_text", "children"),
        Output("precise_text", "children"),
        Output("recall_text", "children"),
    ],
    [
        Input("slider_good", "value"),
        Input("slider_bad", "value"),
        Input("dropdown-select-features", "value"),
    ],
)
def update_text(select_good,select_bad,select_feature):
    train_number_points = train_X.shape[0]
    tot_num = X.shape[0]
    test_num_points = tot_num - train_number_points
    acc_model = acc.mean()
    prec_model = precise.mean()
    rec_model = rec.mean()
    return train_number_points, test_num_points, round(acc_model, 3), round(prec_model, 3), round(rec_model, 3)


@app.callback(
    Output('ouput_barchart', 'figure'),
    [
        Input("my-input1", "value"),
        Input("my-input2", "value"),
        Input("my-input3", "value"),
        Input("my-input4", "value"),
    ]
)
def update_output_div(input_value1, input_value2, input_value3, input_value4):
    index_list, input_data = cal_index(input_value1, input_value2, input_value3, input_value4)
    return plot_bar(index_list, input_data)


def loadDataFromCSV(filePath, report=False):
    raw = pd.read_csv(filePath)
    x = raw.values[:, 1:]
    y = raw.values[:, 0]
    if report:
        print('x:', x.shape, 'y:', y.shape)
    return x, y


# selected features for input
important_list = [7, 3, 14, 17]


def cal_index(input1, input2, input3, input4, k=3):
    '''
    calculate the index of k-th nearest neighbor
    input1,input2,input3,input4: input from the website
    k= number of neighbor
    '''
    x, y = loadDataFromCSV(filePath, report=True)
    dist = np.zeros(len(x))
    x_normed = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    input_data = np.mean(x_normed, axis=0)
    for i in important_list:
        input_data[i] -= x.min(axis=0)[i]
    input_data[important_list[0]], input_data[important_list[1]], input_data[important_list[2]], input_data[
        important_list[3]] = input1, input2, input3, input4
    for i in range(len(x)):
        dist[i] = np.linalg.norm(input_data - x_normed[i])
    return np.argpartition(dist, k)[:k], input_data


def plot_bar(index_list, input_data):
    '''
    plot the grouped bar chart by given index
    also show the input data on figure
    '''
    y = []
    for i in range(3):
        y.append(pd.read_csv(filePath).iloc[index_list[i]].values)
        if y[i][0] == 'Bad':
            y[i][0] = 0
        else:
            y[i][0] = 200

    data = [go.Bar(name='1st nearest neighbor', x=pd.read_csv(filePath).columns.values,
                   y=y[0]),
            go.Bar(name='2nd nearest neighbor', x=pd.read_csv(filePath).columns.values,
                   y=y[1]),
            go.Bar(name='3rd nearest neighbor', x=pd.read_csv(filePath).columns.values,
                   y=y[2]),
            go.Bar(name='input', x=pd.read_csv(filePath).columns.values, y=input_data)]
    fig = go.Figure(data=data)
    fig.update_layout(height=1500)
    return fig


# Main
if __name__ == "__main__":
    app.run_server(debug=False)

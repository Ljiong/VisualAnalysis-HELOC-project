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
from dash.dependencies import Input, Output
from matplotlib import pyplot as plt
import plotly.graph_objs as go


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

FEATURE = X.columns.values
df_0 = pd.read_csv('append2.csv')
df_0 = df_0.iloc[:, 0:4]
df_1 = pd.read_csv('MaxDelq2PublicRecLast12M.csv')
df_1 = df_1.iloc[:,0:2]
df_2 = pd.read_csv('MaxDelqEver.csv')
df_2 = df_2.iloc[:,0:2]
df_3 = pd.read_csv('special.csv')
df_3 = df_3.iloc[:,0:2]


id = X.index.values
Bad_F = features[features['RiskPerformance'] == 'Bad']
Bad_F = Bad_F[~Bad_F.ExternalRiskEstimate.isin([-9])]
Good_F = features[features['RiskPerformance'] == 'Good']
Good_F = Good_F[~Good_F.ExternalRiskEstimate.isin([-9])]
max_external_bad = max(Bad_F.ExternalRiskEstimate)
min_external_bad = min(Bad_F.ExternalRiskEstimate)
max_external_good = max(Good_F.ExternalRiskEstimate)
min_external_good = min(Good_F.ExternalRiskEstimate)

labelDimension = "RiskPerformance"

# load trained model
dirs = 'SAVE_MODEL'
if not os.path.exists(dirs):
    os.makedirs(dirs)
model = joblib.load('./SAVE_MODEL/trained_model.pkl')

# SHAP explainer, only need to execute one time
explainer = shap.TreeExplainer(model)

# Please remove the next few lines if you don't have these images in the folder!
shap_values_sum = explainer.shap_values(X)
shap.summary_plot(shap_values_sum, X, show=False, plot_type="bar",max_display=25)
plt.savefig("full_importance_shapX",dpi=50,bbox_inches='tight')
plt.clf()
shap.summary_plot(shap_values_sum[0], X, show=False,max_display=25)
plt.savefig("bad_importance_shapX",dpi=50,bbox_inches='tight')
plt.clf()
shap.summary_plot(shap_values_sum[1], X, show=False,max_display=25)
plt.savefig("good_importance_shapX",dpi=50,bbox_inches='tight')
plt.clf()


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

        # Prediction
        html.Div(
            [
                html.Div(
                    [
                        html.H5("Model Prediction"),
                        html.P("In this part, you can enter your data and use the model to predict the result. The result will automatical showed above the subscale analysis part. The contribution of each feature is shown in below. Blue represents the contribution in bad result and pink represents the contribution in good result."),
                        html.Br(),
                        "ExternalRiskEstimate:",
                        dcc.Input(
                            id="input_ExternalRiskEstimate",
                            type="number",
                            value = 0,
                            placeholder="0",
                        ),
                        " MSinceOldestTradeOpen:",
                        dcc.Input(
                            id="input_MSinceOldestTradeOpen",
                            type="number",
                            placeholder="0",
                            value=0,

                        ),
                        " MSinceMostRecentTradeOpen:",
                        dcc.Input(
                            id="input_MSinceMostRecentTradeOpen",
                            type="number",
                            placeholder="0",
                            value=0,

                        ),
                        html.Br(),
                        " AverageMInFile:",
                        dcc.Input(
                            id="input_AverageMInFile",
                            type="number",
                            placeholder="0",
                            value=0,

                        ),
                        " NumSatisfactoryTrades:",
                        dcc.Input(
                            id="input_NumSatisfactoryTrades",
                            type="number",
                            placeholder="0",
                            value=0,

                        ),
                        " NumTrades60Ever/DerogPubRec:",
                        dcc.Input(
                            id="input_NumTrades60Ever/DerogPubRec",
                            type="number",
                            placeholder="0",
                            value=0,

                        ),
                        html.Br(),
                        " NumTrades90Ever/DerogPubRec:",
                        dcc.Input(
                            id="input_NumTrades90Ever/DerogPubRec",
                            type="number",
                            placeholder="0",
                            value=0,

                        ),
                        " PercentTradesNeverDelq:",
                        dcc.Input(
                            id="input_PercentTradesNeverDelq",
                            type="number",
                            placeholder="0",
                            value=0,

                        ),
                        " MSinceMostRecentDelq:",
                        dcc.Input(
                            id="input_MSinceMostRecentDelq",
                            type="number",
                            placeholder="0",
                            value=0,

                        ),
                        html.Br(),

                        " MaxDelq/PublicRecLast12M:",
                        dcc.Input(
                            id="input_MaxDelq/PublicRecLast12M",
                            type="number",
                            placeholder="0",
                            value=0,

                        ),
                        " MaxDelqEver:",
                        dcc.Input(
                            id="input_MaxDelqEver",
                            type="number",
                            placeholder="0",
                            value=0,

                        ),
                        " NumTotalTrades:",
                        dcc.Input(
                            id="input_NumTotalTrades",
                            type="number",
                            placeholder="0",
                            value=0,

                        ),
                        html.Br(),

                        " NumTradesOpeninLast12M:",
                        dcc.Input(
                            id="input_NumTradesOpeninLast12M",
                            type="number",
                            placeholder="0",
                            value=0,

                        ),
                        " PercentInstallTrades:",
                        dcc.Input(
                            id="input_PercentInstallTrades",
                            type="number",
                            placeholder="0",
                            value=0,

                        ),
                        " MSinceMostRecentInqexcl7days:",
                        dcc.Input(
                            id="input_MSinceMostRecentInqexcl7days",
                            type="number",
                            placeholder="0",
                            value=0,

                        ),
                        html.Br(),

                        " NumInqLast6M:",
                        dcc.Input(
                            id="input_NumInqLast6M",
                            type="number",
                            placeholder="0",
                            value=0,

                        ),
                        " NumInqLast6Mexcl7days:",
                        dcc.Input(
                            id="input_NumInqLast6Mexcl7days",
                            type="number",
                            placeholder="0",
                            value=0,

                        ),
                        " NetFractionRevolvingBurden:",
                        dcc.Input(
                            id="input_NetFractionRevolvingBurden",
                            type="number",
                            placeholder="0",
                            value=0,

                        ),
                        html.Br(),

                        " NetFractionInstallBurden:",
                        dcc.Input(
                            id="input_NetFractionInstallBurden",
                            type="number",
                            placeholder="0",
                            value=0,

                        ),
                        " NumRevolvingTradesWBalance:",
                        dcc.Input(
                            id="input_NumRevolvingTradesWBalance",
                            type="number",
                            placeholder="0",
                            value=0,

                        ),
                        " NumInstallTradesWBalance:",
                        dcc.Input(
                            id="input_NumInstallTradesWBalance",
                            type="number",
                            placeholder="0",
                            value=0,

                        ),
                        html.Br(),

                        " NumBank/NatlTradesWHighUtilization:",
                        dcc.Input(
                            id="input_NumBank/NatlTradesWHighUtilization",
                            type="number",
                            placeholder="0",
                            value=0,

                        ),
                        " PercentTradesWBalance:",
                        dcc.Input(
                            id="input_PercentTradesWBalance",
                            type="number",
                            placeholder="0",
                            value=0,

                        ),
                    ],
                    className="row, pretty_container",

                ),
                html.Div(
                            [
                                html.H5("Individual Analysis"),
                                html.Br(),
                                html.P("This part is to analyze the contribution of the features. In the upper plot, the force plot shows the most influenced features. In the lower plot, we combined 23 features and divided them into 10 subscale groups. You may find which subscale group contribute more to the good result and which did in the opposite way."),
                                # Local Analyasis
                                html.Div(
                                    [
                                        html.H6("Subscale analysis"),
                                        html.P("Subscale groups:"),
                                        html.P("ExternalRiskEstimate=ExternalRiskEstimate"),
                                        html.P("TradeOpenTime = MSinceOldestTradeOpen+MSinceMostRecentTradeOpen+AverageMInFile"),
                                        html.P("NumSatisfactoryTrades = NumSatisfactoryTrades"),
                                        html.P("TradeFrequency = NumTrades60Ever2DerogPubRec+NumTrades90Ever2DerogPubRec+NumTotalTrades+NumTradesOpeninLast12M"),
                                        html.P("Delinquency = PercentTradesNeverDelq+MSinceMostRecentDelq+MaxDelq2PublicRecLast12M+MaxDelqEver"),
                                        html.P("Installment = PercentInstallTrades+NetFractionInstallBurden+NumInstallTradesWBalance"),
                                        html.P("Inquiry = MSinceMostRecentInqexcl7days+NumInqLast6M+NumInqLast6Mexcl7days"),
                                        html.P("RevolvingBalance = NetFractionRevolvingBurden+NumRevolvingTradesWBalance"),
                                        html.P("Utilization = NumBank2NatlTradesWHighUtilization"),
                                        html.P("TradeWBalance = PercentTradesWBalance"),
                                        dcc.Graph(id="waterfallfig"),
                                        html.H6("Feature contribution"),
                                        html.Iframe(
                                            id="shap_iframe_id1",
                                            srcDoc=None,
                                            style={"scrolling": "no"},
                                            className="iframe_container",
                                        ),
                                    ],
                                    className="twelve columns",
                                ),
                            ],
                            className="row twelve columns pretty_container",
                ),

                html.Div(
                    [
                        html.H5("Related Cases"),
                        html.Br(),
                        # Local Analyasis
                        dash_table.DataTable(
                            id='table',
                            # columns=[{"name": i, "id": i} for i in df.columns],
                            data=[],
                            style_cell_conditional=[
                                {
                                    'if': {'column_id': c},
                                    'textAlign': 'left'
                                } for c in ['Date', 'Region']
                            ],
                            style_table={'overflowX': 'scroll'},
                            style_data_conditional=[
                                {
                                    'if': {
                                        'row_index': 0,  # number | 'odd' | 'even',
                                    },
                                    'backgroundColor': 'rgb(255,199,115)',
                                    'color': 'black'
                                },
                                {
                                    'if': {
                                        'row_index': 1,  # number | 'odd' | 'even',
                                    },
                                    'backgroundColor': 'rgb(255,117,0)',
                                    'color': 'white'
                                },
                                {
                                    'if': {
                                        'filter_query': '{RiskPerformance} = "Good"'
                                    },
                                    'backgroundColor': '#0074D9',
                                    'color': 'white'
                                },
                            ],
                            style_header={
                                # 'backgroundColor': 'rgb(210, 210, 210)',
                                'color': 'white',
                                'fontWeight': 'bold',
                                'backgroundColor': 'grey'
                            },
                            style_data={
                                'height': 'auto',
                                'color': 'black',
                                'backgroundColor': 'white'
                            },
                            style_cell={'fontSize': 14, 'font-family': 'sans-serif', 'margin-right': '8px',
                                        'margin-left': '8px'},

                            style_as_list_view=False,
                        )
                    ],#style={'width':10,},
                    className="row twelve columns pretty_container",
                ),


            ],
                ),

        # Second row
        html.Div(
            [
                # feature importance

                html.Div(
                    id="feature_importance",
                    children=dcc.Loading(
                        className="graph-wrapper",
                        children=[
                            html.H5("Model analysis"),
                            html.Div(
                                [
                                    html.B("Feature Importance in 'Bad' result"),
                                    html.Div(id="bad_feature_importance_fig"),
                                ],
                                className="row four columns",
                            ),

                            html.Div(
                                [
                                    html.B("Feature Importance in 'Good' result"),
                                    html.Div(id="good_feature_importance_fig"),
                                ],
                                className="row four columns",
                            ),
                            html.Div(
                                [
                                    html.B("Feature Importance"),
                                    html.Div(id="feature_importance_fig"),

                                ],
                                className="row four columns",
                            ),
                        ],
                        style={"display": "none"},
                    ),
                    className="pretty_container twelve columns",
                ),
            ],
            className="row flex-display",
        ),

        # Third fow: appendix
        html.Div(
            [
                html.Div([
                    html.H5("Appendix-Feature Explanations"),
                    html.Br(),
                    # Local Analyasis
                    dash_table.DataTable(
                        id='FE',
                        columns=[{"name": i, "id": i} for i in df_0.columns],
                        data=df_0.to_dict('records'),
                        # data = [],
                        style_cell_conditional=[
                            {
                                'if': {'column_id': c},
                                'textAlign': 'left'
                            } for c in ['Date', 'Region']
                        ],
                        style_table={'overflowX': 'scroll'},
                        style_header={
                            # 'backgroundColor': 'rgb(210, 210, 210)',
                            'textAlign': 'center',
                            'color': 'white',
                            'fontWeight': 'bold',
                            'backgroundColor': 'grey'
                        },
                        style_data={
                            'textAlign': 'center',
                            'height': 'auto',
                            'color': 'black',
                            'backgroundColor': 'white'
                        },
                        style_cell={'fontSize': 10, 'font-family': 'sans-serif', 'margin-right': '8px',
                                    'margin-left': '8px'},
                        # style_data_conditional=[
                        #     {
                        #         'if': {'row_index': 'odd'},
                        #         'backgroundColor': 'rgb(220, 220, 220)',
                        #     }
                        # ],
                        style_as_list_view=False,
                    )], ),#style={'width': 1085}
                html.Div([
                    html.B("MaxDelq2PublicRecLast12M Table"),
                    html.Br(),
                    # Local Analyasis
                    dash_table.DataTable(
                        id='MM',
                        columns=[{"name": i, "id": i} for i in df_1.columns],
                        data=df_1.to_dict('records'),
                        # data = [],
                        style_cell_conditional=[
                            {
                                'if': {'column_id': c},
                                'textAlign': 'left'
                            } for c in ['Date', 'Region']
                        ],
                        style_table={'overflowX': 'scroll'},
                        style_header={
                            # 'backgroundColor': 'rgb(210, 210, 210)',
                            'textAlign': 'center',
                            'color': 'white',
                            'fontWeight': 'bold',
                            'backgroundColor': 'grey'
                        },
                        style_data={
                            'textAlign': 'center',
                            'height': 'auto',
                            'color': 'black',
                            'backgroundColor': 'white'
                        },
                        style_cell={'fontSize': 10, 'font-family': 'sans-serif', 'margin-right': '8px',
                                    'margin-left': '8px'},
                        style_as_list_view=False,
                    )], ),#style={'width': 1085}
                html.Div([
                    html.B("MaxDelqEver"),
                    html.Br(),
                    # Local Analyasis
                    dash_table.DataTable(
                        id='MD',
                        columns=[{"name": i, "id": i} for i in df_2.columns],
                        data=df_2.to_dict('records'),
                        # data = [],
                        style_cell_conditional=[
                            {
                                'if': {'column_id': c},
                                'textAlign': 'left'
                            } for c in ['Date', 'Region']
                        ],
                        style_table={'overflowX': 'scroll'},
                        style_header={
                            # 'backgroundColor': 'rgb(210, 210, 210)',
                            'textAlign': 'center',
                            'color': 'white',
                            'fontWeight': 'bold',
                            'backgroundColor': 'grey'
                        },
                        style_data={
                            'textAlign': 'center',
                            'height': 'auto',
                            'color': 'black',
                            'backgroundColor': 'white'
                        },
                        style_cell={'fontSize': 10, 'font-family': 'sans-serif', 'margin-right': '8px',
                                    'margin-left': '8px'},
                        style_as_list_view=False,
                    )], ),#style={'width': 1085}
                html.Div([
                    html.B("Special Value"),
                    html.Br(),
                    # Local Analyasis
                    dash_table.DataTable(
                        id='sv',
                        columns=[{"name": i, "id": i} for i in df_3.columns],
                        data=df_3.to_dict('records'),
                        # data = [],
                        style_cell_conditional=[
                            {
                                'if': {'column_id': c},
                                'textAlign': 'left'
                            } for c in ['Date', 'Region']
                        ],
                        style_table={'overflowX': 'scroll'},
                        style_header={
                            # 'backgroundColor': 'rgb(210, 210, 210)',
                            'textAlign': 'center',
                            'color': 'white',
                            'fontWeight': 'bold',
                            'backgroundColor': 'grey'
                        },
                        style_data={
                            'textAlign': 'center',
                            'height': 'auto',
                            'color': 'black',
                            'backgroundColor': 'white'
                        },
                        style_cell={'fontSize': 10, 'font-family': 'sans-serif', 'margin-right': '8px',
                                    'margin-left': '8px'},

                        style_as_list_view=False,
                    )], ),#style={'width': ''}

            ],
            className="row twelve columns pretty_container",
        )



    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)

# pred_input
@app.callback(
    Output("shap_iframe_id1", "srcDoc"),
    [Input("input_{}".format(_), "value") for _ in FEATURE],
)
def pred_ana(f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23):
    feature_list = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23]
    pred_item = pd.DataFrame([feature_list],columns=FEATURE)
    choosen_instance = pred_item
    shap_v1 = explainer.shap_values(choosen_instance)
    force_plot = shap.force_plot(explainer.expected_value[0], shap_v1[1], choosen_instance)

    shap_html = f"<head>{shap.getjs()}</head><body scroll='no' style='overflow: hidden'>{force_plot.html() }</body>"

    return shap_html

@app.callback(
    Output("waterfallfig", "figure"),
    [Input("input_{}".format(_), "value") for _ in FEATURE],
)
def update_waterfall(f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23):
    # input = [60, 20, 22, -40, -20, -10, -5, 20, 30, -25, 50]
    input_list = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23]

    pred_item = pd.DataFrame([input_list], columns=FEATURE)

    choosen_instance = pred_item

    shap_v1 = explainer.shap_values(choosen_instance)
    sub_shap_val = [round(shap_v1[1][0][0],4), round(shap_v1[1][0][1] + shap_v1[1][0][2] + shap_v1[1][0][3],4), round(shap_v1[1][0][4],4),round(shap_v1[1][0][5] + shap_v1[1][0][6] + shap_v1[1][0][11] + shap_v1[1][0][12],4),round(shap_v1[1][0][7] + shap_v1[1][0][8] + shap_v1[1][0][9] + shap_v1[1][0][10],4),round(shap_v1[1][0][13] + shap_v1[1][0][18] + shap_v1[1][0][20],4),round(shap_v1[1][0][14] + shap_v1[1][0][15] + shap_v1[1][0][16],4), round(shap_v1[1][0][17] + shap_v1[1][0][19],4),round(shap_v1[1][0][21],4), round(shap_v1[1][0][22],4),round(explainer.expected_value[0],4)]
    sum_temp = shap_v1[1][0][0] + shap_v1[1][0][1] + shap_v1[1][0][2] + shap_v1[1][0][3] + shap_v1[1][0][4] + shap_v1[1][0][5] + shap_v1[1][0][6] + shap_v1[1][0][11] + shap_v1[1][0][12] + shap_v1[1][0][7] + shap_v1[1][0][8] + shap_v1[1][0][9] + shap_v1[1][0][10] + shap_v1[1][0][13] + shap_v1[1][0][18] + shap_v1[1][0][20] + shap_v1[1][0][14] + shap_v1[1][0][15] + shap_v1[1][0][16] + shap_v1[1][0][17] + shap_v1[1][0][19] + shap_v1[1][0][21] + shap_v1[1][0][22]
    print(f"sub_shap_val:{len(sub_shap_val)},sum_temp:{sum_temp},explainer:{explainer.expected_value[0]}")

    sub_shap_val.append(round(sum_temp + explainer.expected_value[0],4))
    input = sub_shap_val
    res = model.predict(pred_item)
    #print(f"input:{input}")
    printintlist = []
    for idx in range(0, 12):
        if input[idx] > 0:
            printint = "+" + str(input[idx])
            printintlist.append(printint)
        else:
            printint = str(input[idx])
            printintlist.append(printint)

    figure = go.Figure(go.Waterfall(
        name="Bad:blue,value=0, Good:pink,value=1", orientation="v",
        measure=["relative", "relative", "relative", "relative", "relative", "relative", "relative", "relative",
                 "relative", "relative", "relative","total"],
        x=['ExternalRiskEstimate', 'TradeOpenTime', 'NumSatisfactoryTrades', 'TradeFrequency', 'Delinquency',
           'Installment', 'Inquiry', 'RevolvingBalance', 'Utilization', 'TradeWBalance',"Expected value", 'total'],
        textposition="outside",

        text=[printintlist[0], printintlist[1], printintlist[2], printintlist[3], printintlist[4], printintlist[5],
              printintlist[6], printintlist[7], printintlist[8], printintlist[9], printintlist[10],printintlist[11]],
        y=input,
        connector={"line": {"color": "rgb(0,0,0)"}},
        decreasing={"marker": {"color": "rgb(30, 136, 229)"}},
        increasing={"marker": {"color": "rgb(245, 39, 87)"}},
        totals={"marker": {"color": "grey"}}
    ))

    figure.update_layout(
        title="Prediction result:{}".format(res),
        showlegend = True
    )

    return figure

@app.callback(
    [Output("table", "data"), Output('table', 'columns')],
    [Input("input_{}".format(_), "value") for _ in FEATURE],
)

def createtable(f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23):
    input = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22,f23]
    df = changeform(input)
    return df.to_dict('records'),[{"name": i, "id": i} for i in df.columns]

def calculatemin5id(input):
    b_1 = X
    alldis = []
    min5dis = []
    for i in range(len(b_1)):
        y1 = b_1.ExternalRiskEstimate.values[i]
        y2 = b_1.MSinceOldestTradeOpen.values[i]
        y3 = b_1.MSinceMostRecentTradeOpen.values[i]
        y4 = b_1.AverageMInFile[i]
        y5 = b_1.NumSatisfactoryTrades[i]
        y6=b_1['NumTrades60Ever/DerogPubRec'].values[i]
        y7=b_1['NumTrades90Ever/DerogPubRec'].values[i]
        y8=b_1.PercentTradesNeverDelq.values[i]
        y9=b_1.MSinceMostRecentDelq.values[i]
        y10=b_1['MaxDelq/PublicRecLast12M'].values[i]
        y11=b_1.MaxDelqEver.values[i]
        y12=b_1.NumTotalTrades.values[i]
        y13=b_1.NumTradesOpeninLast12M.values[i]
        y14=b_1.PercentInstallTrades.values[i]
        y15=b_1.MSinceMostRecentInqexcl7days.values[i]
        y16=b_1.NumInqLast6M.values[i]
        y17=b_1.NumInqLast6Mexcl7days.values[i]
        y18=b_1.NetFractionRevolvingBurden.values[i]
        y19=b_1.NetFractionInstallBurden.values[i]
        y20=b_1.NumRevolvingTradesWBalance.values[i]
        y21=b_1.NumInstallTradesWBalance.values[i]
        y22=b_1['NumBank/NatlTradesWHighUtilization'].values[i]
        y23=b_1.PercentTradesWBalance.values[i]
        d=np.sqrt((input[0]-y1)**2+(input[1]-y2)**2+(input[2]-y3)**2+(input[3]-y4)**2+(input[4]-y5)**2+(input[5]-y6)**2+(input[6]-y7)**2+(input[7]-y8)**2+(input[8]-y9)**2+(input[9]-y10)**2+(input[10]-y11)**2+(input[11]-y12)**2+(input[12]-y13)**2+(input[13]-y14)**2+(input[14]-y15)**2+(input[15]-y16)**2+(input[16]-y17)**2+(input[17]-y18)**2+(input[18]-y19)**2+(input[19]-y20)**2+(input[20]-y21)**2+(input[21]-y22)**2+(input[22]-y23)**2)
        # d=np.sqrt((80-y1)**2+(0-y2)**2+(0-y3)**2+(0-y4)**2+(0-y5)**2)
        # print(d)
        alldis.append(d)

    alldis = np.array(alldis)
    # np.argsort(alldis)
    for i in range(0,5):
        min5dis.append(np.argsort(alldis)[i])

    return min5dis


def createTableDf(input):
    min5disid = calculatemin5id(input)
    input.insert(0, 'Null')
    min = features
    df_sample = min.iloc[[min5disid[0], min5disid[1], min5disid[2], min5disid[3], min5disid[4]]].copy()
    df_sample.loc[-1] = input  # adding a row
    df_sample.sort_index(inplace=True)
    df_sample = df_sample.rename(index={-1: 'Current'}).copy()
    predictions = []
    for idx in range(0, 6):
        row = df_sample.iloc[idx]
        instance = row[1:len(row)]
        prediction = model.predict(instance.to_numpy().reshape(1, -1))
        predictions.append(prediction[0])
    df_sample.insert(0, 'Prediction', predictions)

    return df_sample


def changeform(input):
    df = createTableDf(input)
    df.reset_index(inplace=True)
    df = df.rename(columns={'index':'ID'})
    return df

# pred_input
@app.callback(
    Output("shap_iframe_id1", "srcDoc"),
    [Input("input_{}".format(_), "value") for _ in FEATURE],
)

def pred_ana(f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23):
    feature_list = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23]
    pred_item = pd.DataFrame([feature_list],columns=FEATURE)

    choosen_instance = pred_item
    shap_v1 = explainer.shap_values(choosen_instance)
    force_plot = shap.force_plot(explainer.expected_value[0], shap_v1[1], choosen_instance)
    print(f"{explainer.expected_value[0].shape,shap_v1[1].shape}")
    shap_html = f"<head>{shap.getjs()}</head><body scroll='no' style='overflow: hidden'>{force_plot.html()}</body>"

    return shap_html


@app.callback(
    [
        Output("feature_importance_fig", "children"),
        Output("good_feature_importance_fig", "children"),
        Output("bad_feature_importance_fig", "children"),
    ],
    Input("input_{}".format(FEATURE[0]), "value")
)
def update_feature_importance(x1):
    # default feature importance figures
    good_image_filename = 'good_importance_shapX.png'
    good_encoded_image = base64.b64encode(open(good_image_filename, 'rb').read())
    good_importance = html.Img(src='data:image/png;base64,{}'.format(good_encoded_image.decode()))
    bad_image_filename = 'bad_importance_shapX.png'
    bad_encoded_image = base64.b64encode(open(bad_image_filename, 'rb').read())
    bad_importance = html.Img(src='data:image/png;base64,{}'.format(bad_encoded_image.decode()))#,width="33%"height="40%",
    full_image_filename = 'full_importance_shapX.png'
    full_encoded_image = base64.b64encode(open(full_image_filename, 'rb').read())
    full_importance = html.Img(src='data:image/png;base64,{}'.format(full_encoded_image.decode()))
    return full_importance, good_importance, bad_importance



# Main
if __name__ == "__main__":
    app.run_server(debug=False,host = '127.1.1.5')
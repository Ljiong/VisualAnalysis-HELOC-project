from dash import Dash, dcc, html, Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objs as go
#change if you have a different file path
filePath='./data/heloc_dataset_v1.csv'

app = Dash(__name__)
app.layout = html.Div([
    html.H6('plot bar chart based on given input '),
    html.Div([
        "NumTrades90Ever/DerogPubRec: ",
        dcc.Input(id='my-input1', value=10, type='number'),
        "MSinceMostRecentTradeOpen: ",
        dcc.Input(id='my-input2', value=10, type='number'),
        "PercentInstallTrades: ",
        dcc.Input(id='my-input3', value=10, type='number'),
        "NumInqLast6Mexcl7days: ",
        dcc.Input(id='my-input4', value=10, type='number')
    ]),
    html.Br(),
    html.Div(
        dcc.Graph(id='ouput_barchart'),
    )

])


@app.callback(
    Output('ouput_barchart','figure'),
    Input(component_id='my-input1', component_property='value'),
    Input(component_id='my-input2', component_property='value'),
    Input(component_id='my-input3', component_property='value'),
    Input(component_id='my-input4', component_property='value'),
)
def update_output_div(input_value1,input_value2,input_value3,input_value4):
    index_list,input_data = cal_index(input_value1, input_value2, input_value3, input_value4)
    return plot_bar(index_list,input_data)


def loadDataFromCSV(filePath, report=False):
    raw = pd.read_csv(filePath)
    x = raw.values[:,1:]
    y = raw.values[:, 0]
    if report:
        print('x:', x.shape, 'y:', y.shape)
    return x, y

#selected features for input
important_list=[7,3,14,17]

def cal_index(input1,input2,input3,input4,k=3):
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
    input_data[important_list[0]],input_data[important_list[1]],input_data[important_list[2]],input_data[important_list[3]] = input1,input2,input3,input4
    for i in range(len(x)):
        dist[i] = np.linalg.norm(input_data - x_normed[i])
    return np.argpartition(dist, k)[:k], input_data

def plot_bar(index_list,input_data):
    '''
    plot the grouped bar chart by given index
    also show the input data on figure
    '''
    y=[]
    for i in range(3):
        y.append(pd.read_csv(filePath).iloc[index_list[i]].values)
        if y[i][0] == 'Bad':
            y[i][0] = 0
        else:
            y[i][0]= 200

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


if __name__ == '__main__':
    app.run_server(debug=True)
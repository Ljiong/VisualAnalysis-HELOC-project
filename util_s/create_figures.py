import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import plotly.graph_objs as go
import colorlover as cl


def feature_imp(y_importance,features,labelDimension):
    c = y_importance

    features_1 = features.drop(labelDimension, axis = 1)
    x_importances =list(features_1)
    index_1 = [1]
    new = pd.DataFrame(data=c,columns=index_1,index=x_importances)

    new.sort_values(by=index_1,axis=0,ascending=True,inplace=True)

    x_importance=new.index.values
    y = np.sort(y_importance)

    import plotly.graph_objects as go

    fig = go.Figure(
            data=[go.Bar(
                x=y,
                y=x_importance,
                orientation='h')],
            layout=go.Layout(
                plot_bgcolor="#282b38",
                paper_bgcolor="#282b38",
                title="Feature weights",
                font={"color": "#a5b1cd"}
        )
            )

    # fig.show()
    return fig

def CM(model,features,labelDimension):
    y_true = np.array(features[labelDimension])
    X_train = features.drop(labelDimension, axis=1)
    y_prediction = model.predict(X_train.to_numpy())
    matrix = confusion_matrix(y_true, y_prediction)

    tn, fp, fn, tp = matrix.ravel()

    values = [tp, fn, fp, tn]
    label_text = ["True Positive", "False Negative", "False Positive", "True Negative"]
    labels = ["TP", "FN", "FP", "TN"]
    blue = cl.flipper()["seq"]["9"]["Blues"]
    red = cl.flipper()["seq"]["9"]["Reds"]
    colors = ["#13c6e9", blue[1], "#ff916d", "#ff744c"]

    trace0 = go.Pie(
        labels=label_text,
        values=values,
        hoverinfo="label+value+percent",
        textinfo="text+value",
        text=labels,
        sort=False,
        marker=dict(colors=colors),
        insidetextfont={"color": "white"},
        rotation=90,
    )

    layout = go.Layout(
        title="Confusion Matrix",
        margin=dict(l=50, r=50, t=100, b=10),
        legend=dict(bgcolor="#282b38", font={"color": "#a5b1cd"}, orientation="h"),
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        # paper_bgcolor="#FFFFFF",
        font={"color": "#a5b1cd"},
    )

    data = [trace0]
    fig = go.Figure(data=data, layout=layout)

    return fig





def distribution_fig(column,Bad_F,Good_F):
    import plotly.express as px
    df = Bad_F
    df_G = Good_F

    trace1 = go.Histogram(x=df[column],opacity=0.75,marker_color='#EB89B5',name='BAD')
    data = [trace1]
    trace2 = go.Histogram(x=df_G[column],opacity=0.75,name='GOOD')
    data_1 = [trace2]
    layout = go.Layout(
    title="Histogram of "+str(column),
    plot_bgcolor="#282b38",
    paper_bgcolor="#282b38",
    # paper_bgcolor = "#FFFFFF",
    font={"color": "#a5b1cd"},
    xaxis_title_text='Value',
    yaxis_title_text='Count'
    )

    fig = go.Figure(layout=layout)
    fig.add_traces(data=data_1)
    fig.add_traces(data=data)
    return fig


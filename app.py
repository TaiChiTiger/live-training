import sys
sys.path.append('./')
from utils.generate_datasets import SyntheticData
from utils.figures import Visualization
from models.linear_regression import LinearRegressionFromScrach
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from dash import html, dcc, callback, Input, Output, Dash
import dash_bootstrap_components as dbc

sample_size = 200
n_features = 1
noise = 10
seed = 42
sd = SyntheticData()
x, y = sd.lin_sep_reg(sample_size, noise, outliers_ratio=0, seed=seed)

learning_rate = 0.01
epochs = 200
X = np.c_[np.ones(len(x)), x]
lrfs = LinearRegressionFromScrach(learning_rate, epochs, seed)
weight_history, loss_history = lrfs.gradient_descent(X, y)
loss_func = lambda W: lrfs.mean_squared_loss(X, W, y)

lr = LinearRegression()
lr.fit(X, y)
best_weights = np.array([lr.intercept_, lr.coef_[1]])
best_error = mean_squared_error(y, lr.predict(X))

figure = Visualization.plot_animation_reg(x, y, epochs, weight_history, loss_history, 
                                    best_weights, best_error, loss_func)

noisen_1 = {i: '{}'.format(float(i)) for i in range(10, 100, 20)}
noisen = {100: '100.0'}

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

data_card = dbc.Card(
    style={'margin-left': '10px', 'font-size': '14px', 'width': '60rem'},
    children=[
        # dbc.CardHeader("数据集", style={'font-size': '18px'}),
        dbc.CardBody([
            html.Label('选择大小：'),
            dcc.Slider(
                id='datasize-slider',
                min=100, max=500,
                marks={i: '{}'.format(i) for i in range(100, 600, 100)},
                value=200,
                step=None,
                tooltip={"placement": "bottom", "always_visible": True}),
            html.Label('选择噪声水平：'),
            
            dcc.Slider(
                id='noise-slider',
                min=10.0, max=100,
                marks={**noisen_1, **noisen},
                value=10.0,
                step=None,
                tooltip={"placement": "bottom", "always_visible": True})]
            )
    ]
)

param_card = dbc.Card(
    style={'margin-left': '10px', 'font-size': '14px', 'width': '60rem'},
    children=[
        dbc.CardBody([
            html.Label('选择优化方法：'),
            dcc.RadioItems(
                id='opt-radio', 
                options=[{"label": i, "value": i} 
                        for i in ["随机搜索", "坐标下降", 
                        "梯度下降", "牛顿方法", "BFGS"]], 
                value='梯度下降',
                labelStyle={
                    'display': 'inline-block',
                    'padding': '1px',
                    'margin-left': '8px',
                    'margin-right': '50px'},
                style={
                    'display': 'inline-block',
                    'margin-left': '10px',
                    'margin-top': '8px',
                    'display': 'flex'}
            ),
            html.Label('学习率：'),
            dcc.Slider(
                id="lr-slider",
                min=0.01, max=1,
                marks={
                    0.01: '0.01',
                    0.1: '0.1', 
                    0.5: '0.5', 
                    1: '1.0'},
                value=0.1,
                step=None,
                tooltip={"placement": "bottom", "always_visible": True}),
            html.Label('迭代次数：'),
            dcc.Slider(
                id="epochs-slider",
                min=100, max=300,
                marks={i: '{}'.format(i) for i in range(100, 310, 100)},
                value=200,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ]
    )
]),

app.layout = html.Div(
    id='app-container',
    children=[
        html.Div(
            id='title',
            children=[
                html.Br(),
                html.H1("线性回归", className='text-center text-primary, mb-4'),
                html.Hr(),
            ]
        ),
        dbc.Row([
            dbc.Tabs([
                dbc.Tab(tab_id='data-tab', children=data_card, label="数据集", 
                        activeTabClassName="fw-bold fst-italic"),
                dbc.Tab(tab_id='param-tab', children=param_card, label="参数", 
                        activeTabClassName="fw-bold fst-italic")],
                active_tab="param-tab",),
            ],
            justify='center'
        ),
        html.Br(),
        html.Div(
            id='graph',
            children=[
                dbc.Row([
                    dcc.Graph(id='animation-graph', figure=figure, 
                            config= {'displaylogo': False})], 
                    justify='center')]
        ),
        html.Br(),
        html.Br(),
        
    ]
)

@callback(
    Output("animation-graph", "figure"),
    [Input("datasize-slider", "value"),
    Input("noise-slider", "value"),
    Input("lr-slider", "value"),
    Input("epochs-slider", "value")],
)
def update_datasize(datasize, noise, learning_rate, epochs):
    x, y = sd.lin_sep_reg(datasize, noise, outliers_ratio=0, seed=seed)
    # 现场训练
    X = np.c_[np.ones(len(x)), x]
    lr = LinearRegressionFromScrach(learning_rate, epochs, seed)
    weight_history, loss_history = lr.gradient_descent(X, y)
    loss_func = lambda W: lrfs.mean_squared_loss(X, W, y)

    # 目标模型
    lr = LinearRegression()
    lr.fit(X, y)
    best_weights = np.array([lr.intercept_, lr.coef_[1]])
    best_error = mean_squared_error(y, lr.predict(X))

    figure = Visualization.plot_animation_reg(x, y, epochs, weight_history, loss_history, 
                                        best_weights, best_error, loss_func)

    return figure

if __name__ == '__main__':
    app.run_server(debug=False, port=8518)

from dash import Dash, html, dcc, callback_context
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np


import datetime 
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


#lib of financial
import yfinance as yf
from pandas_ta import bbands



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = Dash(__name__, external_stylesheets=external_stylesheets)


# see https://plotly.com/python/px-arguments/ for more options

# Apple stock
df_apple = yf.download('AAPL', 
                        start='2021-01-01',
                        end='2021-12-31',
                        progress=False,
                        auto_adjust=True)

fig = go.Figure(data=[go.Candlestick(x=df_apple.index,
                open=df_apple['Open'],
                high=df_apple['High'],
                low=df_apple['Low'],
                close=df_apple['Close'])])

list_stock = ['AAPL', 'GOOG', 'META']

#ADX candicate defauth = 14. Reason: Lower settings will make the average directional index respond more quickly to price movement but tend to generate more false signals. 
#Higher settings will minimize false signals but make the average directional index a more lagging indicator.

def ema(arr, periods=14, weight=1, init=None):
    leading_na = np.where(~np.isnan(arr))[0][0]
    arr = arr[leading_na:]
    alpha = weight / (periods + (weight-1))
    alpha_rev = 1 - alpha
    n = arr.shape[0]
    pows = alpha_rev**(np.arange(n+1))
    out1 = np.array([])
    if 0 in pows:
        out1 = ema(arr[:int(len(arr)/2)], periods)
        arr = arr[int(len(arr)/2) - 1:]
        init = out1[-1]
        n = arr.shape[0]
        pows = alpha_rev**(np.arange(n+1))
    scale_arr = 1/pows[:-1]
    if init:
        offset = init * pows[1:]
    else:
        offset = arr[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)
    mult = arr*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    out = out[1:] if len(out1) > 0 else out
    out = np.concatenate([out1, out])
    out[:periods] = np.nan
    out = np.concatenate(([np.nan]*leading_na, out))
    return out


def atr(highs, lows, closes, periods=14, ema_weight=1):
    hi = np.array(highs)
    lo = np.array(lows)
    c = np.array(closes)
    tr = np.vstack([np.abs(hi[1:]-c[:-1]),
                    np.abs(lo[1:]-c[:-1]),
                    (hi-lo)[1:]]).max(axis=0)
    atr = ema(tr, periods=periods, weight=ema_weight)
    atr = np.concatenate([[np.nan], atr])
    return atr


def adx(highs, lows, closes, periods=14):
    highs = np.array(highs)
    lows = np.array(lows)
    closes = np.array(closes)
    up = highs[1:] - highs[:-1]
    down = lows[:-1] - lows[1:]
    up_idx = up > down
    down_idx = down > up
    updm = np.zeros(len(up))
    updm[up_idx] = up[up_idx]
    updm[updm < 0] = 0
    downdm = np.zeros(len(down))
    downdm[down_idx] = down[down_idx]
    downdm[downdm < 0] = 0
    _atr = atr(highs, lows, closes, periods)[1:]
    updi = 100 * ema(updm, periods) / _atr
    downdi = 100 * ema(downdm, periods) / _atr
    zeros = (updi + downdi == 0)
    downdi[zeros] = .0000001
    adx = 100 * np.abs(updi - downdi) / (updi + downdi)
    adx = ema(np.concatenate([[np.nan], adx]), periods)
    return adx

df_apple['ADX_14'] = adx(df_apple['High'], 
                            df_apple['Low'],
                            df_apple['Close'])


#SMA and EMA
#fig.add_sma([10,20],width=2,color=['green','lightgreen'],legendgroup=True)
#fig.add_ema(periods=20, color='green')

#RSI
def rsi(df, periods = 14, ema = True):
    """
    Returns a pd.Series with the relative strength index.
    """
    close_delta = df['Close'].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    if ema == True:
	    # Use exponential moving average
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window = periods, adjust=False).mean()
        ma_down = down.rolling(window = periods, adjust=False).mean()
        
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi
df_apple['RSI_14'] = rsi(df_apple)

#MACD
#fig.add_macd()

min_date = '2021-01-01'
max_date = '2021-12-31'
#fig.iplot(asFigure=True)
fig.update_xaxes(range=[min_date, max_date])
fig.update_yaxes(tickprefix='$')

app.layout = html.Div([ 
    dbc.Container([ 
        html.H1("Technical Analysis", style={'align':'center'}),
        dbc.Row([ 
            dbc.Col([ 
                dcc.Dropdown(list_stock,
                            id='stocks-dropdown',
                            value='AAPL',
                            className="disabled",
                            style={'margin-right': '20px'}, 
                            searchable=False
                         ),
                html.Div([
                                     html.Button('1W', id='1W-button',
                                                 n_clicks=0, className='btn-secondary'),
                                     html.Button('1M', id='1M-button',
                                                 n_clicks=0, className='btn-secondary'),
                                     html.Button('3M', id='3M-button',
                                                 n_clicks=0, className='btn-secondary'),
                                     html.Button('6M', id='6M-button',
                                                 n_clicks=0, className='btn-secondary'),
                                     html.Button('1Y', id='1Y-button',
                                                 n_clicks=0, className='btn-secondary'),

                                     ]),
            ],width=6, md=0,lg=0),
            html.Br(),
        dbc.Col([
            
            dcc.Checklist(
                                ['Moving Average',
                                 'Exponential Rolling Mean',
                                 'Bollinger Bands',
                                 'ADX_14',
                                 'RSI_14'],
                                inputStyle={'margin-left': '15px',
                                            'margin-right': '5px'},
                                id='complements-checklist',
                                style={'margin-top': '20px'})

        ],width=6, md=0,lg=0)
        ]),
        html.Br(),
        dcc.Graph(id='price-chart', figure=fig,style={'boder':'None','width':'100%','align':'center'}),
    ], fluid=True)
])

# A function that will change the data being displayed in the candlestick chart in accordance to the stock selected in the dropdown.
# Also, we'll enable the user to adjust the x-axis length and to insert some indicators.
@ app.callback(
    Output('price-chart', 'figure'),
    Input('stocks-dropdown', 'value'),
    Input('complements-checklist', 'value'),
    Input('1W-button', 'n_clicks'),
    Input('1M-button', 'n_clicks'),
    Input('3M-button', 'n_clicks'),
    Input('6M-button', 'n_clicks'),
    Input('1Y-button', 'n_clicks'),
)
def change_price_chart(stock, checklist_values, button_1w, button_1m, button_3m, button_6m, button_1y): #stock
    # Retrieving the stock's data.
    # APPL, META, GOOG stock
    df = yf.download(f'{stock}', 
                        start='2021-01-01',
                        end='2021-12-31',
                        progress=False,
                        auto_adjust=True)
    df['Moving Average'] = df['Close'].rolling(window=20).mean()
    df['Exponential Rolling Mean'] = df['Close'].ewm(
        span=9, adjust=False).mean()
    # df['ADX_14'] = talib.ADX(df['High'], 
    #                         df['Low'],
    #                         df['Close'])
    df['ADX_14'] = adx(df['High'], 
                            df['Low'],
                            df['Close'])
    # df['RSI_14'] = talib.RSI(df['Close'])
    df['RSI_14'] = rsi(df)

    colors = {'Moving Average': '#6fa8dc',
              'Exponential Rolling Mean': '#03396c', 'Bollinger Bands Low': 'darkorange',
              'Bollinger Bands AVG': 'brown',
              'Bollinger Bands High': 'darkorange',
              'ADX_14': 'royalblue',
              'RSI_14': 'firebrick'}
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
    df_bbands = bbands(df['Close'], length=20, std=2)

    if checklist_values != None:
        for metric in checklist_values:

            # Adding the Bollinger Bands' typical three lines.
            if metric == 'Bollinger Bands':
                fig.add_trace(go.Scatter(
                    x=df.index, y=df_bbands.iloc[:, 0],
                    mode='lines', name=metric, line={'color': colors['Bollinger Bands Low'], 'width': 1}))

                fig.add_trace(go.Scatter(
                    x=df.index, y=df_bbands.iloc[:, 1],
                    mode='lines', name=metric, line={'color': colors['Bollinger Bands AVG'], 'width': 1}))

                fig.add_trace(go.Scatter(
                    x=df.index, y=df_bbands.iloc[:, 2],
                    mode='lines', name=metric, line={'color': colors['Bollinger Bands High'], 'width': 1}))
            elif metric == 'ADX_14':
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['ADX_14'], 
                    mode='lines', name=metric, line={'color': colors['ADX_14'], 'width': 1})) 
            elif metric == 'RSI_14':
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['RSI_14'], 
                    mode='lines', name=metric, line={'color': colors['RSI_14'], 'width': 1})) 
            # Plotting any of the other metrics remained, if they are chosen.
            else:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[metric], mode='lines', name=metric, line={'color': colors[metric], 'width': 1}))
    fig.update_layout(
        paper_bgcolor='white',
        font_color='grey',
        height=500,
        width=1400,
        margin=dict(l=10, r=10, b=5, t=5),
        autosize=False,
        showlegend=False
    )
    # Defining the chart's x-axis length according to the button clicked.
    # To do this, we'll alter the 'min_date' and 'max_date' global variables that were defined in the beginning of the script.
    global min_date, max_date
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if '1W-button' in changed_id:
        min_date = df.iloc[-1].name - datetime.timedelta(7)
        max_date = df.iloc[-1].name
    elif '1M-button' in changed_id:
        min_date = df.iloc[-1].name - datetime.timedelta(30)
        max_date = df.iloc[-1].name
    elif '3M-button' in changed_id:
        min_date = df.iloc[-1].name - datetime.timedelta(90)
        max_date = df.iloc[-1].name
    elif '6M-button' in changed_id:
        min_date = df.iloc[-1].name - datetime.timedelta(180)
        max_date = df.iloc[-1].name
    elif '1Y-button' in changed_id:
        min_date = df.iloc[-1].name - datetime.timedelta(365)
        max_date = df.iloc[-1].name
    else:
        min_date = min_date
        max_date = max_date
        fig.update_xaxes(range=[min_date, max_date])
        fig.update_yaxes(tickprefix='$')
        return fig
    # Updating the x-axis length.
    fig.update_xaxes(range=[min_date, max_date])
    fig.update_yaxes(tickprefix='$')
    # fig.update_yaxes(color='white')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)

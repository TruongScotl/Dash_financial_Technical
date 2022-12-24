from dash import Dash, html, dcc, dash_table, callback_context
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go



import cufflinks as cf
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
                        start='2018-01-01',
                        end='2018-12-31',
                        progress=False,
                        auto_adjust=True)
# fig = cf.QuantFig(df_apple, title="Apple's Stock Price",
#                 legend='Top', name='APPL')
fig = go.Figure(data=[go.Candlestick(x=df_apple.index,
                open=df_apple['Open'],
                high=df_apple['High'],
                low=df_apple['Low'],
                close=df_apple['Close'])])

#fig.add_volume()

#boilinger_bands
#fig.add_bollinger_bands(periods=20,boll_std=2,colors=['magenta','grey'],fill=True)

#SMA and EMA
#fig.add_sma([10,20],width=2,color=['green','lightgreen'],legendgroup=True)
#fig.add_ema(periods=20, color='green')

#RSI
#fig.add_rsi(periods=20,color='java')


#MACD
#fig.add_macd()

min_date = '2018-01-01'
max_date = '2018-12-31'
#fig.iplot(asFigure=True)
#fig.update_xaxes(range=[min_date, max_date])
#fig.update_yaxes(tickprefix='$')

app.layout = html.Div(children=[
    html.H1(children='Technical Analysis'),
    html.Div([

        html.Div(children='''
            Dash: A web application framework for Python.
        '''),

        # dcc.Graph(
        #     id='graph1',
        #     figure=fig
        # ),  
    ]),
    # New Div for all elements in the new 'row' of the page
    html.Div([
        dbc.Row([
                    dbc.Col([
                        dcc.Loading(
                            [dcc.Graph(id='price-chart', figure=fig)], id='loading-price-chart',
                            type='dot', color='#1F51FF'),

                    ]),
                    dbc.Row([
                        dbc.Col([

                            # This Div contains the time span buttons for adjustment of the x-axis' length.
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

                                     ], style={'padding': '15px', 'margin-left': '35px'})
                        ], width=4),

                        # The following checklist mentions the indicators available for use in the dashboard.
                         dbc.Col([
                            dcc.Checklist(
                                ['Rolling Mean',
                                 'Exponential Rolling Mean',
                                 'Bollinger Bands'],
                                inputStyle={'margin-left': '15px',
                                            'margin-right': '5px'},
                                id='complements-checklist',
                                style={'margin-top': '20px'})
                        ], width=8)
                    ]),
                ]),
    ]),

        
])

# A function that will change the data being displayed in the candlestick chart in accordance to the stock selected in the dropdown.
# Also, we'll enable the user to adjust the x-axis length and to insert some indicators.
@ app.callback(
    Output('price-chart', 'figure'),
#    Input('stocks-dropdown', 'value'),
    Input('complements-checklist', 'value'),
    Input('1W-button', 'n_clicks'),
    Input('1M-button', 'n_clicks'),
    Input('3M-button', 'n_clicks'),
    Input('6M-button', 'n_clicks'),
    Input('1Y-button', 'n_clicks'),
)
def change_price_chart(checklist_values, button_1w, button_1m, button_3m, button_6m, button_1y): #stock
    # Retrieving the stock's data.
    df = yf.download('AAPL', 
                        start='2018-01-01',
                        end='2018-12-31',
                        progress=False,
                        auto_adjust=True)
    df['Rolling Mean'] = df['Close'].rolling(window=9).mean()
    df['Exponential Rolling Mean'] = df['Close'].ewm(
        span=9, adjust=False).mean()
    colors = {'Rolling Mean': '#6fa8dc',
              'Exponential Rolling Mean': '#03396c', 'Bollinger Bands Low': 'darkorange',
              'Bollinger Bands AVG': 'brown',
              'Bollinger Bands High': 'darkorange'}
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

            # Plotting any of the other metrics remained, if they are chosen.
            else:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[metric], mode='lines', name=metric, line={'color': colors[metric], 'width': 1}))
    fig.update_layout(
        paper_bgcolor='black',
        font_color='grey',
        height=500,
        width=1000,
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

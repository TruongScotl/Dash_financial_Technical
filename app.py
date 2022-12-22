from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

import cufflinks as cf
#lib of financial
import yfinance as yf



app = Dash(__name__)

# see https://plotly.com/python/px-arguments/ for more options


df_apple = yf.download('AAPL', 
                        start='2018-01-01',
                        end='2018-12-31',
                        progress=False,
                        auto_adjust=True)
fig = cf.QuantFig(df_apple, title="Apple's Stock Price",
                legend='Top', name='APPL')
fig.add_volume()
fig.add_sma(periods=20, column='Close', color='red')
fig.add_ema(periods=20, color='green')

fig = fig.iplot(asFigure=True)
app.layout = html.Div(children=[
    html.H1(children='Technical Analysis'),

    html.Div(children='''
        CanddleStick Apple
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
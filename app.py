from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

import cufflinks as cf
#lib of financial
import yfinance as yf


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = Dash(__name__, external_stylesheets=external_stylesheets)


# see https://plotly.com/python/px-arguments/ for more options

# Apple stock
df_apple = yf.download('AAPL', 
                        start='2018-01-01',
                        end='2018-12-31',
                        progress=False,
                        auto_adjust=True)
fig = cf.QuantFig(df_apple, title="Apple's Stock Price",
                legend='Top', name='APPL')
fig.add_volume()

#boilinger_bands
fig.add_bollinger_bands(periods=20,boll_std=2,colors=['magenta','grey'],fill=True)

#SMA and EMA
fig.add_sma([10,20],width=2,color=['green','lightgreen'],legendgroup=True)
fig.add_ema(periods=20, color='green')

#RSI
fig.add_rsi(periods=20,color='java')

#Trendline
date0, date1 = '2018-06-01', '2018-7-30'
fig.add_trendline(date0, date1, on='close', text='Trendline')

#MACD
fig.add_macd()

fig = fig.iplot(asFigure=True)

app.layout = html.Div(children=[
    html.H1(children='Technical Analysis'),

    html.Div(children='''
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])


if __name__ == '__main__':
    app.run_server(debug=True)
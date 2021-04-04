import get_data
from PortfolioOptimization import portfolio_optimization


def plot_stock_data(ticker="AAPL"):
    import dash
    import dash_core_components as dcc
    import dash_html_components as html
    import plotly.graph_objects as go
    from plotly.offline import plot
    from plotly.subplots import make_subplots
    import pandas as pd
    from datetime import datetime
    from dash.dependencies import Input, Output
    import json
    import plotly.express as px

    ticker_list, ticker_names = get_data.get_tickers_list_SP500()

    ticker_list = [{"label": name, "value": ticker} for name, ticker in zip(ticker_names, ticker_list)]
    df = pd.read_csv(f"tickers_data/{ticker}.csv")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Candlestick(
        x=df["datetime"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    ),
        secondary_y=False
    )
    fig.add_trace(go.Bar(x=df["datetime"], y=df["volume"], opacity=1), secondary_y=False)
    fig.layout.yaxis2.showgrid = False
    fig.update_layout(title=f"{ticker}", clickmode="event+select")

    optimal_weights = portfolio_optimization()
    #optimal_weights = pd.DataFrame({"names": optimal_weights.keys(), "values": optimal_weights.values()})
    #print(optimal_weights.head())
    portfolio_fig = make_subplots()
    portfolio_fig.add_trace(go.Pie(
        labels=list(optimal_weights.keys()),
        values=list(optimal_weights.values())
    ))
    # DASH APP
    app = dash.Dash("Stock Data")
    '''
    "data": [{"x": df["datetime"], "open": df["open"], "high": df["high"], "low": df["low"],
                          "close": df["close"],
                          "name":ticker,
                          "type": "candlestick"
                          },
                         {"x": df["datetime"], "y": df["volume"], "type": "bar"}
                         ]
    '''
    app.layout = html.Div([
        dcc.Graph(
            id="basic-interactions",
            figure=fig
        ),
        html.Div(children=[
            dcc.Dropdown(
                id="input-stock",
                options=ticker_list,
                value='AAPL'
            )
        ]),
        dcc.Graph(
            id="portfolio-weights",
            figure=portfolio_fig
        )
    ])

    @app.callback(
        Output('basic-interactions', 'figure'),
        Input('input-stock', 'value'))
    def change_plot(ticker):
        df = pd.read_csv(f"tickers_data/{ticker}.csv")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Candlestick(
            x=df["datetime"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"]
        ),
            secondary_y=True
        )
        fig.add_trace(go.Bar(x=df["datetime"], y=df["volume"]), secondary_y=False)
        fig.layout.yaxis2.showgrid = False
        fig.update_layout(title=f"{ticker}", clickmode="event+select")
        return fig

    app.run_server(debug=True)

if __name__ == "__main__":
    plot_stock_data()
    #get_data.get_tickers_data()
# visualize_return_variance(mu, cov, weights.values())

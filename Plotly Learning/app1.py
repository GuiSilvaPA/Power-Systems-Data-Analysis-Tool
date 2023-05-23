# Import Packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
from base64 import b64encode
import pandas as pd
import plotly.express as px
import io
from dash_save import *


def main():

    port = 9050

    # Incorporate data
    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')
    print(df)

    # Initialize the APP
    app = Dash(__name__)

    buffer     = io.StringIO()
    html_bytes = buffer.getvalue().encode()
    encoded    = b64encode(html_bytes).decode()

    app.layout = html.Div([html.Button('save static', id='save', n_clicks=0),
                           html.Span('', id='saved'),

                           html.Div(children='My First App with Data, Graph, and Controls'),
                           html.Hr(),
                           dcc.RadioItems(options=['pop', 'lifeExp', 'gdpPercap'], value='lifeExp', id='controls-and-radio-item'),
                           dcc.Graph(figure={}, id='controls-and-graph'),
                           dash_table.DataTable(data=df.to_dict('records'), page_size=10),
                           dcc.Graph(figure=px.histogram(df, x='continent', y='lifeExp', histfunc='avg')),
                           html.A(html.Button("Download as HTML"),  id="download", href="data:text/html;base64," + encoded, download="plotly_graph.html")]) 

    # Add controls to build the interaction
    @callback(
        Output(component_id='controls-and-graph', component_property='figure'),
        Input(component_id='controls-and-radio-item', component_property='value')
    )

    def update_graph(col_chosen):
        fig = px.histogram(df, x='continent', y=col_chosen, histfunc='avg')
        return fig
    
    @app.callback(
        Output('saved', 'children'),
        Input('save', 'n_clicks'),
    )
    def save_result(n_clicks):
        if n_clicks == 0:
            return 'not saved'
        else:
            make_static(f'http://127.0.0.1:{port}/')
            return 'saved'

    app.run_server(debug=False, port=port)

if __name__ == '__main__':

    main()
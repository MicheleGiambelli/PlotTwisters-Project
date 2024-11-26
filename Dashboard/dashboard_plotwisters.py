import pandas as pd
import numpy as np
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import scipy.stats as stats
from plotwisters import plotwisters_model  # Importa il tuo modello


models = {
    "GRU": "GRU-best-model.keras",
    "DistilBERT": ""  
}

metrics = {
    "GRU": {
        "Accuracy": "88.7%",
        "Precision": "86%",
        "Recall": "88%",
        "F1-Score": "86%",
        "Precision (macro)": "72%",
        "Recall (macro)": "57%",
        "F1-Score (macro)": "62%"
    },
    "DistilBERT": {
        "Accuracy": "97.5%",
        "Precision": "93.2%",
        "Recall": "92.3%",
        "F1-Score": "92.8%"
    }
}

# Load the combined dataset from CSV
combined_df = pd.read_csv('combined_dataset.csv')
train_df = pd.read_csv("train_df.csv")
val_df = pd.read_csv("val_df.csv")
test_df = pd.read_csv("test_df.csv")

# Create a Dash application
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

navbar = dbc.Navbar(id='navbar', children=[
    dbc.Row([
        dbc.Col(
            dbc.NavbarBrand("PLOTWISTER   NER-TAG   CLASSIFICATION   PROJECT", 
                            style={'color': 'white', 'fontSize': '30px', 'fontFamily': 'Arial, sans-serif',
                                   'textAlign': 'center',
                                   'width': '100%',})
        )
    ], align="center")
], color='#090059')

body_app = dbc.Container([
    html.Br(),
    html.Br(),
    dbc.Row([
        dbc.Col([dbc.Card(id='card_num1', style={'height': '400px'})]),
        dbc.Col([dbc.Card(id='card_num2', style={'height': '150px'})]),
        dbc.Col([dbc.Card(id='card_num3', style={'height': '150px'})])
    ]),
    html.Br(),
    html.Br(),
    dbc.Row([
        dbc.Col([dbc.Card(id='card_num4', style={'height': '350px'})]),
        dbc.Col([dbc.Card(id='card_num6', style={'height': '450px'})])
    ]),
    html.Br(),
    html.Br(),
    dbc.Row([
        #dbc.Col([dbc.Card(id='card_num7', style={'height': '450px'})]),
        dbc.Col([dbc.Card(id='card_num8', style={'height': '450px'})])
    ]),
    
    # INPUT BOX

     html.Br(),
    html.Br(),
    #TO SELECT THE MODEL
    dbc.Row([
        dbc.Col(html.H4('SELECT A NER MODEL'), width={'size': 10, 'offset': 2}),
    ]),

    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='model-selector',
                options=[{'label': key, 'value': key} for key in models.keys()],
                value='',
                style={'width': '80%'}
            ),
            width={'size': 10, 'offset': 2}
        ),
    ]),

    html.Br(),
    #MODEL METRICS
    dbc.Row([
        dbc.Col(html.H4('Model Metrics'), width={'size': 10, 'offset': 2}),
    ]),

    dbc.Row([
        dbc.Col(
            html.Div(
                id='model-metrics',
                style={
                    'border': '1px solid #d6d6d6',
                    'borderRadius': '5px',
                    'padding': '10px',
                    'backgroundColor': '#f9f9f9',
                    'width': '30%',  # Riduci la larghezza
                    'marginLeft': '20%',  # Allineamento a sinistra
                    'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)'
                }
            ),
            width={'size': 10, 'offset': 0}  # Allinea il contenuto con il resto
        ),
    ]),

    html.Br(),
    #TO ENTER THE SENTENCE
    dbc.Row([
        dbc.Col(html.H4('Enter a sentence to process with NER'), width={'size': 10, 'offset': 2}),
    ]),

    dbc.Row([
        dbc.Col(dcc.Textarea(id='sentence-input', value='', style={'height': '100px', 'width': '80%'}),
                width={'size': 10, 'offset': 2}),
    ]),

    html.Br(),
    html.Br(),
    #PROCESS SENTENCE BOTTON
    dbc.Row([
        dbc.Col(dbc.Button(id='process-text', children='Process Sentence', color='dark', n_clicks=0),
                width={'size': 10, 'offset': 2}),
    ]),

    html.Br(),
    html.Br(),

    dbc.Row([
        dbc.Col(
            dcc.Loading(
                id='load-output',
                type='default',
                children=html.Div(id='processed-output', style={'textAlign': 'center', 'color': 'black'})
            )
        ),
    ]),


    html.Br(),
    html.Br(),
    dbc.Row([
        dbc.Col(
            html.Button(id='update-button', n_clicks=0, children='Update Dashboard', className='btn btn-primary', style={'width': '100%'})
        )
    ]),

], style={'backgroundColor': '#f7f7f7'}, fluid=True)

@app.callback([
    Output('card_num1', 'children'),
    Output('card_num2', 'children'),
    Output('card_num3', 'children'),
    Output('card_num4', 'children'),
    Output('card_num6', 'children'),
    #Output('card_num7', 'children'),
    Output('card_num8', 'children')
], [Input('update-button', 'n_clicks')])
def update_cards(n_clicks):
    # Total Tokens
    total_tokens_train = len(train_df)
    total_tokens_val = len(val_df)
    total_tokens_test = len(test_df)
    pie_chart_fig = go.Figure(data=[
        go.Pie(labels=['Train Set', 'Validation Set', 'Test Set'],
               values=[total_tokens_train, total_tokens_val, total_tokens_test],
               hole=.2)
    ])
    pie_chart_fig.update_layout(
        title_text='Total Tokens Distribution',
        legend=dict(orientation='h', y=-0.1),  # Move legend to below the pie chart
        margin=dict(t=50, b=50, l=50, r=50)    # Add more margin to avoid overlapping
    )

    card_content1 = dbc.CardBody([
        html.H6('Total Tokens', style={'fontWeight': 'lighter', 'textAlign': 'center'}),
        dcc.Graph(figure=pie_chart_fig, style={'height': '350px'})
    ])

    # Total Unique Entities
    entity_counts = combined_df['Label'].value_counts()
    card_content2 = dbc.CardBody([
        html.H6('Total Unique Entities', style={'fontWeight': 'lighter', 'textAlign': 'center'}),
        html.Ul([html.Li(f"{entity}: {count}") for entity, count in entity_counts.items()], style={'textAlign': 'center'})
    ])

    # Percentage of Each Entity Type
    entity_percentages = (entity_counts / len(combined_df) * 100).round(2)
    card_content3 = dbc.CardBody([
        html.H6('Entity Type Percentages', style={'fontWeight': 'lighter', 'textAlign': 'center'}),
        html.Ul([html.Li(f"{entity}: {percentage}%") for entity, percentage in entity_percentages.items()], style={'textAlign': 'center'})
    ])

    # Bar Chart of Entity Counts
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']  # Define a list of colors
    fig1 = go.Figure([go.Bar(x=entity_counts.index, y=entity_counts.values, marker_color=colors[:len(entity_counts)])])
    fig1.update_layout(title='Entity Counts', plot_bgcolor='white', yaxis_title='Count', xaxis_title='Entity Type')
    card_content4 = dbc.CardBody([
        html.H6('Entity Counts', style={'fontWeight': 'bold', 'textAlign': 'center'}),
        dcc.Graph(figure=fig1, style={'height': '300px'})
    ])

    # Interactive Bubble Chart for Entity Distribution
    bubble_chart_fig = go.Figure()
    for entity in entity_counts.index:
        top_tokens = combined_df[combined_df['Label'] == entity]['token'].value_counts().head(5)
        bubble_chart_fig.add_trace(go.Scatter(
            x=[entity],
            y=[entity_counts[entity]],
            mode='markers',
            marker=dict(
                size=10 + (entity_counts[entity] / 100),  # Use a fixed minimum size with scaling
                color=colors[entity_counts.index.get_loc(entity)],
                opacity=0.6  # Adjust opacity for better visualization of overlapping bubbles
            ),
            name=entity,
            hoverinfo='text',
            text=[f"Entity: {entity}, Count: {entity_counts[entity]}, Top Tokens: {', '.join(top_tokens.index)}"],
            customdata=[[entity, entity_counts[entity], ', '.join(top_tokens.index)]]
        ))
    bubble_chart_fig.update_layout(
        title='Interactive Entity Distribution',
        xaxis_title='Entity Type',
        yaxis=dict(visible=False),
        plot_bgcolor='white',
        showlegend=False,
        margin=dict(t=50, b=50, l=50, r=50)
    )

    card_content6 = dbc.CardBody([
        html.H6('Interactive Entity Distribution', style={'fontWeight': 'bold', 'textAlign': 'center'}),
        html.Div(id='bubble-hover-data', style={'fontWeight': 'bold', 'fontSize': '14px', 'marginBottom': '10px'}),
        dcc.Graph(id='bubble-chart', figure=bubble_chart_fig, style={'height': '350px'})
    ])

    
    # New Card - Sentence Length Distribution
    


    # New Card - Top Entities for Each Category
    from collections import Counter

    def get_top_entities_by_label(data, label, top_n=5):
        # Filter the data for the specified category
        filtered_data = data[data['Label'] == label]
        # Count tokens within the category
        entity_counts = Counter(filtered_data['token'])
        return entity_counts.most_common(top_n)

    top_loc_entities = get_top_entities_by_label(combined_df, 'LOC')
    top_per_entities = get_top_entities_by_label(combined_df, 'PER')
    top_org_entities = get_top_entities_by_label(combined_df, 'ORG')
    top_o_entities = get_top_entities_by_label(combined_df, 'O')

    def create_bar_chart(top_entities, label):
        entities, counts = zip(*top_entities)
        return go.Bar(x=entities, y=counts, name=label, marker_color='skyblue')

    from plotly.subplots import make_subplots

    fig_top_entities = make_subplots(rows=2, cols=2, subplot_titles=('Top LOC Entities', 'Top PER Entities', 'Top ORG Entities', 'Top O Entities'))

    fig_top_entities.add_trace(create_bar_chart(top_loc_entities, 'LOC'), row=1, col=1)
    fig_top_entities.add_trace(create_bar_chart(top_per_entities, 'PER'), row=1, col=2)
    fig_top_entities.add_trace(create_bar_chart(top_org_entities, 'ORG'), row=2, col=1)
    fig_top_entities.add_trace(create_bar_chart(top_o_entities, 'O'), row=2, col=2)

    fig_top_entities.update_layout(
        title='Top Entities for Each Category',
        showlegend=False,
        plot_bgcolor='white',
        margin=dict(t=50, b=50, l=50, r=50)
    )

    card_content8 = dbc.CardBody([
        html.H6('Top Entities for Each Category', style={'fontWeight': 'bold', 'textAlign': 'center'}),
        dcc.Graph(figure=fig_top_entities, style={'height': '350px'})
    ])

    return card_content1, card_content2, card_content3, card_content4, card_content6, card_content8

@app.callback(
    Output('bubble-hover-data', 'children'),
    [Input('bubble-chart', 'hoverData')]
)
def display_hover_data(hoverData):
    if hoverData is not None:
        point_data = hoverData['points'][0]['customdata']
        entity, count, top_tokens = point_data
        return f"Entity: {entity}, Count: {count}, Top Tokens: {top_tokens}"
    return "Hover over a bubble to see details here."

@app.callback(
    Output('model-metrics', 'children'),
    [Input('model-selector', 'value')]
)
def update_metrics(model_name):
    if model_name in metrics:
        model_metrics = metrics[model_name]
        metrics_list = [
            html.P(f"{metric}: {value}") for metric, value in model_metrics.items()
        ]
        return metrics_list
    return "No metrics available for this model."

@app.callback(
    Output('processed-output', 'children'),
    [Input('process-text', 'n_clicks')],
    [State('model-selector', 'value'), State('sentence-input', 'value')]
)
def processed_output(clicks, model_name, sentence):
    if clicks > 0 and sentence:
        # Chiama la funzione del modello di NER
        try:
            model_path = models[model_name]
            processed_text = plotwisters_model(sentence, model_path, model_name)
            return html.Div(
                children=[html.P(line) for line in processed_text.split('\n')],
                style={
                    'border': '1px solid #d6d6d6',
                    'borderRadius': '5px',
                    'padding': '10px',
                    'backgroundColor': '#f9f9f9',
                    'width': '30%',
                    'margin': 'auto',
                    'textAlign': 'center',
                    'color': '#333',
                    'fontFamily': 'Arial, sans-serif',
                    'lineHeight': '1.5',
                    'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)'

                }
            )
        except Exception as e:
            return f"Error processing sentence: {str(e)}"
    return ""

app.layout = html.Div(id='parent', children=[navbar, body_app])

if __name__ == "__main__":
   app.run_server(debug=True)











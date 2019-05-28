import base64
import datetime
import io
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from urllib.parse import quote
import pandas as pd
import dash_bootstrap_components as dbc
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pickle




navbar1 = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Découvrir l'application", href="/upload")),
    ],
    brand="Claim Severity",
    brand_href="/",
    sticky="top",
)

navbar2 = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Accueil", href="/")),
    ],
    brand="Claim Severity",
    brand_href="/",
    sticky="top",
)


footer = html.Div([
    html.P("Alice Lesbats - Antoine Lozes - William Delaunay",className = "footer")
   
])



app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    footer
    
])






index_page = html.Div([
    navbar1,
    dbc.Container([
        dbc.Row([
            html.P(["Lorsque vous êtes dévasté par un sérieux accident de voitures,vous vous concentrez seulement sur les choses qui comptent le plus : la famille, les amis, et les êtres chers. "]),
            html.Img(src='/assets/image/1.png')

        ],className = "center"),

        dbc.Row([

            html.P(["Vous attaquez à la paperasse avec votre agent d’assurance est la dernière chose sur laquelle vous voulez dépenser de l’énergie ou du temps."]),
            html.Img(src='/assets/image/2.png'),

        ],className = "center"),

        dbc.Row([
            html.P("Cela est pourquoi Allstate, une assurance à la personne aux Etats-Unis, est continuellement à la recherche de nouvelles idées pour améliorer leur service de  réclamations pour plus de 16 millions de ménages qu’ils protègent. "),
            html.Img(src='/assets/image/3.png'),

        ],className = "center"),

        dbc.Row([
            html.P("Allstate est actuellement en développement de méthodes automatisées pour prévoir le coût, d’où l’importance de la sévérité des sinistres, des réclamations."),
            html.Img(src='/assets/image/4.png'),

        ],className= "center"),
        
        
        dbc.Row([html.P("Pour ce Kaggle, nous avons essayé de trouver les algorithmes ayant la meilleure précision de prévision de coût des sinistres.  ")],className = "center")
        
        
    ])
])


page_upload_layout = html.Div([
    navbar2,
    dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop ou ',
                 html.A('Sélectionner un fichier')
                ],className = "upload"),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        
    ),
    dbc.Container([
    

    dbc.Row([

        html.Div(id='output-data-upload')
    ]),

    

    dbc.Row([

        html.A(
            'Télecharger les résultats',
            id='download-link',
            download="resultat.csv",
            href="",
            target="_blank"
        )

    ],className = "center")
    
    

    

    ])
])







def parse_content(contents,filename,date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in content_type:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in content_type:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'Oups il y a eu une erreur avec votre fichier'
        ])

    return html.Div([
        html.P(filename,id= "filename"),
        

        
      
    ])



@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(contents,filename,date):
    if contents is not None:
        children = parse_content(contents,filename,date) 
        return children







@app.callback(
    Output('download-link', 'href'),
    [Input('upload-data', 'contents')]
)
def update_download_link(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        if 'csv' in content_type:
            df = pd.read_csv(io.StringIO(base64.b64decode(content_string).decode('utf-8')))
            
            df = make_prediction(df)
            csv_string = df.to_csv(index=False, encoding='utf-8')
            csv_string = "data:text/csv;charset=utf-8," + quote(csv_string)
            return csv_string

# Update the index
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/upload':
        return page_upload_layout
    
    else:
        return index_page
    

def make_prediction(df):

    model = pickle.load(open("model/lin_reg.sav", 'rb'))
    label = pickle.load(open("model/label.sav",'rb'))   
    df = df.drop(['id'], axis=1)
    
    
    var_cat = []
    for i in range(0, 116):
        #Label encode
        label_encode = LabelEncoder()
        label_encode.fit(label[i])
        cat = label_encode.transform(df.iloc[:,i])
        cat = cat.reshape(df.shape[0], 1)
        #One hot encode 
        feature = OneHotEncoder(sparse=False,n_values=len(label[i])).fit_transform(cat)
        var_cat.append(cat)
        

    var_cat_encodees = np.column_stack(var_cat)


    data_encode = np.concatenate((var_cat_encodees,df.iloc[:,116:].values),axis=1)

    r, c = data_encode.shape

    X = data_encode[:,0:(c-1)]


    df["prediction"] = np.expm1(model.predict(X))


    return df

if __name__ == '__main__':
    app.run_server(debug=True)
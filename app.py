import streamlit as st
import pandas as pd 
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

st.set_page_config(layout="wide", page_title="Real Estate Report", page_icon="🏠")

st.header("Real Estate Report per US Market 🏠", divider= 'blue')

st.sidebar.title("Main Report Filters")

# Load data
cashflow = pd.read_csv('data/cashflows.csv', index_col=0, parse_dates=True).drop(columns=['date_cashflow'])
cashflow['state'] = cashflow['market'].str.split(',').str[1]
cashflow['city'] = cashflow['market'].str.split(',').str[0]

equilibrium = pd.read_csv('data/equilibriums.csv', index_col=0, parse_dates=True).drop(columns=['date_calculated'])
equilibrium['state'] = equilibrium['market'].str.split(',').str[1]
equilibrium['city'] = equilibrium['market'].str.split(',').str[0]

summary = pd.read_csv('data/summaries.csv', index_col=0, parse_dates=True).drop(columns=['date_calculated'])

table = pd.read_csv('data/tables.csv', index_col=0, parse_dates=True).drop(columns=['date_calculated'])
table['state'] = table['market'].str.split(',').str[1]
table['city'] = table['market'].str.split(',').str[0]

# Market Filters
states = summary['state'].unique()
classes = summary['slice'].unique()
horizons = summary['horizon'].unique()

filtro_columnas_mapa = ['irr','equity_multiple','current_price','loan','equity','market_cagr',
                        'noi_cap_rate_compounded','operation_cashflow','market_cap_appreciation_bp','npv','npv/equity','demand_vs_supply','demand_yoy_growth','supply_yoy_growth']
mapa_columns = pd.DataFrame(filtro_columnas_mapa, columns=['columnas'])
mapa_columns.index = mapa_columns['columnas'].str.replace('_',' ').replace('bp','basis point').str.title().str.replace('Yoy','YoY%').str.replace('Npv','Net Present Value').str.replace('Irr', 'IRR')
mapa_columns['unit'] = ['%','x','USD','USD','USD','%','%','%','bp','USD','x','%','%','%']

general = pd.read_csv('data/general_parameters.csv', index_col=0)
general_2 = pd.read_csv('data/general_parameters_2.csv', index_col=0)

def filter_summary(states_f, city_f, slice_f, horizon_f):
    return summary[(summary['state'] == states_f) & (summary['city'] == city_f) & (summary['slice'] == slice_f) & (summary['horizon'] == horizon_f)]

with st.expander('States Filter'):
    st.session_state.selected_states = states
    selected_states = st.multiselect('**Select States**', states, default=st.session_state.selected_states)

# Sidebar filters
with st.sidebar:
    slice = st.selectbox('Select Class', options=classes, index=0)
    st.session_state.slice = slice
    horizon = st.selectbox('Select Horizon (Last Years)', options=horizons, index=0)
    st.session_state.horizon = horizon
    box_selector_pop = ['All','+100K', '+500K', '+1M', '+2M' , '+3M' , '+5M', '+7M', '+10M']
    population = st.selectbox('Select Population', options=box_selector_pop, index=2)
    result = population.replace('+', '').replace('.', '').replace('M', '000000').replace('K', '000').replace('All', '0')
    st.session_state.population = result
    filtro_columnas_mapa_mostrar = st.selectbox('Aspect to classify', options=mapa_columns.index, index=0)
    st.session_state.filtro_columnas_mapa = mapa_columns.loc[filtro_columnas_mapa_mostrar].values[0]
    if st.button('Reset Filters'):
        st.rerun()
    st.markdown('--'*20)

# Filter summary
summary_filtered = summary[(summary['slice'] == st.session_state.slice) & (summary['state'].isin(selected_states)) & (summary['population'] >= int(st.session_state.population))
                           & (summary['horizon'] == st.session_state.horizon)
                           ].copy()
summary_filtered['npv/equity'] = summary_filtered['npv'] / summary_filtered['equity']
if summary_filtered.empty:
    st.error('No data available for the selected filters')
    st.stop()

unidad_columna = mapa_columns.loc[filtro_columnas_mapa_mostrar].values[1]

def transformar_value(column, unidad):
    if unidad in ['%', 'bp']:
        return column.round(4 if unidad == '%' else 0)
    return column.round(2 if unidad == 'x' else 0)

def valor_a_mostrar(column, unidad):
    if unidad == '%':
        return f'{column:.2%}'
    elif unidad == 'x':
        return f'{column:.2f}x'
    elif unidad == 'USD':
        return f'USD {column:,.0f}'
    elif unidad == 'bp':
        return f'{column:.0f} bp'

# Función para asignar colores basados en IRR
def get_color(value, column_name):
    if value > 90:
        return [0, 128, 0, 200]  # Verde más oscuro
    elif value > 75:
        return [144, 238, 144, 200]  # Verde super claro
    elif value > 50:
        return [173, 200, 47, 200]
    elif value > 30:
        return [255, 255, 0, 200]
    elif value > 5:
        return [255, 165, 0, 200] 
    elif value <= 5:
        return [255, 0, 0, 200]

summary_filtered['value'] = transformar_value(summary_filtered[st.session_state.filtro_columnas_mapa], unidad_columna)
summary_filtered['value_show'] = summary_filtered['value'].apply(lambda x: valor_a_mostrar(x, unidad_columna))
## ELIJO LA COLUMNA POR LA CUAL QUIERO FILTRAR
summary_filtered = summary[(summary['slice'] == st.session_state.slice) & (summary['state'].isin(selected_states)) & (summary['population'] >= int(st.session_state.population))
                           & (summary['horizon'] == st.session_state.horizon)
                           ].copy()
summary_filtered['npv/equity'] = summary_filtered['npv'] / summary_filtered['equity']
if summary_filtered.empty:
    st.error('No data available for the selected filters')
    st.stop()

unidad_columna = mapa_columns.loc[filtro_columnas_mapa_mostrar].values[1]

def transformar_value(column, unidad):
    if unidad in ['%', 'bp']:
        return column.round(4 if unidad == '%' else 0)
    return column.round(2 if unidad == 'x' else 0)

def valor_a_mostrar(column, unidad):
    if unidad == '%':
        return f'{column:.2%}'
    elif unidad == 'x':
        return f'{column:.2f}x'
    elif unidad == 'USD':
        return f'USD {column:,.0f}'
    elif unidad == 'bp':
        return f'{column:.0f} bp'
    
# Función para asignar colores basados en IRR
def get_color(value, column_name):
    #if value >  column_name.quantile(0.85):
    if value > 90:
        return [0, 128, 0, 200]  # Verde más oscuro
    #elif value > column_name.quantile(0.7):
    elif value > 75:
        return [144, 238, 144, 200]  # Verde super claro
    #elif value > column_name.quantile(0.4):
    elif value > 50:
        # verde limon
        return [173, 200, 47, 200]
    #elif value > column_name.quantile(0.1):
    elif value > 30:
        # amarillo
        return [255, 255, 0, 200]
    elif value > 5:
        # naranja
        return [255, 165, 0, 200] 
    #elif value <= column_name.quantile(0.1):
    elif value <= 5:
        # rojo
        return [255, 0, 0, 200]
# Añadimos una nueva columna para los colores

summary_filtered['value'] = transformar_value(summary_filtered[st.session_state.filtro_columnas_mapa], unidad_columna)
summary_filtered['value_show'] = summary_filtered['value'].apply(lambda x: valor_a_mostrar(x, unidad_columna))
##3 para el alto de la barra hago un rank y cada uno le asigno su puesto siendo 100 el mayor valor y 1 el menor
summary_filtered['bar_height'] =  summary_filtered['value'].rank(ascending=True, method='max', pct=True) * 100
summary_filtered['color'] = summary_filtered['bar_height'].apply( lambda x: get_color(x, summary_filtered['bar_height']))
summary_filtered['color_no_list'] = summary_filtered['color'].apply(lambda x: f'rgba({x[0]},{x[1]},{x[2]},{x[3]})')
#summary_filtered['bar_height'] = summary_filtered['bar_height'] ** 1.8
summary_filtered['market'] = summary_filtered['market'].str.replace(',', ' - ')
summary_filtered['log_population'] = (summary_filtered['population']) ** 0.5
summary_filtered['population_millions'] = (summary_filtered['population'] / 1_000_000).round(2).astype(str) + 'M'


##########
col1 , col2  = st.columns([1.8, 1.1])

with col1:
    #st.subheader(' per US Market + demographic aggregation for ' + st.session_state.slice + ' buildings')
    st.subheader( f'US Markets classified by {filtro_columnas_mapa_mostrar} + population {st.session_state.slice} buildings')
    # Definimos la capa ColumnLayer
    ### reset view for map 
    st.button('Reset View')
    irr_layer = pdk.Layer(
        "ColumnLayer",
        data=summary_filtered,
        get_position=["longitude", "latitude"],
        get_elevation="bar_height",
        elevation_scale=2500,  # Ajusta según sea necesario para la visibilidad
        radius=20000,  # Ajusta el radio de las columnas
        get_fill_color="color",  # Asignar color basado en la columna calculada
        pickable=True, # Permite seleccionar las barras
        extruded=True,
        auto_highlight=True, 
    )

    population_layer = pdk.Layer(
        "ScatterplotLayer",
        data=summary_filtered,
        get_position=["longitude", "latitude"],
        get_radius="log_population",  # Radio proporcional a la población
        radius_scale=90,  # Ajustar el factor de escala según sea necesario
        get_fill_color= ## Verde con transparencia
        [55, 8, 94, 60],
        pickable=True
    )

    # Configura la vista inicial del mapa
    view_state = pdk.ViewState(
        longitude=-99,
        latitude=38.83,
        zoom=3.4,
        min_zoom=2,
        max_zoom=7,
        pitch=75,  # Reducido para hacer más distinguibles las barras altas
        bearing=23
    )

    lights = pdk.LightSettings(
        number_of_lights= 3)


    # Renderiza el mapa
    st.pydeck_chart(
        pdk.Deck(
            #map_style="mapbox://styles/mapbox/light-v9",
            map_style="mapbox://styles/mapbox/streets-v11",
            initial_view_state=view_state,
            layers=[irr_layer, population_layer],
            tooltip={
                #"html": "<b>City:</b> {city}<br/><b>IRR:</b> {IRR_percentage}%<br/><b>Population:</b> {population}",
                "html": """
                    <b>State:</b> {state}<br/>
                    <b>City:</b> {city}<br/>
                    <b>Value:</b> {value_show}<br/>
                    <b>Population:</b> {population_millions}
                    """,
                "style": {
                    "backgroundColor": "steelblue",
                    "color": "white",
                    "fontSize": "14px",
                    "padding": "10px",
                    "borderRadius": "15px"
                }
            }
            
        ),
        use_container_width=True
        )

with col2:
    subcol1, subcol2 = st.columns([0.2, 1])
    with subcol2:
        st.subheader('Distribution')
        fig = px.histogram(
            summary_filtered, 
            x='value', 
            color='color_no_list', 
            color_discrete_map='identity',
            pattern_shape='city',
            orientation='v', 
            nbins=50,  
            ###3 avoid showing number of observations in the hover
            template='plotly_dark',
        )

        # Ajustar la apariencia del gráfico
        fig.update_layout(
            showlegend=False, 
            yaxis_visible=False, 
            xaxis_title=f'{filtro_columnas_mapa_mostrar}', 
            xaxis_tickformat=',.2f' if unidad_columna == 'USD' else '.2%' if unidad_columna == '%' else '.2f',
            yaxis_title=None,
            bargap=0.1  # Controlar el espacio entre las barras del histograma
        )

        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig, use_container_width=True)
        # Notas del mapa
        st.markdown('''**Map Notes:** The height of the bars is proportional to the value you're filtering, and the color is based on percentiles.  
                The size of the circles is proportional to the population of the city.''')


st.subheader('US Markets Summary', divider= 'blue')

data_show = summary_filtered[['market', 'current_price','market_cagr','noi_cap_rate_compounded',
                               'fixed_interest_rate', 'operation_cashflow','market_cap_appreciation_bp', 'irr', 
                                'npv', 'npv/equity',  'equity_multiple', 'demand_vs_supply',
                                'demand_yoy_growth', 'supply_yoy_growth']]
data_show = data_show.sort_values( st.session_state.filtro_columnas_mapa, ascending=False)
data_show.columns = ['Market', 'Current Price','Market CAGR', 'NOI Cap Rate Compounded',
                        'Fixed Interest Rate', 'Operation Cashflow', 'M. Cap BP', 'IRR', 'NPV', 'NPV/Equity', 'Equity Multiple', 'Demand vs Supply', 'Demand YoY Growth', 'Supply YoY Growth']

for col in ['Current Price','NPV']:
    data_show[col] = data_show[col].apply(lambda x: f'${x:,.0f}')
for col in ['Market CAGR', 'NOI Cap Rate Compounded', 'Fixed Interest Rate', 'Operation Cashflow', 'IRR',]:
    data_show[col] = data_show[col].apply(lambda x: f'{x:.2%}')

data_show['Demand vs Supply'] = data_show['Demand vs Supply'].apply(lambda x: f'{x:.2%}')
data_show['Demand YoY Growth'] = data_show['Demand YoY Growth'].apply(lambda x: f'{x:.2%}')
data_show['Supply YoY Growth'] = data_show['Supply YoY Growth'].apply(lambda x: f'{x:.2%}')


event =  st.dataframe(data_show.set_index('Market')
             , selection_mode=['single-row', 'multi-column'], key='dataframe', on_select='rerun', height=290, use_container_width=True)

if len(event.selection.rows) > 0:
    selection_event = event.selection.rows[-1]
else:
    selection_event = 0

data_selected_market = data_show.iloc[selection_event]['Market']

state = data_selected_market.split(' - ')[1]
city = data_selected_market.split(' - ')[0]

st.header('Individual Market Analysis', divider= 'blue')
st.subheader(f'{data_selected_market} Market Analysis')




cashflow_filtered = cashflow.query('state == @state & city == @city & slice == @slice & horizon == @horizon')
cashflow_filtered_show = cashflow_filtered[['price','equity','revenue','debt_payment','loan_payoff','valuation','cashflow']].copy()
for i in cashflow_filtered_show.columns:
    cashflow_filtered_show[i] = cashflow_filtered_show[i].apply(lambda x: f'${x:,.0f}') 
cashflow_filtered_show.columns = ['Price','Equity','Revenue','Debt Payment','Loan Payoff','Valuation','Cashflow']
cashflow_filtered_show.index = cashflow_filtered_show.index.strftime('%Y-%m-%d')

st.dataframe(use_container_width=True, data=cashflow_filtered_show)



summary_filtered_city = filter_summary(state, city, slice, horizon)
equilibrium_filtered = equilibrium.query('state == @state & city == @city & slice == @slice & horizon == @horizon')

st.subheader('Market Summary')
st.write(summary_filtered_city)

import streamlit as st
import os
import sys
import importlib.util
import inspect
import pandas as pd

# Paths
NBA_PATH = 'nba_api-master/src/nba_api/stats/endpoints'
CBB_MENS_PATH = 'CBBpy-master/src/cbbpy/mens_scraper.py'
CBB_WOMENS_PATH = 'CBBpy-master/src/cbbpy/womens_scraper.py'

sys.path.append('nba_api-master/src')
sys.path.append('CBBpy-master/src')

def get_nba_endpoints():
    endpoints = []
    for file in os.listdir(NBA_PATH):
        if file.endswith('.py') and file != '__init__.py':
            module_name = file.replace('.py', '')
            spec = importlib.util.spec_from_file_location(module_name, os.path.join(NBA_PATH, file))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ == module.__name__:
                    endpoints.append(name)
    return endpoints

def get_nba_columns(endpoint_name):
    try:
        mod = __import__(f'nba_api.stats.endpoints.{endpoint_name}', fromlist=[''])
        instance = getattr(mod, endpoint_name)()
        return list(instance.get_data_frames()[0].columns)
    except Exception as e:
        return [str(e)]

def load_cbb_module(path):
    spec = importlib.util.spec_from_file_location('scraper', path)
    scraper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scraper)
    return scraper

def get_cbb_endpoints(scraper):
    return [name for name, obj in inspect.getmembers(scraper, inspect.isfunction)]

def get_cbb_columns(scraper, func_name):
    func = getattr(scraper, func_name)
    try:
        result = func(*[None]*len(inspect.signature(func).parameters))
        if isinstance(result, pd.DataFrame):
            return list(result.columns)
        elif isinstance(result, tuple):
            return [list(df.columns) for df in result if isinstance(df, pd.DataFrame)]
        else:
            return ['No columns found']
    except Exception as e:
        return [str(e)]

st.title('NBA/NCAAB API Explorer')

api_choice = st.selectbox('Select API', ['NBA', 'NCAAB'])

if st.button('Get Endpoints'):
    if api_choice == 'NBA':
        endpoints = get_nba_endpoints()
        selected_endpoint = st.selectbox('Select NBA Endpoint', endpoints)
        if st.button('Get Columns'):
            columns = get_nba_columns(selected_endpoint)
            st.write(columns)
            if st.button('Save Results'):
                pd.DataFrame(columns, columns=['Columns']).to_csv('nba_columns.csv', index=False)
                st.success('NBA Columns saved as nba_columns.csv')
    else:
        scraper_choice = st.selectbox('Select CBB Type', ['mens', 'womens'])
        scraper = load_cbb_module(CBB_MENS_PATH if scraper_choice == 'mens' else CBB_WOMENS_PATH)
        endpoints = get_cbb_endpoints(scraper)
        selected_endpoint = st.selectbox('Select CBB Endpoint', endpoints)
        if st.button('Get Columns'):
            columns = get_cbb_columns(scraper, selected_endpoint)
            st.write(columns)
            if st.button('Save Results'):
                pd.DataFrame(columns, columns=['Columns']).to_csv('cbb_columns.csv', index=False)
                st.success('CBB Columns saved as cbb_columns.csv')

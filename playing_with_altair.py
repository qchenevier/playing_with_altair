import requests

import pandas as pd
import numpy as np
import altair as alt

from io import BytesIO

#%% Configure viz library
alt.data_transformers.enable('default')
alt.theme.themes.enable('opaque')

#%% Load data
binary_file = requests.get('https://gitlab.com/qchenevier/test_datasets/raw/master/test_ds2.csv').content
csv_file = BytesIO(binary_file)
df_raw = pd.read_csv(csv_file)

#%% Load data form disk
# df_raw = pd.read_csv('test_ds2.csv')

#%% Data preparation
def rename_columns(df):
    df.columns = [c.replace('.', '_') for c in df.columns]
    return df

df = (
    df_raw
    .pipe(rename_columns)
    .assign(capital_variation=lambda df: df.capital_gain - df.capital_loss)
    .assign(income_int=lambda df: (df.income == '>50K').astype(int))
)
df.head().T

#%%
alt.Chart(df.sample(5000), width=500, height=500).mark_circle().encode(
    x='education_num',
    y='capital_variation',
).interactive()

#%%
alt.Chart(df.sample(5000), width=500, height=500).mark_circle().encode(
    x='education_num:O',
    y='capital_variation:Q',
).interactive()
# interactivity much better, but is there overplotting ?

#%%
alt.Chart(df.sample(5000), width=500, height=500).mark_circle(opacity=0.1).encode(
    x='education_num:O',
    y='capital_variation:Q',
).interactive()
# there seems to be overplotting, let's count the dots

#%%
alt.Chart(df.sample(5000), width=500, height=500).mark_circle().encode(
    x='education_num:O',
    y='capital_variation:Q',
    size='count()'
).interactive()
# overplotting confirmed, let's filter data

#%%
alt.Chart(
    df
    .sample(5000)
    .loc[lambda df: df.capital_variation != 0],
    width=500,
    height=500
).mark_circle().encode(
    x='education_num:O',
    y='capital_variation:Q',
    size='count()'
).interactive()
# still some overplotting, let's bin the data

#%%
alt.Chart(
    df
    .sample(5000)
    .loc[lambda df: df.capital_variation != 0],
    width=500,
    height=500,
).mark_circle().encode(
    x='education_num:O',
    y=alt.Y('capital_variation:Q', bin=True),
    size='count()'
).interactive()
# much better but the bins are too big

#%%
alt.Chart(
    df
    .sample(5000)
    .loc[lambda df: df.capital_variation != 0],
    width=500,
    height=500,
).mark_circle().encode(
    x='education_num:O',
    y=alt.Y('capital_variation:Q', bin=alt.Bin(maxbins=30)),
    size='count()'
).interactive()
# nice ! let's try to add some info: income !

#%%
alt.Chart(
    df
    .sample(5000)
    .loc[lambda df: df.capital_variation != 0],
    width=500,
    height=500,
).mark_circle().encode(
    x='education_num:O',
    y=alt.Y('capital_variation:Q', bin=alt.Bin(maxbins=30)),
    size='count()',
    color='income',
).interactive()
# overplotting again ! let's try coloring (works only with binary target) !

#%%
alt.Chart(
    df
    .sample(5000)
    .loc[lambda df: df.capital_variation != 0],
    width=500,
    height=500,
).mark_circle().encode(
    x='education_num:O',
    y=alt.Y('capital_variation:Q', bin=alt.Bin(maxbins=30)),
    size='count()',
    color='mean(income_int)',
).interactive()
# nice ! let's try facetting !

#%%
alt.Chart(
    df
    .sample(5000)
    .loc[lambda df: df.capital_variation != 0],
    width=500,
    height=500,
).mark_circle().encode(
    x='education_num:O',
    y=alt.Y('capital_variation:Q', bin=alt.Bin(maxbins=30)),
    size='count()',
    color='income',
).facet(
    column='income'
).interactive()
# facetting is nice ! let's play with bars

#%%
alt.Chart(
    df
    .sample(5000),
    width=500,
    height=500,
).mark_bar().encode(
    x='education_num:O',
    y='count(income)',
    color='income:N',
).interactive()
# well... difficult to compare with stacked bars ! let's try with columns !

#%%
alt.Chart(
    df
    .sample(5000),
    width=500,
    height=500,
).mark_bar().encode(
    x='income:N',
    y='count(income)',
    color='income:N',
    column='education_num:O',
).interactive()
# hmm, let's reduce the size of each subplot !

#%%
alt.Chart(
    df
    .sample(5000),
).mark_bar().encode(
    x='income:N',
    y='count(income)',
    color='income:N',
    column='education_num:O',
).interactive()
# hmm, let's clean unnecessary info !

#%%
alt.Chart(
    df
    .sample(5000),
).mark_bar().encode(
    x=alt.X('income:N', axis=alt.Axis(title='')),
    y='count(income)',
    color='income:N',
    column='education_num:O',
).configure_view(stroke='transparent').interactive()
# nice, why not visualize ratio ?

#%%
alt.Chart(
    df
    .sample(5000),
    width=500,
    height=500,
).mark_bar().encode(
    x='education_num:O',
    y=alt.Y('count(income)', stack='normalize'),
    color='income:N',
    tooltip='count(income)',
)
# Nice ! education_num is really helping ! but what are the classes which contain the most elements ?

#%%
alt.Chart(
    df
    .sample(5000),
    width=500,
    height=500,
).mark_bar().encode(
    x='education_num:O',
    y=alt.Y('count(income)', stack='normalize'),
    color='income:N',
    tooltip='count(income)',
    opacity='count(income)',
)
# Nice ! let's try another column !

#%%
df.columns.tolist()

#%%
column_to_plot = [
    'age',
    'workclass',
    # 'fnlwgt',
    # 'education',
    'education_num',
    'marital_status',
    # 'occupation',
    'relationship',
    'sex',
    # 'capital_gain',
    # 'capital_loss',
    'hours_per_week',
    # 'native_country',
    # 'income',
    # 'capital_variation',
    # 'income_int'
]

#%%
alt.Chart(
    df
    .sample(5000),
    width=500,
    height=500,
).mark_bar().encode(
    x=alt.X(alt.repeat('row'), type='nominal'),
    y=alt.Y('count(income)', stack='normalize'),
    color='income:N',
    tooltip='count(income)',
    # opacity='count(income)',
).repeat(
    row=column_to_plot
)
# Nice ! education_num is really helping !

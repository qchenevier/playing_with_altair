import requests

import pandas as pd
import numpy as np
import altair as alt

from io import BytesIO

#%% Configure viz library
alt.data_transformers.enable('csv') # to be able to plot more than 1000 rows
alt.theme.themes.enable('opaque')

#%% Load data
binary_file = requests.get('https://gitlab.com/qchenevier/test_datasets/raw/master/test_ds2.csv').content
csv_file = BytesIO(binary_file)
df_raw = pd.read_csv(csv_file)

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
alt.Chart(df.sample(4000), width=500, height=500).mark_circle().encode(
    x='capital_variation',
    y='education_num',
    color='income',
    tooltip=['capital_variation']
).interactive()

#%%
alt.Chart(df.sample(4000), width=500, height=500).mark_circle().encode(
    alt.X('capital_variation:Q', bin=alt.Bin(maxbins=60)),
    alt.Y('education_num:Q', bin=alt.Bin(maxbins=60)),
    # alt.Color('count(capital_variation):Q', scale=alt.Scale(scheme='greenblue'))
    color='income',
    size='count(capital_variation):Q',
    tooltip=['capital_variation']
).interactive()

#%%
def shorthand(column, column_type):
    return '{column}:{type}'.format(column=column, type=column_type[0].upper())

columns_and_altair_type = (
    df.dtypes
    .drop([
        'income',
        'income_int',
        'native_country',
        'education',
        'capital_loss',
        'capital_gain',
        'education',
    ])
    .reset_index()
    .rename(columns={
        'index': 'column',
        0: 'column_type',
    })
    .assign(column_type=lambda df: df.column_type.replace({
        'int64': 'quantitative',
        'object': 'nominal',
    }))
    .assign(shorthand=lambda df: df.apply(lambda x: shorthand(x.column, x.column_type), axis=1))
    .to_dict('records')
)

#%%
columns = [c['shorthand'] for c in columns_and_altair_type]

alt.Chart(df.sample(500)).mark_circle(opacity=0.3).encode(
    alt.X(alt.repeat('column')),
    alt.Y(alt.repeat('row')),
    color='income',
).properties(
    width=250,
    height=250,
).repeat(
    row=list(columns),
    column=list(reversed(columns)),
).interactive().save('scatter_matrix.html')

#%%
data = df.sample(5000)

ruler = alt.Chart(data).mark_rule(color='red').encode(
    alt.Y('mean(income_int)', type='quantitative'),
).properties(
    width=500,
    height=500,
).interactive()

def compute_histo(data, col, col_type):
    if col_type == 'quantitative':
        X = alt.X(col, bin=alt.Bin(maxbins=100), type=col_type)
    else:
        X = alt.X(col, type=col_type)
    return alt.Chart(data).mark_bar().encode(
        X,
        alt.Y('mean(income_int)', type='quantitative'),
        alt.Color('count(income_int)', scale=alt.Scale(domain=[0,200])),
    ).properties(
        width=500,
        height=500,
    ).interactive()

alt.vconcat(*[ruler + compute_histo(data, c['column'], c['column_type']) for c in columns_and_altair_type])

#%%
alt.Chart(df.sample(500)).mark_rect(interpolate='step').encode(
    alt.X('age', bin=alt.Bin(maxbins=100), type='quantitative'),
    alt.Y('mean(income_int)', type='quantitative'),
    color='mean(income_int)',
).properties(
    width=500,
    height=500,
).interactive()

#%%
alt.Chart(df.sample(1000))


#%%
def compute_X(c, keep_axis=True):
    additional_kwargs = {}
    if c['column_type'] == 'quantitative':
        additional_kwargs.update({'bin': alt.Bin(maxbins=40)})
    if not keep_axis:
        additional_kwargs.update({'axis': None})
    return alt.X(c['column'], type=c['column_type'], **additional_kwargs)

def compute_Y(c, keep_axis=True):
    additional_kwargs = {}
    if c['column_type'] == 'quantitative':
        additional_kwargs.update({'bin': alt.Bin(maxbins=40)})
    if not keep_axis:
        additional_kwargs.update({'axis': None})
    return alt.Y(c['column'], type=c['column_type'], **additional_kwargs)

X_vs_Y_chart = alt.Chart().mark_circle(opacity=0.3).encode(
    color=alt.Color('income_int', scale=alt.Scale(scheme='spectral')),
).properties(
    width=200,
    height=200,
).interactive()

histo_chart = alt.Chart().mark_rect().encode(
    color=alt.Color('mean(income_int)', scale=alt.Scale(scheme='spectral')),
).properties(
    width=200,
    height=200,
).interactive()

chart = alt.vconcat(data=df.sample(1000))

first_column_index = 0
last_line_index = len(columns_and_altair_type) - 1

for y_index, y_column in enumerate(columns_and_altair_type):
    row = alt.hconcat()
    for x_index, x_column in enumerate(columns_and_altair_type):
        x_encoding = compute_X(x_column, keep_axis=(y_index == last_line_index))
        y_encoding = compute_X(y_column, keep_axis=(x_index == first_column_index))
        row |= base.encode(x=x_encoding, y=y_encoding)
    chart &= row
chart.save('scatter_matrix_binned.html')


#%%
def compute_X(c, keep_axis=True, bins=False):
    additional_kwargs = {}
    if (c['column_type'] == 'quantitative') and bins:
        additional_kwargs.update({'bin': alt.Bin(maxbins=bins)})
    if not keep_axis:
        additional_kwargs.update({'axis': None})
    return alt.X(c['column'], type=c['column_type'], **additional_kwargs)

def compute_Y(c, keep_axis=True, bins=False, histo=False):
    additional_kwargs = {}
    if (c['column_type'] == 'quantitative') and bins and not histo:
        additional_kwargs.update({'bin': alt.Bin(maxbins=bins)})
    if not keep_axis:
        additional_kwargs.update({'axis': None})
    if not histo:
        return alt.Y(c['column'], type=c['column_type'], **additional_kwargs)
    else:
        return alt.Y('count()', stack=None, **additional_kwargs)

X_vs_Y_chart = alt.Chart().mark_circle(opacity=0.3).encode(
    color='income',
).properties(
    width=200,
    height=200,
).interactive()

histo_chart = alt.Chart().mark_area(opacity=0.3, interpolate='step').encode(
    color='income',
).properties(
    width=200,
    height=200,
).interactive()

chart = alt.vconcat(data=df.sample(1000))

first_column_index = 0
last_line_index = len(columns_and_altair_type) - 1

for y_index, y_column in enumerate(columns_and_altair_type):
    row = alt.hconcat()
    for x_index, x_column in enumerate(columns_and_altair_type):
        y_encoding = compute_Y(y_column, keep_axis=(x_index == first_column_index))
        if x_index == y_index:
            x_encoding = compute_X(x_column, keep_axis=(y_index == last_line_index), bins=40)
            y_encoding = compute_Y(y_column, keep_axis=(x_index == first_column_index), histo=True)
            new_chart = histo_chart.encode(x=x_encoding, y=y_encoding)
        else:
            x_encoding = compute_X(x_column, keep_axis=(y_index == last_line_index))
            new_chart = X_vs_Y_chart.encode(x=x_encoding, y=y_encoding)
        row |= new_chart
    chart &= row
chart.save('scatter_matrix.html')

#%%
def compute_classes_percentages(df, class_column, target_column):
    return (
        df
        .groupby([class_column, target_column])
        .size()
        .reset_index()
        .rename(columns={0: 'n'})
        .groupby(class_column)
        .apply(lambda df: df.assign(percentage=lambda df: df.n / df.n.sum()))
        .reset_index(drop=True)
        .assign(plot_percentage=lambda df: np.where(df.income == '<=50K', -df.percentage, df.percentage))
    )

#%%
alt.Chart(df.pipe(compute_classes_percentages, 'occupation', 'income'), width=500, height=500).mark_bar().encode(
    x='plot_percentage:Q',
    y='occupation:N',
    color='income:N',
    tooltip=['plot_percentage', 'occupation'],
).interactive()

#%%

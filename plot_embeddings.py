import pandas as pd
from scipy import sparse
import psycopg2
from sqlalchemy import create_engine

from bokeh.plotting import figure, show, save, output_file
from bokeh.models import HoverTool, value, LabelSet, Legend, ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral3

stitch = create_engine(
    'postgresql://',
    echo=True,
    pool_recycle=300,
    echo_pool=True,
    creator=lambda _: psycopg2.connect(service='rockets-stitch'))

slave = psycopg2.connect(service="rockets-slave")

localdb = create_engine(
    'postgresql://',
    echo=False,
    pool_recycle=300,
    creator=lambda _: psycopg2.connect(service="rockets-local"))

# df_combine = pd.read_sql_query("SELECT * FROM dw.product_embeddings", stitch)
# df_combine = pd.read_sql_query("SELECT * FROM dwh.product_embeddings", localdb)
# skus = set(df_combine['sku'].unique())
# sku_list = '(' + ', '.join(["'" + str(sku) + "'" for sku in skus]) + ')'

# g = pd.read_sql_query(
#     """
#     SELECT v.id          AS variant_id,
#         v.sku,
#         genders.value AS gender
#     FROM ((spree_variants v
#         LEFT JOIN spree_product_properties genders ON ((genders.product_id = v.product_id)))
#         JOIN spree_properties g_property ON (((g_property.id = genders.property_id) AND
#                                             ((g_property.name) :: text = 'gender' :: text))))
#     WHERE v.sku IN {skus};
# """.format(skus=sku_list), slave)

# df_combine = df_combine.merge(
#     g, how='left', left_on=['id', 'sku'], right_on=['variant_id', 'sku'])
# df_combine.drop('variant_id', axis=1, inplace=True)

gs = pd.read_sql_query(
    """
    SELECT -- v.id AS variant_id,
        DISTINCT
        (regexp_split_to_array(v.sku, '-'))[1] AS mid,
        -- s.presentation,
        g.gender
    FROM spree_variants v
    -- LEFT JOIN variant_sizes s ON v.id = s.spree_variant_id
    LEFT JOIN variant_genders g ON v.id = g.spree_variant_id
    WHERE v.sku LIKE ANY (ARRAY{skus})
    -- WHERE v.sku IN {skus};
""".format(skus=sku_list), slave)

df_combine = pd.merge(
    df_combine,
    gs,
    how='left', on='mid')
    # left_on=['id', 'sku'],
    # right_on=['variant_id', 'sku'])
df_combine.dropna(inplace=True)

# df_combine.drop('variant_id', axis=1, inplace=True)

source = ColumnDataSource(
    data=dict(
        x=df_combine['x-umap'],
        y=df_combine['y-umap'],
        # gender=df_combine['gender'],
        imgs=df_combine['url'],
        title=df_combine['mid']))

title = 'T-SNE visualization of product embeddings'

plot_lda = figure(
    plot_width=1000,
    plot_height=1000,
    title=title,
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None,
    y_axis_type=None,
    min_border=1)

wt = list(df_combine['gender'].unique())

plot_lda.scatter(
    x='x',
    y='y',
    # legend='gender',
    source=source,
    alpha=0.9,
    size=10)
    # fill_color=factor_cmap('gender', palette=Spectral3, factors=wt))

# hover tools
hover = plot_lda.select(dict(type=HoverTool))
hover.tooltips = """
    <div>
        <div>
            <span style="font-size: 17px; font-weight: bold;">@title</span></div>
        <div>
            <img src="@imgs" height="82" alt="@imgs" width="82" style="float: left; margin: 0px 5px 5px 0px;" border="2"></img>
        </div>
    </div>
"""

plot_lda.legend.orientation = "horizontal"
plot_lda.legend.location = "top_center"

output_file('product_embeddings.html', title=title)
show(plot_lda)
# save(plot_lda)  # , 'product_embeddings.html')

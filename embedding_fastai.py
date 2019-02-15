"""
from this webpage: https://towardsdatascience.com/collaborative-filtering-and-embeddings-part-2-919da17ecefb
"""

import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sklearn.manifold import TSNE
import numpy as np

import torch
from bokeh.plotting import figure, show, output_notebook, save
from bokeh.models import HoverTool, value, LabelSet, Legend, ColumnDataSource
# output_notebook()
# from fastai.learner import *
from fastai.learner import math
from fastai.learner import metrics
from fastai.learner import sns
from fastai.learner import to_np

# from fastai.column_data import *
from fastai.column_data import get_cv_idxs
from fastai.column_data import CollabFilterDataset

slave = psycopg2.connect(service="rockets-slave")

r = pd.read_sql_query(
    """
    SELECT b.kid_profile_id   AS kid_id,
        v.id               AS variant_id,
        CASE
            WHEN ri.id IS NULL OR ri.reception_status = 'expired' THEN 1
            ELSE -1 END AS kept,
        v.sku
    FROM boxes b
            JOIN spree_orders o ON b.order_id = o.id
            JOIN spree_line_items si ON si.order_id = o.id
            LEFT JOIN spree_inventory_units iu ON iu.line_item_id = si.id
            LEFT JOIN spree_return_items ri ON ri.inventory_unit_id = iu.id
            LEFT JOIN spree_variants v ON v.id = si.variant_id
    WHERE b.state = 'final'  -- otherwise kept is meaningless
    AND v.sku <> 'X001-K09-A'
    AND b.approved_at < (CURRENT_DATE - INTERVAL '2 weeks') :: date
    AND b.approved_at > '2018-01-01';
""", slave)

val_idxs = get_cv_idxs(len(r))
wd = 2e-4  #weight decay
n_factors = 50  #dimension of embedding vector

cf = CollabFilterDataset.from_data_frame("./", r, "kid_id", "variant_id",
                                         "kept")
learn = cf.get_learner(n_factors, val_idxs, 64, opt_fn=optim.Adam)
learn.fit(1e-2, 2, wds=wd, cycle_len=1, cycle_mult=2, use_wd_sched=True)
preds = learn.predict()  #prediction on validation
y = learn.data.val_y
print(math.sqrt(metrics.mean_squared_error(y, preds)))  #calculating RMSE
sns.jointplot(preds, y, kind='hex', stat_func=None)

# torch.save(learn, 'test2')

# Now the plotting part
items = r.loc[:, ['variant_id', 'sku']]
skus = items.set_index('variant_id')['sku'].to_dict()  # titles
g = r.groupby('variant_id')['kept'].count()  # number of ratings
top_items = g.sort_values(
    ascending=False
).index.values[:3000]  #top 3000 movies based on number of ratings
top_item_idx = np.array(
    [cf.item2idx[o] for o in top_items]
)  #id for the top movies to link it to the embedding and bias matrix created by the model

m = learn.model  #retrieving the model
item_emb = to_np(m.i(
    V(top_item_idx)))  #converting the torch embedding to numpy matrix

#reducing the dimension of movie embeddings from 50 to 2
tsne = TSNE(
    n_components=2, verbose=1, perplexity=30, n_iter=1000, learning_rate=10)
tsne_results = tsne.fit_transform(item_emb)

#visualizing t-sne components of embeddings using Bokeh
df_combine = pd.DataFrame([skus[i] for i in top_items])
df_combine.columns = ['sku']
df_combine['x-tsne'] = tsne_results[:, 0]
df_combine['y-tsne'] = tsne_results[:, 1]

title = 'T-SNE visualization of embeddings'

plot_lda = figure(
    plot_width=1000,
    plot_height=600,
    title=title,
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None,
    y_axis_type=None,
    min_border=1)

plot_lda.scatter(x='x', y='y', source=source, alpha=0.4, size=10)

# hover tools
hover = plot_lda.select(dict(type=HoverTool))
hover.tooltips = {"content": "sku: @title"}

show(plot_lda)
save(plot_lda, '{}.html'.format(title))

# bias now
item_bias = to_np(m.ib(
    V(top_item_idx)))  #extracting item bias and converting it to numpy matrix
item_ratings = [(b[0], skus[i]) for i, b in zip(top_items, item_bias)]
sorted(
    item_ratings, key=lambda o: o[0])[:15]  #worst items based on bias ranking
sorted(
    item_ratings, key=lambda o: o[0],
    reverse=True)[:15]  #top items based on bias ranking

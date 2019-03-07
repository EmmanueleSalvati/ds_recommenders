import pandas as pd
from scipy import sparse
import psycopg2
import argparse
from sqlalchemy import create_engine
from sklearn.manifold import TSNE
import umap

from sklearn.feature_extraction import DictVectorizer
from helpers import *
import numpy as np

from lightfm import LightFM
from skopt import forest_minimize

from lightfm.evaluation import precision_at_k
from lightfm.evaluation import recall_at_k
from lightfm.evaluation import auc_score

from sklearn.cluster import KMeans

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# parser = argparse.ArgumentParser()
# parser.add_argument('-o', '--optimize', help='optimize parameters', action='store_true')
# args = parser.parse_args()

redshift = create_engine(
    'postgresql://',
    echo=False,
    pool_recycle=300,
    echo_pool=True,
    creator=lambda _: psycopg2.connect(service='rockets-redshift'))

slave = create_engine(
    'postgresql://',
    echo=False,
    pool_recycle=300,
    echo_pool=True,
    creator=lambda _: psycopg2.connect(service='rockets-slave'))

k = pd.read_sql_query(
    """
    SELECT fkbc.kid_profile_id,
        fkbc.box_id,
        dk.gender,
        fbkp.top_size
    FROM dw.fact_kid_box_count fkbc
            JOIN dw.fact_boxes b ON fkbc.box_id = b.box_id
            JOIN dw.fact_box_kid_preferences fbkp ON fkbc.box_id = fbkp.box_id
            JOIN dw.dim_kid dk ON fkbc.kid_profile_id = dk.kid_profile_id
    WHERE kid_final_box_rank IS NOT NULL
        AND kid_final_box_rank > 1
        AND b.approved_at BETWEEN '2018-01-01' AND '2018-12-31'
""", redshift)
k['top_size'] = k['top_size'].astype('int')
b = k['box_id'].values
boxes = '(' + ', '.join([str(box) for box in b]) + ')'

d = pd.read_sql_query(
    """
    SELECT box_id,
        variant_id,
        fbsk.sku,
        dc.style_number,
        dc.color_code,
        dc.size_code,
        dc.division,
        dc.color_family,
        dc.color_name,
        dc.category,
        dc.image_url
        -- dc.shot_type
    FROM dw.fact_box_sku_keep fbsk
    LEFT join dw.dim_canon dc on fbsk.sku = dc.sku
    WHERE box_id IN {boxes}
        AND fbsk.sku <> 'X001-K09-A'
        AND kept = 1.0
        AND dc.shot_type = 'front'
""".format(boxes=boxes), redshift)

# remove a deleted beanie
d = d.loc[d['style_number'].notnull(), ]


df = pd.merge(k, d, how='inner', on='box_id')
# df['kept'] = df['kept'].replace(0, -1).astype('int')
import sys
sys.exit()

g = df.loc[df['gender'] == 'girls', ]
b = df.loc[df['gender'] == 'boys', ]

X = threshold_interactions_df(g, 'kid_profile_id', 'style_number', 8, 8)
X.sort_values(by=['kid_profile_id', 'style_number'], inplace=True)

likes, kid_to_idx, idx_to_kid, style_to_idx, idx_to_style = \
    df_to_matrix(X, 'kid_profile_id', 'style_number')
train, test, user_index = train_test_split(likes, 5, fraction=0.2)


eval_train = train.copy()
non_eval_users = list(set(range(train.shape[0])) - set(user_index))

eval_train = eval_train.tolil()
for u in non_eval_users:
    eval_train[u, :] = 0.0
eval_train = eval_train.tocsr()

model = LightFM(
    learning_rate=0.05, loss='warp', no_components=25, random_state=2018)
model.fit(train, epochs=0)
# user_features=user_features_concat,
# item_features=item_features_concat)

iterarray = range(1, 5, 1)
model, train_patk, test_patk = \
    patk_learning_curve(model, train, test, eval_train, iterarray, k=20)
# user_features=user_features_concat, item_features=item_features_concat, k=20)

# Plot train on left
ax = plt.subplot(1, 2, 1)
fig = ax.get_figure()
sns.despine(fig)
plot_patk(iterarray, train_patk, 'Train', k=20)

# Plot test on right
ax = plt.subplot(1, 2, 2)
fig = ax.get_figure()
sns.despine(fig)
plot_patk(iterarray, test_patk, 'Test', k=20)

plt.tight_layout()

# a la lightfm
# precision = precision_at_k(model, test, eval_train, 20,
#                            user_features_concat).mean()
precision_test = precision_at_k(model, test, eval_train, 20).mean()
recall_test = recall_at_k(model, test, eval_train, 20).mean()

# recall = recall_at_k(model, test, eval_train, 20,
#                      user_features_concat).mean()
# auc = auc_score(model, test, eval_train, user_features_concat).mean()

auc_train = auc_score(model, train, num_threads=2).mean()
auc_test = auc_score(model, test, eval_train, num_threads=2).mean()



# skus = set(df['mid'].unique())
# sku_list = '[' + ', '.join(["'" + sku + "-%'" for sku in skus]) + ']'

# images = pd.read_sql_query(
#     """
#     SELECT
#         DISTINCT
#         (regexp_split_to_array(v.sku, '-'))[1] AS mid,
#         'https://res.cloudinary.com/roa-canon/image/upload/w_339/' || s.public_id AS url
#     FROM shots s
#             JOIN spree_variants v ON s.variant_id = v.id
#     WHERE v.sku LIKE ANY (ARRAY{skus})
#         AND shot_type = 'front'
# """.format(skus=sku_list), slave)

# images = images.groupby('mid').first().reset_index()

# user_emb = model.user_embeddings  # 50 feature embedding per user:  (23592, 50) ndarray
# this is 7531 x 50 ndarray (50 embeddings per item)
item_emb = model.item_embeddings

top_styles = g.groupby('style_number').count().sort_values(
    by='kid_profile_id', ascending=False).index.values[:100]
top_idx = np.array([style_to_idx[sku] for sku in top_styles])
top_item_emb = item_emb[top_idx]

tsne = TSNE(
    n_components=2, verbose=1, perplexity=30, n_iter=1000, learning_rate=10)
tsne_results = tsne.fit_transform(top_item_emb)

embedding = umap.UMAP(
    n_neighbors=150, min_dist=0.5, random_state=12).fit_transform(top_item_emb)

images = d.groupby('style_number').first().reset_index()

df_combine = images.loc[images['style_number'].isin(top_styles), 'image_url'].to_frame()

# df_combine = images.loc[images['mid'].isin(top_skus),]
df_combine['x-tsne'] = tsne_results[:, 0]
df_combine['y-tsne'] = tsne_results[:, 1]
df_combine['x-umap'] = embedding[:, 0]
df_combine['y-umap'] = embedding[:, 1]

# df_combine.to_sql(
#     "product_embeddings",
#     stitch,
#     schema='dw',
#     if_exists='replace',
#     index=False,
#     chunksize=1000)


# user embeddings to cluster
user_emb = model.user_embeddings

kmeans = KMeans(n_clusters=10, random_state=0).fit(user_emb)
preds = kmeans.predict(user_emb)
tsne_kmeans = TSNE(
    n_components=2, verbose=1, perplexity=30, n_iter=1000, learning_rate=10)
preds_2D = tsne_kmeans.fit_transform(user_emb)


centers = kmeans.cluster_centers_

# build user_features
# user_dlist = [{} for _ in uid_to_idx]
# for uuid in idx_to_uid:
# size = kids.loc[kids['uid'] == idx_to_uid[uuid], 'size'].values[0]
# gender = kids.loc[kids['uid'] == idx_to_uid[uuid], 'gender'].values[0]
# user_key = '{}_{}'.format(gender, str(size))
# user_dlist[uuid][user_key] = 1
# user_dlist[uuid][gender] = 1

# dv = DictVectorizer()
# user_features = dv.fit_transform(user_dlist)

# eye = sp.eye(user_features.shape[0], user_features.shape[0]).tocsr()
# user_features_concat = sp.hstack((eye, user_features))
# user_features_concat = user_features_concat.tocsr().astype(np.float32)

# build item_features
# item_dlist = [{} for _ in mid_to_idx]
# for muid in idx_to_mid:
# size = d.loc[d['mid'] == idx_to_mid[muid], 'presentation'].values[0]
# gender = d.loc[d['mid'] == idx_to_mid[muid], 'gender'].values[0]
# item_key = '{}_{}'.format(gender, str(size))
# item_dlist[muid][item_key] = 1
# item_dlist[muid][gender] = 1

# fv = DictVectorizer()
# item_features = fv.fit_transform(item_dlist)

# item_eye = sp.eye(item_features.shape[0], item_features.shape[0]).tocsr()
# item_features_concat = sp.hstack((item_eye, item_features))
# item_features_concat = item_features_concat.tocsr().astype(np.float32)

# if args.optimize:
#     space = [
#         (15, 25),  # no_components
#         ('warp', 'bpr', 'warp-kos')  # loss function
#     ]

#     def objective(params):
#         # unpack
#         no_components, loss = params

#         model = LightFM(
#             loss=loss, random_state=2016, no_components=no_components)
#         model.fit(train, verbose=True)

#         patks = lightfm.evaluation.precision_at_k(
#             model, test, train_interactions=eval_train, k=5, num_threads=4)
#         mapatk = np.mean(patks)
#         # Make negative because we want to _minimize_ objective
#         out = -mapatk
#         # Handle some weird numerical shit going on
#         if np.abs(out + 1) < 0.01 or out < -1.0:
#             return 0.0
#         else:
#             return out

#     res_fm = forest_minimize(
#         objective, space, n_calls=250, random_state=0, verbose=True)

#     print('Maximimum p@k found: {:6.5f}'.format(-res_fm.fun))
#     print('Optimal parameters:')
#     params = ['no_components', 'loss']
#     for (p, x_) in zip(params, res_fm.x):
#         print('{}: {}'.format(p, x_))

# else:

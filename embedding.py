import pandas as pd
from scipy import sparse
import psycopg2
from sqlalchemy import create_engine
# from sklearn.manifold import TSNE
from sklearn.feature_extraction import DictVectorizer
from helpers import *
import numpy as np

from lightfm.data import Dataset
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import recall_at_k
from lightfm.evaluation import auc_score
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

slave = psycopg2.connect(service="rockets-slave")

k = pd.read_sql_query(
    """
    SELECT p.id                                                       AS uid,
        p.gender,
        b.id                                                       AS box_id,
        b.approved_at,
        rank() OVER (PARTITION BY p.id ORDER BY b.approved_at ASC) AS box_number,
        cast(p.preferences->'size_preferences'->>'tops' as int) as size
    FROM kid_profiles p
            LEFT JOIN boxes b ON p.id = b.kid_profile_id
    WHERE b.state = 'final';
""", slave)

d = pd.read_sql_query(
    """
    SELECT b.kid_profile_id AS uid,
        v.id             AS mid
    FROM boxes b
            JOIN spree_orders o ON b.order_id = o.id
            JOIN spree_line_items si ON si.order_id = o.id
            LEFT JOIN spree_inventory_units iu ON iu.line_item_id = si.id
            LEFT JOIN spree_return_items ri ON ri.inventory_unit_id = iu.id
            LEFT JOIN spree_variants v ON v.id = si.variant_id
    WHERE b.state = 'final'
    AND v.sku <> 'X001-K09-A'
    AND b.approved_at < (CURRENT_DATE - INTERVAL '2 weeks') :: date
    AND b.approved_at > '2017-01-01'
    AND (ri.id IS NULL OR ri.reception_status = 'expired');
""", slave)

# consider only kids that received more than one box
kids_to_consider = set(k.loc[k['box_number'] > 1, 'uid'].unique())

d = d.loc[d['uid'].isin(kids_to_consider), ]
d = threshold_interactions_df(d, 'uid', 'mid', 8, 8)
d.sort_values(by='uid', inplace=True)

kids = k.loc[k['uid'].isin(set(d['uid'].unique())), [
    'uid', 'gender', 'size']].drop_duplicates()

# kids = k.loc[k['uid'].isin(set(d['uid'].unique())), ]
kids.sort_values(by='uid', inplace=True)

likes, uid_to_idx, idx_to_uid, mid_to_idx, idx_to_mid = \
    df_to_matrix(d, 'uid', 'mid')
train, test, user_index = train_test_split(likes, 5, fraction=0.2)

eval_train = train.copy()
non_eval_users = list(set(range(train.shape[0])) - set(user_index))

eval_train = eval_train.tolil()
for u in non_eval_users:
    eval_train[u, :] = 0.0
eval_train = eval_train.tocsr()

# build user_features
user_dlist = [{} for _ in uid_to_idx]
for uuid in idx_to_uid:
    size = kids.loc[kids['uid'] == idx_to_uid[uuid], 'size'].values[0]
    gender = kids.loc[kids['uid'] == idx_to_uid[uuid], 'gender'].values[0]
    user_key = '{}_{}'.format(gender, str(size))
    user_dlist[uuid][user_key] = 1

dv = DictVectorizer()
user_features = dv.fit_transform(user_dlist)

# eye = sp.eye(user_features.shape[0], user_features.shape[0]).tocsr()
# user_features_concat = sp.hstack((eye, user_features))
# user_features_concat = user_features_concat.tocsr().astype(np.float32)

model = LightFM(
    learning_rate=0.05,
    loss='warp',
    no_components=50,
    random_state=2018)
model.fit(train, epochs=0, user_features=user_features)

iterarray = range(1, 15, 1)
model, train_patk, test_patk = \
    patk_learning_curve(model, train, test, eval_train,
                        iterarray, user_features, k=20)

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
# train_precision = precision_at_k(model, train, k=10).mean()
# test_precision = precision_at_k(model, test, k=10).mean()

# train_auc = auc_score(model, train).mean()
# test_auc = auc_score(model, test).mean()

# print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
# print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

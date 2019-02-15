import pandas as pd

import argparse

import numpy as np
import psycopg2
from spotlight.interactions import Interactions
from spotlight.cross_validation import user_based_train_test_split
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.evaluation import rmse_score
from spotlight.evaluation import precision_recall_score

slave = psycopg2.connect(service="rockets-slave")

parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--explicit', help='explicit models', action='store_true')

args = parser.parse_args()

if args.explicit:
    print("Explicit model")
    r = pd.read_sql_query(
        """
        SELECT b.kid_profile_id   AS "user",
            v.id               AS item,
            CASE
                WHEN ri.id IS NULL OR ri.reception_status = 'expired' THEN 1
                ELSE -1 END AS rating,
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
        AND b.approved_at > '2017-01-01';
    """, slave)


    row_idx = np.int32(r['user'].values)
    col_idx = np.int32(r['item'].values)
    data = np.float32(r['rating'].values)
    dataset = Interactions(row_idx, col_idx, data)

    train, test = user_based_train_test_split(dataset)

    model = ExplicitFactorizationModel(
        n_iter=3,
        loss="logistic",
        embedding_dim=50,
        learning_rate=1e-2,
        l2=1e-9,
        batch_size=1024)

    model.fit(train, verbose=True)
    # rmse = rmse_score(model, test)
    p, r = precision_recall_score(model, test, train)

    # top_skus = \
    #     r.groupby('variant_id')['sku'].count().sort_values(ascending=False).\
    #     index.values[:3000]

else:
    print("Implicit model")
    d = pd.read_sql_query(
        """
        SELECT b.kid_profile_id AS "user",
            v.id             AS item
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

    row_idx = np.int32(d['user'].values)
    col_idx = np.int32(d['item'].values)
    dataset = Interactions(row_idx, col_idx)

    train, test = user_based_train_test_split(dataset)

    model = ImplicitFactorizationModel(
        n_iter=3,
        loss="adaptive_hinge",
        embedding_dim=50,
        learning_rate=1e-2,
        l2=1e-9,
        batch_size=1024)

    model.fit(train, verbose=True)
    # rmse = rmse_score(model, test)
    p, r = precision_recall_score(model, test, train)

"""Shared data loading for the Anime Industry Analytics site.

Rebuilds the exact analysis frame used in the Kaggle notebook (cleand2.ipynb)
from the published UI dataset, so every number shown on the site is the same
number that appears in the project book — balanced_score included.
"""

import ast
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
MODELS = os.path.join(ROOT, 'models')
SITE = os.path.join(ROOT, 'site')

LIST_COLS = ['genres', 'themes', 'studios', 'producers', 'demographics']

# balanced_score = 0.40 score + 0.25 members + 0.20 favorites + 0.15 popularity
# (project book, Table 2.2)
BSCORE_WEIGHTS = {'score_pct': 0.40, 'members_pct': 0.25,
                  'favorites_pct': 0.20, 'popularity_pct': 0.15}


def parse_list_col(x):
    if isinstance(x, list):
        return x
    if not isinstance(x, str):
        return []
    try:
        v = ast.literal_eval(x)
        return v if isinstance(v, list) else []
    except (ValueError, SyntaxError):
        return []


def _pct_rank(series):
    """Fractional percentile rank in (0, 1] — higher value, higher percentile."""
    return pd.Series(rankdata(series, method='average') / len(series), index=series.index)


def load_frame():
    """The 8,227-row working dataset with balanced_score recomputed from the book's formula."""
    df = pd.read_csv(os.path.join(DATA, 'df_clean_ui_10_9_2026_latest.csv'), encoding='utf-8-sig')
    for c in LIST_COLS:
        df[c] = df[c].apply(parse_list_col)

    # members and favorites are power-law distributed; log1p before ranking
    df['members_pct'] = _pct_rank(np.log1p(df['members']))
    df['favorites_pct'] = _pct_rank(np.log1p(df['favorites']))
    df['score_pct'] = _pct_rank(df['score'])
    # popularity is a rank: lower number = more popular, so negate before ranking
    df['popularity_pct'] = _pct_rank(-df['popularity'])
    df['balanced_score'] = sum(df[c] * w for c, w in BSCORE_WEIGHTS.items())

    df['year'] = df['year'].astype('float64')
    return df


def modeling_split(df, train_end=2022, val_end=2023):
    """Chronological split used for the success classifier (notebook cell 162)."""
    dm = df[(df['year'] >= 2010) & (df['status'] == 'Finished Airing')].copy().reset_index(drop=True)
    train = dm[dm['year'] <= train_end].copy()
    val = dm[(dm['year'] > train_end) & (dm['year'] <= val_end)].copy()
    test = dm[dm['year'] > val_end].copy()
    threshold = train['balanced_score'].quantile(0.75)
    for s in (train, val, test):
        s['is_success_b'] = (s['balanced_score'] >= threshold).astype(int)
    return dm, train, val, test, threshold

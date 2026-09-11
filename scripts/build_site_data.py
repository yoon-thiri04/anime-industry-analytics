"""Precompute everything the static pages of the site display.

Descriptive mining, association rules, storyline clusters, model evaluation,
global SHAP and the Holt-Winters forecasts are all deterministic given the
dataset and the saved models, so they are computed once here and written to
site/anime_data.js. Only the two interactive predictive tools — the
recommender and the success lab — call the live API at request time.

    python3 scripts/build_site_data.py
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

import feature_builder  # noqa: E402
from common import DATA, MODELS, SITE, load_frame, modeling_split  # noqa: E402

feature_builder.register()

import joblib  # noqa: E402

T0 = time.time()


def step(msg):
    print(f'[{time.time() - T0:6.1f}s] {msg}', flush=True)


def r(x, n=4):
    """JSON-safe rounding — numpy scalars and NaN included."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), n)


OUT = {}

# ══════════════════════════════════════════════════════════════════
# 1 — DATASET & DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════════
step('Loading working dataset')
df = load_frame()
dm, train_df, val_df, test_df, SUCCESS_THRESHOLD = modeling_split(df)

years = df['year'].dropna().astype(int)
OUT['GLOB'] = {
    'raw_records': 19547,          # Jikan extraction before filtering (book §2.1)
    'raw_attributes': 33,
    'records': int(len(df)),
    'attributes': int(len(df.columns) - 5),   # minus the 5 derived percentile columns
    'year_min': int(years.min()),
    'year_max': int(years.max()),
    'n_genres': int(df['genres'].explode().nunique()),
    'n_themes': int(df['themes'].explode().nunique()),
    'n_studios': int(df['studios'].explode().nunique()),
    'n_producers': int(df['producers'].explode().nunique()),
    'n_members': int(df['members'].sum()),
    'success_threshold': r(SUCCESS_THRESHOLD),
    'bscore_mean': r(df['balanced_score'].mean(), 3),
    'bscore_std': r(df['balanced_score'].std(), 3),
    'bscore_corr_score': r(df['balanced_score'].corr(df['score']), 3),
}

step('Distributions')


def hist(series, bins=40, log=False):
    v = pd.to_numeric(series, errors='coerce').dropna().values
    if log:
        v = np.log1p(v)
    counts, edges = np.histogram(v, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    return {'x': [r(c, 3) for c in centres], 'y': [int(c) for c in counts]}


OUT['DIST'] = {
    'score': hist(df['score'], 40),
    'rank': hist(df['rank'], 40),
    'members_raw': hist(df['members'], 40),
    'members_log': hist(df['members'], 40, log=True),
    'scored_by_raw': hist(df['scored_by'], 40),
    'scored_by_log': hist(df['scored_by'], 40, log=True),
    'favorites_raw': hist(df['favorites'], 40),
    'favorites_log': hist(df['favorites'], 40, log=True),
    'balanced': hist(df['balanced_score'], 40),
}

step('Format structure')
fmt = (df.groupby('type')
         .agg(n=('mal_id', 'count'),
              avg_episodes=('episodes', 'mean'),
              med_episodes=('episodes', 'median'),
              avg_duration=('duration_minutes', 'mean'),
              med_duration=('duration_minutes', 'median'),
              avg_score=('score', 'mean'),
              avg_balanced=('balanced_score', 'mean'))
         .reset_index()
         .sort_values('n', ascending=False))
OUT['FORMAT'] = [{'type': row['type'], 'n': int(row['n']),
                  'avg_episodes': r(row['avg_episodes'], 1), 'med_episodes': r(row['med_episodes'], 1),
                  'avg_duration': r(row['avg_duration'], 1), 'med_duration': r(row['med_duration'], 1),
                  'avg_score': r(row['avg_score'], 2), 'avg_balanced': r(row['avg_balanced'], 3)}
                 for _, row in fmt.iterrows()]

step('Genre release trend')
d_ts = df[(df['year'] >= 2010) & (df['year'] <= 2025)].copy()
genre_year = d_ts.explode('genres').groupby(['year', 'genres']).size().reset_index(name='count')
top10_genres = genre_year.groupby('genres')['count'].sum().nlargest(10).index.tolist()
trend_years = list(range(2010, 2026))
OUT['GENRE_TREND'] = {
    'years': trend_years,
    'series': {g: [int(genre_year[(genre_year['genres'] == g) & (genre_year['year'] == y)]['count'].sum())
                   for y in trend_years] for g in top10_genres},
}

step('Season / source / demographic breakdowns')


def explode_counts(col, n=15):
    vc = df[col].explode().dropna().value_counts().head(n)
    ex = df.explode(col)
    agg = ex[ex[col].isin(vc.index)].groupby(col).agg(
        avg_balanced=('balanced_score', 'mean'),
        avg_score=('score', 'mean'),
        med_members=('members', 'median')).reindex(vc.index)
    return [{'name': str(k), 'n': int(v),
             'avg_balanced': r(agg.loc[k, 'avg_balanced'], 3),
             'avg_score': r(agg.loc[k, 'avg_score'], 2),
             'med_members': int(agg.loc[k, 'med_members'])} for k, v in vc.items()]


def flat_counts(col, n=20, min_n=1):
    g = df.groupby(col).agg(n=('mal_id', 'count'), avg_balanced=('balanced_score', 'mean'),
                            avg_score=('score', 'mean'), med_members=('members', 'median'))
    g = g[g['n'] >= min_n].sort_values('n', ascending=False).head(n)
    return [{'name': str(k), 'n': int(row['n']), 'avg_balanced': r(row['avg_balanced'], 3),
             'avg_score': r(row['avg_score'], 2), 'med_members': int(row['med_members'])}
            for k, row in g.iterrows()]


OUT['TOP'] = {
    'genres': explode_counts('genres', 22),
    'themes': explode_counts('themes', 20),
    'studios': explode_counts('studios', 20),
    'producers': explode_counts('producers', 20),
    'demographics': explode_counts('demographics', 8),
    'sources': flat_counts('source', 14),
    'seasons': flat_counts('season', 4),
    'ratings': flat_counts('rating', 6),
    'broadcast_days': flat_counts('broadcast_day', 7),
}

# Studio benchmark: track record only counts where there is enough history
studio_ex = df.explode('studios')
sb = (studio_ex.groupby('studios')
      .agg(n=('mal_id', 'count'), avg_balanced=('balanced_score', 'mean'),
           avg_score=('score', 'mean'), med_members=('members', 'median'))
      .query('n >= 10').sort_values('avg_balanced', ascending=False))
OUT['STUDIO_BENCH'] = [{'name': str(k), 'n': int(v['n']), 'avg_balanced': r(v['avg_balanced'], 3),
                        'avg_score': r(v['avg_score'], 2), 'med_members': int(v['med_members'])}
                       for k, v in sb.head(25).iterrows()]

step('balanced_score re-ranking examples')
d = df.copy()
d['score_rank'] = d['score'].rank(ascending=False)
d['balanced_rank'] = d['balanced_score'].rank(ascending=False)
d['rank_shift'] = d['balanced_rank'] - d['score_rank']


def shift_rows(sub):
    return [{'title': row['title'], 'score': r(row['score'], 2), 'members': int(row['members']),
             'popularity': int(row['popularity']), 'favorites': int(row['favorites']),
             'balanced': r(row['balanced_score'], 3), 'image': row['image_url']}
            for _, row in sub.iterrows()]


OUT['BALANCED'] = {
    'weights': [{'k': 'score', 'w': 40, 'what': 'Critical / audience reception', 'how': 'Percentile rank among rated titles'},
                {'k': 'members', 'w': 25, 'what': 'Total reach — everyone who added the title to a list', 'how': 'Percentile rank of log1p(members)'},
                {'k': 'favorites', 'w': 20, 'what': 'Depth of fan enthusiasm', 'how': 'Percentile rank of log1p(favorites)'},
                {'k': 'popularity', 'w': 15, 'what': "MAL's own popularity rank", 'how': 'Percentile rank of −popularity'}],
    'demoted': shift_rows(d.sort_values('rank_shift', ascending=False).head(6)),
    'promoted': shift_rows(d.sort_values('rank_shift', ascending=True).head(6)),
    'corr': r(d['balanced_score'].corr(d['score']), 3),
}

# Correlation matrix over the engagement signals
corr_cols = ['score', 'scored_by', 'rank', 'popularity', 'members', 'favorites',
             'episodes', 'duration_minutes', 'year', 'balanced_score']
cm = df[corr_cols].astype(float).corr().round(3)
OUT['CORR'] = {'labels': corr_cols, 'z': [[r(v, 3) for v in row] for row in cm.values]}

# ══════════════════════════════════════════════════════════════════
# 2 — ASSOCIATION RULE MINING (FP-Growth)
# ══════════════════════════════════════════════════════════════════
step('FP-Growth association rules')
from mlxtend.frequent_patterns import association_rules, fpgrowth  # noqa: E402
from mlxtend.preprocessing import TransactionEncoder  # noqa: E402

hi_cut, mid_cut = d['balanced_score'].quantile(0.75), d['balanced_score'].quantile(0.40)


def bscore_bin(s):
    if s >= hi_cut:
        return 'Balanced_High'
    if s >= mid_cut:
        return 'Balanced_Mid'
    return 'Balanced_Low'


d['bscore_bin'] = d['balanced_score'].apply(bscore_bin)

theme_counts = d['themes'].explode().value_counts()
top_themes = set(theme_counts[theme_counts >= 15].index)
d['themes_clean'] = d['themes'].apply(lambda lst: [t for t in lst if t in top_themes])

producer_counts = d['producers'].explode().value_counts()
top_producers = set(producer_counts[producer_counts >= 20].index)
d['producers_clean'] = d['producers'].apply(lambda lst: [p for p in lst if p in top_producers])

studio_counts = d['studios'].explode().value_counts()
top_studios_rules = set(studio_counts[studio_counts >= 10].index)
d['studios_clean'] = d['studios'].apply(lambda lst: [s for s in lst if s in top_studios_rules])

OUT['BSCORE_BINS'] = {'hi_cut': r(hi_cut), 'mid_cut': r(mid_cut),
                      'counts': {k: int(v) for k, v in d['bscore_bin'].value_counts().items()}}


def mine(transactions, min_support, min_conf=0.30):
    te = TransactionEncoder()
    basket = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)
    freq = fpgrowth(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(freq, metric='confidence', min_threshold=min_conf)
    rules = rules[rules['lift'] > 1.0].sort_values(['lift', 'confidence'], ascending=False).reset_index(drop=True)
    return basket, freq, rules


def pretty(itemset):
    return sorted(str(i).replace('Genre_', '').replace('Theme_', '')
                  .replace('Producer_', '').replace('Studio_', '') for i in itemset)


def rule_rows(rules, limit=12):
    return [{'a': pretty(row['antecedents']), 'c': pretty(row['consequents']),
             'support': r(row['support']), 'confidence': r(row['confidence']),
             'lift': r(row['lift'])} for _, row in rules.head(limit).iterrows()]


# 2a — genre ↔ theme co-occurrence
txn_gt = d.apply(lambda row: [f'Genre_{g}' for g in row['genres']] +
                             [f'Theme_{t}' for t in row['themes_clean']], axis=1).tolist()
_, freq_gt, rules_gt_all = mine(txn_gt, 0.015)
only_gt = (rules_gt_all['antecedents'].apply(lambda x: all(str(i).startswith(('Genre_', 'Theme_')) for i in x)) &
           rules_gt_all['consequents'].apply(lambda x: all(str(i).startswith(('Genre_', 'Theme_')) for i in x)))
rules_gt = rules_gt_all[only_gt].sort_values('lift', ascending=False)

# 2b — producer + genre/theme → Balanced_High
txn_prod = d.apply(lambda row: [f'Genre_{g}' for g in row['genres']] +
                               [f'Theme_{t}' for t in row['themes_clean']] +
                               [f'Producer_{p}' for p in row['producers_clean']] +
                               [row['bscore_bin']], axis=1).tolist()
_, freq_prod, rules_prod = mine(txn_prod, 0.01)
prod_high = rules_prod[
    rules_prod['antecedents'].apply(lambda x: any(str(i).startswith('Producer_') for i in x) and
                                    any(str(i).startswith(('Genre_', 'Theme_')) for i in x)) &
    rules_prod['consequents'].apply(lambda x: set(map(str, x)) == {'Balanced_High'})
].sort_values(['lift', 'confidence', 'support'], ascending=False)

# 2c — studio + genre/theme → Balanced_High
txn_studio = d.apply(lambda row: [f'Genre_{g}' for g in row['genres']] +
                                 [f'Theme_{t}' for t in row['themes_clean']] +
                                 [f'Studio_{s}' for s in row['studios_clean']] +
                                 [row['bscore_bin']], axis=1).tolist()
_, freq_studio, rules_studio = mine(txn_studio, 0.005)
studio_high = rules_studio[
    rules_studio['antecedents'].apply(lambda x: any(str(i).startswith('Studio_') for i in x) and
                                      any(str(i).startswith(('Genre_', 'Theme_')) for i in x)) &
    rules_studio['consequents'].apply(lambda x: set(map(str, x)) == {'Balanced_High'})
].sort_values(['lift', 'confidence', 'support'], ascending=False)

# 2d — producer → genre/theme specialisation (no outcome token)
prod_spec = rules_prod[
    rules_prod['antecedents'].apply(lambda x: any(str(i).startswith('Producer_') for i in x)) &
    rules_prod['consequents'].apply(lambda x: all(str(i).startswith(('Genre_', 'Theme_')) for i in x))
].sort_values('lift', ascending=False)

OUT['RULES'] = {
    'genre_theme': rule_rows(rules_gt, 16),
    'producer_high': rule_rows(prod_high, 12),
    'studio_high': rule_rows(studio_high, 12),
    'producer_spec': rule_rows(prod_spec, 12),
    # a wider slice of each space, so the support/confidence scatter shows the
    # whole rule cloud rather than only the handful rendered as cards
    'cloud': {'genre_theme': rule_rows(rules_gt, 160),
              'producer_high': rule_rows(prod_high, 160),
              'studio_high': rule_rows(studio_high, 160),
              'producer_spec': rule_rows(prod_spec, 160)},
    'params': {'gt': {'support': 0.015, 'conf': 0.30, 'n_freq': int(len(freq_gt)), 'n_rules': int(len(rules_gt))},
               'producer': {'support': 0.01, 'conf': 0.30, 'n_freq': int(len(freq_prod)), 'n_rules': int(len(rules_prod))},
               'studio': {'support': 0.005, 'conf': 0.30, 'n_freq': int(len(freq_studio)), 'n_rules': int(len(rules_studio))}},
}
step(f'  rules — gt {len(rules_gt)}, producer→High {len(prod_high)}, studio→High {len(studio_high)}')

step('Producer × studio collaborations')
dn2 = d[(d['producers'].apply(len) >= 1) & (d['studios'].apply(len) >= 1)]
rows2 = [{'producer': p, 'studio': s, 'balanced_score': row['balanced_score'], 'members': row['members']}
         for _, row in dn2.iterrows() for p in set(row['producers']) for s in set(row['studios'])]
ps_collab = (pd.DataFrame(rows2).groupby(['producer', 'studio'])
             .agg(num_titles=('balanced_score', 'count'), avg_balanced=('balanced_score', 'mean'),
                  avg_members=('members', 'mean')).reset_index())
ps_collab = ps_collab[ps_collab['num_titles'] >= 5].sort_values('avg_balanced', ascending=False)

collab = []
for _, row in ps_collab.head(12).iterrows():
    pair = d[d['producers'].apply(lambda x: row['producer'] in x) &
             d['studios'].apply(lambda x: row['studio'] in x)]
    collab.append({'producer': row['producer'], 'studio': row['studio'], 'n': int(row['num_titles']),
                   'avg_balanced': r(row['avg_balanced'], 3), 'avg_members': int(row['avg_members']),
                   'titles': pair.sort_values('balanced_score', ascending=False)['title'].head(8).tolist()})
OUT['COLLAB'] = collab

# ══════════════════════════════════════════════════════════════════
# 3 — STORYLINE CLUSTERS (TF-IDF → SVD → K-Means)
# ══════════════════════════════════════════════════════════════════
step('Storyline clusters')
dt = pd.read_pickle(os.path.join(DATA, 'anime_recommendation_data.pkl'))
tfidf = joblib.load(os.path.join(MODELS, 'tfidf_vectorizer.pkl'))
svd = joblib.load(os.path.join(MODELS, 'svd.pkl'))

X_tfidf = tfidf.transform(dt['synopsis_clean'])
X_svd = svd.transform(X_tfidf)
terms = tfidf.get_feature_names_out()

from sklearn.preprocessing import normalize  # noqa: E402
X_cluster = normalize(X_svd[:, :30])

# image URLs live in the UI csv, not in the recommendation frame
img_by_id = dict(zip(df['mal_id'], df['image_url']))

clusters = []
for c in sorted(dt['nlp_cluster'].unique()):
    idx = dt.index[dt['nlp_cluster'] == c]
    mean_vec = np.asarray(X_tfidf[idx].mean(axis=0)).ravel()
    top_terms = [str(terms[i]) for i in mean_vec.argsort()[::-1][:8]]
    top_genres = dt.loc[idx, 'genres'].explode().value_counts().head(4)
    ex = dt.loc[idx].sort_values('balanced_score', ascending=False).head(6)
    clusters.append({
        'id': int(c), 'n': int(len(idx)), 'terms': top_terms,
        'genres': [{'name': str(k), 'n': int(v)} for k, v in top_genres.items()],
        'avg_balanced': r(dt.loc[idx, 'balanced_score'].mean(), 3),
        'examples': [{'mal_id': int(row['mal_id']), 'title': row['title'],
                      'image': img_by_id.get(row['mal_id']), 'balanced': r(row['balanced_score'], 3)}
                     for _, row in ex.iterrows()],
    })

from scipy.stats import chi2_contingency  # noqa: E402
dtg = dt[['nlp_cluster', 'genres']].explode('genres').reset_index(drop=True)
dtg['genres'] = dtg['genres'].fillna('Unknown')
ctab = pd.crosstab(dtg['nlp_cluster'], dtg['genres'])
chi2, p, dof, _ = chi2_contingency(ctab)
n_tab = ctab.sum().sum()
cramers_v = np.sqrt(chi2 / (n_tab * (min(ctab.shape) - 1)))

# silhouette scan, k = 8..40 — the objective choice of k from the notebook
sil_cache = os.path.join(DATA, 'silhouette_scan.json')
if os.path.exists(sil_cache):
    sil = json.load(open(sil_cache))
else:
    from sklearn.cluster import KMeans  # noqa: E402
    from sklearn.metrics import silhouette_score  # noqa: E402
    ks, scores = list(range(8, 41)), []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_k = km.fit_predict(X_cluster)
        scores.append(r(silhouette_score(X_cluster, labels_k, sample_size=3000, random_state=42), 4))
        step(f'  k={k} silhouette={scores[-1]}')
    sil = {'k': ks, 'score': scores}
    json.dump(sil, open(sil_cache, 'w'))

best_k = int(sil['k'][int(np.argmax(sil['score']))])

from sklearn.decomposition import PCA  # noqa: E402
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_cluster)
samp = np.random.RandomState(42).choice(len(dt), size=min(2600, len(dt)), replace=False)
OUT['CLUSTERS'] = {
    'k': best_k,
    'best_silhouette': max(sil['score']),
    'silhouette_curve': sil,
    'chi2': r(chi2, 1), 'p': ('< 0.001' if p < 1e-3 else r(p, 6)), 'dof': int(dof),
    'cramers_v': r(cramers_v, 3),
    'n_docs': int(len(dt)),
    'variance_100': r(float(svd.explained_variance_ratio_.sum()) * 100, 1),
    'variance_30': r(float(svd.explained_variance_ratio_[:30].sum()) * 100, 1),
    'items': clusters,
    'scatter': {'x': [r(v, 3) for v in X_pca[samp, 0]], 'y': [r(v, 3) for v in X_pca[samp, 1]],
                'c': [int(v) for v in dt['nlp_cluster'].values[samp]],
                't': [str(v) for v in dt['title'].values[samp]]},
}
step(f'  {best_k} clusters, silhouette {max(sil["score"])}, Cramer V {r(cramers_v, 3)}')

json.dump(OUT, open(os.path.join(DATA, '_partial_a.json'), 'w'))
step('Part A written')

# ══════════════════════════════════════════════════════════════════
# 4 — SUCCESS CLASSIFIER EVALUATION
# ══════════════════════════════════════════════════════════════════
step('Rebuilding the held-out feature matrices')
fb = joblib.load(os.path.join(MODELS, 'feature_builder.pkl'))
scaler = joblib.load(os.path.join(MODELS, 'standard_scaler.pkl'))

X_train = fb.transform(train_df)
X_test = fb.transform(test_df)
y_train = train_df['is_success_b'].values
y_test = test_df['is_success_b'].values
Xte_s = scaler.transform(X_test)

MODEL_FILES = [
    ('Logistic Regression', 'logistic_regression.pkl', True),
    ('Decision Tree', 'decision_tree.pkl', False),
    ('Random Forest', 'random_forest.pkl', False),
    ('XGBoost', 'xgboost.pkl', False),
    ('XGBoost (tuned)', 'best_xgb.pkl', False),
]
loaded = {name: joblib.load(os.path.join(MODELS, f)) for name, f, _ in MODEL_FILES}

from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,  # noqa: E402
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

step('Evaluating 5 models on the held-out (>= 2024) test split')
metrics, roc, pr, cms = [], {}, {}, {}
for name, _f, scaled in MODEL_FILES:
    model = loaded[name]
    Xte = Xte_s if scaled else X_test
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]
    metrics.append({'model': name,
                    'accuracy': r(accuracy_score(y_test, y_pred)),
                    'precision': r(precision_score(y_test, y_pred, zero_division=0)),
                    'recall': r(recall_score(y_test, y_pred, zero_division=0)),
                    'f1': r(f1_score(y_test, y_pred, zero_division=0)),
                    'roc_auc': r(roc_auc_score(y_test, y_prob)),
                    'avg_precision': r(average_precision_score(y_test, y_prob))})
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    keep = np.linspace(0, len(fpr) - 1, min(220, len(fpr))).astype(int)
    roc[name] = {'fpr': [r(v, 4) for v in fpr[keep]], 'tpr': [r(v, 4) for v in tpr[keep]],
                 'auc': r(roc_auc_score(y_test, y_prob))}
    p_, rc_, _ = precision_recall_curve(y_test, y_prob)
    keep = np.linspace(0, len(p_) - 1, min(220, len(p_))).astype(int)
    pr[name] = {'precision': [r(v, 4) for v in p_[keep]], 'recall': [r(v, 4) for v in rc_[keep]],
                'ap': r(average_precision_score(y_test, y_prob))}
    cms[name] = [[int(v) for v in row] for row in confusion_matrix(y_test, y_pred)]
    step(f'  {name:22s} acc={metrics[-1]["accuracy"]:.3f} f1={metrics[-1]["f1"]:.3f} auc={metrics[-1]["roc_auc"]:.3f}')

OUT['MODELS'] = {
    'metrics': metrics, 'roc': roc, 'pr': pr, 'cm': cms,
    'n_features': int(X_train.shape[1]),
    'threshold': r(SUCCESS_THRESHOLD),
    'splits': [
        {'name': 'Train', 'range': '≤ 2022', 'n': int(len(train_df)),
         'pos': int(train_df['is_success_b'].sum()), 'rate': r(train_df['is_success_b'].mean(), 3)},
        {'name': 'Validation', 'range': '2023', 'n': int(len(val_df)),
         'pos': int(val_df['is_success_b'].sum()), 'rate': r(val_df['is_success_b'].mean(), 3)},
        {'name': 'Test (held out)', 'range': '≥ 2024', 'n': int(len(test_df)),
         'pos': int(test_df['is_success_b'].sum()), 'rate': r(test_df['is_success_b'].mean(), 3)},
    ],
    'feature_groups': [
        {'name': 'Genre one-hot', 'n': len(fb.genre_vocab), 'detail': 'top genres in the training split'},
        {'name': 'Theme one-hot', 'n': len(fb.theme_vocab), 'detail': 'top themes in the training split'},
        {'name': 'Demographic', 'n': len(fb.demo_vocab), 'detail': 'Shounen, Seinen, Shoujo, Josei, Kids'},
        {'name': 'Source material', 'n': len(fb.source_vocab), 'detail': 'manga, light novel, original, game …'},
        {'name': 'Format / rating / season', 'n': len(fb.type_vocab) + len(fb.rating_vocab) + len(fb.season_vocab),
         'detail': 'TV, Movie, ONA, OVA · age rating · airing season'},
        {'name': 'Historical track record', 'n': 2, 'detail': 'studio and source mean balanced_score, fit on train only'},
        {'name': 'Production metadata', 'n': 7, 'detail': 'episodes, duration, year, list lengths, tag counts'},
        {'name': 'Storyline embedding', 'n': fb.svd_dim, 'detail': 'TF-IDF → TruncatedSVD on the synopsis'},
    ],
}

# ══════════════════════════════════════════════════════════════════
# 5 — GLOBAL SHAP
# ══════════════════════════════════════════════════════════════════
step('Global SHAP over the test split')
import shap  # noqa: E402

best_xgb = loaded['XGBoost (tuned)']
explainer = shap.TreeExplainer(best_xgb)
sv = explainer.shap_values(X_test)
mean_abs = np.abs(sv).mean(axis=0)
order = np.argsort(-mean_abs)[:20]
feat_names = list(X_test.columns)

beeswarm = []
rs = np.random.RandomState(42)
# 300 points per feature keeps the beeswarm readable and light enough to draw
# as plain SVG, which works everywhere — scattergl silently fails without WebGL
samp_idx = rs.choice(len(X_test), size=min(300, len(X_test)), replace=False)
for i in order:
    col = X_test.iloc[samp_idx, i].values.astype(float)
    lo, hi = np.nanpercentile(col, 5), np.nanpercentile(col, 95)
    norm = np.zeros_like(col) if hi <= lo else np.clip((col - lo) / (hi - lo), 0, 1)
    beeswarm.append({'feature': feat_names[i],
                     'shap': [r(v, 4) for v in sv[samp_idx, i]],
                     'norm': [r(v, 3) for v in norm],
                     'raw': [r(v, 3) for v in col]})

OUT['SHAP'] = {
    'base_value': r(float(explainer.expected_value), 4),
    'bar': [{'feature': feat_names[i], 'mean_abs': r(mean_abs[i])} for i in order],
    'beeswarm': beeswarm,
    'n_test': int(len(X_test)),
}

# XGBoost's own gain-based importance, for contrast with SHAP
booster_imp = best_xgb.get_booster().get_score(importance_type='gain')
gain = sorted(((k, v) for k, v in booster_imp.items()), key=lambda kv: -kv[1])[:20]
OUT['GAIN'] = [{'feature': k, 'gain': r(v, 2)} for k, v in gain]

# ══════════════════════════════════════════════════════════════════
# 6 — HOLT-WINTERS FORECASTS
# ══════════════════════════════════════════════════════════════════
step('Holt-Winters forecasts')
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # noqa: E402


def holt_block(series_by_genre, genres, label):
    hist_years = list(range(2010, 2026))
    fut_years = [2026, 2027, 2028]
    out = {'years': hist_years, 'future_years': fut_years, 'label': label,
           'history': {}, 'forecast': {}, 'backtest': [], 'backtest_curve': {}}
    for g in genres:
        s = series_by_genre[g]
        out['history'][g] = [r(v, 1) for v in s.values]

        train_ts, test_ts = s.loc[:2022], s.loc[2023:2025]
        bt = ExponentialSmoothing(train_ts, trend='add', damped_trend=True).fit()
        fc_bt = bt.forecast(3)
        mae = float(np.mean(np.abs(fc_bt.values - test_ts.values)))
        rmse = float(np.sqrt(np.mean((fc_bt.values - test_ts.values) ** 2)))
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = float(np.mean(np.abs((test_ts.values - fc_bt.values) /
                                        np.where(test_ts.values == 0, np.nan, test_ts.values))) * 100)
        out['backtest'].append({'genre': g, 'mae': r(mae, 1), 'rmse': r(rmse, 1), 'mape': r(mape, 2),
                                'actual': [r(v, 1) for v in test_ts.values],
                                'forecast': [r(v, 1) for v in fc_bt.values]})
        out['backtest_curve'][g] = {'years': [2023, 2024, 2025],
                                    'forecast': [r(v, 1) for v in fc_bt.values],
                                    'actual': [r(v, 1) for v in test_ts.values]}

        full = ExponentialSmoothing(s, trend='add', damped_trend=True).fit()
        fc = full.forecast(3)
        out['forecast'][g] = [r(v, 1) for v in fc.values]
    out['backtest'].sort(key=lambda x: x['mae'])
    return out


# 6a — release volume
gy = d_ts.explode('genres').groupby(['year', 'genres']).size().reset_index(name='count')
rel_genres = gy.groupby('genres')['count'].sum().nlargest(6).index.tolist()
rel_series = {g: gy[gy['genres'] == g].set_index('year')['count']
              .reindex(range(2010, 2026), fill_value=0).astype(float) for g in rel_genres}
OUT['FORECAST_RELEASE'] = holt_block(rel_series, rel_genres, 'Titles released')

# 6b — audience demand (median members)
gm = (d_ts.explode('genres').dropna(subset=['genres', 'members'])
      .groupby(['year', 'genres'])['members'].median().reset_index(name='median_members'))
dem_genres = gm.groupby('genres')['median_members'].median().nlargest(6).index.tolist()
dem_series = {g: gm[gm['genres'] == g].set_index('year')['median_members']
              .reindex(range(2010, 2026)).interpolate().bfill().ffill().astype(float) for g in dem_genres}
OUT['FORECAST_DEMAND'] = holt_block(dem_series, dem_genres, 'Median MAL members')

# ══════════════════════════════════════════════════════════════════
# 7 — HOME-PAGE HIGHLIGHTS
# ══════════════════════════════════════════════════════════════════
step('Home highlights')
top_titles = df.sort_values('balanced_score', ascending=False).head(14)
OUT['SPOTLIGHT'] = [{'mal_id': int(row['mal_id']), 'title': row['title'],
                     'english': None if pd.isna(row['title_english']) else row['title_english'],
                     'image': row['image_url'], 'year': None if pd.isna(row['year']) else int(row['year']),
                     'type': row['type'], 'score': r(row['score'], 2),
                     'balanced': r(row['balanced_score'], 3), 'genres': row['genres'][:3]}
                    for _, row in top_titles.iterrows()]

# a wall of real cover art for the landing hero, drawn from the highest-reach titles
wall = (df[df['image_url'].notna()].sort_values('members', ascending=False)
        .head(160).sample(frac=1.0, random_state=7))
OUT['POSTERS'] = wall['image_url'].tolist()

best_row = max(metrics, key=lambda m: m['f1'])
OUT['HOME'] = {
    'titles': int(len(df)),
    'studios': OUT['GLOB']['n_studios'],
    'producers': OUT['GLOB']['n_producers'],
    'years_span': f"{OUT['GLOB']['year_min']}–{OUT['GLOB']['year_max']}",
    'clusters': OUT['CLUSTERS']['k'],
    'features': OUT['MODELS']['n_features'],
    'best_model': best_row['model'],
    'best_auc': best_row['roc_auc'],
    'best_acc': best_row['accuracy'],
    'rules_mined': OUT['RULES']['params']['studio']['n_rules'],
}

# ══════════════════════════════════════════════════════════════════
step('Writing site/anime_data.js')
payload = json.dumps(OUT, separators=(',', ':'), allow_nan=False)
with open(os.path.join(SITE, 'anime_data.js'), 'w') as f:
    f.write('/* Generated by scripts/build_site_data.py — do not edit by hand. */\n')
    f.write('const ANIME_DATA = ' + payload + ';\n')
size_mb = os.path.getsize(os.path.join(SITE, 'anime_data.js')) / 1e6
step(f'Done — {size_mb:.2f} MB')

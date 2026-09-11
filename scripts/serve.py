"""Anime Industry Analytics — static site + live predictive API.

    python3 scripts/serve.py            # http://localhost:8100

Descriptive mining is precomputed into site/anime_data.js by
scripts/build_site_data.py. This server exists for the two things that must
run the real model files on demand:

  POST /api/predict    FeatureBuilder -> 5 trained classifiers + SHAP TreeExplainer
  GET  /api/recommend  hybrid 0.6 storyline + 0.4 genre/theme cosine similarity
"""

import os
import sys

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import feature_builder  # noqa: E402
from common import DATA, MODELS, SITE, load_frame  # noqa: E402

feature_builder.register()

import joblib  # noqa: E402
import shap  # noqa: E402

app = Flask(__name__, static_folder=None)

# ══════════════════════════════════════════════════════════════════
# Load everything once at start-up
# ══════════════════════════════════════════════════════════════════
print('Loading catalogue …', flush=True)
DF = load_frame()
DF['year_i'] = DF['year'].fillna(0).astype(int)
DF['_search'] = (DF['title'].fillna('') + ' ¦ ' + DF['title_english'].fillna('') + ' ¦ ' +
                 DF['title_japanese'].fillna('') + ' ¦ ' + DF['title_synonym'].fillna('')).str.lower()

print('Loading recommender …', flush=True)
DT = pd.read_pickle(os.path.join(DATA, 'anime_recommendation_data.pkl'))
X_STORY = np.load(os.path.join(DATA, 'X_recommend.npy'))          # already L2-normalised
GT = np.load(os.path.join(DATA, 'GT.npy')).astype(np.float32)
_gt_norm = np.linalg.norm(GT, axis=1, keepdims=True)
GT_N = GT / np.where(_gt_norm == 0, 1.0, _gt_norm)                # cosine via dot product
X_STORY = X_STORY.astype(np.float32)

MLB_G = joblib.load(os.path.join(MODELS, 'genre_mlb.pkl'))
MLB_T = joblib.load(os.path.join(MODELS, 'theme_mlb.pkl'))

DT_SEARCH = pd.DataFrame({
    'title': DT['title'].fillna('').str.lower(),
    'english': DT['title_english'].fillna('').str.lower(),
    'japanese': DT['title_japanese'].fillna('').str.lower(),
    'synonym': DT['title_synonym'].fillna('').astype(str).str.lower(),
})

# UI-only columns (artwork, trailers) live in the csv, not the recommendation frame
EXTRA = DF.set_index('mal_id')[['image_url', 'url', 'trailer_embed_url']]
IMG = EXTRA['image_url'].to_dict()
MAL_URL = EXTRA['url'].to_dict()
TRAILER = EXTRA['trailer_embed_url'].to_dict()

print('Loading classifiers …', flush=True)
FB = joblib.load(os.path.join(MODELS, 'feature_builder.pkl'))
SCALER = joblib.load(os.path.join(MODELS, 'standard_scaler.pkl'))
CLASSIFIERS = [
    ('Logistic Regression', joblib.load(os.path.join(MODELS, 'logistic_regression.pkl')), True, 0.5),
    ('Decision Tree', joblib.load(os.path.join(MODELS, 'decision_tree.pkl')), False, 0.5),
    ('Random Forest', joblib.load(os.path.join(MODELS, 'random_forest.pkl')), False, 0.5),
    ('XGBoost', joblib.load(os.path.join(MODELS, 'xgboost.pkl')), False, 0.5),
    ('XGBoost (tuned)', joblib.load(os.path.join(MODELS, 'best_xgb.pkl')), False, 0.5),
]
BEST = dict((n, m) for n, m, _s, _t in CLASSIFIERS)['XGBoost (tuned)']
EXPLAINER = shap.TreeExplainer(BEST)
print('Ready.', flush=True)


def nn(v):
    """NaN -> None so jsonify produces valid JSON."""
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, float) and pd.isna(v):
        return None
    return v


def card(row):
    return {
        'mal_id': int(row['mal_id']),
        'title': row['title'],
        'english': None if pd.isna(row.get('title_english')) else row['title_english'],
        'japanese': None if pd.isna(row.get('title_japanese')) else row['title_japanese'],
        'image': None if pd.isna(row.get('image_url')) else row['image_url'],
        'type': row['type'],
        'source': row['source'],
        'year': None if pd.isna(row['year']) else int(row['year']),
        'season': None if pd.isna(row.get('season')) else row['season'],
        'episodes': nn(row.get('episodes')),
        'duration': nn(row.get('duration_minutes')),
        'score': nn(row.get('score')),
        'members': nn(row.get('members')),
        'favorites': nn(row.get('favorites')),
        'popularity': nn(row.get('popularity')),
        'rank': nn(row.get('rank')),
        'balanced': round(float(row['balanced_score']), 4),
        'genres': list(row['genres']),
        'themes': list(row['themes']),
        'studios': list(row['studios']),
        'producers': list(row['producers']),
        'demographics': list(row['demographics']),
        'rating': None if pd.isna(row.get('rating')) else row['rating'],
        'synopsis': None if pd.isna(row.get('synopsis_clean')) else row['synopsis_clean'],
        'url': None if pd.isna(row.get('url')) else row['url'],
        'trailer': None if pd.isna(row.get('trailer_embed_url')) else row['trailer_embed_url'],
    }


# ══════════════════════════════════════════════════════════════════
# Metadata + catalogue
# ══════════════════════════════════════════════════════════════════
@app.get('/api/meta')
def meta():
    def opts(col, min_n=1):
        vc = DF[col].explode().dropna().value_counts()
        return [{'name': str(k), 'n': int(v)} for k, v in vc[vc >= min_n].items()]

    return jsonify({
        'genres': opts('genres'),
        'themes': opts('themes'),
        'studios': opts('studios', 3),
        'producers': opts('producers', 3),
        'demographics': opts('demographics'),
        'types': [{'name': str(k), 'n': int(v)} for k, v in DF['type'].value_counts().items()],
        'sources': [{'name': str(k), 'n': int(v)} for k, v in DF['source'].value_counts().items()],
        'ratings': [{'name': str(k), 'n': int(v)} for k, v in DF['rating'].value_counts().items()],
        'seasons': [{'name': str(k), 'n': int(v)} for k, v in DF['season'].value_counts().items()],
        'year_min': int(DF['year'].min()), 'year_max': int(DF['year'].max()),
        'total': int(len(DF)),
        # the classifier only understands the vocabularies it was fitted on
        'model_vocab': {
            'genres': sorted(FB.genre_vocab),
            'themes': sorted(FB.theme_vocab),
            'studios': sorted(FB.studio_vocab),
            'all_studios': sorted(FB.studio_perf.index.astype(str).tolist()),
            'demographics': sorted(FB.demo_vocab),
            'sources': sorted(FB.source_vocab),
            'types': sorted(FB.type_vocab),
            'ratings': sorted(FB.rating_vocab),
            'seasons': sorted(FB.season_vocab),
            'producers': sorted(DF['producers'].explode().dropna().value_counts().head(400).index.tolist()),
            'studio_perf': {str(k): round(float(v), 4) for k, v in FB.studio_perf.items()},
            'source_perf': {str(k): round(float(v), 4) for k, v in FB.source_perf.items()},
            'global_mean': round(float(FB.global_bscore_mean), 4),
        },
    })


SORTS = {'balanced': ('balanced_score', False), 'score': ('score', False),
         'members': ('members', False), 'favorites': ('favorites', False),
         'newest': ('year', False), 'oldest': ('year', True), 'title': ('title', True)}


@app.get('/api/catalogue')
def catalogue():
    a = request.args
    sub = DF
    q = (a.get('q') or '').strip().lower()
    if q:
        sub = sub[sub['_search'].str.contains(q, regex=False)]
    for field, col in (('genre', 'genres'), ('theme', 'themes'),
                       ('studio', 'studios'), ('producer', 'producers'),
                       ('demographic', 'demographics')):
        vals = [v for v in a.getlist(field) if v]
        for v in vals:
            sub = sub[sub[col].apply(lambda lst, v=v: v in lst)]
    for field, col in (('type', 'type'), ('source', 'source'), ('season', 'season'), ('rating', 'rating')):
        vals = [v for v in a.getlist(field) if v]
        if vals:
            sub = sub[sub[col].isin(vals)]
    if a.get('year_min'):
        sub = sub[sub['year_i'] >= int(a['year_min'])]
    if a.get('year_max'):
        sub = sub[sub['year_i'] <= int(a['year_max'])]
    if a.get('min_score'):
        sub = sub[sub['score'] >= float(a['min_score'])]

    col, asc = SORTS.get(a.get('sort', 'balanced'), SORTS['balanced'])
    sub = sub.sort_values(col, ascending=asc, na_position='last')

    page = max(1, int(a.get('page', 1)))
    per = min(60, max(1, int(a.get('per', 24))))
    total = len(sub)
    rows = sub.iloc[(page - 1) * per: page * per]
    return jsonify({'total': int(total), 'page': page, 'per': per,
                    'pages': int(np.ceil(total / per)) if total else 0,
                    'results': [card(r) for _, r in rows.iterrows()]})


@app.get('/api/anime/<int:mal_id>')
def anime(mal_id):
    row = DF[DF['mal_id'] == mal_id]
    if row.empty:
        return jsonify({'error': 'not found'}), 404
    out = card(row.iloc[0])
    in_reco = DT.index[DT['mal_id'] == mal_id]
    out['recommendable'] = bool(len(in_reco))
    if len(in_reco):
        out['nlp_cluster'] = int(DT.loc[in_reco[0], 'nlp_cluster'])
    return jsonify(out)


@app.get('/api/search')
def search():
    """Autocomplete across primary, English, Japanese and synonym titles."""
    q = (request.args.get('q') or '').strip().lower()
    if len(q) < 2:
        return jsonify({'results': []})
    hit = DT_SEARCH.apply(lambda col: col.str.contains(q, regex=False)).any(axis=1)
    idx = DT.index[hit]
    if not len(idx):
        return jsonify({'results': [], 'message': f'No title found containing "{q}".'})
    sub = DT.loc[idx].sort_values('balanced_score', ascending=False).head(12)
    return jsonify({'results': [{
        'mal_id': int(r['mal_id']), 'title': r['title'],
        'english': None if pd.isna(r['title_english']) else r['title_english'],
        'japanese': None if pd.isna(r['title_japanese']) else r['title_japanese'],
        'year': None if pd.isna(r['year']) else int(r['year']), 'type': r['type'],
        'image': IMG.get(int(r['mal_id'])), 'score': nn(r['score']),
        'balanced': round(float(r['balanced_score']), 4),
    } for _, r in sub.iterrows()]})


# ══════════════════════════════════════════════════════════════════
# Hybrid content-based recommender
# ══════════════════════════════════════════════════════════════════
def _minmax(v):
    lo, hi = float(v.min()), float(v.max())
    return np.zeros_like(v) if hi == lo else (v - lo) / (hi - lo)


@app.get('/api/recommend')
def recommend():
    a = request.args
    top_n = min(24, max(1, int(a.get('top_n', 8))))
    w_story = float(a.get('w_story', 0.6))
    w_genre = round(1.0 - w_story, 4)

    if a.get('mal_id'):
        idx = DT.index[DT['mal_id'] == int(a['mal_id'])]
        if not len(idx):
            return jsonify({'error': 'This title is not in the recommendation index '
                                     '(it needs a synopsis of at least 150 characters).'}), 404
        i = int(idx[0])
        matches = []
    else:
        q = (a.get('q') or '').strip().lower()
        if not q:
            return jsonify({'error': 'Provide q or mal_id'}), 400
        hit = DT_SEARCH.apply(lambda col: col.str.contains(q, regex=False)).any(axis=1)
        idx = DT.index[hit]
        if not len(idx):
            return jsonify({'error': f'No title found containing "{a.get("q")}".'}), 404
        i = int(idx[0])
        matches = [{'mal_id': int(DT.loc[j, 'mal_id']), 'title': DT.loc[j, 'title']} for j in idx[:8]]

    # rows of X_STORY are L2-normalised, so the dot product IS the cosine similarity
    story = X_STORY @ X_STORY[i]
    gt = GT_N @ GT_N[i]
    story_n, gt_n = _minmax(story), _minmax(gt)
    combined = w_story * story_n + w_genre * gt_n
    combined[i] = -1

    order = np.argsort(combined)[::-1][:top_n]
    query = DT.loc[i]
    results = []
    for j in order:
        row = DT.loc[int(j)]
        mid = int(row['mal_id'])
        shared_g = sorted(set(row['genres']) & set(query['genres']))
        shared_t = sorted(set(row['themes']) & set(query['themes']))
        results.append({
            'mal_id': mid, 'title': row['title'],
            'english': None if pd.isna(row['title_english']) else row['title_english'],
            'image': IMG.get(mid), 'url': MAL_URL.get(mid),
            'year': None if pd.isna(row['year']) else int(row['year']), 'type': row['type'],
            'score': nn(row['score']), 'members': nn(row['members']),
            'balanced': round(float(row['balanced_score']), 4),
            'genres': list(row['genres']), 'themes': list(row['themes']),
            'studios': list(row['studios']),
            'synopsis': (row['synopsis_clean'] or '')[:340],
            'story_similarity': round(float(story_n[j]), 4),
            'genre_theme_similarity': round(float(gt_n[j]), 4),
            'combined_similarity': round(float(combined[j]), 4),
            'shared_genres': shared_g, 'shared_themes': shared_t,
            'nlp_cluster': int(row['nlp_cluster']),
        })

    qid = int(query['mal_id'])
    return jsonify({
        'query': {'mal_id': qid, 'title': query['title'],
                  'english': None if pd.isna(query['title_english']) else query['title_english'],
                  'japanese': None if pd.isna(query['title_japanese']) else query['title_japanese'],
                  'image': IMG.get(qid), 'url': MAL_URL.get(qid),
                  'trailer': TRAILER.get(qid),
                  'year': None if pd.isna(query['year']) else int(query['year']),
                  'type': query['type'], 'score': nn(query['score']),
                  'members': nn(query['members']),
                  'balanced': round(float(query['balanced_score']), 4),
                  'genres': list(query['genres']), 'themes': list(query['themes']),
                  'studios': list(query['studios']),
                  'nlp_cluster': int(query['nlp_cluster']),
                  'synopsis': query['synopsis_clean']},
        'weights': {'story': w_story, 'genre_theme': w_genre},
        'other_matches': matches,
        'results': results,
    })


# ══════════════════════════════════════════════════════════════════
# Pre-release success classifier + SHAP
# ══════════════════════════════════════════════════════════════════
PRETTY = {'studio_hist_balanced': 'Studio track record (mean balanced_score)',
          'source_hist_balanced': 'Source-material track record',
          'studio_is_known': 'Studio is in the top-30 vocabulary',
          'num_studios': 'Number of studios', 'num_producers': 'Number of producers',
          'num_genres': 'Number of genres', 'num_themes': 'Number of themes',
          'episodes': 'Episode count', 'duration_minutes': 'Episode duration (min)',
          'year': 'Release year'}


def pretty_feature(name):
    if name in PRETTY:
        return PRETTY[name]
    for prefix, label in (('genre_', 'Genre'), ('theme_', 'Theme'), ('demo_', 'Demographic'),
                          ('source_', 'Source'), ('type_', 'Format'), ('rating_', 'Rating'),
                          ('season_', 'Season')):
        if name.startswith(prefix):
            return f'{label}: {name[len(prefix):]}'
    if name.startswith('story_svd_'):
        return f'Storyline component #{name.rsplit("_", 1)[1]}'
    return name


@app.post('/api/predict')
def predict():
    body = request.get_json(force=True) or {}

    def lst(key):
        v = body.get(key) or []
        return [str(x) for x in v] if isinstance(v, list) else [str(v)]

    row = {
        'genres': lst('genres'), 'themes': lst('themes'), 'demographics': lst('demographics'),
        'studios': lst('studios'), 'producers': lst('producers'),
        'source': body.get('source') or 'Original',
        'type': body.get('type') or 'TV',
        'rating': body.get('rating') or 'PG-13 - Teens 13 or older',
        'season': (body.get('season') or 'spring').lower(),
        'episodes': float(body.get('episodes') or 12),
        'duration_minutes': float(body.get('duration_minutes') or 23),
        'year': float(body.get('year') or 2026),
        'synopsis_clean': body.get('synopsis') or '',
    }
    X = FB.transform(pd.DataFrame([row]))

    preds = []
    for name, model, scaled, thr in CLASSIFIERS:
        Xi = SCALER.transform(X) if scaled else X
        p = float(model.predict_proba(Xi)[0, 1])
        preds.append({'model': name, 'probability': round(p, 4),
                      'prediction': int(p >= thr), 'threshold': thr})

    sv = EXPLAINER.shap_values(X)[0]
    base = float(EXPLAINER.expected_value)
    order = np.argsort(-np.abs(sv))[:14]
    contrib = [{'feature': X.columns[i], 'label': pretty_feature(X.columns[i]),
                'value': round(float(X.iloc[0, i]), 4),
                'shap': round(float(sv[i]), 4)} for i in order]
    other = float(sv.sum() - sum(sv[i] for i in order))
    logit = base + float(sv.sum())

    champion = next(p for p in preds if p['model'] == 'XGBoost (tuned)')
    return jsonify({
        'predictions': preds,
        'champion': champion,
        'shap': {'base_value': round(base, 4), 'contributions': contrib,
                 'other_features': round(other, 4), 'logit': round(logit, 4),
                 'probability': round(1 / (1 + np.exp(-logit)), 4)},
        'features_built': int(X.shape[1]),
        'echo': {k: row[k] for k in ('genres', 'themes', 'studios', 'producers',
                                     'source', 'type', 'season', 'year', 'episodes')},
    })


# ══════════════════════════════════════════════════════════════════
# Static site
# ══════════════════════════════════════════════════════════════════
@app.get('/')
def home():
    return send_from_directory(SITE, 'index.html')


@app.get('/favicon.ico')
def favicon():
    """Browsers ask for this by default; hand them the SVG the pages already use."""
    return send_from_directory(os.path.join(SITE, 'assets'), 'favicon.svg', mimetype='image/svg+xml')


@app.get('/<path:path>')
def static_files(path):
    return send_from_directory(SITE, path)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', 8100)), debug=False, threaded=True)

"""FeatureBuilder — verbatim copy of the class defined in the Kaggle notebook
(cleand2.ipynb, cell 165).

models/feature_builder.pkl was pickled from `__main__` inside the notebook, so
joblib looks for `__main__.FeatureBuilder` when loading it. `register()` below
re-attaches this class under that name so the fitted object unpickles with its
training-split vocabularies, historical aggregates, TF-IDF and SVD intact — the
builder is never re-fitted here.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


class FeatureBuilder:
    """Builds a pre-release-knowable feature matrix. Every historical-average statistic
    is fit on the TRAIN split only, then applied unchanged to val/test."""

    def __init__(self, top_genre_n=25, top_theme_n=20, top_studio_n=30, svd_dim=20):
        self.top_genre_n, self.top_theme_n, self.top_studio_n, self.svd_dim = top_genre_n, top_theme_n, top_studio_n, svd_dim

    def fit(self, train):
        self.genre_vocab = train['genres'].explode().value_counts().head(self.top_genre_n).index.tolist()
        self.theme_vocab = train['themes'].explode().value_counts().head(self.top_theme_n).index.tolist()
        studio_vc = train['studios'].explode().value_counts()
        self.studio_vocab = studio_vc[studio_vc >= 10].head(self.top_studio_n).index.tolist()
        self.demo_vocab = train['demographics'].explode().dropna().unique().tolist()
        self.source_vocab = train['source'].dropna().unique().tolist()
        self.type_vocab = train['type'].dropna().unique().tolist()
        self.rating_vocab = train['rating'].dropna().unique().tolist()
        self.season_vocab = train['season'].dropna().unique().tolist()

        self.studio_perf = train.explode('studios').groupby('studios')['balanced_score'].mean()
        self.global_bscore_mean = train['balanced_score'].mean()
        self.source_perf = train.groupby('source')['balanced_score'].mean()

        self.episodes_median = train['episodes'].median()
        self.duration_median = train['duration_minutes'].median()

        meta_stop = {
            'season', 'seasons', 'episode', 'episodes', 'ova', 'ona', 'special', 'specials', 'movie', 'movies',
            'film', 'films', 'short', 'shorts', 'recap', 'sequel', 'tv', 'series', 'anime', 'adaptation', 'adapted',
            'bundled', 'volume', 'volumes', 'manga', 'light', 'novel', 'novels', 'game', 'games', 'video', 'anniversary',
            'collaboration', 'mini', 'characters', 'character', 'based', 'second', 'third', 'fourth', 'fifth', 'sixth',
            'part', 'story', 'stories', 'animated', 'animation', 'source', 'studio', 'aired',
        }
        stopwords = list(ENGLISH_STOP_WORDS.union(meta_stop))
        self.tfidf = TfidfVectorizer(max_features=3000, stop_words=stopwords, ngram_range=(1, 2), min_df=3, max_df=0.4)
        X_text = self.tfidf.fit_transform(train['synopsis_clean'].fillna(''))
        self.svd = TruncatedSVD(n_components=self.svd_dim, random_state=42)
        self.svd.fit(X_text)
        return self

    def transform(self, data):
        out = pd.DataFrame(index=data.index)
        for g in self.genre_vocab: out[f'genre_{g}'] = data['genres'].apply(lambda lst: int(g in lst))
        for t in self.theme_vocab: out[f'theme_{t}'] = data['themes'].apply(lambda lst: int(t in lst))
        for dmg in self.demo_vocab: out[f'demo_{dmg}'] = data['demographics'].apply(lambda lst: int(dmg in lst))
        for s in self.source_vocab: out[f'source_{s}'] = (data['source'] == s).astype(int)
        for ty in self.type_vocab: out[f'type_{ty}'] = (data['type'] == ty).astype(int)
        for r in self.rating_vocab: out[f'rating_{r}'] = (data['rating'] == r).astype(int)
        for se in self.season_vocab: out[f'season_{se}'] = (data['season'] == se).astype(int)

        def studio_hist(lst):
            vals = [self.studio_perf.loc[s] for s in lst if s in self.studio_perf.index]
            return np.mean(vals) if vals else self.global_bscore_mean
        out['studio_hist_balanced'] = data['studios'].apply(studio_hist)
        out['studio_is_known'] = data['studios'].apply(lambda lst: int(any(s in self.studio_vocab for s in lst)))
        out['num_studios'] = data['studios'].apply(len)
        out['num_producers'] = data['producers'].apply(len)
        out['source_hist_balanced'] = data['source'].map(self.source_perf).fillna(self.global_bscore_mean)

        out['episodes'] = data['episodes'].fillna(self.episodes_median)

        out['duration_minutes'] = data['duration_minutes'].fillna(self.duration_median)
        out['year'] = data['year']
        out['num_genres'] = data['genres'].apply(len)
        out['num_themes'] = data['themes'].apply(len)

        X_text = self.tfidf.transform(data['synopsis_clean'].fillna(''))
        X_svd = self.svd.transform(X_text)
        for i in range(self.svd_dim): out[f'story_svd_{i}'] = X_svd[:, i]
        return out


def register():
    """Make `__main__.FeatureBuilder` resolve to this class so the notebook pickle loads."""
    main = sys.modules.get('__main__')
    if main is not None and not hasattr(main, 'FeatureBuilder'):
        main.FeatureBuilder = FeatureBuilder

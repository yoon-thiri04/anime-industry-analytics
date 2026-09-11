# Anime Industry Analytics System — Web UI

A presentation and inference layer over the IS-212 *Data & Knowledge Mining* project
(**Yoon Thiri Aung, YKPT-22425, University of Computer Studies, Yangon**).

Nine pages covering the whole pipeline from the project book: the dataset, descriptive mining,
association rules, storyline clustering, the hybrid recommender, the pre-release success
classifier with SHAP, model evaluation, and the genre demand forecast.

## Run it

```bash
cd ~/anime_industry_analytics
python3 scripts/serve.py
```

open <http://localhost:8100>. Set `PORT` to use a different port.

## Pages

| Page | What it shows | Source |
|---|---|---|
| `index.html` | Landing page, headline figures, pipeline overview | — |
| `catalogue.html` | All 8,227 titles, filterable and sortable, with cover art and trailers | live API |
| `descriptive.html` | Provenance, cleaning, distributions, format structure, `balanced_score`, genre trend | precomputed |
| `patterns.html` | FP-Growth rules across four transaction spaces + producer x studio partnerships | precomputed |
| `storylines.html` | 32 synopsis clusters, silhouette scan, PCA map, chi-square / Cramer's V validation | precomputed |
| `recommend.html` | Hybrid content-based recommender with an adjustable storyline/tag blend | **live models** |
| `predict.html` | Success Lab — score an unreleased concept, with a live SHAP waterfall | **live models** |
| `models.html` | Five classifiers, ROC/PR curves, confusion matrices, global SHAP, gain importance | precomputed |
| `forecast.html` | Damped Holt-Winters demand and volume forecasts, with a 2023-2025 backtest | precomputed |


## Descriptive mining is deterministic given the dataset and the saved models, so it is computed once at build time and written to `site/anime_data.js`:

```bash
python3 scripts/build_site_data.py
```

That script rebuilds the analysis frame from `data/df_clean_ui_10_9_2026_latest.csv`,
recomputing `balanced_score` from the project book's formula rather than reading a stored column —
which is why the site reproduces the book: 8,227 titles, mean 0.500 / sd 0.260, r = 0.845 against
raw score, a 5,165 / 480 / 1,014 chronological split, a success threshold of 0.6946, k = 32
clusters at silhouette 0.1488, and chi-square = 10,372 with Cramer's V = 0.180.

`scripts/serve.py` loads the real model files
and runs them per request:

- `POST /api/predict` — `FeatureBuilder.transform` then the five classifiers, then
  `shap.TreeExplainer` on the tuned XGBoost. The waterfall on the Success Lab is computed,
  not illustrated.
- `GET /api/recommend` — cosine similarity over `X_recommend.npy` (100-component storyline SVD)
  and `GT.npy` (multi-hot genre + theme), min-max scaled per query and blended 0.6 / 0.4.

Other endpoints: `/api/meta`, `/api/catalogue`, `/api/anime/<mal_id>`, `/api/search`.

## Layout

```
data/      the published UI dataset, recommendation frame and similarity matrices
models/    the 11 pickles exported from the Kaggle notebook
scripts/   feature_builder.py | common.py | build_site_data.py | serve.py
site/      the static site (HTML, style.css, anime.js, generated anime_data.js)
docs/      the project book and the source notebook
```

`scripts/feature_builder.py` is a verbatim copy of the `FeatureBuilder` class from the notebook.
`models/feature_builder.pkl` was pickled from `__main__`, so `register()` re-attaches the class
under that name; the fitted object is never re-fitted.

## Requirements

Python 3.12 with `flask`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `xgboost`, `shap`,
`statsmodels`, `mlxtend`, `joblib`.

On macOS `xgboost` needs an OpenMP runtime (`libomp.dylib`). If `import xgboost` fails with
`Library not loaded: @rpath/libomp.dylib`, either `brew install libomp`, or point it at the copy
that ships with scikit-learn:

```bash
DYLD_LIBRARY_PATH=$(python3 -c "import sklearn,os;print(os.path.join(os.path.dirname(sklearn.__file__),'.dylibs'))") python3 scripts/serve.py
```

## A note on the evaluation figures

Every number on `models.html` is recomputed at build time by loading the saved models and
rebuilding the held-out split — none of it is transcribed from the book. The figures land within
about 0.01 of Table 4.1; the residual is library-version drift in the scoring environment, and it
does not change the model ranking or any conclusion. The dataset, split, threshold, clustering and
forecast figures reproduce exactly.

# MAGPIE — TF-IDF + Logistic Regression Baseline Report

> Classical floor model. Any neural approach that cannot clearly beat these scores has not justified its added complexity.

## Summary: Baseline vs. Majority-Class Floor

| Task | Metric | TF-IDF+LR | Majority-Class Floor | Best C | n_test |
|------|--------|-----------|---------------------|--------|--------|
| Task 1 — Epistemic (sent) | Macro-F1 (95% CI) | 0.6164 [0.582, 0.650] | 31.7% | 1 | 2558 |
| Task 1 — Epistemic (token) | Span-F1 | 0.7745 | — | 10 | 1682 examples |
| Task 2 — Bias Binary | Macro-F1 (95% CI) | 0.6665 [0.615, 0.717] | 33.7% | 1 | 368 |
| Task 2 — Bias Binary | AUC-ROC | 0.7713 | 0.5000 | — | — |
| Task 2 — Bias Multiclass | Macro-F1 (95% CI) | 0.4264 [0.412, 0.441] | 18.8% | 1 | 3405 |
| Task 3 — Emotion | Micro-F1 | 0.7470 | — | 10 | 25560 |
| Task 3 — Emotion | Macro-F1 (95% CI) | 0.6509 [0.642, 0.659] | — | — | — |
| Task 3 — Emotion | Hamming Loss | 0.0521 | — | — | — |

## Task 1 — Epistemic Certainty

**Data**: Wikipedia Uncertainty corpus + FactBank (Szeged XML)  
**Classes**: asserted (0), hedged (1), speculative (2)  
**Split**: document-level 80/10/10, seed=42

### Per-class scores (sentence head)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| asserted | 0.9688 | 0.8825 | 0.9236 |
| hedged | 0.4472 | 0.7129 | 0.5496 |
| speculative | 0.2776 | 0.5821 | 0.3759 |

### Token head — BioScope sliding-window (±2 tokens)

- Token macro-F1: **0.6918**
- Cue-F1 (binary): **0.3996**
- Span-F1 (BioScope): **0.7745**
- Best C (token model): 10
- Test windows: 40253 | Test examples: 1682


### Top discriminative n-grams (sentence head)

- **asserted**: `often`, `usually`, `many`, `said to`, `is said`
- **hedged**: `probably`, `likely`, `perhaps`, `possibly`, `that`
- **speculative**: `said`, `would`, `says`, `allegedly`, `if`

## Task 2 — Political Bias

### Binary (biased / neutral)

**Data**: BABE (train+val, ~2.9K) + BASIL (train+val, ~6.3K)  
**Test**: BABE test split only (matches unified model eval)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| neutral | 0.7124 | 0.5829 | 0.6412 |
| biased | 0.6372 | 0.7569 | 0.6919 |

**Top n-grams:**
- **neutral**: `said`, `mr`, `trump said`, `both`, `case`
- **biased**: `trump`, `is`, `his`, `and`, `of`

### Multiclass (5-class credibility spectrum)

> Classes are a credibility spectrum derived from LIAR truthfulness scores (0.0=true→class 0, 1.0=pants-fire→class 4) and FakeNewsNet (real→0, fake→4). MAGPIE preprocessed LIAR lacks speaker party affiliation, so strict political orientation (left/right) is not supported.

**Classes** (0→most credible, 4→least credible):
highly_credible, mostly_credible, mixed, mostly_unreliable, highly_unreliable

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| highly_credible | 0.8790 | 0.6738 | 0.7628 |
| mostly_credible | 0.2206 | 0.4084 | 0.2865 |
| mixed | 0.0000 | 0.0000 | 0.0000 |
| mostly_unreliable | 0.3836 | 0.5721 | 0.4592 |
| highly_unreliable | 0.6079 | 0.6400 | 0.6235 |

**Top n-grams:**
- **highly_credible**: `season`, `2018`, `her`, `how`, `star`
- **mostly_credible**: `the`, `percent`, `cut`, `says`, `000`
- **mixed**: `(class not in training data)`
- **mostly_unreliable**: `says`, `the`, `no`, `we`, `government`
- **highly_unreliable**: `breaking`, `trump`, `report`, `jenner`, `kardashian`

## Task 3 — Emotional Framing

**Data**: GoEmotions + dair-ai/emotion + TweetEval + MAGPIE 84_emotion_tweets  
**Threshold**: 0.5 (tuned on dev set)  
**Best C**: 10

### Per-emotion F1

| Emotion | Prevalence | Precision | Recall | F1 |
|---------|-----------|-----------|--------|----|
| anger | 0.125 | 0.6838 | 0.8325 | 0.7509 |
| anticipation | 0.255 | 0.7451 | 0.8637 | 0.8001 |
| disgust | 0.030 | 0.5899 | 0.8108 | 0.6829 |
| fear | 0.061 | 0.6506 | 0.8217 | 0.7262 |
| joy | 0.132 | 0.6998 | 0.8495 | 0.7674 |
| love | 0.052 | 0.5197 | 0.7037 | 0.5978 |
| optimism | 0.009 | 0.2954 | 0.4504 | 0.3568 |
| pessimism | 0.007 | 0.2500 | 0.3095 | 0.2766 |
| sadness | 0.096 | 0.6973 | 0.8174 | 0.7526 |
| surprise | 0.047 | 0.5853 | 0.7943 | 0.6740 |
| trust | 0.206 | 0.7074 | 0.8558 | 0.7746 |

**Top n-grams per emotion:**
- **anger**: `hate`, `hit`, `words`, `bout`, `hot`
- **anticipation**: `time`, `hope`, `thought`, `start`, `finally`
- **disgust**: `weird`, `boy`, `toxic`, `mess`, `ugly`
- **fear**: `change`, `hearing`, `government`, `difficult`, `fire`
- **joy**: `love`, `special`, `create`, `beautiful`, `baby`
- **love**: `thanks`, `sympathetic`, `tender`, `delicate`, `longing`
- **optimism**: `hopefully`, `hope`, `hoping`, `optimism`, `fear`
- **pessimism**: `disappointed`, `unfortunately`, `disappointing`, `disappointment`, `upset`
- **sadness**: `black`, `lost`, `late`, `crying`, `cry`
- **surprise**: `surprised`, `unique`, `break`, `chance`, `guess`
- **trust**: `show`, `real`, `swear`, `team`, `school`

## Artifact / Leakage Notes

Inspecting top discriminative features is mandatory for detecting dataset artifacts (e.g. source-name leakage in LIAR, near-duplicate sentences in BASIL). If source names (e.g. newspaper names) appear in top features, the model is learning domain identity rather than bias signal.

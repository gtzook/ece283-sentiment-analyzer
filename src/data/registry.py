"""
Dataset registry for MAGPIE datasets.

Each DatasetMeta describes one dataset's schema: task type per label column,
raw-to-canonical label mapping, and whether it is news-domain relevant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskType(str, Enum):
    BINARY_CLS = "binary_classification"
    MULTICLASS_CLS = "multiclass_classification"
    MULTILABEL_CLS = "multilabel_classification"
    REGRESSION = "regression"
    SEQUENCE_LABEL = "sequence_labeling"


@dataclass
class LabelColumn:
    """Describes a single label column within a dataset CSV."""

    col: str
    task_type: TaskType
    # Number of distinct classes (1 for regression scalars)
    num_classes: int
    # Maps raw CSV value (as string) -> canonical int (or float for regression)
    label_map: dict[str, int | float]
    # Semantic: +1 = high value means positive/good, -1 = high means negative/bad, 0 = no polarity
    sentiment_polarity: int = 0


@dataclass
class DatasetMeta:
    """All metadata needed to load and interpret one MAGPIE dataset."""

    id: str                    # matches the directory name, e.g. "10_BABE"
    name: str                  # short human-readable name
    description: str
    text_col: str = "text"
    label_columns: list[LabelColumn] = field(default_factory=list)
    news_relevant: bool = False  # True for datasets whose text is from news articles/headlines
    extra_cols: list[str] = field(default_factory=list)  # other columns to keep (e.g. biased_words)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, DatasetMeta] = {}

def _reg(meta: DatasetMeta) -> DatasetMeta:
    REGISTRY[meta.id] = meta
    return meta


# --- News-domain datasets ---------------------------------------------------

_reg(DatasetMeta(
    id="10_BABE",
    name="BABE",
    description="Bias Annotations By Experts — 3,700 news sentences with word and sentence-level bias labels.",
    news_relevant=True,
    extra_cols=["biased words"],
    label_columns=[
        LabelColumn(
            col="label",
            task_type=TaskType.BINARY_CLS,
            num_classes=2,
            # Raw: 1=biased, 2=not biased → canonical 0=not biased, 1=biased
            label_map={"1": 1, "2": 0},
            sentiment_polarity=-1,  # biased = bad/negative framing
        )
    ],
))

_reg(DatasetMeta(
    id="9_BASIL",
    name="BASIL",
    description="Bias Annotation Scheme for Information Lateralization — 300 news articles with sentence-level bias type and targeting aim.",
    news_relevant=True,
    label_columns=[
        LabelColumn(
            col="label",
            task_type=TaskType.MULTICLASS_CLS,
            num_classes=3,
            # 0=lexical bias, 1=informational bias, 2=non-biased
            label_map={"0": 0, "1": 1, "2": 2},
            sentiment_polarity=0,
        ),
        LabelColumn(
            col="aim",
            task_type=TaskType.MULTICLASS_CLS,
            num_classes=3,
            # 0=not direct, 1=direct targeting, 2=non-biased
            label_map={"0": 0, "1": 1, "2": 2},
            sentiment_polarity=0,
        ),
    ],
))

_reg(DatasetMeta(
    id="12_PHEME",
    name="PHEME",
    description="5,221 Twitter rumours and non-rumours posted during breaking news events.",
    news_relevant=True,
    label_columns=[
        LabelColumn(
            col="label",
            task_type=TaskType.BINARY_CLS,
            num_classes=2,
            label_map={"0": 0, "1": 1},
            sentiment_polarity=0,
        ),
        LabelColumn(
            col="veracity_label",
            task_type=TaskType.MULTICLASS_CLS,
            num_classes=3,
            # 0=false, 1=true, 2=unknown
            label_map={"0": 0, "1": 1, "2": 2},
            sentiment_polarity=1,  # 1=true is positive credibility
        ),
    ],
))

_reg(DatasetMeta(
    id="19_MultiDimNews",
    name="MultiDimNews",
    description="2,057 news sentences annotated across four binary dimensions: bias, subjectivity, framing, hidden assumptions.",
    news_relevant=True,
    label_columns=[
        LabelColumn(col="label_bias",           task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=-1),
        LabelColumn(col="label_subj",           task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=0),
        LabelColumn(col="label_framing",        task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=0),
        LabelColumn(col="label_hidden_assumpt", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=0),
    ],
))

_reg(DatasetMeta(
    id="22_NewsWCL50",
    name="NewsWCL50",
    description="News articles annotated for within-community language bias.",
    news_relevant=True,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=-1),
    ],
))

_reg(DatasetMeta(
    id="25_FakeNewsNet",
    name="FakeNewsNet",
    description="23,196 news headlines labeled for fake news detection.",
    news_relevant=True,
    label_columns=[
        LabelColumn(
            col="label",
            task_type=TaskType.BINARY_CLS,
            num_classes=2,
            label_map={"0": 0, "1": 1},  # 0=real, 1=fake
            sentiment_polarity=-1,  # fake = negative credibility
        )
    ],
))

_reg(DatasetMeta(
    id="26_neutralizing-bias",
    name="NeutralizingBias",
    description="Pairs of biased and neutral sentences for bias neutralization.",
    news_relevant=True,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=-1),
    ],
))

_reg(DatasetMeta(
    id="72_LIAR",
    name="LIAR",
    description="12,791 PolitiFact statements with six-class truthfulness, collapsed to a continuous score and binary label.",
    news_relevant=True,
    label_columns=[
        LabelColumn(
            col="label",
            task_type=TaskType.REGRESSION,
            num_classes=1,
            # Continuous 0.0 (true) → 1.0 (pants-fire); identity map (values already floats)
            label_map={},  # no remapping needed for continuous
            sentiment_polarity=-1,  # higher score = less truthful = more negative
        ),
        LabelColumn(
            col="label_binary",
            task_type=TaskType.BINARY_CLS,
            num_classes=2,
            label_map={"0": 0, "1": 1},  # 0=true, 1=false
            sentiment_polarity=-1,
        ),
    ],
))

_reg(DatasetMeta(
    id="96_Bu-NEMO",
    name="Bu-NEMO",
    description="News media objectivity dataset.",
    news_relevant=True,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=0),
    ],
))

_reg(DatasetMeta(
    id="42_GoodNewsEveryone",
    name="GoodNewsEveryone",
    description="5,000 news headlines annotated for emotion cue and experiencer spans.",
    news_relevant=True,
    extra_cols=["cue_pos", "experiencer_pos"],
    label_columns=[
        # Sequence labels: stored as string representations of position spans
        LabelColumn(col="cue_pos",          task_type=TaskType.SEQUENCE_LABEL, num_classes=2, label_map={}, sentiment_polarity=0),
        LabelColumn(col="experiencer_pos",  task_type=TaskType.SEQUENCE_LABEL, num_classes=2, label_map={}, sentiment_polarity=0),
    ],
))

# --- General sentiment datasets ---------------------------------------------

_reg(DatasetMeta(
    id="99_SST2",
    name="SST2",
    description="Stanford Sentiment Treebank — 9,614 movie review sentences with binary sentiment.",
    news_relevant=False,
    label_columns=[
        LabelColumn(
            col="label",
            task_type=TaskType.BINARY_CLS,
            num_classes=2,
            label_map={"0": 0, "1": 1},  # 0=positive, 1=negative
            sentiment_polarity=-1,  # higher label = more negative
        )
    ],
))

_reg(DatasetMeta(
    id="63_semeval2014",
    name="SemEval2014",
    description="5,794 aspect-based sentiment samples from laptop/restaurant reviews.",
    news_relevant=False,
    label_columns=[
        LabelColumn(
            col="label",
            task_type=TaskType.MULTICLASS_CLS,
            num_classes=3,
            label_map={"-1": 0, "0": 1, "1": 2},  # canonical: 0=neg, 1=neu, 2=pos
            sentiment_polarity=1,
        )
    ],
))

_reg(DatasetMeta(
    id="100_Amazon_reviews",
    name="AmazonReviews",
    description="Amazon product reviews with star ratings as sentiment proxy.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.REGRESSION, num_classes=1, label_map={}, sentiment_polarity=1),
    ],
))

_reg(DatasetMeta(
    id="101_IMDB",
    name="IMDB",
    description="IMDB movie reviews with binary sentiment.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=1),
    ],
))

_reg(DatasetMeta(
    id="103_MPQA",
    name="MPQA",
    description="Multi-Perspective Question Answering — opinion/subjectivity annotations.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=0),
    ],
))

_reg(DatasetMeta(
    id="31_SUBJ",
    name="SUBJ",
    description="Subjectivity dataset: 10,000 sentences labeled subjective vs. objective.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=0),
    ],
))

# --- Emotion datasets -------------------------------------------------------

_reg(DatasetMeta(
    id="84_emotion_tweets",
    name="EmotionTweets",
    description="~198k tweets with 8-class Plutchik emotion labels (anger, anticipation, disgust, fear, joy, sadness, surprise, trust).",
    news_relevant=False,
    label_columns=[
        LabelColumn(
            col="label",
            task_type=TaskType.MULTICLASS_CLS,
            num_classes=8,
            # 1-indexed in raw → 0-indexed canonical
            label_map={str(i): i - 1 for i in range(1, 9)},
            sentiment_polarity=0,
        )
    ],
))

# --- Bias / stereotype datasets ---------------------------------------------

_reg(DatasetMeta(
    id="33_CrowSPairs",
    name="CrowSPairs",
    description="Crowdsourced stereotype pairs for measuring social biases in LMs.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=0),
    ],
))

_reg(DatasetMeta(
    id="64_StereoSet",
    name="StereoSet",
    description="Measures stereotypical bias in LMs across gender, race, religion, profession.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.MULTICLASS_CLS, num_classes=3, label_map={"0": 0, "1": 1, "2": 2}, sentiment_polarity=0),
    ],
))

_reg(DatasetMeta(
    id="109_stereotype",
    name="Stereotype",
    description="Stereotype detection dataset.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=0),
    ],
))

# --- Gender bias datasets ---------------------------------------------------

_reg(DatasetMeta(
    id="18_GAP",
    name="GAP",
    description="Gender-balanced Wikipedia pronoun resolution dataset.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=0),
    ],
))

_reg(DatasetMeta(
    id="105_RtGender",
    name="RtGender",
    description="Response to gender dataset — social media responses annotated for gender-targeted content.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=0),
    ],
))

_reg(DatasetMeta(
    id="116_MDGender",
    name="MDGender",
    description="Multi-dimensional gender bias dataset.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.MULTICLASS_CLS, num_classes=3, label_map={"0": 0, "1": 1, "2": 2}, sentiment_polarity=0),
    ],
))

_reg(DatasetMeta(
    id="117_Funpedia",
    name="Funpedia",
    description="Funpedia dataset for gender bias in text generation.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=0),
    ],
))

_reg(DatasetMeta(
    id="118_WizardsOfWikipedia",
    name="WizardsOfWikipedia",
    description="Knowledge-grounded dialogue dataset annotated for gender bias.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=0),
    ],
))

# --- Hate speech / toxicity datasets ----------------------------------------

_reg(DatasetMeta(
    id="40_JIGSAW",
    name="JIGSAW",
    description="Jigsaw toxic comment classification — Wikipedia talk page comments.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=-1),
    ],
))

_reg(DatasetMeta(
    id="86_OffensiveLanguage",
    name="OffensiveLanguage",
    description="OffComText — offensive language detection in tweets.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=-1),
    ],
))

_reg(DatasetMeta(
    id="87_OnlineHarassmentDataset",
    name="OnlineHarassment",
    description="Online harassment detection in Twitter data.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=-1),
    ],
))

_reg(DatasetMeta(
    id="88_HatespeechTwitter",
    name="HatespeechTwitter",
    description="Hate speech detection on Twitter.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=-1),
    ],
))

_reg(DatasetMeta(
    id="92_HateXplain",
    name="HateXplain",
    description="Hate speech dataset with rationale explanations.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.MULTICLASS_CLS, num_classes=3, label_map={"0": 0, "1": 1, "2": 2}, sentiment_polarity=-1),
    ],
))

_reg(DatasetMeta(
    id="891_WikiDetoxToxicity",
    name="WikiDetoxToxicity",
    description="Wikipedia talk page comments annotated for toxicity.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=-1),
    ],
))

_reg(DatasetMeta(
    id="892_WikiDetoxAggressionAndAttack",
    name="WikiDetoxAggression",
    description="Wikipedia talk page comments annotated for aggression and personal attacks.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=-1),
    ],
))

# --- Stance datasets --------------------------------------------------------

_reg(DatasetMeta(
    id="119_SemEval2023Task4",
    name="SemEval2023Task4",
    description="Value detection in arguments.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=0),
    ],
))

_reg(DatasetMeta(
    id="120_SemEval2023Task3",
    name="SemEval2023Task3",
    description="News genre and framing detection.",
    news_relevant=True,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.MULTICLASS_CLS, num_classes=3, label_map={"0": 0, "1": 1, "2": 2}, sentiment_polarity=0),
    ],
))

_reg(DatasetMeta(
    id="124_SemEval2016Task6",
    name="SemEval2016Task6",
    description="Detecting stance in tweets toward six targets.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.MULTICLASS_CLS, num_classes=3, label_map={"0": 0, "1": 1, "2": 2}, sentiment_polarity=0),
    ],
))

_reg(DatasetMeta(
    id="125_MultiTargetStance",
    name="MultiTargetStance",
    description="Multi-target stance detection.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.MULTICLASS_CLS, num_classes=3, label_map={"0": 0, "1": 1, "2": 2}, sentiment_polarity=0),
    ],
))

_reg(DatasetMeta(
    id="126_WTWT",
    name="WTWT",
    description="Will-They-Won-T merger stance detection on Twitter.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.MULTICLASS_CLS, num_classes=4, label_map={"0": 0, "1": 1, "2": 2, "3": 3}, sentiment_polarity=0),
    ],
))

_reg(DatasetMeta(
    id="127_VaccineLies",
    name="VaccineLies",
    description="Vaccine misinformation detection on Twitter.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=-1),
    ],
))

# --- Remaining datasets (schemas to be verified) ----------------------------

_reg(DatasetMeta(
    id="03_CW_HARD",
    name="CW_HARD",
    description="Clickbait and hyperpartisan detection dataset.",
    news_relevant=True,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=0),
    ],
))

_reg(DatasetMeta(
    id="38_starbucks",
    name="Starbucks",
    description="Starbucks-related social media posts annotated for sentiment.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.MULTICLASS_CLS, num_classes=3, label_map={"0": 0, "1": 1, "2": 2}, sentiment_polarity=1),
    ],
))

_reg(DatasetMeta(
    id="75_RedditBias",
    name="RedditBias",
    description="Reddit comments annotated for demographic and ideological bias.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=-1),
    ],
))

_reg(DatasetMeta(
    id="80_DebateEffects",
    name="DebateEffects",
    description="Argumentative effects and persuasion in political debate transcripts.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=0),
    ],
))

_reg(DatasetMeta(
    id="91_WikiMadlibs",
    name="WikiMadlibs",
    description="Wikipedia-derived dataset for measuring gender and occupational stereotypes.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=0),
    ],
))

_reg(DatasetMeta(
    id="104_TRAC2",
    name="TRAC2",
    description="Aggression identification in social media (English + Hindi).",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.MULTICLASS_CLS, num_classes=3, label_map={"0": 0, "1": 1, "2": 2}, sentiment_polarity=-1),
    ],
))

_reg(DatasetMeta(
    id="108_MeTooMA",
    name="MeTooMA",
    description="#MeToo movement tweets annotated for stance and hate speech.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.BINARY_CLS, num_classes=2, label_map={"0": 0, "1": 1}, sentiment_polarity=0),
    ],
))

_reg(DatasetMeta(
    id="128_GWSD",
    name="GWSD",
    description="Graded word sense disambiguation dataset.",
    news_relevant=False,
    label_columns=[
        LabelColumn(col="label", task_type=TaskType.REGRESSION, num_classes=1, label_map={}, sentiment_polarity=0),
    ],
))

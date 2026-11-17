"""The "Project" tab of the Streamlit app: the full write-up, read from docs/data/site.json.

Every number comes from site.json (written by scripts/export_site_data.py from the result files);
only prose lives here. Needs no model weights, features or audio, so it runs on any machine with the repo.
"""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "docs" / "data" / "site.json"
IMG = ROOT / "docs" / "img"
REPO = "https://github.com/yashvardhan-unlimited/ES-335-Music-genre-classifier"
BLUE, GREY, INK, GRID = "#2a78d6", "#8a8984", "#52514e", "#e4e3df"
BLUES = ["#e9f2fd", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SPLIT_ORDER = ["A", "B", "Bp", "D", "C"]
SPLIT_LABEL = {"A": "A window-random", "B": "B clip-random", "Bp": "B' fault-filtered",
               "D": "D artist-grouped", "C": "C published artist-disjoint"}


@st.cache_data
def load_site() -> dict:
    return json.loads(SITE.read_text(encoding="utf-8"))


def _show(chart: alt.Chart) -> None:
    st.altair_chart(chart.configure_view(strokeWidth=0)
                    .configure_axis(gridColor=GRID, domainColor=GRID, tickColor=GRID, labelColor=INK,
                                    titleColor=INK, labelFontSize=12, titleFontSize=12)
                    .configure_legend(labelColor=INK, titleColor=INK), width="stretch", theme=None)


def _fig(name: str, caption: str) -> None:
    if (IMG / name).exists():
        st.image(str(IMG / name), caption=caption)


def question(S: dict) -> None:
    st.header("The question, and what one row is")
    st.markdown(
        "**Question.** How well can a model predict a 30-second clip's genre from the audio alone, and how much of "
        "that accuracy survives when the test set contains no artist the model heard in training?\n\n"
        "**One row.** We predict one **30 s clip** (the unit of every reported score). A model trains on one "
        "**3 s window** of audio with its clip's genre as the label: 10 rows per clip, so 4,420 training rows from "
        "442 training clips. All ten windows of a clip stay in the same part, so a clip is never split between "
        "train and test.\n\n"
        "**Why it matters.** GTZAN is the standard benchmark for this task, yet it holds several clips per artist "
        "and some repeated recordings. A random split can reward recognising a performer or a recording instead of "
        "a genre. This page measures that gap.")


def answer(S: dict) -> None:
    st.header("The answer in one number")
    h, c = S["headline"], S["c"]
    models = {m["model"]: m for m in c["models"]}
    svm, strict, ladB = models["svm"], c["strict"].get("cnn"), S["ladder"].get("cnn", {}).get("B")
    part = "test" if S["part"] == "test" else "validation"
    a, b, d, e = st.columns(4)
    a.metric(f"CNN, split C ({part}, {h['n_clips']} clips)", f"{h['macro_f1']:.2f}",
             f"95% CI {h['ci_low']:.2f} to {h['ci_high']:.2f}", delta_color="off")
    b.metric("Best classical (RBF SVM)", f"{svm['macro_f1']:.2f}",
             f"95% CI {svm['ci_low']:.2f} to {svm['ci_high']:.2f}", delta_color="off")
    d.metric("Chance floor", f"{h['floor']:.2f}", "stratified dummy", delta_color="off")
    if ladB:
        e.metric("CNN on a leaky split (B)", f"{ladB['mean']:.2f}", "same sizes, artists shared", delta_color="off")
    p, drop = c["paired"].get("cnn - svm"), S["drops"].get("cnn: Bp - C")
    st.markdown(
        f"On clips by artists it never heard, the CNN reaches a macro-F1 of **{h['macro_f1']:.2f}**: far above "
        f"chance, far from perfect. "
        + (f"It differs from the best classical model by {p['diff']:+.2f} (paired 95% interval {p['ci_low']:.2f} to "
           f"{p['ci_high']:.2f}). " if p else "")
        + (f"Trained the same way on a random clip split of the same clean pool (B'), where artists are shared, the "
           f"CNN scores {drop['diff']:.2f} higher than on C: that is how much a careless split would have flattered "
           f"it." if drop else ""))
    st.caption("Macro-F1 runs from 0 to 1. " + (
        "Final models were refit on training plus validation clips before the single test run."
        if S["fit_on_C"] == "train+valid" else "Models trained on the training part only."))


def data(S: dict) -> None:
    st.header("Data")
    a = S["data"]
    st.markdown(
        "**GTZAN** (Tzanetakis & Cook, 2002): 1,000 clips of 30 s, 100 per genre, 22,050 Hz mono, in 10 genres: "
        "blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae and rock. Downloaded from a Kaggle "
        "mirror; the audio is not republished.")
    st.subheader("Size and key variables")
    st.table(pd.DataFrame([
        ("Clip (1,000 files; 999 readable)", "30 s of audio at 22,050 Hz mono (about 660,000 samples). Length varies by under 0.7 s."),
        ("Genre (target)", "One of 10 labels, 100 clips each (99 for jazz, because one file is unreadable)."),
        ("Artist (grouping variable)", "321 distinct artist names from gtzan-bind; 14 clips have no artist label. "
                                       "Used only to build splits and bootstrap intervals, never as a model input."),
        ("Window (training row)", "A 3 s slice: 57 handcrafted features for the classical models, or a 128-band "
                                  "log-mel image for the CNN."),
        ("Split part", "train, valid or test, assigned per clip. Split C has 442 / 197 / 290 usable clips."),
    ], columns=["Variable", "What it is"]).set_index("Variable"))
    st.subheader("Quality issues")
    st.markdown(
        f"- **Audit:** {a['n_files']} files, {a['n_loadable']} load ({', '.join(a['failures'])} has no valid WAV "
        "header and is dropped).\n"
        "- **Exact duplicates:** 14 groups of clips have byte-identical audio (for example metal.00034 and "
        "metal.00094). Under a random split one copy can sit in train and the other in test.\n"
        "- **Faults found by listening** (Sturm, 2013): the same song in several clips, mislabelled genres and "
        "distorted clips. Removing them leaves 929 usable clips for splits B' and D.\n"
        "- **Many clips per artist:** a random split puts the same artist, and sometimes the same recording, on "
        "both sides. This is the leakage this project measures.\n"
        "- **Uneven lengths:** clips run from 29.93 s to 30.65 s (nine are under 30 s). Short clips are "
        "zero-padded and long ones trimmed to exactly 30 s, so every clip gives ten equal windows.\n"
        "- Artist names come from gtzan-bind, matched automatically, not by listening. Credits that share a "
        "performer (\"Bob Marley & The Wailers\" and \"Bob Marley\") count as one artist cluster.")
    _fig("mel_per_genre.png", "What the CNN sees: one log-mel spectrogram per genre (fixed dB reference, so "
                              "loudness differences are kept).")


def method(S: dict) -> None:
    st.header("Method")
    st.subheader("Preprocessing, and why")
    st.markdown(
        "- **Mono, 22,050 Hz, 3 s windows:** GTZAN's native format. A 3 s window is long enough to hear rhythm and "
        "timbre and gives ten rows per clip, so more training rows for the classical models. Other window lengths "
        "were compared on validation only.\n"
        "- **Unreadable and faulty clips removed:** one clip (jazz.00054) cannot be decoded, and split C uses the "
        "published fault-filtered list (Kereliuk et al.) so repeated and distorted recordings do not inflate scores.\n"
        "- **Scalers and normalisation fitted on training data only,** so nothing about validation or test leaks "
        "into training.\n"
        "- **Log-mel in dB with a fixed reference,** not per-clip normalisation, so loudness, which differs by "
        "genre, is kept (see Limitations).\n"
        "- **Features:** 57 handcrafted numbers per window (20 MFCC means and variances, chroma, spectral shape, "
        "zero-crossing rate, loudness, harmonic/percussive energy, clip tempo) for the classical models; a 128-band "
        "log-mel spectrogram for the CNN.")
    st.subheader("Baselines and models compared")
    st.table(pd.DataFrame([
        ("Dummy, most frequent", "Baseline: always predicts one genre."),
        ("Dummy, stratified", "Baseline: guesses genres at random in proportion to class frequency."),
        ("k-nearest neighbours", "Simple, no training: copies the genre of the nearest windows."),
        ("Logistic regression", "Linear model on the 57 features, the simplest learned boundary."),
        ("RBF SVM", "Non-linear boundary on the same features."),
        ("XGBoost", "Boosted trees on the same features, strong on tabular data."),
        ("CNN on log-mels", "Learns its own features from the spectrogram; small 4-block network trained from "
                            "scratch with random crops and SpecAugment, 3 seeds."),
    ], columns=["Model", "Role"]).set_index("Model"))
    st.subheader("Evaluation: splits and metrics")
    st.markdown(
        "Five splits keep the same sizes (442 / 197 / 290 clips) and change only the leakage: **A** shuffles windows "
        "(a clip sits on both sides), **B** shuffles whole clips, **B'** does the same after removing faulty clips, "
        "**D** keeps each artist in one part (random), and **C** is the published artist-disjoint split, the main "
        "result. Validation picks hyperparameters (by clip macro-F1, never k-fold over windows, which would put a "
        "clip's windows on both sides); test is scored once.\n\n"
        "- **Unit:** one prediction per 30 s clip: a model scores every window and the clip's prediction is the "
        "average of its ten probability vectors (soft vote).\n"
        "- **Primary metric:** clip-level macro-F1, so every genre counts equally. 95% intervals come from a "
        "bootstrap that resamples *artists*, because clips by one artist are not independent. Secondary: accuracy, "
        "per-genre F1, confusion matrices.\n"
        "- **Test lock:** the code refuses to read the test part until the git tag `pre-test-freeze` exists. The tag "
        "was made after every choice was written down, and test was scored once.")
    rows = [(v["short"], v["name"], v["description"], f"{v['test_share_artist_in_train']:.0%}")
            for v in S["splits"].values()]
    st.dataframe(pd.DataFrame(rows, columns=["Split", "Name", "How it is built", "Test clips sharing an artist with train"]),
                 hide_index=True, width="stretch")
    _fig("val_svm_heatmap.png", "Tuning on validation only: the SVM's grid on split C.")
    _fig("analysis_learning_curves.png", "More training artists, better scores on new artists: neither model has "
                                         "levelled off at 442 clips.")


def ladder(S: dict) -> None:
    st.header("The leakage ladder: same models, same sizes, less leakage")
    st.markdown("Every split has exactly 442 training, 197 validation and 290 test clips, and every model is tuned "
                "the same way on each. Only what can leak from training into the test set changes, one step at a "
                "time, so each fall estimates how much that kind of leakage was inflating the score. "
                "**Pick a model** to see it.")
    labels = {S["labels"][m]: m for m in S["ladder"]}
    model = labels[st.selectbox("Model", list(labels), index=list(labels.values()).index("cnn"), key="ladder_model")]
    rows = [dict(split=SPLIT_LABEL[s], **{k: v[k] for k in ("mean", "ci_low", "ci_high", "n_runs")})
            for s, v in S["ladder"][model].items()]
    df = pd.DataFrame(rows)
    order = [SPLIT_LABEL[s] for s in SPLIT_ORDER]
    x = alt.X("split:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, grid=False))
    base = alt.Chart(df).encode(x=x, tooltip=["split", alt.Tooltip("mean:Q", format=".3f"),
                                              alt.Tooltip("ci_low:Q", format=".3f"),
                                              alt.Tooltip("ci_high:Q", format=".3f"), "n_runs"])
    y = alt.Y("mean:Q", title="test clip macro-F1" if S["part"] == "test" else "valid clip macro-F1",
              scale=alt.Scale(domain=[0, 1]))
    _show(alt.layer(base.mark_rule(color=GREY).encode(y="ci_low:Q", y2="ci_high:Q"),
                    base.mark_line(color=BLUE, strokeWidth=2).encode(y=y),
                    base.mark_point(color=BLUE, size=90, filled=True).encode(y=y)).properties(height=340))
    st.caption("Points are the mean over runs (split seeds, or CNN seeds); the bars are 95% intervals. "
               "Splits B, B' and D use five seeds each.")
    steps = [("A", "B", "windows to whole clips"), ("B", "Bp", "remove faulty clips"),
             ("Bp", "D", "keep each artist in one part"), ("D", "C", "random artists to the published split"),
             ("Bp", "C", "B' to C in one step")]
    drops = [(f"{a}→{b}: {what}", S["drops"].get(f"{model}: {a} - {b}")) for a, b, what in steps]
    st.dataframe(pd.DataFrame([(n, f"{d['diff']:+.3f}", f"{d['ci_low']:+.3f} to {d['ci_high']:+.3f}")
                               for n, d in drops if d], columns=["Step", "Fall in macro-F1", "95% interval"]),
                 hide_index=True, width="stretch")


def models(S: dict) -> None:
    st.header("Every model on split C")
    part = "Test" if S["part"] == "test" else "Validation"
    st.markdown(f"{part} clip macro-F1 with artist-cluster 95% intervals. Grey bars are the dummy floors.")
    df = pd.DataFrame(S["c"]["models"]).assign(floor=lambda d: d["model"].str.startswith("dummy"))
    y = alt.Y("label:N", sort=None, title=None, axis=alt.Axis(ticks=False))
    base = alt.Chart(df).encode(y=y, tooltip=["label", alt.Tooltip("macro_f1:Q", format=".3f"),
                                              alt.Tooltip("ci_low:Q", format=".3f"),
                                              alt.Tooltip("ci_high:Q", format=".3f"), "n_runs"])
    bar = base.mark_bar(height=16, cornerRadiusEnd=4).encode(
        x=alt.X("macro_f1:Q", title="clip macro-F1", scale=alt.Scale(domain=[0, 1])),
        color=alt.condition(alt.datum.floor, alt.value(GREY), alt.value(BLUE)))
    ci = base.mark_rule(color=INK).encode(x="ci_low:Q", x2="ci_high:Q")
    _show(alt.layer(bar, ci).properties(height=32 * len(df) + 40))
    p = S["c"]["paired"].get("cnn - svm")
    if p:
        st.caption(f"CNN minus SVM, paired over the same artist resamples: {p['diff']:+.3f} "
                   f"(95% interval {p['ci_low']:+.3f} to {p['ci_high']:+.3f}). The models' intervals overlap heavily.")
    with st.expander("Table view"):
        st.dataframe(df.drop(columns="floor"), hide_index=True, width="stretch")


def errors(S: dict) -> None:
    st.header("Where the model goes wrong")
    st.markdown("Rows are the true genre, columns the predicted genre. **Pick a model, then pick a true and a "
                "predicted genre** to list the clips in that cell, most confident first. Confident mistakes are "
                "the interesting ones.")
    G = S["genres"]
    keys = list(S["confusion"])
    run = S["confusion"][st.selectbox("Model run", keys, format_func=lambda k: S["confusion"][k]["label"],
                                      index=keys.index("cnn_s0"), key="cm_run")]
    cm = pd.DataFrame(run["confusion"], index=G, columns=G)
    long = cm.rename_axis("true").reset_index().melt("true", var_name="predicted", value_name="clips")
    long["share"] = long["clips"] / long.groupby("true")["clips"].transform("sum").clip(lower=1)
    long["text"] = long["clips"].map(lambda v: str(v) if v else "")
    enc = dict(x=alt.X("predicted:N", sort=G, axis=alt.Axis(labelAngle=-45, grid=False)),
               y=alt.Y("true:N", sort=G, title="true genre", axis=alt.Axis(grid=False)))
    cell = alt.Chart(long).encode(**enc, tooltip=["true", "predicted", "clips", alt.Tooltip("share:Q", format=".0%")])
    left, right = st.columns([3, 2], gap="large")
    with left:
        _show(alt.layer(cell.mark_rect(stroke="#ffffff", strokeWidth=2).encode(
            color=alt.Color("share:Q", title="share of row", scale=alt.Scale(domain=[0, 1], range=BLUES))),
            cell.mark_text(fontSize=11).encode(text="text:N", color=alt.condition(
                alt.datum.share > 0.5, alt.value("#ffffff"), alt.value("#1f1f1d")))).properties(height=420))
    off = long[long["true"] != long["predicted"]].sort_values("clips", ascending=False).iloc[0]
    with right:
        t = st.selectbox("True genre", G, index=G.index(off["true"]), key=f"t_{run['label']}")
        p = st.selectbox("Predicted genre", G, index=G.index(off["predicted"]), key=f"p_{run['label']}")
        clips = run["cells"].get(f"{t}|{p}", [])
        total = int(cm.loc[t].sum())
        st.markdown(f"**{t} predicted as {p}:** {len(clips)} of {total} clips")
        if clips:
            st.dataframe(pd.DataFrame(clips, columns=["clip id", "top score"]), hide_index=True, width="stretch")
        else:
            st.caption("No clips in this cell.")
    st.subheader("F1 per genre")
    f1 = pd.DataFrame({"genre": G, "f1": [run["per_genre_f1"][g] for g in G],
                       "clips": [run["support"][g] for g in G]})
    _show(alt.Chart(f1).mark_bar(height=16, cornerRadiusEnd=4, color=BLUE).encode(
        y=alt.Y("genre:N", sort=None, title=None), x=alt.X("f1:Q", title="F1", scale=alt.Scale(domain=[0, 1])),
        tooltip=["genre", alt.Tooltip("f1:Q", format=".3f"), "clips"]).properties(height=300))
    st.caption("With 13 to 32 clips per genre, one clip moves a genre's F1 by several points, so read small "
               "differences as noise.")


def limitations() -> None:
    st.header("Limitations: what this does not show")
    st.markdown(
        "- **One small, old dataset.** GTZAN is about 1,000 clips, mostly from around 2000, partly from radio and "
        "CDs of uneven quality. A score here says little about today's streaming catalogue.\n"
        "- **One label per clip, ten Western genres.** Real songs mix genres, and the labels are one person's "
        "judgement; some are wrong (Sturm 2013).\n"
        "- **Artist labels are matched automatically.** They come from gtzan-bind, not from listening, so some "
        "links between clips may be missed. Split C still shares artist clusters between train and test for a few "
        "test clips by these names; the C-strict figure removes them.\n"
        "- **\"Audio alone\" can still mean shortcuts.** We keep absolute loudness on purpose, and recording quality, "
        "mastering and era all differ by genre in GTZAN. The model may use them; we have not separated those cues "
        "from genre.\n"
        "- **The validation part made many choices** (hyperparameters, CNN epochs, XGBoost trees, window length, "
        "voting), so validation scores are optimistic. That is why the headline is a test score.\n"
        "- **Unequal runs.** The CNN has three training runs on C; the classical models are deterministic, one run "
        "each. The random splits B, B' and D have five draws each; C is one published draw.\n"
        "- **One fixed split is noisy.** The five random artist-disjoint draws (D) differ a lot from each other, so "
        "a single split, including C, can land anywhere in that range.\n"
        "- **Some genres rest on very few artists.** In C's validation part all hip-hop clips are one artist, and "
        "blues and jazz have three artists each, so a per-genre score there describes a handful of musicians. "
        "The intervals resample artists, which is why they are wide.")


def reproduce() -> None:
    st.header("Reproduce")
    st.markdown(f"[Repository]({REPO}) · [README (setup, data access)]({REPO}/blob/main/README.md) · "
                f"[DECISIONS.md]({REPO}/blob/main/DECISIONS.md) · [AI-use disclosure]({REPO}/blob/main/AI_USE.md) · "
                f"[Full results]({REPO}/tree/main/results) · [Study notes]({REPO}/tree/main/learn)")
    st.code("git clone " + REPO + "\ncd ES-335-Music-genre-classifier\n"
            "make setup data splits features tune     # download, audit, splits, features, tune (hours on a CPU)\n"
            "python scripts/evaluate_part.py --part valid\n"
            "python scripts/evaluate_part.py --part test     # refuses to run without the pre-test-freeze tag\n"
            "python scripts/export_site_data.py              # numbers for this page\n"
            "make tests", language="bash")
    st.caption("Every number on this tab is read from docs/data/site.json, which scripts/export_site_data.py "
               "writes from the result files. Design decisions and rejected alternatives: DECISIONS.md.")
    st.markdown("**References**\n\n"
                "- G. Tzanetakis and P. Cook, \"Musical genre classification of audio signals,\" IEEE Trans. Speech "
                "and Audio Processing, 2002.\n"
                "- B. L. Sturm, \"The GTZAN dataset: its contents, its faults, their effects on evaluation, and its "
                "future use,\" arXiv:1306.1461, 2013.\n"
                "- C. Kereliuk, B. L. Sturm and J. Larsen, \"Deep learning and music adversaries,\" IEEE Trans. "
                "Multimedia, 2015 (fault-filtered split).")


def project_tab() -> None:
    """Render the whole write-up, top to bottom."""
    S = load_site()
    question(S)
    answer(S)
    data(S)
    method(S)
    ladder(S)
    models(S)
    errors(S)
    limitations()
    reproduce()

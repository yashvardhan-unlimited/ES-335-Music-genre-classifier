"""Local web app: classify an upload or a sample clip, and browse every validation result.

Run from repo root: .venv/Scripts/python.exe -m streamlit run app.py
Logic lives in src/webapp.py. Only validation scores exist until the pre-test-freeze tag.
"""
import io

import altair as alt
import pandas as pd
import streamlit as st

from src import webapp as w
from src.splits import is_test_unlocked

BLUE, GREY, INK, GRID = "#2a78d6", "#8a8984", "#52514e", "#e4e3df"
BLUES = ["#e9f2fd", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]     # sequential, light to dark
LINE_COLORS = {"cnn": "#e87ba4", "svm": "#1baf7a", "logreg": "#eb6834", "xgb": "#eda100", "knn": "#2a78d6"}
LINE_SHAPES = {"cnn": "circle", "svm": "square", "logreg": "triangle-up", "xgb": "diamond", "knn": "cross"}
SPLIT_LABELS = {"A": "A window-random", "B": "B clip-random", "Bp": "B' fault-filtered", "C": "C artist-disjoint"}
FIGURES = [("cnn_curves_C.png", "CNN training curves on split C"),
           ("analysis_learning_curves.png", "Learning curves: score against training artists"),
           ("analysis_window_length.png", "Window length"),
           ("analysis_feature_groups.png", "Which feature groups matter"),
           ("permutation_importance.png", "Permutation importance"),
           ("analysis_calibration.png", "Calibration"),
           ("analysis_voting.png", "Soft against majority vote"),
           ("leakage_lab.png", "Leakage lab: three deliberate mistakes")]
UPLOAD_TYPES = ["wav", "mp3", "flac", "ogg", "aac", "m4a"]

st.set_page_config(page_title="GTZAN genre classifier", layout="wide")


@st.cache_resource
def classifier() -> w.Classifier:
    return w.load_classifier()


@st.cache_data
def results() -> pd.DataFrame:
    return w.load_results()


@st.cache_data
def samples() -> pd.DataFrame:
    return w.sample_clips()


@st.cache_data
def predict_sample(clip_id: str) -> dict:
    return w.classify_sample(classifier(), clip_id)


@st.cache_data(max_entries=8)
def predict_upload(data: bytes) -> dict:
    return w.classify_audio(classifier(), io.BytesIO(data))


def show(chart: alt.Chart) -> None:
    """Shared chart styling: hairline grid, recessive axes, one surface."""
    st.altair_chart(chart.configure_view(strokeWidth=0)
                    .configure_axis(gridColor=GRID, domainColor=GRID, tickColor=GRID, labelColor=INK, titleColor=INK,
                                    labelFontSize=12, titleFontSize=12)
                    .configure_legend(labelColor=INK, titleColor=INK, labelFontSize=12), width="stretch", theme=None)


def bars(data: pd.DataFrame, label: str, value: str, axis_title: str, labelled: int = 0) -> alt.LayerChart:
    """Ranked horizontal bars; only the first `labelled` rows carry a printed value."""
    data = data.assign(text=[f"{v:.2f}" if i < labelled else "" for i, v in enumerate(data[value])])
    y = alt.Y(f"{label}:N", sort=None, title=None, axis=alt.Axis(ticks=False))
    x = alt.X(f"{value}:Q", title=axis_title, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(tickCount=5))
    base = alt.Chart(data).encode(y=y, x=x, tooltip=[f"{label}:N", alt.Tooltip(f"{value}:Q", format=".3f")])
    color = alt.condition(alt.datum.floor, alt.value(GREY), alt.value(BLUE)) if "floor" in data else alt.value(BLUE)
    return alt.layer(base.mark_bar(height=16, cornerRadiusEnd=4).encode(color=color),
                     base.mark_text(align="left", dx=5, color=INK, fontSize=12).encode(text="text:N"),
                     ).properties(height=30 * len(data) + 44)


def show_prediction(result: dict, truth: str | None = None) -> None:
    """Top genre, runners-up, all ten scores, and what the numbers do and do not mean."""
    scores, clf = result["scores"], classifier()
    st.subheader(f"Not sure: best guess {scores.index[0]}" if result["unsure"] else f"Prediction: {scores.index[0]}")
    top = ", ".join(f"{g} {s:.0%}" for g, s in scores.head(w.APP["top_k"]).items())
    st.write(f"Top {w.APP['top_k']}: {top}. {result['windows_agreeing']} of {result['n_windows']} windows "
             f"picked {scores.index[0]}.")
    if result["unsure"]:
        st.info(f"The top score is under {w.APP['min_confidence']:.0%}. On validation clips, answers this weak were "
                "right less than half the time, so treat this as a guess.")
    if truth is not None:
        (st.success if truth == scores.index[0] else st.warning)(
            f"{'Correct' if truth == scores.index[0] else 'Wrong'}: GTZAN labels this clip {truth}.")
    data = scores.rename("score").rename_axis("genre").reset_index()
    show(bars(data, "genre", "score", "model score (soft vote over 10 windows)", w.APP["top_k"]))
    st.caption(f"Scores are not calibrated probabilities: this CNN runs overconfident (clip ECE {clf.clip_ece:.3f} "
               f"on valid), so read them as a ranking.")


def classify_tab() -> None:
    clf = classifier()
    st.write(f"Serving the CNN run with the best validation score on split C (run seed {clf.seed}, "
             f"macro-F1 {clf.valid_macro_f1:.3f}). It knows only GTZAN's ten genres, so anything else gets the "
             "closest of those. Audio stays on this computer.")
    mode = st.radio("Audio", ["Try a sample clip", "Upload your own"], horizontal=True, label_visibility="collapsed")
    left, right = st.columns(2, gap="large")
    if mode == "Try a sample clip":
        clips = samples()
        names = {f"{c.genre}: {c.clip_id}": c for c in clips.itertuples()}
        with left:
            clip = names[st.selectbox("Sample clip", list(names))]
            st.audio(clip.path)
            st.caption("Random clips from the validation part. The model never trained on them, "
                       "though validation chose when training stopped. No test clips are used.")
        with right:
            show_prediction(predict_sample(clip.clip_id), clip.genre)
        return
    with left:
        upload = st.file_uploader("Audio file", type=UPLOAD_TYPES, help=f"At least 3 seconds. Only the first "
                                  f"{w.APP['max_seconds'] // 60} minutes are used.")
        if upload is not None:
            st.audio(upload)
    if upload is not None:
        with right:
            try:
                with st.spinner("Classifying..."):
                    result = predict_upload(upload.getvalue())
            except ValueError as e:
                st.error(str(e))
            else:
                show_prediction(result)


def overview(table: pd.DataFrame, res: pd.DataFrame) -> None:
    best, floor = table.iloc[0], table.loc["dummy_stratified", "macro_f1"]
    trained = table.drop(["dummy_stratified", "dummy_most_frequent"]).index
    c = w.ladder(res).set_index(["model", "split"])["mean"]
    drop = (c.xs("A", level="split") - c.xs("C", level="split")).reindex(trained)
    a, b, d, e = st.columns(4)
    a.metric(f"Best macro-F1 ({w.MODEL_NAMES[best.name]})", f"{best['macro_f1']:.3f}")
    b.metric("Its accuracy", f"{best['accuracy']:.3f}")
    d.metric("Random-guess floor", f"{floor:.3f}")
    e.metric(f"Leakage drop, A to C ({w.MODEL_NAMES[drop.idxmax()]})", f"{drop.max():.3f}")


def models_section(table: pd.DataFrame) -> None:
    st.subheader("Models on split C (unseen artists)")
    data = table.reset_index().assign(name=lambda d: d["model"].map(w.MODEL_NAMES),
                                      floor=lambda d: d["model"].str.startswith("dummy"))
    show(bars(data, "name", "macro_f1", "valid clip macro-F1", labelled=len(data)))
    with st.expander("Table"):
        st.dataframe(data[["name", "macro_f1", "sd", "accuracy", "majority_macro_f1", "window_accuracy", "runs"]]
                     .rename(columns={"name": "model", "macro_f1": "macro-F1 (soft vote)", "sd": "sd over runs",
                                      "majority_macro_f1": "macro-F1 (majority)", "window_accuracy": "window accuracy"}),
                     hide_index=True, width="stretch")


def ladder_section(res: pd.DataFrame) -> None:
    st.subheader("How much survives as leakage is removed")
    data = w.ladder(res)
    data = data[data["model"].isin(LINE_COLORS)].assign(
        name=lambda d: d["model"].map(w.MODEL_NAMES), step=lambda d: d["split"].map(SPLIT_LABELS))
    names = [w.MODEL_NAMES[m] for m in LINE_COLORS]
    x = alt.X("step:N", sort=list(SPLIT_LABELS.values()), title=None, axis=alt.Axis(labelAngle=0, grid=False))
    y = alt.Y("mean:Q", title="valid clip macro-F1", scale=alt.Scale(domain=[0, 1]))
    color = alt.Color("name:N", title=None, sort=names,
                      scale=alt.Scale(domain=names, range=list(LINE_COLORS.values())))
    shape = alt.Shape("name:N", title=None, sort=names,
                      scale=alt.Scale(domain=names, range=list(LINE_SHAPES.values())))
    tip = ["name:N", "step:N", alt.Tooltip("mean:Q", format=".3f"), alt.Tooltip("sd:Q", format=".3f"), "runs:Q"]
    base = alt.Chart(data).encode(x=x, y=y, color=color, tooltip=tip)
    show(alt.layer(base.mark_line(strokeWidth=2), base.mark_point(size=90, filled=True, opacity=1, stroke="#ffffff",
                                                                  strokeWidth=2).encode(shape=shape)).properties(height=380))
    st.caption("A leaks most (windows of one clip on both sides), C least. B and B' average 5 split seeds. "
               "A gap in a line means that run has not finished.")
    with st.expander("Table"):
        table = data.assign(cell=lambda d: d["mean"].map("{:.3f}".format).where(d["mean"].notna(), "not run")
                            + (" +/- " + d["sd"].map("{:.3f}".format)).where(d["sd"].notna(), ""))
        st.dataframe(table.pivot(index="name", columns="step", values="cell").reindex(index=names)[
            list(SPLIT_LABELS.values())], width="stretch")


def genre_section(res: pd.DataFrame) -> None:
    st.subheader("Per genre on split C")
    labels = {w.MODEL_NAMES[m]: m for m in LINE_COLORS}
    model = labels[st.selectbox("Model", list(labels))]
    left, right = st.columns(2, gap="large")
    with left:
        f1 = w.genre_f1(res, model).rename("f1").rename_axis("genre").reset_index()
        show(bars(f1, "genre", "f1", "valid clip F1", labelled=len(f1)).properties(height=400))
    with right:
        cm = w.confusion(res, model)
        data = cm.rename_axis("true").reset_index().melt("true", var_name="predicted", value_name="share")
        data = data.assign(text=data["share"].map(lambda v: f"{v:.0%}" if v >= 0.1 else ""))
        axis = dict(sort=w.GENRES)
        base = alt.Chart(data).encode(
            x=alt.X("predicted:N", title="predicted", axis=alt.Axis(labelAngle=-45, grid=False), **axis),
            y=alt.Y("true:N", title="true genre", axis=alt.Axis(grid=False), **axis),
            tooltip=["true:N", "predicted:N", alt.Tooltip("share:Q", format=".0%")])
        show(alt.layer(base.mark_rect(stroke="#ffffff", strokeWidth=2).encode(
            color=alt.Color("share:Q", title="share of row", scale=alt.Scale(domain=[0, 1], range=BLUES))),
            base.mark_text(fontSize=11).encode(text="text:N", color=alt.condition(
                alt.datum.share > 0.5, alt.value("#ffffff"), alt.value("#1f1f1d")))).properties(height=400))
    with st.expander("Tables"):
        st.dataframe(f1, hide_index=True, width="stretch")
        st.dataframe(cm.style.format("{:.0%}"), width="stretch")


def results_tab() -> None:
    res = results()
    if is_test_unlocked():
        st.info("The pre-test-freeze tag exists. This page still shows validation scores only.")
    else:
        st.info("All scores here are **validation** scores, which are slightly optimistic because validation chose "
                "the settings. Test scores and bootstrap confidence intervals stay locked until the pre-test-freeze tag.")
    missing = w.missing_runs(res)
    if missing:
        st.warning(f"The validation protocol is unfinished: {len(missing)} runs are missing "
                   f"({', '.join(missing)}), so those cells are blank.")
    table = w.model_table(res)
    overview(table, res)
    models_section(table)
    ladder_section(res)
    genre_section(res)
    st.subheader("Analyses")
    tabs = st.tabs([title for _, title in FIGURES])
    for tab, (name, title) in zip(tabs, FIGURES):
        path = w.RESULTS / "figures" / name
        if path.exists():
            tab.image(str(path), caption=title)


st.title("GTZAN genre classifier")
try:
    classifier()
except FileNotFoundError as e:
    st.error(f"Trained model not found ({e.filename}). Run the CNN training first.")
    st.stop()
classify, dashboard = st.tabs(["Classify", "Results"])
with classify:
    classify_tab()
with dashboard:
    results_tab()

"""Streamlit front end for the skin-lesion screening prototype.

Run with::

    streamlit run app/streamlit_app.py

Design notes:

* The "not a diagnosis" disclaimer is visible at all times -- at the top of the
  page, in the sidebar, and inside every generated report.
* A photo that fails the quality gate produces a retake prompt instead of a
  risk score. The user can override, but only deliberately, and the report then
  carries the caution.
* If PanDerm's weights are missing, or the model has not been trained, the UI
  says so plainly rather than presenting a confident-looking number from a
  fallback or untrained model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.pipeline import AnalysisResult, LesionAnalyzer  # noqa: E402


PAGE_TITLE = "Skin Lesion Risk Screening"
DISCLAIMER_SHORT = (
    "**This is a student project, not a medical device.** It does not diagnose "
    "anything. Never use it to rule out skin cancer."
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model (first run downloads weights)...")
def get_analyzer() -> LesionAnalyzer:
    """Build the analyzer once and reuse it across reruns.

    Streamlit re-executes the whole script on every interaction, so without
    caching the backbone would reload on each click.
    """
    return LesionAnalyzer(strict=False)


@st.cache_data
def get_config_snapshot() -> dict:
    """Return the loaded config as a plain dict (cached across reruns)."""
    return load_config().as_dict()


@st.cache_data
def find_sample_pairs() -> list[dict]:
    """Find prepared images to offer as one-click samples.

    Returns a list of ``{"label", "path", "previous"}`` dicts drawn from the
    unified CSV. ``previous`` points at another photo of the same lesion when
    one exists, so the sample can demonstrate change detection too. Returns an
    empty list when no data has been prepared yet.
    """
    import pandas as pd

    cfg = load_config()
    csv_path = cfg.path("paths.unified_csv")
    if not csv_path.exists():
        # Hosted deployments clone the repo without the prepared dataset (it is
        # gitignored). The demo images are deterministic from the config seed,
        # so regenerate them once rather than committing ~29 MB of blobs.
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
            from make_demo_dataset import generate

            with st.spinner("Preparing sample photos (first run only)..."):
                generate(cfg, per_class=6, size=256, merge=True)
        except Exception:  # noqa: BLE001 - samples are optional, never fatal
            return []
        if not csv_path.exists():
            return []

    try:
        frame = pd.read_csv(csv_path)
    except Exception:  # noqa: BLE001
        return []

    samples: list[dict] = []
    for risk in cfg.get("labels.risk_classes"):
        subset = frame[frame["risk_label"] == risk]
        if subset.empty:
            continue
        row = subset.iloc[0]
        path = load_config().resolve(row["image_path"])
        if not Path(path).exists():
            continue

        previous: Optional[str] = None
        siblings = subset[(subset["lesion_id"] == row["lesion_id"])
                          & (subset["image_path"] != row["image_path"])]
        if not siblings.empty:
            candidate = load_config().resolve(siblings.iloc[0]["image_path"])
            if Path(candidate).exists():
                previous = str(candidate)

        samples.append({
            "label": f"{risk} risk example ({row['source']})",
            "path": str(path),
            "previous": previous,
            "source": str(row["source"]),
        })
    return samples


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------
def render_header() -> None:
    """Render the page title and the always-visible disclaimer."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🔬", layout="wide")
    st.title("🔬 Skin Lesion Risk Screening")
    st.caption("Explainable classification and risk assessment - research prototype")
    st.error(
        DISCLAIMER_SHORT + " If you are worried about a mark on your skin, or it "
        "is changing, bleeding, itching or not healing, see a doctor regardless "
        "of what this tool says."
    )


def render_sidebar(analyzer: LesionAnalyzer) -> dict:
    """Render the sidebar and return the collected patient inputs.

    Returns:
        A dict with keys ``age``, ``sex``, ``site``, ``frame_width_mm``,
        ``require_quality`` and ``explain``.
    """
    cfg = analyzer.cfg

    st.sidebar.header("About this photo")
    age = st.sidebar.number_input(
        "Age", min_value=0, max_value=110, value=45, step=1,
        help="Age matters: the same lesion carries different risk at 20 and 70.",
    )
    sex_options = [s for s in cfg.get("metadata.sex_vocab")]
    sex = st.sidebar.selectbox("Sex", sex_options,
                               index=sex_options.index("unknown")
                               if "unknown" in sex_options else 0)
    site_options = list(cfg.get("metadata.site_vocab"))
    site_labels = {
        "head_neck": "Head or neck",
        "trunk": "Trunk (chest, back, abdomen)",
        "upper_extremity": "Arm or hand",
        "lower_extremity": "Leg or foot",
        "palms_soles": "Palm, sole or nail",
        "genital": "Genital area",
        "unknown": "Not sure",
    }
    site = st.sidebar.selectbox(
        "Where on the body?", site_options,
        index=site_options.index("trunk") if "trunk" in site_options else 0,
        format_func=lambda s: site_labels.get(s, s.replace("_", " ").title()),
    )

    st.sidebar.divider()
    st.sidebar.subheader("Optional: give the photo a scale")
    st.sidebar.caption(
        "A photo has no scale, so we cannot measure the lesion in millimetres "
        "unless you tell us how wide the picture is in real life. Photograph a "
        "ruler alongside the lesion and enter the width here."
    )
    use_scale = st.sidebar.checkbox("I know how wide the photo is", value=False)
    frame_width_mm: Optional[float] = None
    if use_scale:
        frame_width_mm = st.sidebar.number_input(
            "Photo width in real life (mm)", min_value=5.0, max_value=500.0,
            value=40.0, step=5.0,
        )

    st.sidebar.divider()
    st.sidebar.subheader("Settings")
    require_quality = st.sidebar.checkbox(
        "Reject poor-quality photos", value=True,
        help="Turn off to analyse a photo that failed the quality checks. "
             "The result will be less reliable and the report will say so.",
    )
    explain = st.sidebar.checkbox("Compute Grad-CAM heatmap", value=True)

    st.sidebar.divider()
    st.sidebar.subheader("System status")
    backbone = analyzer.model.backbone
    if backbone.is_fallback:
        st.sidebar.warning(
            f"Using the fallback backbone (`{backbone.name}`). This is a "
            "general ImageNet model, not the PanDerm dermatology model. See "
            "the README to enable PanDerm."
        )
    else:
        st.sidebar.success(f"Using PanDerm: `{backbone.name}`")

    if not analyzer.is_trained:
        st.sidebar.error(
            "No trained checkpoint found. Run `python src/model/train.py` -- "
            "until then, risk levels are meaningless."
        )
    else:
        st.sidebar.info(f"Model ready ({backbone.feature_dim}-dim features)")

    st.sidebar.divider()
    st.sidebar.caption(DISCLAIMER_SHORT)

    return {
        "age": age,
        "sex": sex,
        "site": site,
        "frame_width_mm": frame_width_mm,
        "require_quality": require_quality,
        "explain": explain,
    }


def render_quality_failure(result: AnalysisResult) -> None:
    """Render the retake prompt shown when a photo fails the quality gate."""
    st.warning("### Please take another photo")
    st.write(result.quality.reason)

    with st.expander("What we checked"):
        for check in result.quality.checks:
            icon = "✅" if check.passed else "❌"
            st.write(f"{icon} **{check.name}** - measured {check.value:.1f} "
                     f"(threshold: {check.threshold})")

    st.info(
        "**Tips for a good photo:** good even light (no direct flash), the "
        "lesion centred and filling maybe a third of the frame, camera about "
        "15-20 cm away, and hold still until it focuses."
    )
    st.caption("You can turn off the quality check in the sidebar to analyse "
               "this photo anyway - but the result will be less reliable.")


def render_risk_banner(result: AnalysisResult) -> None:
    """Render the headline risk level, recommendation and confidence."""
    report = result.report
    assert report is not None

    palette = {"Low": ("🟢", "success"), "Moderate": ("🟠", "warning"),
               "High": ("🔴", "error")}
    icon, kind = palette.get(report.risk_level, ("⚪", "info"))

    message = (f"## {icon} Risk level: {report.risk_level}\n\n"
               f"**What to do:** {report.recommendation}")
    getattr(st, kind)(message)

    columns = st.columns(len(report.probabilities) + 1)
    columns[0].metric("Confidence", f"{report.confidence:.0%}",
                      help="How sure the model is of its own prediction.")
    for column, (label, probability) in zip(columns[1:], report.probabilities.items()):
        column.metric(label, f"{probability:.0%}")

    if report.escalation_reasons:
        st.info("**We raised the advice above the model's own prediction "
                f"({report.model_risk_level}) because:** "
                + "; ".join(report.escalation_reasons) + ".")


def render_explanations(result: AnalysisResult) -> None:
    """Render the image panel, ABCDE breakdown and change comparison."""
    report = result.report
    assert report is not None

    left, right = st.columns(2)
    with left:
        st.subheader("Your photo")
        st.image(result.image, width='stretch')
    with right:
        st.subheader("Where the model looked")
        if result.gradcam is not None:
            st.image(result.gradcam.overlay, width='stretch')
            ratio = result.gradcam.focus_ratio
            if ratio is not None:
                st.caption(
                    f"{result.gradcam.peak_inside_lesion:.0%} of the model's "
                    f"attention fell on the lesion, which covers "
                    f"{result.gradcam.lesion_area_fraction:.0%} of the photo - "
                    f"{ratio:.1f}x more concentrated than chance. Red is where "
                    "the model looked hardest."
                )
            else:
                st.caption("The heatmap for this image was not informative.")
        else:
            st.info("Heatmap not computed.")

    if result.abcde is not None:
        st.subheader("What we looked at - the ABCDE rule")
        st.write(result.abcde.summary())
        for feature in result.abcde.features:
            if not feature.measurable:
                icon = "❔"
            elif feature.flagged:
                icon = "⚠️"
            else:
                icon = "✅"
            with st.expander(f"{icon} **{feature.letter}** - {feature.title}",
                             expanded=feature.flagged):
                st.write(feature.sentence)
                if feature.score is not None and feature.threshold is not None:
                    st.caption(f"Measured {feature.score:.3f} "
                               f"(flagged above {feature.threshold:.3f})")

    if result.change is not None:
        st.subheader("Compared with your earlier photo")
        if result.change.changed:
            st.warning(f"**Meaningful change detected.** {result.change.reason}")
        else:
            st.success(f"**No significant change.** {result.change.reason}")
        with st.expander("Signal breakdown"):
            for signal in result.change.signals:
                icon = "⚠️" if signal.changed else "✅"
                st.write(f"{icon} **{signal.name}**: {signal.value:.3f} "
                         f"(threshold {signal.threshold:.3f}) - {signal.description}")

    if report.notes:
        st.subheader("Please note")
        for note in report.notes:
            st.warning(note)


def render_downloads(result: AnalysisResult) -> None:
    """Render the full report and its download buttons."""
    report = result.report
    assert report is not None

    st.subheader("Full report")
    with st.expander("Read the full report", expanded=False):
        st.markdown(report.to_markdown(), unsafe_allow_html=False)

    left, middle, right = st.columns(3)
    left.download_button("Download report (Markdown)",
                         report.to_markdown(include_images=True),
                         file_name="lesion_report.md", mime="text/markdown")
    middle.download_button("Download report (HTML)",
                           report.to_html(include_images=True),
                           file_name="lesion_report.html", mime="text/html")
    right.download_button("Download data (JSON)", report.to_json(),
                          file_name="lesion_report.json", mime="application/json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for the Streamlit app."""
    render_header()

    try:
        analyzer = get_analyzer()
    except Exception as exc:  # noqa: BLE001 - show the user something actionable
        st.exception(exc)
        st.error("Could not load the model. Check the README for setup steps.")
        return

    inputs = render_sidebar(analyzer)

    st.subheader("1. Upload a photo of the lesion")
    uploaded = st.file_uploader(
        "A clear, well-lit photo with the lesion centred",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="current_photo",
    )

    st.subheader("2. Optional: add an earlier photo of the same lesion")
    st.caption(
        "Change over time is one of the strongest warning signs there is, and "
        "it is the one thing we cannot check from a single photo."
    )
    previous = st.file_uploader(
        "An older photo of the same lesion",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="previous_photo",
    )

    # Demo path: lets the app be exercised without a phone photo to hand.
    samples = find_sample_pairs()
    if uploaded is None and samples:
        st.subheader("...or try one of the prepared examples")
        use_sample = st.checkbox("Use a sample photo", value=False, key="use_sample")
        if use_sample:
            labels = [s["label"] for s in samples]
            chosen = st.selectbox("Which example?", labels, key="sample_choice")
            sample = samples[labels.index(chosen)]
            uploaded = sample["path"]
            if previous is None and sample["previous"]:
                previous = sample["previous"]
                st.caption("This example includes an earlier photo of the same "
                           "lesion, so change detection will run too.")
            if sample["source"] == "synthetic_demo":
                st.warning(
                    "This is a **synthetic** demo image - a procedurally drawn "
                    "blob, not a real lesion. It exists to show the pipeline "
                    "working, and says nothing about clinical accuracy."
                )

    if uploaded is None:
        st.info("Upload a photo above to get started.")
        with st.expander("How to take a good photo"):
            st.markdown(
                "- Use bright, even light. Daylight near a window works well.\n"
                "- Avoid direct flash: it washes out colour, which matters here.\n"
                "- Hold the camera about 15-20 cm from the skin.\n"
                "- Centre the lesion; let it fill roughly a third of the frame.\n"
                "- Keep still until the camera focuses.\n"
                "- Include a ruler beside the lesion if you want a size estimate."
            )
        return

    st.subheader("3. Result")
    if not analyzer.is_trained:
        st.error(
            "**No trained model found.** Run `python src/model/train.py` first. "
            "Anything below comes from an untrained network and means nothing."
        )

    with st.spinner("Analysing..."):
        try:
            result = analyzer.analyse(
                uploaded,
                age=inputs["age"],
                sex=inputs["sex"],
                site=inputs["site"],
                previous_image=previous,
                frame_width_mm=inputs["frame_width_mm"],
                require_quality=inputs["require_quality"],
                explain=inputs["explain"],
            )
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)
            st.error("Analysis failed. See the traceback above.")
            return

    if result.stopped_at_quality_gate:
        st.image(result.image, caption="The photo you uploaded", width=360)
        render_quality_failure(result)
        return

    render_risk_banner(result)
    st.divider()
    render_explanations(result)
    st.divider()
    render_downloads(result)

    st.divider()
    st.caption(analyzer.cfg.get("report.disclaimer"))


if __name__ == "__main__":
    main()

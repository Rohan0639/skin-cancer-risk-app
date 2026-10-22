"""Smoke tests for the Streamlit app.

Streamlit's ``AppTest`` runs the script headlessly and exposes the rendered
elements, so these check that the app actually renders -- including the full
analysis path via the built-in sample selector -- rather than merely importing.
"""

from __future__ import annotations

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

APP_PATH = "app/streamlit_app.py"
TIMEOUT = 600


@pytest.fixture(scope="module")
def app():
    """Render the app once with no photo uploaded."""
    return AppTest.from_file(APP_PATH, default_timeout=TIMEOUT).run()


@pytest.mark.slow
class TestInitialRender:
    def test_renders_without_exception(self, app):
        assert not list(app.exception)

    def test_shows_the_title(self, app):
        assert any("Skin Lesion Risk Screening" in t.value for t in app.title)

    def test_disclaimer_is_visible_up_front(self, app):
        """The disclaimer must be on screen before any result is shown."""
        banners = [e.value for e in app.error] + [e.value for e in app.sidebar.error]
        assert any("not a medical device" in text.lower() for text in banners)

    def test_collects_the_patient_metadata(self, app):
        assert len(app.sidebar.number_input) >= 1     # age
        assert len(app.sidebar.selectbox) >= 2        # sex, site

    def test_offers_both_upload_slots(self, app):
        headers = [s.value for s in app.subheader]
        assert any("Upload a photo" in h for h in headers)
        assert any("earlier photo" in h for h in headers)

    def test_reports_which_backbone_is_in_use(self, app):
        status = ([w.value for w in app.sidebar.warning]
                  + [s.value for s in app.sidebar.success])
        assert any("PanDerm" in text for text in status), \
            "The UI must always say whether PanDerm or the fallback is running"

    def test_prompts_for_a_photo(self, app):
        assert any("Upload a photo" in i.value for i in app.info)


@pytest.mark.slow
class TestSampleAnalysisPath:
    """Drives the full analysis through the app's built-in sample selector.

    ``AppTest`` cannot simulate a file upload, so the sample selector is what
    makes the render-a-result path testable at all -- as well as being the
    quickest way to demo the tool.
    """

    @pytest.fixture(scope="class")
    def analysed(self):
        at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT).run()
        if not at.checkbox(key="use_sample"):
            pytest.skip("No prepared data; run data/make_demo_dataset.py first.")
        at.checkbox(key="use_sample").set_value(True).run()
        return at

    def test_runs_without_exception(self, analysed):
        assert not list(analysed.exception)

    def test_offers_one_sample_per_risk_class(self, analysed):
        options = analysed.selectbox(key="sample_choice").options
        assert len(options) >= 1
        assert any("risk example" in option for option in options)

    def test_shows_a_risk_level_and_recommendation(self, analysed):
        banners = [e.value for kind in ("success", "warning", "error")
                   for e in getattr(analysed, kind)]
        assert any("Risk level" in text for text in banners)
        assert any("What to do" in text for text in banners)

    def test_shows_a_confidence_metric(self, analysed):
        labels = [m.label for m in analysed.metric]
        assert "Confidence" in labels

    def test_renders_photo_and_heatmap(self, analysed):
        headers = [s.value for s in analysed.subheader]
        assert "Your photo" in headers
        assert "Where the model looked" in headers

    def test_renders_the_abcde_section(self, analysed):
        assert any("ABCDE" in s.value for s in analysed.subheader)

    def test_offers_the_report_for_download(self, analysed):
        assert any("Full report" in s.value for s in analysed.subheader)

    def test_flags_synthetic_samples_as_synthetic(self, analysed):
        """A synthetic demo image must never be mistaken for a real lesion."""
        options = analysed.selectbox(key="sample_choice").options
        if not any("synthetic" in option for option in options):
            pytest.skip("Prepared data is not the synthetic demo set.")
        warnings = [w.value for w in analysed.warning]
        assert any("synthetic" in text.lower() for text in warnings)

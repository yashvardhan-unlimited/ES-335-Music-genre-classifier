"""The Project tab renders every section from docs/data/site.json without an exception."""
from streamlit.testing.v1 import AppTest


def test_project_tab_renders_and_is_interactive():
    at = AppTest.from_file("../app.py", default_timeout=120).run()
    assert not at.exception
    assert len(at.header) >= 9
    next(s for s in at.selectbox if s.key == "ladder_model").select("RBF SVM")
    next(s for s in at.selectbox if s.key == "cm_run").select("svm_s0")
    assert not at.run().exception

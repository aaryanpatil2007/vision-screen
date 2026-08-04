from visionscreen.data.pmc_cases import parse_figures

FIG = """<fig id="f1"><label>Figure 2</label><caption><p>Preoperative photograph
of a 26-year-old male patient with esotropia of 18 prism diopters at distance.
</p></caption><graphic xlink:href="case-f2.jpg"/></fig>"""

AMBIGUOUS = """<fig id="f2"><caption><p>Appearance before (30 PD) and after
(4 PD) surgery.</p></caption><graphic xlink:href="c.jpg"/></fig>"""

NOT_A_PHOTO = """<fig id="f3"><caption><p>Schematic diagram of the surgical
approach; deviation was 20 prism diopters.</p></caption>
<graphic xlink:href="d.jpg"/></fig>"""

NO_NUMBER = """<fig id="f4"><caption><p>Preoperative photograph of the
patient.</p></caption><graphic xlink:href="e.jpg"/></fig>"""


def test_extracts_photograph_and_measurement():
    got = parse_figures(FIG, "PMC1")
    assert len(got) == 1
    assert got[0]["deviation_pd"] == 18
    assert got[0]["graphic"] == "case-f2.jpg"
    assert "26-year-old" in got[0]["caption"]


def test_rejects_ambiguous_multi_value_captions():
    """Two numbers means the caption does not say which one the photo shows."""
    assert parse_figures(AMBIGUOUS, "PMC2") == []


def test_rejects_non_photographs():
    assert parse_figures(NOT_A_PHOTO, "PMC3") == []


def test_rejects_captions_without_a_measurement():
    assert parse_figures(NO_NUMBER, "PMC4") == []


def test_caption_excludes_processing_metadata():
    """JATS carries status text that naive tag-stripping pulls in."""
    noisy = FIG.replace("<fig id=\"f1\">",
                        "<fig id=\"f1\"><processing-meta>pmc-status live</processing-meta>")
    got = parse_figures(noisy, "PMC5")
    assert len(got) == 1
    assert "pmc-status" not in got[0]["caption"]

"""Parsing documents that came from outside: what is refused, and by whom.

Every XML this service reads is someone else's — an uploaded GPX or SVG, and
the CityGML it downloads. The two attacks that live in the format are an entity
that reads a local file back out, and one that expands into gigabytes. libxml2
happens to refuse both already; these tests are what make that a promise of
this program rather than of whichever version is installed.
"""
from __future__ import annotations

import io

import pytest
from lxml import etree

from app.core import safexml
from app.core.gpx import parse_gpx


def doc_reading(path) -> bytes:
    """A document that tries to read `path` back out through an entity."""
    return (f'<?xml version="1.0"?>\n'
            f'<!DOCTYPE r [<!ENTITY x SYSTEM "file://{path}">]>\n'
            f'<r><v>&x;</v></r>').encode()


BOMB = b'''<?xml version="1.0"?>
<!DOCTYPE r [
 <!ENTITY a "AAAAAAAAAA">
 <!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">
 <!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;">
 <!ENTITY d "&c;&c;&c;&c;&c;&c;&c;&c;&c;&c;">
 <!ENTITY e "&d;&d;&d;&d;&d;&d;&d;&d;&d;&d;">
]>
<r><v>&e;</v></r>'''


# ---- what a hostile document cannot do -------------------------------------

def test_an_entity_never_reaches_the_file_it_names(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("SENSITIVE")
    root = safexml.fromstring(doc_reading(secret))
    assert b"SENSITIVE" not in etree.tostring(root)
    assert root[0].text is None          # left unexpanded, not resolved


def test_an_entity_is_not_expanded_even_when_it_is_harmless():
    """Refusing to expand at all is what makes the blow-up impossible; there is
    no threshold to be on the wrong side of."""
    root = safexml.fromstring(b'<!DOCTYPE r [<!ENTITY e "XX">]><r><v>&e;</v></r>')
    assert root[0].text is None


def test_an_entity_that_expands_into_itself_gets_nowhere():
    root = safexml.fromstring(BOMB)
    assert len(etree.tostring(root)) < 200


def test_a_document_nested_past_all_reason_is_refused():
    """`huge_tree=False` is what keeps libxml2's own ceiling on depth; the
    option exists mainly so that nobody turns it off without meaning to."""
    deep = b"<a>" + b"<b>" * 2000 + b"</b>" * 2000 + b"</a>"
    with pytest.raises(etree.XMLSyntaxError):
        safexml.fromstring(deep)


# ---- what an ordinary document still does ----------------------------------

def test_the_predefined_entities_are_untouched():
    """`&amp;` and its four siblings are part of XML itself, not declarations,
    and real files are full of them."""
    root = safexml.fromstring(b"<r><v>Ka &amp; Ku &lt;1&gt;</v></r>")
    assert root[0].text == "Ka & Ku <1>"


def test_streaming_a_tag_honours_the_same_refusals(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("SENSITIVE")
    body = doc_reading(secret).replace(b"<r>", b"<r><v>plain</v>")
    got = [e.text for _, e in safexml.iterparse(io.BytesIO(body), "v")]
    assert got == ["plain", None]


# ---- spotting the declarations expat would expand --------------------------

def test_declared_entities_are_reported_by_name():
    root = safexml.fromstring(b'<!DOCTYPE r [<!ENTITY a "A"><!ENTITY b "B">]><r/>')
    assert safexml.entity_declarations(root) == ["a", "b"]


def test_an_ordinary_document_declares_nothing():
    assert safexml.entity_declarations(safexml.fromstring(b"<r/>")) == []


# ---- the uploaded GPX, through the real entry point ------------------------

def test_a_gpx_cannot_read_a_file_off_the_server(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("SENSITIVE")
    doc = (f'<!DOCTYPE gpx [<!ENTITY x SYSTEM "file://{secret}">]>'
           f'<gpx><trk><trkseg>'
           f'<trkpt lat="35.5" lon="139.5"><name>&x;</name></trkpt>'
           f'<trkpt lat="35.6" lon="139.6"/></trkseg></trk></gpx>').encode()
    # The track still parses — the entity is simply never resolved, so there is
    # nothing to leak and no reason to reject an otherwise good file.
    track = parse_gpx(doc)
    assert track.lats == [35.5, 35.6]

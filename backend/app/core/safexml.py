"""Parsing XML that came from somewhere else.

Every document this service reads is someone else's: the GPX and the SVG a
caller uploads, and the CityGML it downloads from PLATEAU. Two attacks live in
the format itself — a SYSTEM entity that reads a local file back out, and an
entity that expands into gigabytes — and both are refused here by asking, not
by being lucky about a library's defaults.

libxml2 does already refuse them: out of the box it will not load an external
subset, and it caps entity amplification. But nothing in this program asked, so
the safety was a property of the version that happened to be installed, and an
upgrade or a differently built wheel could take it away with every test still
passing. Naming the options is what stops that being silent.

The standard library's expat — which `svgelements` parses with, and which this
module cannot configure — is the weaker of the two: it permits any expansion at
all below 8 MiB and a hundredfold above that, so an upload at the 5 MB ceiling
could still become hundreds of megabytes of text. `entity_declarations` is here
for the caller that therefore has to look for the declarations itself.
"""
from __future__ import annotations

from lxml import etree

# What is refused, and why:
#   resolve_entities  nothing is ever expanded, so neither a SYSTEM entity nor
#                     a self-referential one can do anything at all
#   load_dtd          no external subset is read...
#   no_network        ...and nothing would be fetched over the network anyway
#   huge_tree         keeps libxml2's own ceilings on depth and node count,
#                     which is exactly what huge_tree=True exists to remove
#   recover           a malformed document is an error to report, not
#                     something to guess the rest of
_HARDENED = dict(resolve_entities=False, load_dtd=False, no_network=True,
                 huge_tree=False, recover=False)


def parser() -> etree.XMLParser:
    """A parser that resolves and expands nothing it is told to."""
    return etree.XMLParser(**_HARDENED)


def fromstring(data: bytes):
    """Parse a whole document held in memory."""
    return etree.fromstring(data, parser=parser())


def iterparse(source, tag):
    """Stream one tag out of a document without holding all of it."""
    return etree.iterparse(source, tag=tag, **_HARDENED)


def entity_declarations(root) -> list[str]:
    """Names of the entities a document declares in its own internal subset.

    Ordinary files declare none — a design tool, a GPS watch and PLATEAU all
    emit documents whose `internalDTD` is simply absent.
    """
    dtd = root.getroottree().docinfo.internalDTD
    return [] if dtd is None else [e.name for e in dtd.iterentities()]

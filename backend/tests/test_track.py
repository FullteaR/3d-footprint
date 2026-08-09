"""The GPX track as a raised ridge that hugs the terrain."""
from __future__ import annotations

import numpy as np
import pytest

from app.core.gpx import Track
from app.core.track import track_ridge

from conftest import DEG, LAT0, LON0


def straight_track(n=12) -> Track:
    """West to east across the middle of the fixture grids."""
    lons = np.linspace(LON0 + 0.1 * DEG, LON0 + 0.9 * DEG, n)
    return Track(lats=[LAT0 + 0.5 * DEG] * n, lons=lons.tolist())


def test_the_ridge_is_a_watertight_solid(hill_proj):
    mesh = track_ridge(straight_track(), hill_proj)
    assert mesh.is_watertight
    assert mesh.volume > 0


def test_the_ridge_stands_the_asked_height_above_the_terrain(hill_proj):
    """It has to follow the relief, so "height" is measured against the
    surface under each station, not against z=0."""
    track = straight_track()
    mesh = track_ridge(track, hill_proj, width_mm=1.2, height_mm=1.5)
    surface = hill_proj.sample_z(np.asarray(track.lons), np.asarray(track.lats))
    assert mesh.vertices[:, 2].max() == pytest.approx(surface.max() + 1.5)


def test_the_ridge_is_embedded_so_it_merges_with_the_ground(hill_proj):
    """A ridge that only touched the surface would print as a floating line."""
    track = straight_track()
    mesh = track_ridge(track, hill_proj, embed_mm=0.6)
    surface = hill_proj.sample_z(np.asarray(track.lons), np.asarray(track.lats))
    assert mesh.vertices[:, 2].min() == pytest.approx(surface.min() - 0.6)


def test_the_ridge_is_the_asked_width(hill_proj):
    mesh = track_ridge(straight_track(), hill_proj, width_mm=2.0)
    lo, hi = mesh.bounds
    assert hi[1] - lo[1] == pytest.approx(2.0)


def test_duplicate_points_do_not_break_the_tangents(hill_proj):
    """Loggers repeat a fix when you stand still; the sweep drops those."""
    lon = LON0 + 0.5 * DEG
    track = Track(lats=[LAT0 + 0.5 * DEG] * 4,
                  lons=[lon, lon, lon + 0.2 * DEG, lon + 0.2 * DEG])
    assert track_ridge(track, hill_proj).is_watertight


def test_a_track_that_stands_still_has_nothing_to_sweep(hill_proj):
    """A sliver that only grazes the model border ends up like this; the
    caller drops it rather than failing the whole generation."""
    lon, lat = LON0 + 0.5 * DEG, LAT0 + 0.5 * DEG
    with pytest.raises(ValueError, match="too few distinct points"):
        track_ridge(Track(lats=[lat, lat], lons=[lon, lon]), hill_proj)

"""The print projection and the terrain solid.

The headline property is watertightness *per colour*: a multi-material printer
needs each colour to be its own closed solid running the full height, so every
body that comes out of `terrain_solid` has to be a volume on its own.
"""
from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon, box

from app.core.mesh import _douglas_peucker, make_projection, terrain_solid
from app.core.region import Region

from conftest import DEG, LAT0, LON0, N, make_grid


# ---- projection ------------------------------------------------------------

def test_the_lowest_valid_elevation_is_the_z_datum(island_grid, make_proj):
    """z=0 is the lowest ground there is data for — the no-data ring around
    the island must not drag the datum down with it."""
    proj = make_proj(island_grid)
    assert proj.emin == pytest.approx(np.nanmin(island_grid.elev))
    assert proj.z_of(proj.emin) == 0.0


def test_no_data_is_filled_to_the_datum_not_left_nan(island_grid, make_proj):
    proj = make_proj(island_grid)
    assert np.isfinite(proj.filled).all()
    assert proj.filled.min() == pytest.approx(proj.emin)


def test_an_area_with_no_elevation_at_all_is_an_error(make_proj):
    with pytest.raises(ValueError, match="no valid elevation"):
        make_proj(make_grid(np.full((4, 4), np.nan)))


def test_size_mm_maps_onto_the_longest_edge(flat_grid, make_proj):
    proj = make_proj(flat_grid, size_mm=120.0)
    width = proj.x_of(flat_grid.lons.max()) - proj.x_of(flat_grid.lons.min())
    height = proj.y_of(flat_grid.lats.max()) - proj.y_of(flat_grid.lats.min())
    assert max(width, height) == pytest.approx(120.0)


def test_span_m_overrides_the_grid_extent(flat_grid, make_proj):
    """The requested outline is what size_mm applies to — the fetched grid can
    snap a hair wider at DEM pixel edges, and the scale readout must not."""
    proj = make_proj(flat_grid, size_mm=100.0, span_m=1000.0)
    assert proj.scale == pytest.approx(0.1)


@pytest.mark.parametrize("vertical_scale", [1.0, 8.0])
def test_horizontal_and_vertical_mappings_invert(flat_grid, make_proj,
                                                 vertical_scale):
    proj = make_proj(flat_grid, vertical_scale=vertical_scale)
    assert proj.lon_of(proj.x_of(139.004)) == pytest.approx(139.004)
    assert proj.lat_of(proj.y_of(35.004)) == pytest.approx(35.004)
    assert proj.elev_of(proj.z_of(42.0)) == pytest.approx(42.0)


def test_sampling_hits_the_grid_nodes_exactly(hill_grid, make_proj):
    proj = make_proj(hill_grid)
    z = proj.sample_z(hill_grid.lons, hill_grid.lats[N // 2])
    assert z == pytest.approx(proj.z_of(hill_grid.elev[N // 2]))


def test_sampling_between_nodes_is_bilinear(make_proj):
    grid = make_grid(np.array([[0.0, 10.0], [0.0, 10.0]]))
    proj = make_proj(grid, vertical_scale=1.0)
    mid = (grid.lons[0] + grid.lons[1]) / 2
    assert proj.sample_z(mid, grid.lats[0]) == pytest.approx(proj.z_of(5.0))


# ---- the nameplate pocket --------------------------------------------------

def footprint(proj, half_deg=DEG / 8):
    c_lon = (proj.grid.lons[0] + proj.grid.lons[-1]) / 2
    c_lat = (proj.grid.lats[0] + proj.grid.lats[-1]) / 2
    return box(c_lon - half_deg, c_lat - half_deg, c_lon + half_deg, c_lat + half_deg)


def test_flatten_under_cuts_a_flat_floor_and_leaves_the_rest(hill_proj):
    poly = footprint(hill_proj)
    before = hill_proj.filled.copy()
    hill_proj.flatten_under(poly, -1.0)

    lon_g, lat_g = np.meshgrid(hill_proj.grid.lons, hill_proj.grid.lats)
    inside = shapely.contains_xy(poly, lon_g, lat_g)
    assert inside.any()
    assert hill_proj.filled[inside] == pytest.approx(hill_proj.elev_of(-1.0))
    assert hill_proj.filled[~inside] == pytest.approx(before[~inside])


def test_flatten_under_a_footprint_off_the_grid_is_a_no_op(hill_proj):
    before = hill_proj.filled.copy()
    hill_proj.flatten_under(box(100.0, 10.0, 100.1, 10.1), -1.0)
    assert hill_proj.filled == pytest.approx(before)


def test_z_range_reads_every_cell_under_the_footprint(hill_proj):
    """A plaque has to clear every peak beneath it, not just the sampled ones."""
    poly = footprint(hill_proj, DEG / 3)
    lo, hi = hill_proj.z_range_under(poly)
    assert lo < hi
    centre = float(hill_proj.sample_z(*poly.centroid.coords[0]))
    assert lo <= centre <= hi + 1e-9


def test_z_range_of_a_footprint_smaller_than_a_cell_falls_back_to_a_sample(
        hill_proj):
    tiny = footprint(hill_proj, DEG * 1e-5)
    lo, hi = hill_proj.z_range_under(tiny)
    assert lo == pytest.approx(hi)


# ---- the terrain solid -----------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("grid_name", ["flat_grid", "hill_grid", "island_grid"])
def test_one_colour_gives_one_watertight_solid(grid_name, make_proj, request):
    proj = make_proj(request.getfixturevalue(grid_name))
    (body,) = terrain_solid(proj)
    assert body.labels == "terrain"
    assert body.mesh.is_watertight
    assert body.mesh.volume > 0


@pytest.mark.slow
@pytest.mark.parametrize("naturalize", [True, False])
def test_every_colour_is_its_own_watertight_solid(hill_proj, split_cats,
                                                  naturalize):
    """Both the contour cut and the raw cell-edge mode have to close."""
    bodies = terrain_solid(hill_proj, split_cats, naturalize=naturalize)
    assert {b.labels for b in bodies} == {"water", "forest"}
    for b in bodies:
        assert b.mesh.is_watertight, f"{b.labels} is open"
        assert b.mesh.volume > 0


@pytest.mark.slow
def test_the_solid_runs_from_the_base_up_to_the_terrain(hill_proj):
    (body,) = terrain_solid(hill_proj)
    lo, hi = body.mesh.bounds
    assert lo[2] == pytest.approx(-hill_proj.base_thickness_mm)
    assert hi[2] == pytest.approx(hill_proj.z_of(hill_proj.filled.max()))


@pytest.mark.slow
def test_the_contour_cut_moves_the_border_off_the_grid(hill_proj, split_cats):
    """Smoothed colours must not follow cell edges — that is the whole point
    of the marching-squares cut. Raw mode, by contrast, must stay on them."""
    lons = hill_proj.grid.lons
    node_x = {round(float(hill_proj.x_of(l)), 6) for l in lons}

    def border_xs(naturalize):
        water = next(b for b in terrain_solid(hill_proj, split_cats,
                                              naturalize=naturalize)
                     if b.labels == "water")
        return water.mesh.vertices[:, 0].max()

    assert round(border_xs(False), 6) in node_x
    assert round(border_xs(True), 6) not in node_x


@pytest.mark.slow
def test_a_clip_keeps_the_solid_inside_the_outline(hill_grid, make_proj):
    region = Region(bbox=(LON0, LAT0, LON0 + DEG, LAT0 + DEG), shape="hex")
    proj = make_proj(hill_grid, span_m=region.span_m)
    clip = region.polygon_mm(proj)
    (body,) = terrain_solid(proj, clip=clip)
    assert body.mesh.is_watertight
    xy = body.mesh.vertices[:, :2]
    assert shapely.contains_xy(clip.buffer(1e-6), xy[:, 0], xy[:, 1]).all()
    assert body.mesh.volume < terrain_solid(proj)[0].mesh.volume


def test_an_outline_that_misses_the_terrain_is_an_error(hill_proj):
    with pytest.raises(ValueError, match="does not overlap"):
        terrain_solid(hill_proj, clip=box(1e6, 1e6, 1e6 + 1, 1e6 + 1))


# ---- polyline simplification ----------------------------------------------

def test_douglas_peucker_keeps_only_the_ends_of_a_straight_run():
    pts = np.column_stack([np.arange(10.0), np.zeros(10)])
    keep = _douglas_peucker(pts, 0.5)
    assert keep.tolist() == [True] + [False] * 8 + [True]


def test_douglas_peucker_keeps_a_real_corner():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 5.0], [3.0, 0.0], [4.0, 0.0]])
    assert _douglas_peucker(pts, 0.5)[2]

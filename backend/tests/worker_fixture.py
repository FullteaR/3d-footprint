"""Network-free subprocess fixture for job lifecycle and actual mesh exports."""
import json
import subprocess
import sys
import time
from pathlib import Path

directory = Path(sys.argv[1])
mode = (directory / "input.gpx").read_bytes()
if mode.startswith(b"sleep"):
    if mode == b"sleep-child":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        (directory / "child.pid").write_text(str(child.pid))
    time.sleep(120)
elif mode == b"crash":
    sys.exit(9)
else:
    import numpy as np
    from app.api import routes
    from app.core.terrain import ElevationGrid
    from app.core import report
    from app.job_worker import run

    grid = ElevationGrid(elev=np.full((16, 16), 5.0),
                         lons=np.linspace(139, 139.01, 16), lats=np.linspace(35, 35.01, 16))
    routes.fetch_elevation_grid = lambda *a, **k: grid
    routes.category_grid = lambda *a, **k: None
    report.warn("buildings", "fetch_failed")
    run(directory)

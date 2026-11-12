"""Exercise the StellarSolver wrapper without requiring index files."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from stdpipe import astrometry


@pytest.fixture
def cli(tmp_path):
    """A subprocess fixture accepting the CLI's supported option names."""
    executable = tmp_path / "solver with spaces"
    executable.write_text(
        "#!" + sys.executable + "\n" + '''import argparse
import numpy as np
from astropy.io import fits

parser = argparse.ArgumentParser()
for option in ["save-fits", "index-files", "config", "ra", "dec", "degrees",
               "scale-low", "scale-high", "scale-units", "depth", "parity",
               "built-in-profile"]:
    parser.add_argument("--" + option)
parser.add_argument("image")
args = parser.parse_args()
image = fits.getdata(args.image)
assert image.shape == (6, 8)
assert np.array_equal(image, np.arange(48).reshape(6, 8))
header = fits.Header({
    "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "CRPIX1": 4.0, "CRPIX2": 3.0,
    "CRVAL1": float(args.ra or 123), "CRVAL2": float(args.dec or -20),
    "CDELT1": -1 / 3600, "CDELT2": 1 / 3600, "CROTA2": 30.0,
})
fits.writeto(args.save_fits, image, header, overwrite=True)
print("Solved successfully")
'''
    )
    executable.chmod(0o755)
    return executable


@pytest.mark.unit
@pytest.mark.parametrize("get_header", [False, True])
def test_solve_image(cli, tmp_path, get_header):
    image = np.arange(48).reshape(6, 8)
    messages = []
    workdir = tmp_path / "work ; with spaces"
    extra = {"depth": 500, "parity": "neg", "built-in-profile": "4-SmallScaleSolving"}
    result = astrometry.blind_match_stellarsolver(
        image, get_header=get_header, center_ra=0.0, center_dec=-30.0,
        radius=2.0, scale_lower=0.5, scale_upper=2.0,
        index_dir=tmp_path / "index files", config=tmp_path / "solver config",
        extra=extra, _exe=cli, _workdir=workdir, verbose=messages.append,
    )
    assert isinstance(result, fits.Header if get_header else WCS)
    wcs = WCS(result) if get_header else result
    np.testing.assert_allclose(wcs.all_pix2world(3, 2, 0), [0.0, -30.0], atol=1e-10)
    assert (workdir / "solved.fits").exists()
    np.testing.assert_array_equal(fits.getdata(workdir / "image.fits"), image)
    assert extra == {"depth": 500, "parity": "neg", "built-in-profile": "4-SmallScaleSolving"}
    assert "Solved successfully" in messages


@pytest.mark.unit
def test_blind_solve_and_cleanup(cli, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(cli.parent) + os.pathsep + os.environ["PATH"])
    assert astrometry.blind_match_stellarsolver(
        np.arange(48).reshape(6, 8), _exe=cli.name, _tmpdir=tmp_path,
    ).is_celestial
    assert list(tmp_path.iterdir()) == [cli]


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["exit", "missing", "noncelestial", "corrupt", "timeout", "oserror"])
def test_failure_cleanup(cli, tmp_path, monkeypatch, failure):
    def run(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if failure == "oserror":
            raise OSError("Cannot execute solver")
        if failure in ["exit", "noncelestial"]:
            output = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--save-fits="))
            fits.writeto(output, np.zeros((6, 8)))
        if failure == "corrupt":
            output = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--save-fits="))
            Path(output).write_text("Failed FITS export")
        return subprocess.CompletedProcess(command, 1 if failure == "exit" else 0, "", "")

    monkeypatch.setattr(astrometry.subprocess, "run", run)
    assert astrometry.blind_match_stellarsolver(
        np.zeros((6, 8)), _exe=cli, _tmpdir=tmp_path, timeout=0.1,
    ) is None
    assert list(tmp_path.iterdir()) == [cli]


@pytest.mark.unit
def test_stale_output_is_not_reused(cli, tmp_path, monkeypatch):
    workdir = tmp_path / "work"
    image = np.arange(48).reshape(6, 8)
    assert astrometry.blind_match_stellarsolver(image, _exe=cli, _workdir=workdir)
    monkeypatch.setattr(
        astrometry.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "", ""),
    )
    assert astrometry.blind_match_stellarsolver(image, _exe=cli, _workdir=workdir) is None
    assert not (workdir / "solved.fits").exists()


@pytest.mark.unit
def test_missing_executable(tmp_path):
    assert astrometry.blind_match_stellarsolver(
        np.zeros((6, 8)), _exe=tmp_path / "missing", _tmpdir=tmp_path,
    ) is None
    assert not list(tmp_path.iterdir())


@pytest.mark.unit
@pytest.mark.parametrize("kwargs", [
    {"image": np.zeros(6)}, {"center_ra": 1.0}, {"center_dec": -20.0},
    {"scale_lower": 1.0}, {"scale_upper": 2.0},
])
def test_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        astrometry.blind_match_stellarsolver(**{"image": np.zeros((6, 8)), **kwargs})

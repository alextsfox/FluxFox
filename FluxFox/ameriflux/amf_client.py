# Author: Alex Fox
# Created: 2026-06-17
"""
Client for retrieving AmeriFlux data.
"""


from dataclasses import dataclass
from time import sleep
import zipfile
import tempfile
from io import StringIO
from pathlib import Path
from typing import List, Literal, Tuple
import requests

import pandas as pd


_INTENDED_USE_MAP = {
    "synthesis": "Research - Multi-site synthesis",
    "remote_sensing": "Research - Remote sensing",
    "model": "Research - Land model/Earth system model",
    "other_research": "Research - Other",
    "education": "Education (Teacher or Student)",
    "other": "Other",
}

_AMERIFLUX_DOWNLOAD_URL = "https://amfcdn.lbl.gov/api/v1/data_download"
_MEAS_HEIGHT_URL = "https://ftp.fluxdata.org/.ameriflux_downloads/measurement_height/BASE_MeasurementHeight_20260527.csv"


@dataclass
class AmerifluxData:
    data: pd.DataFrame
    bif: pd.DataFrame
    zmeas: pd.DataFrame
    lat: float
    lon: float
    elev: float

def retrieve_ameriflux(
    site_id: str,
    username: str,
    email: str,
    intended_use: Literal["synthesis", "remote_sensing", "model", "other_research", "education", "other"],
    use_descr: str,
) -> AmerifluxData:
    """Retrieve AmeriFlux BASE-BADM data for a single site.

    Parameters
    ----------
    site_id : str
        The AmeriFlux site ID to retrieve data for.
    username : str
        Your AmeriFlux username.
    email : str
        Your email address.
    intended_use : Literal["synthesis", "remote_sensing", "model", "other_research", "education", "other"]
        The intended use of the data.
    use_descr : str
        A description of how the data will be used.

    Returns
    -------
    AmerifluxData
        data  : half-hourly (or hourly) timeseries with a DatetimeIndex.
        bif   : full site metadata from the BIF Excel file.
        zmeas : measurement heights for this site.
        lat   : site latitude (degrees).
        lon   : site longitude (degrees).
        elev  : site elevation (metres).

    Examples
    --------
    >>> result = retrieve_ameriflux(
    ...     site_id="US-NR1",
    ...     username="your_username",
    ...     email="your_email@example.com",
    ...     intended_use="education",
    ...     use_descr="FluxFox example"
    ... )
    >>> result.data.head()
    >>> result.bif.head()
    >>> result.zmeas.head()
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        try:
            zip_paths = _download_base_data(site_id, username, email, intended_use, use_descr, out_dir)

            if not zip_paths:
                raise ValueError(f"No data downloaded for site '{site_id}'.")

            data, bif_path = _extract_base_zip(zip_paths[0])

            bif = _load_bif(bif_path) if (bif_path and bif_path.exists()) else None
        except Exception as e:
            raise RuntimeError(
                "Error retrieving data! Double-check your username and email are valid "
                "and that the site you requested exists. "
                "If the error persists, contact alextsfox@gmail.com"
            ) from e

        zmeas = _fetch_measurement_heights(site_id)
        lat, lon, elev = _extract_location(bif) if bif is not None else (None, None, None)

    return AmerifluxData(data=data, bif=bif, zmeas=zmeas, lat=lat, lon=lon, elev=elev)


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _download_base_data(
    site_id: str,
    username: str,
    email: str,
    intended_use: str,
    use_descr: str,
    out_dir: Path,
    data_product: str = "BASE-BADM",
    data_policy: str = "CCBY4.0",
    timeout: int = 120,
) -> List[Path]:
    if data_policy not in {"CCBY4.0", "LEGACY"}:
        raise ValueError("data_policy must be 'CCBY4.0' or 'LEGACY'")
    if intended_use not in _INTENDED_USE_MAP:
        raise ValueError(f"Invalid intended_use: {intended_use!r}")

    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "user_id": username,
        "user_email": email,
        "data_product": data_product,
        "data_policy": data_policy,
        "site_ids": [site_id],
        "intended_use": _INTENDED_USE_MAP[intended_use],
        "description": f"{use_descr} [FluxFox download]",
        "is_test": "",
    }

    response = requests.post(
        _AMERIFLUX_DOWNLOAD_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    sleep(3)
    response.raise_for_status()

    urls = [
        entry["url"]
        for entry in response.json().get("data_urls", [])
        if "url" in entry
    ]

    if not urls:
        return []

    downloaded: List[Path] = []
    session = requests.Session()
    for url in urls:
        filename = url.split("/")[-1].split("?")[0]
        output_path = out_dir / filename
        with session.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        downloaded.append(output_path)

    return downloaded


def _extract_base_zip(zip_path: Path) -> Tuple[pd.DataFrame | None, Path | None]:
    data_df = None
    bif_path = None

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        zf.extractall(zip_path.parent)

        csv_files = [f for f in names if f.endswith(".csv")]
        bif_files = [f for f in names if f.endswith(".xlsx") and "BIF" in f.upper()]

        if csv_files:
            data_df = pd.read_csv(
                zip_path.parent / csv_files[0], skiprows=2, na_values=-9999
            )
            data_df["TIMESTAMP"] = pd.to_datetime(
                data_df["TIMESTAMP_START"], format="%Y%m%d%H%M"
            )
            data_df = (
                data_df
                .drop(columns=["TIMESTAMP_START", "TIMESTAMP_END"])
                .set_index("TIMESTAMP")
                .sort_index()
            )
            float_cols = data_df.select_dtypes(include="float").columns
            data_df[float_cols] = data_df[float_cols].astype("float32")

        if bif_files:
            bif_path = zip_path.parent / bif_files[0]

    return data_df, bif_path


def _load_bif(bif_path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_excel(bif_path)
    except Exception as e:
        print(f"Warning: could not load metadata from {bif_path.name}: {e}")
        return None


def _extract_location(bif: pd.DataFrame) -> Tuple[float | None, float | None, float | None]:
    try:
        lat  = float(bif.query("VARIABLE == 'LOCATION_LAT'").iat[0, -1])
        lon  = float(bif.query("VARIABLE == 'LOCATION_LONG'").iat[0, -1])
        elev = float(bif.query("VARIABLE == 'LOCATION_ELEV'").iat[0, -1])
        return lat, lon, elev
    except Exception as e:
        print(f"Warning: could not extract location from BIF: {e}")
        return None, None, None


def _fetch_measurement_heights(site_id: str) -> pd.DataFrame | None:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    try:
        response = requests.get(_MEAS_HEIGHT_URL, headers=headers)
        response.raise_for_status()
        sleep(3)
        all_heights = pd.read_csv(StringIO(response.text))
        return all_heights.query(f"Site_ID == '{site_id}'").reset_index(drop=True)
    except requests.exceptions.RequestException as err:
        print(f"Warning: could not retrieve measurement heights: {err}")
        print("Contact alextsfox@gmail.com for help.")
        return None
    except Exception as err:
        print(f"Warning: unexpected error fetching measurement heights: {err}")
        print("Contact alextsfox@gmail.com for help.")
        return None
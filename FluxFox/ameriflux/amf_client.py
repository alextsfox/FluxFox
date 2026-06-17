# Author: Alex Fox
# Created: 2026-06-17
"""
Client for retrieving AmeriFlux data.
"""


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


class AmerifluxRetriever:
    """AmeriFlux BASE-BADM client.

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

    Attributes
    ----------
    data : pd.DataFrame
        Half-hourly (or hourly) timeseries with a DatetimeIndex.
    metadata : pd.DataFrame
        Full site metadata from the BIF Excel file.
    zmeas : pd.DataFrame
        Measurement heights for this site.

    Examples
    --------
    >>> retriever = AmerifluxRetriever(
    ...     site_id="US-NR1",
    ...     username="your_username",
    ...     email="your_email@example.com",
    ...     intended_use="synthesis",
    ...     use_descr="FluxFox example"
    ... )
    >>> retriever.data.head()
    >>> retriever.metadata.head()
    >>> retriever.zmeas.head()
    """

    def __init__(
            self, 
            site_id: str, 
            username: str, 
            email: str, 
            intended_use: Literal["synthesis", "remote_sensing", "model", "other_research", "education", "other"],
            use_descr: str
        ):

        self.site_id = site_id
        self.username = username
        self.email = email
        self.intended_use = intended_use
        self.use_descr = use_descr
        self.data: pd.DataFrame | None = None
        self.metadata: pd.DataFrame | None = None
        self.zmeas: pd.DataFrame | None = None

        # Keep a reference so the directory isn't deleted while the object is alive
        self._tmpdir = tempfile.TemporaryDirectory()
        self._out_dir = Path(self._tmpdir.name)

        try:
            zip_paths = self._download_base_data()

            if not zip_paths:
                raise ValueError(f"No data downloaded for site '{self.site_id}'.")

            self.data, bif_path = self._extract_base_zip(zip_paths[0])

            if bif_path and bif_path.exists():
                self.metadata = self._load_metadata_from_bif(bif_path)
        except Exception as e:
            raise RuntimeError(
                "Error retrieving data! Double-check your username and email are valid "
                "and that the site you requested exists. "
                "If the error persists, contact alextsfox@gmail.com"
            ) from e

        self._fetch_measurement_heights()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download_base_data(
        self,
        data_product: str = "BASE-BADM",
        data_policy: str = "CCBY4.0",
        timeout: int = 120,
    ) -> List[Path]:
        if data_policy not in {"CCBY4.0", "LEGACY"}:
            raise ValueError("data_policy must be 'CCBY4.0' or 'LEGACY'")
        if self.intended_use not in _INTENDED_USE_MAP:
            raise ValueError(f"Invalid intended_use: {self.intended_use!r}")

        self._out_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "user_id": self.username,
            "user_email": self.email,
            "data_product": data_product,
            "data_policy": data_policy,
            "site_ids": [self.site_id],
            "intended_use": _INTENDED_USE_MAP[self.intended_use],
            "description": f"{self.use_descr} [FluxFox download]",
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
            output_path = self._out_dir / filename
            with session.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            downloaded.append(output_path)

        return downloaded

    def _extract_base_zip(self, zip_path: Path) -> Tuple[pd.DataFrame | None, Path | None]:
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

    def _load_metadata_from_bif(self, bif_path: Path) -> pd.DataFrame | None:
        try:
            return pd.read_excel(bif_path)
        except Exception as e:
            print(f"Warning: could not load metadata from {bif_path.name}: {e}")
            return None

    def _fetch_measurement_heights(self) -> None:
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
            self.zmeas = all_heights.query(f"Site_ID == '{self.site_id}'").reset_index(drop=True)
        except requests.exceptions.RequestException as err:
            print(f"Warning: could not retrieve measurement heights: {err}")
            print("Contact alextsfox@gmail.com for help.")
        except Exception as err:
            print(f"Warning: unexpected error fetching measurement heights: {err}")
            print("Contact alextsfox@gmail.com for help.")
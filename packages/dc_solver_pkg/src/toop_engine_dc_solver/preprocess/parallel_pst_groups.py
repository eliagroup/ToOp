"""Helpers for loading and generating parallel PST group definitions."""

import csv
import io

import numpy as np
import structlog
from beartype.typing import Sequence
from fsspec import AbstractFileSystem
from jaxtyping import Bool
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS

logger = structlog.get_logger(__name__)

PARALLEL_PSTS_CSV_HEADERS = ("pst_id", "group")


def load_or_create_parallel_pst_group_mask(
    filesystem: AbstractFileSystem,
    pst_ids: Sequence[str | int],
) -> Bool[np.ndarray, " n_parallel_pst_groups n_controllable_pst"]:
    """Load the parallel PST grouping from CSV or generate a default one.

    Parameters
    ----------
    filesystem : AbstractFileSystem
        Filesystem rooted at the preprocessing directory.
    pst_ids : Sequence[str | int]
        Ordered controllable PST ids the output mask should align with.

    Returns
    -------
    Bool[np.ndarray, " n_parallel_pst_groups n_controllable_pst"]
        A boolean group mask aligned with ``pst_ids``. Each column belongs to exactly one group.
    """
    pst_id_list = [str(pst_id) for pst_id in pst_ids]
    if not pst_id_list:
        return np.zeros((0, 0), dtype=bool)

    file_path = PREPROCESSING_PATHS["parallel_psts_file_path"]
    if not filesystem.exists(file_path):
        logger.warning(
            "parallel_psts.csv is missing. Generating a default file with one group per PST.",
            file_path=file_path,
        )
        _write_default_parallel_psts_csv(filesystem=filesystem, file_path=file_path, pst_ids=pst_id_list)
        return np.eye(len(pst_id_list), dtype=bool)

    with filesystem.open(file_path, "r", encoding="utf-8") as file:
        csv_content = file.read()

    if not csv_content.strip():
        logger.warning(
            (
                "parallel_psts.csv is empty. Generating a default file with one group per PST. "
                "Fill parallel_psts.csv with PST id and group name columns to configure parallel optimization groups."
            ),
            file_path=file_path,
        )
        _write_default_parallel_psts_csv(filesystem=filesystem, file_path=file_path, pst_ids=pst_id_list)
        return np.eye(len(pst_id_list), dtype=bool)

    pst_group_rows = _read_parallel_pst_rows(csv_content=csv_content, file_path=file_path)
    pst_to_group = _build_pst_group_mapping(pst_group_rows=pst_group_rows, file_path=file_path)

    missing_pst_ids = [pst_id for pst_id in pst_id_list if pst_id not in pst_to_group]
    if missing_pst_ids:
        raise ValueError(f"parallel_psts.csv does not contain all controllable PST ids. Missing ids: {missing_pst_ids}.")

    unknown_pst_ids = [pst_id for pst_id in pst_to_group if pst_id not in set(pst_id_list)]
    if unknown_pst_ids:
        logger.warning(
            "parallel_psts.csv contains PST ids that are not part of the controllable PST set. Ignoring them.",
            file_path=file_path,
            pst_ids=unknown_pst_ids,
        )

    ordered_group_names: list[str] = []
    group_index_by_name: dict[str, int] = {}
    for pst_id in pst_id_list:
        group_name = pst_to_group[pst_id]
        if group_name not in group_index_by_name:
            group_index_by_name[group_name] = len(ordered_group_names)
            ordered_group_names.append(group_name)

    parallel_pst_group_mask = np.zeros((len(ordered_group_names), len(pst_id_list)), dtype=bool)
    for pst_index, pst_id in enumerate(pst_id_list):
        parallel_pst_group_mask[group_index_by_name[pst_to_group[pst_id]], pst_index] = True

    return parallel_pst_group_mask


def _build_pst_group_mapping(pst_group_rows: list[tuple[str, str]], file_path: str) -> dict[str, str]:
    """Build a mapping from PST id to group name and check for duplicates.

    Parameters
    ----------
    pst_group_rows : list[tuple[str, str]]
        List of (pst_id, group_name) tuples parsed from the CSV.
    file_path : str
        Path to the CSV file, used for error messages.

    Returns
    -------
    dict[str, str]
        Mapping from PST id to group name.

    Raises
    ------
    ValueError
        If a duplicate PST id is found.

    """
    pst_to_group: dict[str, str] = {}
    for pst_id, group_name in pst_group_rows:
        if pst_id in pst_to_group:
            raise ValueError(f"Duplicate PST id '{pst_id}' found in {file_path}.")
        pst_to_group[pst_id] = group_name
    return pst_to_group


def _read_parallel_pst_rows(csv_content: str, file_path: str) -> list[tuple[str, str]]:
    """Parse the parallel PST CSV into ordered rows."""
    reader = csv.reader(io.StringIO(csv_content))
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        return []

    header = rows[0]
    if len(header) < 2:
        raise ValueError(f"{file_path} must contain at least two columns for PST id and group, got header {header}.")

    parsed_rows: list[tuple[str, str]] = []
    for row in rows[1:]:
        if len(row) < 2:
            raise ValueError(f"Every row in {file_path} must contain PST id and group, got row {row}.")
        pst_id = row[0].strip()
        group_name = row[1].strip()
        if not pst_id or not group_name:
            raise ValueError(f"Every row in {file_path} must have non-empty PST id and group values, got row {row}.")
        parsed_rows.append((pst_id, group_name))

    return parsed_rows


def _write_default_parallel_psts_csv(filesystem: AbstractFileSystem, file_path: str, pst_ids: Sequence[str]) -> None:
    """Write a default CSV where each PST is its own optimization group."""
    with filesystem.open(file_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(PARALLEL_PSTS_CSV_HEADERS)
        for pst_id in pst_ids:
            writer.writerow((pst_id, pst_id))

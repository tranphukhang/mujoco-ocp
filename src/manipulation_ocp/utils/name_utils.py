from __future__ import annotations


def filter_names_by_prefix(
    names: list[str] | tuple[str, ...],
    prefix: str,
) -> list[str]:
    """Return names that start with a given prefix."""
    return [name for name in names if name.startswith(prefix)]


def filter_names_by_suffix(
    names: list[str] | tuple[str, ...],
    suffix: str,
) -> list[str]:
    """Return names that end with a given suffix."""
    return [name for name in names if name.endswith(suffix)]


def filter_names_containing(
    names: list[str] | tuple[str, ...],
    text: str,
) -> list[str]:
    """Return names that contain a given text."""
    return [name for name in names if text in name]


def require_names(
    available_names: list[str] | tuple[str, ...],
    required_names: list[str] | tuple[str, ...],
    *,
    label: str = "names",
) -> None:
    """
    Validate that all required names exist in available_names.

    Raises
    ------
    ValueError
        If at least one required name is missing.
    """
    available_set = set(available_names)

    missing_names = [
        name for name in required_names
        if name not in available_set
    ]

    if missing_names:
        raise ValueError(f"Missing {label}: {missing_names}")


def get_name_indices(
    available_names: list[str] | tuple[str, ...],
    selected_names: list[str] | tuple[str, ...],
    *,
    label: str = "names",
) -> list[int]:
    """
    Get indices of selected names in available_names.

    This preserves the order of selected_names.
    """
    name_to_index = {
        name: index
        for index, name in enumerate(available_names)
    }

    missing_names = [
        name for name in selected_names
        if name not in name_to_index
    ]

    if missing_names:
        raise ValueError(f"Missing {label}: {missing_names}")

    return [name_to_index[name] for name in selected_names]


def get_prefixed_name_indices(
    available_names: list[str] | tuple[str, ...],
    prefix: str,
) -> list[int]:
    """
    Get indices of all names that start with a given prefix.
    """
    return [
        index
        for index, name in enumerate(available_names)
        if name.startswith(prefix)
    ]


def assert_unique_names(
    names: list[str] | tuple[str, ...],
    *,
    label: str = "names",
) -> None:
    """
    Validate that all names are unique.

    Raises
    ------
    ValueError
        If duplicated names are found.
    """
    seen: set[str] = set()
    duplicates: list[str] = []

    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)

    if duplicates:
        raise ValueError(f"Duplicated {label}: {duplicates}")
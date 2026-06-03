import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

from .peak_list import PeakList


class PeakListCollection:
    """
    Container for managing multiple PeakList objects.

    This provides a dict-like interface keyed by a user-supplied name and is
    intentionally lightweight compared to RelaxationFitCollection. Typical use
    cases are keeping separate peak lists for different spectra, processing
    stages, or grouping rules.

    Args:
        name: Optional collection name for bookkeeping
    """

    def __init__(self, name: Optional[str] = None):
        self._peaklists: Dict[str, PeakList] = {}
        self.name = name

    def add(self, key: str, peaklist: PeakList) -> None:
        """
        Add or replace a PeakList under the provided key.

        Args:
            key: Identifier for the peak list (e.g., spectrum name)
            peaklist: PeakList instance to store

        Raises:
            TypeError: If peaklist is not a PeakList
        """
        if not isinstance(peaklist, PeakList):
            raise TypeError(f"Expected PeakList, got {type(peaklist)}")
        self._peaklists[key] = peaklist

    def remove(self, key: str) -> None:
        """Remove a peak list by key."""
        del self._peaklists[key]

    def get(self, key: str, default=None) -> Optional[PeakList]:
        """Return the PeakList for key or default if missing."""
        return self._peaklists.get(key, default)

    def clear(self) -> None:
        """Remove all stored peak lists."""
        self._peaklists.clear()

    def keys(self) -> List[str]:
        """Return list of keys."""
        return list(self._peaklists.keys())

    def values(self) -> List[PeakList]:
        """Return list of PeakList objects."""
        return list(self._peaklists.values())

    def items(self) -> List[tuple[str, PeakList]]:
        """Return list of (key, PeakList) tuples."""
        return list(self._peaklists.items())

    def filter(self, predicate) -> "PeakListCollection":
        """
        Filter peak lists using a predicate.

        Args:
            predicate: Callable that accepts (key, PeakList) and returns bool

        Returns:
            New PeakListCollection containing matching entries
        """
        filtered = PeakListCollection(name=f"{self.name}_filtered" if self.name else None)
        for key, peaklist in self._peaklists.items():
            if predicate(key, peaklist):
                filtered.add(key, peaklist)
        return filtered

    def total_peaks(self) -> int:
        """Return the total number of peaks across all stored lists."""
        return sum(len(pl) for pl in self._peaklists.values())

    def to_dict(self) -> dict:
        """Serialize collection to a dictionary."""
        return {
            "name": self.name,
            "count": len(self._peaklists),
            "peaklists": {key: pl.to_dict() for key, pl in self._peaklists.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PeakListCollection":
        """Create a collection from its dictionary representation."""
        collection = cls(name=data.get("name"))
        for key, pl_data in data.get("peaklists", {}).items():
            peaklist = PeakList.from_dict(pl_data)
            collection.add(key, peaklist)
        return collection

    def save(self, file_path: Union[str, Path], indent: int = 2) -> None:
        """Save collection to a JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "PeakListCollection":
        """Load collection from a JSON file."""
        file_path = Path(file_path)
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __len__(self) -> int:
        return len(self._peaklists)

    def __getitem__(self, key: str) -> PeakList:
        return self._peaklists[key]

    def __setitem__(self, key: str, peaklist: PeakList) -> None:
        self.add(key, peaklist)

    def __delitem__(self, key: str) -> None:
        self.remove(key)

    def __contains__(self, key: str) -> bool:
        return key in self._peaklists

    def __iter__(self) -> Iterator[tuple[str, PeakList]]:
        return iter(self._peaklists.items())

    def __repr__(self) -> str:
        name_str = f"'{self.name}', " if self.name else ""
        return f"PeakListCollection({name_str}{len(self._peaklists)} peaklists)"

    def __str__(self) -> str:
        if not self._peaklists:
            return (
                f"PeakListCollection('{self.name}' - empty)"
                if self.name
                else "PeakListCollection(empty)"
            )
        name_str = f"'{self.name}' - " if self.name else ""
        return f"PeakListCollection({name_str}{len(self._peaklists)} peaklists)"

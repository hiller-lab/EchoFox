from typing import List, Optional, Union, Callable, Iterator, Tuple
import numpy as np
from collections import defaultdict
import json
from pathlib import Path

from .peak import NmrPeak


class PeakList:
    """
    Container for managing collections of NMR peaks.

    PeakList provides a list-like interface for managing multiple Peak objects,
    with additional methods for searching, filtering, sorting, and analyzing
    peak collections.

    Examples:
        # Create empty peak list
        peak_list = PeakList()

        # Add peaks
        peak_list.add(Peak([7.26], nuclei=['1H']))
        peak_list.add(Peak([8.5, 120.2], nuclei=['1H', '15N']))

        # Access peaks
        first_peak = peak_list[0]
        num_peaks = len(peak_list)

        # Iterate over peaks
        for peak in peak_list:
            print(peak)

        # Filter peaks
        h_peaks = peak_list.filter_by_nucleus('1H', dimension=0)
        assigned_peaks = peak_list.filter_assigned()

        # Find closest peak to a position
        closest = peak_list.find_closest([7.25, 120.0])

        # Sort by chemical shift
        peak_list.sort_by_shift(dimension=0)

    Args:
        peaks: Optional initial list of peaks
        name: Optional name for the peak list
    """

    def __init__(
        self,
        peaks: Optional[List[NmrPeak]] = None,
        name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        self._peaks: List[NmrPeak] = []
        self.name = name
        self._metadata = metadata or {}
        self._experiment_id: Optional[int] = None

        if peaks is not None:
            for peak in peaks:
                self.add(peak)

    @property
    def peaks(self) -> List[NmrPeak]:
        """Returns a copy of the peak list."""
        return self._peaks.copy()

    def add(self, peak: NmrPeak) -> None:
        """
        Add a peak to the list.

        Args:
            peak: Peak object to add

        Raises:
            TypeError: If peak is not a Peak object
        """
        if not isinstance(peak, NmrPeak):
            raise TypeError(f"Expected Peak object, got {type(peak)}")
        self._peaks.append(peak)

    def remove(self, peak: NmrPeak) -> None:
        """
        Remove a peak from the list.

        Args:
            peak: Peak object to remove

        Raises:
            ValueError: If peak is not in the list
        """
        self._peaks.remove(peak)

    def remove_at(self, index: int) -> NmrPeak:
        """
        Remove and return peak at specified index.

        Args:
            index: Index of peak to remove

        Returns:
            Removed peak

        Raises:
            IndexError: If index is out of range
        """
        return self._peaks.pop(index)

    def clear(self) -> None:
        """Remove all peaks from the list."""
        self._peaks.clear()

    def extend(self, peaks: Union["PeakList", List[NmrPeak]]) -> None:
        """
        Add multiple peaks to the list.

        Args:
            peaks: PeakList or list of Peak objects
        """
        if isinstance(peaks, PeakList):
            peaks = peaks.peaks
        for peak in peaks:
            self.add(peak)

    def filter_by_dimension(self, ndim: int) -> "PeakList":
        """
        Filter peaks by number of dimensions.

        Args:
            ndim: Number of dimensions

        Returns:
            New PeakList containing only peaks with specified dimensions
        """
        filtered = [peak for peak in self._peaks if peak.ndim == ndim]
        return PeakList(filtered, name=f"{self.name}_dim{ndim}" if self.name else None)

    def filter_by_nucleus(
        self, nucleus: str, dimension: Optional[int] = None
    ) -> "PeakList":
        """
        Filter peaks by nucleus type.

        Args:
            nucleus: Nucleus label (e.g., '1H', '13C', '15N')
            dimension: If specified, only check this dimension. Otherwise check all dimensions.

        Returns:
            New PeakList containing only peaks with specified nucleus
        """
        filtered = []
        for peak in self._peaks:
            if peak.nuclei is None:
                continue

            if dimension is not None:
                # Check specific dimension
                if dimension < peak.ndim and peak.get_nucleus(dimension) == nucleus:
                    filtered.append(peak)
            else:
                # Check any dimension
                if nucleus in peak.nuclei:
                    filtered.append(peak)

        return PeakList(filtered, name=f"{self.name}_{nucleus}" if self.name else None)

    def filter_by_assignment(
        self, assignment_pattern: str, dimension: Optional[int] = None
    ) -> "PeakList":
        """
        Filter peaks by assignment pattern.

        Args:
            assignment_pattern: Assignment string or pattern (supports partial matching)
            dimension: If specified, only check this dimension. Otherwise check all dimensions.

        Returns:
            New PeakList containing matching peaks
        """
        filtered = []
        for peak in self._peaks:
            if peak.assignments is None:
                continue

            if dimension is not None:
                # Check specific dimension
                assignment = peak.get_assignment(dimension)
                if assignment and assignment_pattern in assignment:
                    filtered.append(peak)
            else:
                # Check any dimension
                if any(
                    assignment and assignment_pattern in assignment
                    for assignment in peak.assignments
                    if assignment is not None
                ):
                    filtered.append(peak)

        return PeakList(
            filtered, name=f"{self.name}_{assignment_pattern}" if self.name else None
        )

    def filter_assigned(self, dimension: Optional[int] = None) -> "PeakList":
        """
        Filter peaks that have assignments.

        Args:
            dimension: If specified, only check this dimension. Otherwise check any dimension.

        Returns:
            New PeakList containing only assigned peaks
        """
        filtered = []
        for peak in self._peaks:
            if peak.assignments is None:
                continue

            if dimension is not None:
                # Check specific dimension
                if peak.get_assignment(dimension) is not None:
                    filtered.append(peak)
            else:
                # Check if any dimension is assigned
                if any(a is not None for a in peak.assignments):
                    filtered.append(peak)

        return PeakList(filtered, name=f"{self.name}_assigned" if self.name else None)

    def filter_unassigned(self) -> "PeakList":
        """
        Filter peaks that have no assignments.

        Returns:
            New PeakList containing only unassigned peaks
        """
        filtered = [
            peak
            for peak in self._peaks
            if peak.assignments is None or all(a is None for a in peak.assignments)
        ]
        return PeakList(filtered, name=f"{self.name}_unassigned" if self.name else None)

    def filter_by_shift_range(
        self, dimension: int, min_ppm: float, max_ppm: float
    ) -> "PeakList":
        """
        Filter peaks by chemical shift range in a specific dimension.

        Args:
            dimension: Dimension index (0-based)
            min_ppm: Minimum chemical shift (ppm)
            max_ppm: Maximum chemical shift (ppm)

        Returns:
            New PeakList containing peaks within the specified range
        """
        filtered = []
        for peak in self._peaks:
            if dimension < peak.ndim:
                shift = peak.get_chemical_shift(dimension).ppm
                if min_ppm <= shift <= max_ppm:
                    filtered.append(peak)

        return PeakList(filtered, name=f"{self.name}_range" if self.name else None)

    def filter_by_intensity(
        self, min_intensity: Optional[float] = None, max_intensity: Optional[float] = None
    ) -> "PeakList":
        """
        Filter peaks by intensity range.

        Args:
            min_intensity: Minimum intensity (inclusive)
            max_intensity: Maximum intensity (inclusive)

        Returns:
            New PeakList containing peaks within the specified intensity range
        """
        filtered = []
        for peak in self._peaks:
            if peak.intensity is None:
                continue

            if min_intensity is not None and peak.intensity < min_intensity:
                continue
            if max_intensity is not None and peak.intensity > max_intensity:
                continue

            filtered.append(peak)

        return PeakList(
            filtered, name=f"{self.name}_intensity_filtered" if self.name else None
        )

    def filter(self, predicate: Callable[[NmrPeak], bool]) -> "PeakList":
        """
        Filter peaks using a custom predicate function.

        Args:
            predicate: Function that takes a Peak and returns True if it should be included

        Returns:
            New PeakList containing peaks that satisfy the predicate

        Example:
            # Filter peaks with intensity > 1000
            high_intensity = peak_list.filter(lambda p: p.intensity and p.intensity > 1000)
        """
        filtered = [peak for peak in self._peaks if predicate(peak)]
        return PeakList(filtered, name=f"{self.name}_filtered" if self.name else None)

    def find_closest(
        self,
        position: List[float],
        max_distance: Optional[float] = None,
        weights: Optional[List[float]] = None,
    ) -> Optional[Tuple[NmrPeak, float]]:
        """
        Find the closest peak to a given position.

        Args:
            position: Target position as list of chemical shifts (one per dimension)
            max_distance: Maximum distance to consider (peaks farther away are ignored)
            weights: Optional weights for each dimension in distance calculation

        Returns:
            Tuple of (closest_peak, distance) or None if no peaks found

        Raises:
            ValueError: If peak list is empty
        """
        if not self._peaks:
            return None

        # Create a temporary peak at the target position
        target_peak = NmrPeak(position)

        min_distance = float("inf")
        closest_peak = None

        for peak in self._peaks:
            # Only compare peaks with same dimensions
            if peak.ndim != target_peak.ndim:
                continue

            try:
                distance = peak.distance_to(target_peak, weights=weights)
                if distance < min_distance:
                    if max_distance is None or distance <= max_distance:
                        min_distance = distance
                        closest_peak = peak
            except Exception:
                # Skip peaks that can't be compared (e.g., nucleus mismatch)
                continue

        if closest_peak is None:
            return None

        return closest_peak, min_distance

    def find_within_tolerance(
        self, position: List[float], tolerances: Union[float, List[float]]
    ) -> "PeakList":
        """
        Find all peaks within tolerance of a given position.

        Args:
            position: Target position as list of chemical shifts
            tolerances: Single tolerance or list of tolerances per dimension

        Returns:
            New PeakList containing peaks within tolerance
        """
        target_peak = NmrPeak(position)
        matches = []

        for peak in self._peaks:
            if peak.ndim != target_peak.ndim:
                continue

            try:
                if peak.is_within_tolerance(target_peak, tolerances):
                    matches.append(peak)
            except Exception:
                continue

        return PeakList(matches, name=f"{self.name}_matches" if self.name else None)

    def sort_by_shift(
        self, dimension: int, reverse: bool = False, inplace: bool = True
    ) -> Optional["PeakList"]:
        """
        Sort peaks by chemical shift in a specific dimension.

        Args:
            dimension: Dimension index (0-based)
            reverse: If True, sort in descending order
            inplace: If True, sort this list. If False, return a new sorted list.

        Returns:
            None if inplace=True, otherwise new sorted PeakList
        """

        def get_shift(peak: NmrPeak) -> float:
            if dimension < peak.ndim:
                return peak.get_chemical_shift(dimension).ppm
            return float("inf")  # Put peaks without this dimension at the end

        if inplace:
            self._peaks.sort(key=get_shift, reverse=reverse)
            return None
        else:
            sorted_peaks = sorted(self._peaks, key=get_shift, reverse=reverse)
            return PeakList(sorted_peaks, name=self.name)

    def sort_by_intensity(
        self, reverse: bool = True, inplace: bool = True
    ) -> Optional["PeakList"]:
        """
        Sort peaks by intensity.

        Args:
            reverse: If True (default), sort in descending order (highest first)
            inplace: If True, sort this list. If False, return a new sorted list.

        Returns:
            None if inplace=True, otherwise new sorted PeakList
        """

        def get_intensity(peak: NmrPeak) -> float:
            return peak.intensity if peak.intensity is not None else float("-inf")

        if inplace:
            self._peaks.sort(key=get_intensity, reverse=reverse)
            return None
        else:
            sorted_peaks = sorted(self._peaks, key=get_intensity, reverse=reverse)
            return PeakList(sorted_peaks, name=self.name)

    def sort_by_assignment(
        self, dimension: int = 0, reverse: bool = False, inplace: bool = True
    ) -> Optional["PeakList"]:
        """
        Sort peaks by assignment label.

        Args:
            dimension: Dimension index for assignment (0-based)
            reverse: If True, sort in descending order
            inplace: If True, sort this list. If False, return a new sorted list.

        Returns:
            None if inplace=True, otherwise new sorted PeakList
        """

        def get_assignment(peak: NmrPeak) -> str:
            assignment = peak.get_assignment(dimension)
            return assignment if assignment is not None else ""

        if inplace:
            self._peaks.sort(key=get_assignment, reverse=reverse)
            return None
        else:
            sorted_peaks = sorted(self._peaks, key=get_assignment, reverse=reverse)
            return PeakList(sorted_peaks, name=self.name)

    def group_by_assignment(self, dimension: int = 0) -> dict[str, "PeakList"]:
        """
        Group peaks by assignment in a specific dimension.

        Args:
            dimension: Dimension index (0-based)

        Returns:
            Dictionary mapping assignment strings to PeakList objects
        """
        groups = defaultdict(list)

        for peak in self._peaks:
            assignment = peak.get_assignment(dimension)
            key = assignment if assignment is not None else "unassigned"
            groups[key].append(peak)

        return {
            key: PeakList(peaks, name=f"{self.name}_{key}" if self.name else key)
            for key, peaks in groups.items()
        }

    def statistics(self) -> dict:
        """
        Calculate statistics for the peak list.

        Returns:
            Dictionary containing various statistics
        """
        if not self._peaks:
            return {
                "count": 0,
                "dimensions": None,
                "assigned_count": 0,
                "intensities": None,
            }

        intensities = [p.intensity for p in self._peaks if p.intensity is not None]
        assigned_count = sum(
            1
            for p in self._peaks
            if p.assignments and any(a is not None for a in p.assignments)
        )

        # Count peaks by dimension
        dim_counts = defaultdict(int)
        for peak in self._peaks:
            dim_counts[peak.ndim] += 1

        stats = {
            "count": len(self._peaks),
            "dimensions": dict(dim_counts),
            "assigned_count": assigned_count,
            "unassigned_count": len(self._peaks) - assigned_count,
        }

        if intensities:
            stats["intensities"] = {
                "min": min(intensities),
                "max": max(intensities),
                "mean": np.mean(intensities),
                "median": np.median(intensities),
                "std": np.std(intensities),
            }

        return stats

    def to_dict(self) -> dict:
        """
        Convert peak list to dictionary representation.

        Returns:
            Dictionary containing peak list data
        """
        return {
            "name": self.name,
            "count": len(self._peaks),
            "peaks": [peak.to_dict() for peak in self._peaks],
            "metadata": self._metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PeakList":
        """
        Create PeakList from dictionary representation.

        Args:
            data: Dictionary containing peak list data

        Returns:
            PeakList object
        """
        peaks = [NmrPeak.from_dict(peak_data) for peak_data in data["peaks"]]
        return cls(peaks, name=data.get("name"), metadata=data.get("metadata"))

    def save(self, file_path: Union[str, Path], indent: int = 2) -> None:
        """
        Save peak list to a JSON file.

        Args:
            file_path: Path to the output file
            indent: JSON indentation for readability (default: 2)

        Example:
            peak_list.save('peaks.json')
            peak_list.save('/path/to/peaks.json', indent=4)
        """
        file_path = Path(file_path)

        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "PeakList":
        """
        Load peak list from a JSON file.

        Args:
            file_path: Path to the input file

        Returns:
            PeakList object

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON

        Example:
            peak_list = PeakList.load('peaks.json')
            peak_list = PeakList.load('/path/to/peaks.json')
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Peak list file not found: {file_path}")

        with open(file_path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def __len__(self) -> int:
        """Return number of peaks in the list."""
        return len(self._peaks)

    def __getitem__(self, index: Union[int, slice]) -> Union[NmrPeak, "PeakList"]:
        """
        Get peak(s) by index or slice.

        Args:
            index: Integer index or slice

        Returns:
            Peak if index is int, PeakList if index is slice
        """
        if isinstance(index, slice):
            return PeakList(self._peaks[index], name=self.name)
        return self._peaks[index]

    def __setitem__(self, index: int, peak: NmrPeak) -> None:
        """Set peak at index."""
        if not isinstance(peak, NmrPeak):
            raise TypeError(f"Expected Peak object, got {type(peak)}")
        self._peaks[index] = peak

    def __delitem__(self, index: int) -> None:
        """Delete peak at index."""
        del self._peaks[index]

    def __iter__(self) -> Iterator[NmrPeak]:
        """Iterate over peaks."""
        return iter(self._peaks)

    def __contains__(self, peak: NmrPeak) -> bool:
        """Check if peak is in the list."""
        return peak in self._peaks

    def __repr__(self) -> str:
        """Return detailed representation."""

        # Create a defaultdict to count occurrences of each ndim value
        dim_counts = defaultdict(int)

        # Iterate over each peak in the peaklist and count its ndim
        for peak in self._peaks:
            dim_counts[peak.ndim] += 1  # Increment count for each ndim

        # Generate the formatted output (this part sorts the ndim counts)
        dim_info = ", ".join(
            f"{dim}D: {count}" for dim, count in sorted(dim_counts.items())
        )

        name_str = f"'{self.name}', " if self.name else ""
        return f"PeakList({name_str}{len(self._peaks)} peaks, {dim_info})"

    def __str__(self) -> str:
        """Return string representation."""
        if not self._peaks:
            return f"PeakList('{self.name}' - empty)" if self.name else "PeakList(empty)"

        name_str = f"'{self.name}' - " if self.name else ""
        return f"PeakList({name_str}{len(self._peaks)} peaks)"

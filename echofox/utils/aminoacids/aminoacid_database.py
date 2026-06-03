"""Aminoacids database schema configuration"""

import logging
import os
from typing import Any

import yaml

from .exceptions import (
    AminoAcidNotFoundError,
    SchemaError,
    SchemaFileNotFoundError,
    SchemaMissingFieldError,
    SchemaParseError,
)

log = logging.getLogger(__name__)


class AminoAcidsDatabase:
    """Load and manage amino acids schema from YAML configuration"""

    def __init__(self, schema_path: str | None = None):
        """
        Initialize schema loader

        Parameters
        ----------
        schema_path : str, optional
            Path to a schema YAML file. If None, use the default location.
        """
        if schema_path is None:
            # Default to amino_acids.yaml in the same directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            schema_path = os.path.join(current_dir, "aminoacids.yaml")

        self.schema_path = schema_path
        self.amino_acids = []
        self._aa_map = {}  # Maps one_letter code -> amino acid definition
        self._three_letter_map = {}  # Maps three_letter code -> amino acid definition
        self._load_schema()

    def _load_schema(self):
        """Load schema from the YAML file"""
        try:
            with open(self.schema_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if "amino_acids" not in data:
                raise SchemaMissingFieldError("Schema file missing 'amino_acids' section")

            self.amino_acids = data["amino_acids"]

            # Build amino acid maps for quick lookup
            self._aa_map = {aa["one_letter"]: aa for aa in self.amino_acids}
            self._three_letter_map = {aa["three_letters"]: aa for aa in self.amino_acids}

            log.info(f"Loaded {len(self.amino_acids)} amino acid definitions from {self.schema_path}")

        except FileNotFoundError as e:
            log.error(f"Schema file not found: {self.schema_path}")
            raise SchemaFileNotFoundError(f"Schema file not found: {self.schema_path}") from e
        except yaml.YAMLError as e:
            log.error(f"Error parsing schema YAML: {e}")
            raise SchemaParseError(f"Error parsing schema YAML: {e}") from e
        except (SchemaMissingFieldError, SchemaFileNotFoundError, SchemaParseError):
            raise
        except Exception as e:
            log.error(f"Error loading schema: {e}")
            raise SchemaError(f"Error loading schema: {e}") from e

    def get_aa_codes(self) -> list[str]:
        """
        Get the list of amino acids in one-letter code in order

        Returns
        -------
        list of str
            One-letter code for all amino acids
        """
        return [aa["one_letter"] for aa in self.amino_acids]

    def get_aa_by_code(self, one_letter: str, strict: bool = True) -> dict[str, Any] | None:
        """
        Get amino acid from a one-letter code

        Parameters
        ----------
        one_letter : str
            One-letter code of the amino acid
        strict : bool, optional
            If True, raise AminoAcidNotFoundError when not found.
            If False, return None when not found. Default is True.

        Returns
        -------
        dict or None
            Amino acid definition if found, None if not found and strict=False

        Raises
        ------
        AminoAcidNotFoundError
            If amino acid is not found and strict=True
        """
        aa = self._aa_map.get(one_letter)
        if aa is None and strict:
            raise AminoAcidNotFoundError(one_letter, search_type="code")
        return aa

    def get_aa_by_full_name(self, full_name: str, strict: bool = True) -> dict[str, Any] | None:
        """
        Get amino acid from the full name

        Parameters
        ----------
        full_name : str
            Full name of the amino acid
        strict : bool, optional
            If True, raise AminoAcidNotFoundError when not found.
            If False, return None when not found. Default is True.

        Returns
        -------
        dict or None
            Amino acid definition if found, None if not found and strict=False

        Raises
        ------
        AminoAcidNotFoundError
            If amino acid is not found and strict=True
        """
        for aa in self.amino_acids:
            if aa["name"] == full_name:
                return aa
        if strict:
            raise AminoAcidNotFoundError(full_name, search_type="name")
        return None

    def get_full_name(self, one_letter: str, strict: bool = True) -> str:
        """
        Get the full name from a one-letter code

        Parameters
        ----------
        one_letter : str
            One-letter code of the amino acid
        strict : bool, optional
            If True, raise AminoAcidNotFoundError when not found.
            If False, return one_letter when not found. Default is True.

        Returns
        -------
        str
            Full name, or one_letter if not found and strict=False

        Raises
        ------
        AminoAcidNotFoundError
            If amino acid is not found and strict=True
        """
        aa = self._aa_map.get(one_letter)
        if aa is None and strict:
            raise AminoAcidNotFoundError(one_letter, search_type="code")
        return aa["name"] if aa else one_letter

    def get_code(self, full_name: str, strict: bool = True) -> str:
        """
        Get one-letter code from the full name

        Parameters
        ----------
        full_name : str
            Full name of the amino acid
        strict : bool, optional
            If True, raise AminoAcidNotFoundError when not found.
            If False, return full_name when not found. Default is True.

        Returns
        -------
        str
            One-letter code, or full_name if not found and strict=False

        Raises
        ------
        AminoAcidNotFoundError
            If amino acid is not found and strict=True
        """
        aa = self.get_aa_by_full_name(full_name, strict=strict)
        return aa["one_letter"] if aa else full_name

    def get_three_letter_code(self, one_letter: str, strict: bool = True) -> str:
        """
        Convert one-letter code to three-letter code

        Parameters
        ----------
        one_letter : str
            One-letter code of the amino acid
        strict : bool, optional
            If True, raise AminoAcidNotFoundError when not found.
            If False, return one_letter when not found. Default is True.

        Returns
        -------
        str
            Three-letter code, or one_letter if not found and strict=False

        Raises
        ------
        AminoAcidNotFoundError
            If amino acid is not found and strict=True
        """
        aa = self._aa_map.get(one_letter)
        if aa is None and strict:
            raise AminoAcidNotFoundError(one_letter, search_type="code")
        return aa["three_letters"] if aa else one_letter

    def get_one_letter_code(self, three_letters: str, strict: bool = True) -> str:
        """
        Convert a three-letter code to a one-letter code

        Parameters
        ----------
        three_letters : str
            Three-letter code of the amino acid
        strict : bool, optional
            If True, raise AminoAcidNotFoundError when not found.
            If False, return three_letters when not found. Default is True.

        Returns
        -------
        str
            One-letter code, or three_letters if not found and strict=False

        Raises
        ------
        AminoAcidNotFoundError
            If amino acid is not found and strict=True
        """
        aa = self._three_letter_map.get(three_letters)
        if aa is None and strict:
            raise AminoAcidNotFoundError(three_letters, search_type="three_letter_code")
        return aa["one_letter"] if aa else three_letters


# Global instance (singleton pattern)
AA_DATABASE = AminoAcidsDatabase()


def get_aa_database() -> AminoAcidsDatabase:
    """
    Get a global schema instance (singleton)

    Returns
    -------
    AminoAcidsDatabase
        Global schema loader instance
    """
    global AA_DATABASE
    if AA_DATABASE is None:
        AA_DATABASE = AminoAcidsDatabase()
    return AA_DATABASE


def reload_schema():
    """Reload schema from a file (useful after modifications)"""
    global AA_DATABASE
    AA_DATABASE = None
    return get_aa_database()

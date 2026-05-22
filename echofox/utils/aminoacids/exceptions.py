"""Custom exceptions for the amino acids package"""


class AminoAcidError(Exception):
    """Base exception for all amino acid-related errors"""
    pass


class SchemaError(AminoAcidError):
    """Raised when there is an error loading or parsing the schema file"""
    pass


class SchemaFileNotFoundError(SchemaError):
    """Raised when the schema YAML file is not found"""
    pass


class SchemaParseError(SchemaError):
    """Raised when the schema YAML file cannot be parsed"""
    pass


class SchemaMissingFieldError(SchemaError):
    """Raised when required fields are missing from the schema"""
    pass


class AminoAcidNotFoundError(AminoAcidError):
    """Raised when an amino acid cannot be found by code or name"""

    def __init__(self, identifier: str, search_type: str = "code"):
        """
        Initialize exception

        Parameters
        ----------
        identifier : str
            The amino acid identifier that was not found
        search_type : str, optional
            Type of search performed ("code", "name", etc.)
        """
        self.identifier = identifier
        self.search_type = search_type
        super().__init__(f"Amino acid not found: {identifier} (searched by {search_type})")


class InvalidAminoAcidCodeError(AminoAcidError):
    """Raised when an invalid amino acid code is provided"""

    def __init__(self, code: str):
        """
        Initialize exception

        Parameters
        ----------
        code : str
            The invalid amino acid code
        """
        self.code = code
        super().__init__(f"Invalid amino acid code: {code}")


class InvalidAminoAcidNameError(AminoAcidError):
    """Raised when an invalid amino acid name is provided"""

    def __init__(self, name: str):
        """
        Initialize exception

        Parameters
        ----------
        name : str
            The invalid amino acid name
        """
        self.name = name
        super().__init__(f"Invalid amino acid name: {name}")

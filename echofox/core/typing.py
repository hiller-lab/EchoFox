# typing
from typing import Literal, TypeAlias, get_args, get_origin

from echofox.core.colors import Color

Colors: TypeAlias = list[Color]
Number: TypeAlias = int | float


# functions to check types


def is_newtype_instance(value: object, newtype: type) -> bool:
    """Function to check if a variable follows a NewType or a Union of NewTypes."""

    # If it's a union, check against all contained types
    if hasattr(newtype, "__args__"):  # Handles UnionTypes
        return any(is_newtype_instance(value, t) for t in newtype.__args__)

    # Check if it's a NewType (NewTypes have __supertype__)
    if hasattr(newtype, "__supertype__"):
        return isinstance(value, newtype.__supertype__)

    # Otherwise, do a normal isinstance check
    return isinstance(value, newtype)


def is_typealias_instance(value, expected_type) -> bool:
    """Check if a value follows the expected TypeAlias structure."""

    origin = get_origin(expected_type)  # Extracts base type (list, tuple, etc.)
    args = get_args(expected_type)  # Extracts inner types (e.g., str, int, etc.)

    if origin is tuple:
        return (
            isinstance(value, tuple)  # Must be a tuple
            and len(value) == len(args)  # Must have the same number of elements
            and all(
                is_typealias_instance(v, t) for v, t in zip(value, args)
            )  # Element-wise validation
        )

    elif origin is list:
        return (
            isinstance(value, list)  # Must be a list
            and all(
                is_typealias_instance(item, args[0]) for item in value
            )  # Recursively check each item
        )

    elif origin is Literal:
        return value in args  # Ensure value matches one of the allowed Literals

    # Otherwise, fallback to normal isinstance check
    return isinstance(value, expected_type)

import re

from echofox.core.typing import Number


def convert_to_inches(value: str | Number):
    """
    Converts a string containing a number with a unit (mm, cm, m) into inches.
    If a single number is provided **without** a unit, it is returned as a float.

    Supported units:
    - "mm" (millimeters)
    - "cm" (centimeters)
    - "m" (meters)

    Returns:
        float: The value in inches if a unit is provided.
        float: The raw number if no unit is provided.
    """

    # Conversion factors
    conversion_factors = {
        "mm": 0.0393701,  # 1 mm = 0.0393701 inches
        "cm": 0.393701,  # 1 cm = 0.393701 inches
        "m": 39.3701,  # 1 m = 39.3701 inches
    }

    value = str(value).strip()

    # Check if input is just a number without a unit
    if re.match(r"^\d+(\.\d+)?$", value):
        return float(value)  # Return number as-is

    # Check for a number with a unit
    match = re.match(r"^([\d\.]+)\s*(mm|cm|m)\s*$", value, re.IGNORECASE)
    if match:
        number, unit = match.groups()
        number = float(number)
        unit = unit.lower()
        return number * conversion_factors[unit]  # Convert to inches

    # If input is invalid
    raise ValueError(
        f"Invalid input format: {value}. Expected format: '10mm', '5 cm', '2m', or '10'."
    )

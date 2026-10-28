from decimal import Decimal

from campus_rag.tools.numbers import chinese_integer, normalize_cell, parse_decimal


def test_chinese_numbers_and_units():
    assert chinese_integer("一千零二") == 1002
    assert parse_decimal("1,234.50元") == Decimal("1234.50")
    assert normalize_cell("12.5％") == {
        "raw_value": "12.5％",
        "normalized_value": "12.5%",
        "numeric_value": "0.125",
        "unit": "%",
    }


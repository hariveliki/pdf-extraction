"""The numeric cross-check - the layer that catches quiet VLM failures."""

from __future__ import annotations

from decimal import Decimal

import pytest

from pdfx.config import ValidationConfig
from pdfx.validate import (
    FindingKind,
    Severity,
    compare_numbers,
    extract_numbers,
    parse_number,
    severity_at_least,
)

CFG = ValidationConfig()


# ---- number parsing -------------------------------------------------------


@pytest.mark.parametrize(
    "token,expected",
    [
        ("42", Decimal("42")),
        ("-17", Decimal("-17")),
        ("0.5", Decimal("0.5")),
        ("21.4", Decimal("21.4")),
        # Swiss apostrophe grouping - unambiguous.
        ("1'013.25", Decimal("1013.25")),
        ("14'827.60", Decimal("14827.6")),
        ("1'248", Decimal("1248")),
        # Mixed separators - the rightmost is the decimal point.
        ("1.234,56", Decimal("1234.56")),
        ("1,234.56", Decimal("1234.56")),
        ("1.234.567,89", Decimal("1234567.89")),
    ],
)
def test_unambiguous_numbers_parse_to_one_value(token, expected):
    tok = parse_number(token)
    assert tok is not None, token
    assert expected in tok.values


def test_a_single_separator_before_three_digits_stays_ambiguous():
    """1,234 is 1234 in Zurich and 1.234 in Berlin. Keep both readings."""
    tok = parse_number("1,234")
    assert tok is not None
    assert tok.values == {Decimal("1234"), Decimal("1.234")}


def test_forcing_a_locale_collapses_the_ambiguity():
    assert parse_number("1,234", locale="ch").values == {Decimal("1234")}
    assert parse_number("1,234", locale="de").values == {Decimal("1.234")}


def test_two_decimal_places_is_never_a_group_separator():
    tok = parse_number("21.40")
    assert tok.values == {Decimal("21.4")}


def test_version_strings_are_not_numbers():
    assert parse_number("1.2.3") is None


def test_non_numeric_input_is_rejected():
    assert parse_number("") is None
    assert parse_number("CHF") is None


def test_extraction_finds_numbers_in_running_text():
    text = "Reference pressure: 1'013.25 hPa across 1'248 samples, -0.37 % drift"
    values = [v for tok in extract_numbers(text) for v in tok.values]
    assert Decimal("1013.25") in values
    assert Decimal("1248") in values
    assert Decimal("-0.37") in values


# ---- comparison -----------------------------------------------------------


def test_identical_readings_produce_no_findings():
    text = "Total invoiced: CHF 14'827.60 over 1'248 samples"
    result = compare_numbers(text, text, CFG)
    assert result.checked
    assert result.findings == []
    assert result.max_severity is Severity.NONE
    assert result.match_rate == 1.0


def test_reformatted_numbers_still_match():
    """The VLM dropping thousands separators is not an error."""
    det = "Total invoiced: CHF 14'827.60"
    vlm = "Total invoiced: CHF 14827.60"
    result = compare_numbers(det, vlm, CFG)
    assert result.findings == []


def test_transposed_digits_are_caught_at_high_severity():
    """The headline failure mode: 14'827.60 read back as 14'287.60."""
    det = "Total invoiced: CHF 14'827.60"
    vlm = "Total invoiced: CHF 14'287.60"
    result = compare_numbers(det, vlm, CFG)

    kinds = [f.kind for f in result.findings]
    assert FindingKind.TRANSPOSED_DIGITS in kinds
    assert result.max_severity is Severity.HIGH
    finding = next(f for f in result.findings if f.kind is FindingKind.TRANSPOSED_DIGITS)
    assert finding.deterministic == "14'827.60"
    assert finding.vlm == "14'287.60"


def test_a_single_substituted_digit_is_caught():
    """8 read as 3 is the classic OCR confusion."""
    result = compare_numbers("Sample count: 1'248", "Sample count: 1'243", CFG)
    kinds = [f.kind for f in result.findings]
    assert FindingKind.SUBSTITUTED_DIGIT in kinds
    assert result.max_severity is Severity.HIGH


def test_a_dropped_number_is_reported_as_missing():
    det = "Pressure 1'013.25 hPa, deviation -0.37 %"
    vlm = "Pressure 1'013.25 hPa"
    result = compare_numbers(det, vlm, CFG)
    missing = [f for f in result.findings if f.kind is FindingKind.MISSING_FROM_VLM]
    assert len(missing) == 1
    assert missing[0].deterministic == "-0.37"
    assert missing[0].severity is Severity.HIGH


def test_an_invented_number_is_reported():
    det = "Pressure 1'013.25 hPa"
    vlm = "Pressure 1'013.25 hPa, deviation -0.37 %"
    result = compare_numbers(det, vlm, CFG)
    extra = [f for f in result.findings if f.kind is FindingKind.NOT_IN_DETERMINISTIC]
    assert len(extra) == 1
    assert extra[0].vlm == "-0.37"


def test_tesseract_disagreements_are_graded_lower_than_the_text_layer():
    """A noisy second opinion should not scream as loudly as ground truth."""
    det, vlm = "Total 14'827.60", "Total 14'287.60"
    authoritative = compare_numbers(det, vlm, CFG, source="text_layer")
    second_opinion = compare_numbers(det, vlm, CFG, source="tesseract")
    assert authoritative.max_severity is Severity.HIGH
    assert second_opinion.max_severity is Severity.MEDIUM


def test_single_digit_noise_is_low_severity():
    """Page numbers and list markers should not dominate the report."""
    result = compare_numbers("item 7 of the list", "item of the list", CFG)
    assert result.max_severity is Severity.LOW


def test_tolerance_absorbs_rounding_differences():
    cfg = ValidationConfig(abs_tolerance=0.05)
    result = compare_numbers("Temperature 21.40 C", "Temperature 21.38 C", cfg)
    assert result.findings == []


def test_pages_without_numbers_are_reported_as_such():
    result = compare_numbers("no digits here", "no digits here either", CFG)
    assert result.checked
    assert result.note == "no numbers on this page"


def test_match_counting_is_one_to_one():
    """Two occurrences of a figure must not both match a single output copy."""
    result = compare_numbers("100 and 100", "100", CFG)
    assert result.matched == 1
    assert result.deterministic_count == 2
    assert len(result.findings) == 1


# ---- severity gating ------------------------------------------------------


def test_severity_threshold_gating():
    assert severity_at_least(Severity.HIGH, "high")
    assert severity_at_least(Severity.HIGH, "medium")
    assert not severity_at_least(Severity.LOW, "medium")
    assert not severity_at_least(Severity.HIGH, "none")


def test_unknown_threshold_is_rejected():
    with pytest.raises(ValueError, match="unknown severity threshold"):
        severity_at_least(Severity.HIGH, "critical")

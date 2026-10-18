"""Deterministic numeric cross-check.

The design point this whole module exists for: rule-based parsers fail loudly,
so you notice. VLM parsers fail *quietly* - fluent, well-formatted output with
two digits swapped in a total. On invoices, measurement reports and spec sheets
that is the expensive failure.

So every page the VLM reads is also read deterministically (its own text layer
where it has one, otherwise Tesseract), the numbers from both are compared, and
disagreements are reported with a severity instead of being averaged away.

Locale handling matters here: Swiss documents write 1'234.56, German ones
1.234,56 and US ones 1,234.56. Rather than guess a document locale, ambiguous
tokens carry every plausible interpretation and match if any interpretation
agrees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Iterable

from .config import ValidationConfig

# Separators that are never decimal points: Swiss apostrophe, typographic
# apostrophe, no-break space, narrow no-break space, thin space.
GROUP_ONLY_CHARS = "'’   "

# A numeric run: digits, optionally interleaved with separators, ending in a digit.
NUMBER_RE = re.compile(
    r"[-+−]?\d(?:[\d" + re.escape(GROUP_ONLY_CHARS) + r".,]*\d)?"
)


class Severity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_SEVERITY_ORDER = {
    Severity.NONE: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
}


class FindingKind(str, Enum):
    TRANSPOSED_DIGITS = "transposed_digits"
    SUBSTITUTED_DIGIT = "substituted_digit"
    MISSING_FROM_VLM = "missing_from_vlm"
    NOT_IN_DETERMINISTIC = "not_in_deterministic"


@dataclass(frozen=True)
class NumberToken:
    """One numeric run, with every locale interpretation that fits it."""

    text: str
    values: frozenset[Decimal]
    digits: str

    @property
    def is_single_digit(self) -> bool:
        return len(self.digits) <= 1


@dataclass
class Finding:
    kind: FindingKind
    severity: Severity
    detail: str
    deterministic: str | None = None
    vlm: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind.value,
            "severity": self.severity.value,
            "detail": self.detail,
        }
        if self.deterministic is not None:
            out["deterministic"] = self.deterministic
        if self.vlm is not None:
            out["vlm"] = self.vlm
        return out


@dataclass
class PageValidation:
    page_index: int
    checked: bool
    source: str = "none"
    deterministic_count: int = 0
    vlm_count: int = 0
    matched: int = 0
    findings: list[Finding] = field(default_factory=list)
    note: str = ""

    @property
    def max_severity(self) -> Severity:
        if not self.findings:
            return Severity.NONE
        return max((f.severity for f in self.findings), key=lambda s: _SEVERITY_ORDER[s])

    @property
    def match_rate(self) -> float | None:
        if not self.checked or self.deterministic_count == 0:
            return None
        return self.matched / self.deterministic_count

    def to_dict(self) -> dict[str, Any]:
        rate = self.match_rate
        return {
            "checked": self.checked,
            "source": self.source,
            "deterministic_numbers": self.deterministic_count,
            "vlm_numbers": self.vlm_count,
            "matched": self.matched,
            "match_rate": None if rate is None else round(rate, 4),
            "max_severity": self.max_severity.value,
            "findings": [f.to_dict() for f in self.findings],
            "note": self.note,
        }


# ---- number parsing -------------------------------------------------------


def _interpretations(core: str) -> set[Decimal]:
    """Every plausible Decimal reading of a separator-bearing numeric string.

    `core` has already had sign and group-only characters removed, so it
    contains digits, '.' and ',' only.
    """
    has_dot = "." in core
    has_comma = "," in core

    def to_decimal(s: str) -> Decimal | None:
        try:
            return Decimal(s).normalize()
        except (InvalidOperation, ValueError):
            return None

    out: set[Decimal] = set()

    if not has_dot and not has_comma:
        d = to_decimal(core)
        return {d} if d is not None else set()

    if has_dot and has_comma:
        # Mixed separators are unambiguous: the rightmost one is the decimal
        # point and the other is grouping. 1.234,56 -> 1234.56
        dec_sep = "," if core.rfind(",") > core.rfind(".") else "."
        grp_sep = "." if dec_sep == "," else ","
        if core.count(dec_sep) > 1:
            return set()  # e.g. an IP address or version string
        d = to_decimal(core.replace(grp_sep, "").replace(dec_sep, "."))
        return {d} if d is not None else set()

    sep = "." if has_dot else ","
    parts = core.split(sep)

    if core.count(sep) > 1:
        # Repeated single separator: grouping, if the groups are well formed.
        if all(len(p) == 3 for p in parts[1:]) and 1 <= len(parts[0]) <= 3:
            d = to_decimal("".join(parts))
            if d is not None:
                out.add(d)
        return out

    head, tail = parts[0], parts[1]

    # Decimal reading is always plausible.
    d = to_decimal(f"{head}.{tail}")
    if d is not None:
        out.add(d)

    # Grouping reading only when the shape fits: 1-3 digits, then exactly 3.
    if len(tail) == 3 and 1 <= len(head) <= 3:
        d = to_decimal(head + tail)
        if d is not None:
            out.add(d)

    return out


def parse_number(token: str, locale: str = "auto") -> NumberToken | None:
    """Turn a numeric run into a NumberToken, or None if it is not a number.

    locale: "auto" keeps ambiguous readings open; "ch"/"us" force '.' as the
    decimal point and "de" forces ','.
    """
    raw = token.strip()
    if not raw:
        return None

    negative = raw[0] in "-−"
    core = raw.lstrip("+-−")
    for ch in GROUP_ONLY_CHARS:
        core = core.replace(ch, "")

    if not core or not any(c.isdigit() for c in core):
        return None

    if locale == "auto":
        values = _interpretations(core)
    elif locale in {"ch", "us", "de"}:
        # Forced locale: the decimal point is known, so there is exactly one
        # reading and grouping separators are simply dropped.
        if locale == "de":
            core_forced = core.replace(".", "").replace(",", ".")
        else:
            core_forced = core.replace(",", "")
        try:
            values = {Decimal(core_forced).normalize()}
        except (InvalidOperation, ValueError):
            values = set()
    else:
        raise ValueError(f"unknown number_locale {locale!r}; expected auto, ch, de or us")

    if not values:
        return None

    if negative:
        values = {-v for v in values}

    digits = "".join(c for c in core if c.isdigit())
    return NumberToken(text=raw, values=frozenset(values), digits=digits)


def extract_numbers(text: str, locale: str = "auto") -> list[NumberToken]:
    """Every numeric run in a block of text, in document order."""
    tokens: list[NumberToken] = []
    for match in NUMBER_RE.finditer(text or ""):
        tok = parse_number(match.group(0), locale=locale)
        if tok is not None:
            tokens.append(tok)
    return tokens


# ---- comparison -----------------------------------------------------------


def _values_agree(a: NumberToken, b: NumberToken, tolerance: float) -> bool:
    if a.values & b.values:
        return True
    if tolerance > 0:
        tol = Decimal(str(tolerance))
        return any(abs(x - y) <= tol for x in a.values for y in b.values)
    return False


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 2:
        return 99
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _classify_pair(det: NumberToken, vlm: NumberToken) -> tuple[FindingKind, str] | None:
    """Decide whether two unmatched tokens are a corrupted version of each other."""
    if det.digits == vlm.digits:
        # Same digits, different value: a separator moved. Real, but usually a
        # formatting artefact rather than a corrupted figure.
        return None
    if sorted(det.digits) == sorted(vlm.digits) and len(det.digits) > 1:
        return (
            FindingKind.TRANSPOSED_DIGITS,
            f"same digits in a different order: {det.text!r} vs {vlm.text!r}",
        )
    if len(det.digits) == len(vlm.digits) and _edit_distance(det.digits, vlm.digits) == 1:
        return (
            FindingKind.SUBSTITUTED_DIGIT,
            f"one digit differs: {det.text!r} vs {vlm.text!r}",
        )
    return None


def compare_numbers(
    deterministic_text: str,
    vlm_text: str,
    cfg: ValidationConfig,
    page_index: int = 0,
    source: str = "text_layer",
) -> PageValidation:
    """Cross-check the numbers in VLM output against a deterministic reading.

    `source` records how authoritative the deterministic side is: a page's own
    embedded text layer is ground truth, Tesseract output is a second opinion
    and its disagreements are graded lower.
    """
    authoritative = source == "text_layer"

    det_tokens = extract_numbers(deterministic_text, cfg.number_locale)
    vlm_tokens = extract_numbers(vlm_text, cfg.number_locale)

    result = PageValidation(
        page_index=page_index,
        checked=True,
        source=source,
        deterministic_count=len(det_tokens),
        vlm_count=len(vlm_tokens),
    )

    if not det_tokens and not vlm_tokens:
        result.note = "no numbers on this page"
        return result

    # Greedy one-to-one matching on value.
    unmatched_vlm = list(vlm_tokens)
    unmatched_det: list[NumberToken] = []

    for det in det_tokens:
        hit = next(
            (i for i, v in enumerate(unmatched_vlm)
             if _values_agree(det, v, cfg.abs_tolerance)),
            None,
        )
        if hit is None:
            unmatched_det.append(det)
        else:
            unmatched_vlm.pop(hit)
            result.matched += 1

    # Leftovers: try to pair them up as corruptions before calling them
    # missing or invented.
    still_extra = list(unmatched_vlm)
    truly_missing: list[NumberToken] = []

    for det in unmatched_det:
        paired = None
        for i, vlm in enumerate(still_extra):
            verdict = _classify_pair(det, vlm)
            if verdict is not None:
                paired = (i, verdict)
                break
        if paired is None:
            truly_missing.append(det)
            continue

        idx, (kind, detail) = paired
        vlm = still_extra.pop(idx)
        sev = Severity.HIGH if authoritative else Severity.MEDIUM
        result.findings.append(
            Finding(
                kind=kind,
                severity=sev,
                detail=detail,
                deterministic=det.text,
                vlm=vlm.text,
            )
        )

    for det in truly_missing:
        if det.is_single_digit:
            sev = Severity.LOW
        elif authoritative:
            sev = Severity.HIGH
        else:
            sev = Severity.MEDIUM
        result.findings.append(
            Finding(
                kind=FindingKind.MISSING_FROM_VLM,
                severity=sev,
                detail=f"{det.text!r} is in the deterministic reading but absent "
                       f"from the model output",
                deterministic=det.text,
            )
        )

    for vlm in still_extra:
        if vlm.is_single_digit:
            sev = Severity.LOW
        elif authoritative:
            sev = Severity.MEDIUM
        else:
            sev = Severity.LOW
        result.findings.append(
            Finding(
                kind=FindingKind.NOT_IN_DETERMINISTIC,
                severity=sev,
                detail=f"{vlm.text!r} appears in the model output but not in the "
                       f"deterministic reading",
                vlm=vlm.text,
            )
        )

    return result


def skipped(page_index: int, note: str) -> PageValidation:
    """A page that was not cross-checked, and why."""
    return PageValidation(page_index=page_index, checked=False, note=note)


def worst_severity(validations: Iterable[PageValidation]) -> Severity:
    worst = Severity.NONE
    for v in validations:
        if _SEVERITY_ORDER[v.max_severity] > _SEVERITY_ORDER[worst]:
            worst = v.max_severity
    return worst


def severity_at_least(actual: Severity, threshold: str) -> bool:
    """True when `actual` meets or exceeds a configured threshold string."""
    if threshold == "none":
        return False
    try:
        want = Severity(threshold)
    except ValueError as exc:
        raise ValueError(
            f"unknown severity threshold {threshold!r}; expected none, medium or high"
        ) from exc
    return _SEVERITY_ORDER[actual] >= _SEVERITY_ORDER[want]


# ---- Tesseract second opinion ---------------------------------------------


def tesseract_text(png_bytes: bytes) -> str | None:
    """OCR a rendered page with Tesseract, or None if it is not installed.

    Only used for pages with no text layer at all, where there would otherwise
    be nothing to check the VLM against.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return None

    import io

    try:
        return pytesseract.image_to_string(Image.open(io.BytesIO(png_bytes)))
    except Exception:
        # Missing binary, unsupported language pack, etc. A failed second
        # opinion must never fail the run.
        return None

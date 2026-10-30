"""Read-only report assembly for endmember export checks.

This module does not compare models, load files/plugins, or generate databases.
Callers declare the checks in scope and supply their findings. Comparison adapters
can be added separately without making the report specific to unary elements.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import json
from typing import Literal, TypeAlias


JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)
FindingStatus: TypeAlias = Literal[
    "passed", "failed", "not_checked", "unsupported", "not_applicable"
]
Severity: TypeAlias = Literal["info", "warning", "error"]

_STATUSES = ("passed", "failed", "not_checked", "unsupported", "not_applicable")
_SEVERITIES = ("info", "warning", "error")


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")


def _json_snapshot(value: JSONValue, name: str) -> JSONValue:
    """Copy JSON evidence without retaining aliases or accepting NaN/Infinity."""
    try:
        encoded = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain finite JSON-compatible values; convert symbolic "
            "expressions and array/scalar objects explicitly before reporting."
        ) from exc

    def validate_keys(item, location):
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError(f"{location} keys must be strings.")
                validate_keys(child, f"{location}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                validate_keys(child, f"{location}[{index}]")

    # Serialization rejects cycles before this traversal. Validate the original
    # keys because json.dumps otherwise silently converts integer keys to strings.
    validate_keys(value, name)
    return json.loads(encoded)


@dataclass(frozen=True)
class EndmemberTarget:
    """Identify one endmember, preserving sublattice order and vacancy labels.

    Each sublattice contains one species. A compound can have distinct species on
    different sublattices, e.g. (("AL",), ("NI",)). Species labels are opaque;
    no composition, reference phase, site ratios, or normalization is inferred.
    """

    phase: str
    constituent_array: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        _require_text(self.phase, "phase")
        if (
            not isinstance(self.constituent_array, (list, tuple))
            or not self.constituent_array
        ):
            raise ValueError(
                "constituent_array must be a nonempty sublattice sequence."
            )
        normalized = []
        for sublattice in self.constituent_array:
            if not isinstance(sublattice, (list, tuple)) or len(sublattice) != 1:
                raise ValueError(
                    "Each endmember sublattice must contain exactly one species."
                )
            _require_text(sublattice[0], "constituent species")
            normalized.append(tuple(sublattice))
        object.__setattr__(self, "constituent_array", tuple(normalized))

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a JSON-compatible target without collapsing sublattices."""
        return {
            "phase": self.phase,
            "constituent_array": [
                list(sublattice) for sublattice in self.constituent_array
            ],
        }


@dataclass(frozen=True)
class AuditCheck:
    """Declare a named check for one target; this pair is its unique identity."""

    check_id: str
    target: EndmemberTarget

    def __post_init__(self) -> None:
        _require_text(self.check_id, "check_id")
        if not isinstance(self.target, EndmemberTarget):
            raise TypeError("target must be an EndmemberTarget.")

    def to_dict(self) -> dict[str, JSONValue]:
        return {"check_id": self.check_id, "target": self.target.to_dict()}


@dataclass(frozen=True)
class AuditFinding:
    """Record one check outcome, independently of any fit-acceptance policy.

    Use stable codes for machine consumers and a message for human readers.
    Evidence and expected/observed values are JSON snapshots, not live database,
    model, array, or symbolic objects. None means no value supplied or JSON null.
    """

    check: AuditCheck
    code: str
    status: FindingStatus
    message: str
    severity: Severity = "info"
    model: str | None = None
    parameter: str | None = None
    expected: JSONValue = None
    observed: JSONValue = None
    evidence: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.check, AuditCheck):
            raise TypeError("check must be an AuditCheck.")
        _require_text(self.code, "code")
        _require_text(self.message, "message")
        if self.status not in _STATUSES:
            raise ValueError(f"status must be one of {_STATUSES}.")
        if self.severity not in _SEVERITIES:
            raise ValueError(f"severity must be one of {_SEVERITIES}.")
        for name in ("model", "parameter"):
            if getattr(self, name) is not None:
                _require_text(getattr(self, name), name)
        if not isinstance(self.evidence, dict):
            raise TypeError("evidence must be a dictionary with string keys.")
        for name in ("expected", "observed", "evidence"):
            object.__setattr__(self, name, _json_snapshot(getattr(self, name), name))

    def to_dict(self) -> dict[str, JSONValue]:
        """Return detached evidence so report consumers cannot mutate the finding."""
        return {
            **self.check.to_dict(),
            "code": self.code,
            "status": self.status,
            "severity": self.severity,
            "message": self.message,
            "model": self.model,
            "parameter": self.parameter,
            "expected": _json_snapshot(self.expected, "expected"),
            "observed": _json_snapshot(self.observed, "observed"),
            "evidence": _json_snapshot(self.evidence, "evidence"),
        }


def audit_endmember_export(
    planned_checks: Iterable[AuditCheck],
    findings: Iterable[AuditFinding] = (),
    *,
    provenance: Mapping[str, JSONValue] | None = None,
) -> dict[str, JSONValue]:
    """Assemble a versioned report from an explicit plan and supplied findings.

    This initial API is a report builder, not an implementation of export
    comparisons. It performs no file/plugin/database access and makes no approval
    decision. One finding is allowed per planned check. Missing findings become
    not_checked; undeclared and duplicate results raise errors rather than being
    ignored. Output follows plan order regardless of the findings' input order.

    An empty plan is not_checked, never passed. A report is complete only when a
    nonempty plan has no not_checked or unsupported outcomes. A failed check
    makes the overall status failed even if other checks are still incomplete.
    An all-not_applicable plan is complete but not a pass. A passed report means
    only that the declared scope passed, not that the export or fit is approved.
    """
    plan = tuple(planned_checks)
    if any(not isinstance(check, AuditCheck) for check in plan):
        raise TypeError("planned_checks must contain AuditCheck objects.")
    if len(set(plan)) != len(plan):
        raise ValueError("planned_checks contains a duplicate check_id/target pair.")
    declared = set(plan)
    supplied = {}
    for finding in findings:
        if not isinstance(finding, AuditFinding):
            raise TypeError("findings must contain AuditFinding objects.")
        if finding.check not in declared:
            raise ValueError(
                f"Finding for undeclared check: {finding.check.check_id!r}."
            )
        if finding.check in supplied:
            raise ValueError(
                f"Duplicate finding for check: {finding.check.check_id!r}."
            )
        supplied[finding.check] = finding

    results = []
    for check in plan:
        finding = supplied.get(check)
        if finding is None:
            finding = AuditFinding(
                check=check,
                code="AUDIT_CHECK_NOT_RUN",
                status="not_checked",
                message="No result was supplied for this planned check.",
            )
        results.append(finding.to_dict())

    counts = {
        status: sum(item["status"] == status for item in results)
        for status in _STATUSES
    }
    complete = bool(plan) and not (counts["not_checked"] or counts["unsupported"])
    if not plan:
        status = "not_checked"
    elif counts["failed"]:
        status = "failed"
    elif not complete:
        status = "incomplete"
    elif counts["passed"]:
        status = "passed"
    else:
        status = "not_applicable"
    if provenance is not None and not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping with string keys.")
    return {
        "schema_version": "1.0",
        "scope": [check.to_dict() for check in plan],
        "summary": {
            "status": status,
            "complete": complete,
            "checks_planned": len(plan),
            "counts": counts,
            "severity_counts": {
                severity: sum(item["severity"] == severity for item in results)
                for severity in _SEVERITIES
            },
        },
        "findings": results,
        "provenance": _json_snapshot(
            dict(provenance) if provenance is not None else {}, "provenance"
        ),
    }

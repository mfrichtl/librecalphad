"""The report contract is independent of live databases, plugins, and elements."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import subprocess
import sys

from libreCalphad.databases.endmember_audit import (
    AuditCheck,
    AuditFinding,
    EndmemberTarget,
    audit_endmember_export,
)
import pytest


@pytest.fixture
def check():
    return AuditCheck("native.THETA", EndmemberTarget("BCC_A2", (("FE",), ("VA",))))


def finding(check, status="passed", **kwargs):
    return AuditFinding(
        check, code="NATIVE_PARAMETER_COMPARED", status=status,
        message="Compared the supplied values.", **kwargs,
    )


@pytest.mark.parametrize(
    "phase,constituents",
    [
        ("BCC_A2", (("FE",), ("VA",))),
        ("B2", (("AL",), ("NI",))),
        ("B2", (("NI",), ("AL",))),
        ("COMPOUND", (("AL2O3",),)),
        ("THREE_SUBLATTICES", (("CR",), ("FE",), ("C",))),
    ],
)
def test_targets_preserve_sublattices_and_species(phase, constituents):
    original = [list(sublattice) for sublattice in constituents]
    target = EndmemberTarget(phase, original)
    original[0][0] = "CHANGED"

    assert target.constituent_array == constituents
    assert target.to_dict() == {
        "phase": phase, "constituent_array": [list(row) for row in constituents],
    }
    with pytest.raises(FrozenInstanceError):
        target.phase = "CHANGED"


@pytest.mark.parametrize("phase", [None, 1, "", "  "])
def test_target_requires_phase(phase):
    with pytest.raises(ValueError, match="phase must be a nonempty string"):
        EndmemberTarget(phase, (("FE",),))


@pytest.mark.parametrize(
    "constituents",
    [None, [], "FE", ["FE"], [()], [("FE", "NI")], [(None,)], [(1,)], [("",)]],
)
def test_target_rejects_invalid_or_mixed_configurations(constituents):
    with pytest.raises(ValueError):
        EndmemberTarget("BCC_A2", constituents)


def test_empty_plan_never_passes():
    report = audit_endmember_export([])

    assert report["schema_version"] == "1.0"
    assert report["scope"] == []
    assert report["findings"] == []
    assert report["summary"]["status"] == "not_checked"
    assert report["summary"]["complete"] is False
    assert report["summary"]["checks_planned"] == 0
    assert not any(report["summary"]["counts"].values())
    assert report["provenance"] == {}
    assert "approved" not in report


def test_missing_check_results_are_explicit(check):
    report = audit_endmember_export([check])

    assert report["summary"]["status"] == "incomplete"
    assert report["summary"]["complete"] is False
    assert report["summary"]["counts"]["not_checked"] == 1
    result = report["findings"][0]
    assert result["check_id"] == check.check_id
    assert result["target"] == check.target.to_dict()
    assert result["status"] == "not_checked"
    assert result["code"] == "AUDIT_CHECK_NOT_RUN"
    assert result["message"]


@pytest.mark.parametrize(
    "status,overall,complete",
    [
        ("passed", "passed", True),
        ("failed", "failed", True),
        ("not_checked", "incomplete", False),
        ("unsupported", "incomplete", False),
        ("not_applicable", "not_applicable", True),
    ],
)
def test_summary_distinguishes_status_and_completeness(
    check, status, overall, complete
):
    report = audit_endmember_export([check], [finding(check, status)])

    assert report["summary"]["status"] == overall
    assert report["summary"]["complete"] is complete
    assert report["summary"]["counts"][status] == 1
    assert sum(report["summary"]["counts"].values()) == 1


def test_failure_does_not_hide_incomplete_scope(check):
    second = AuditCheck("reference.registry", check.target)
    report = audit_endmember_export([check, second], [finding(check, "failed")])

    assert report["summary"]["status"] == "failed"
    assert report["summary"]["complete"] is False
    assert report["summary"]["counts"]["failed"] == 1
    assert report["summary"]["counts"]["not_checked"] == 1


def test_same_check_can_apply_to_distinct_multicomponent_endmembers():
    targets = [
        EndmemberTarget("B2", (("AL",), ("NI",))),
        EndmemberTarget("B2", (("NI",), ("AL",))),
        EndmemberTarget("BCC_A2", (("FE",), ("VA",))),
    ]
    plan = [AuditCheck("reference.registry", target) for target in targets]
    results = [finding(plan[0]), finding(plan[1], "not_applicable"), finding(plan[2])]
    original = deepcopy(results)

    report = audit_endmember_export(iter(plan), iter(reversed(results)))

    assert report["scope"] == [item.to_dict() for item in plan]
    assert [row["target"] for row in report["findings"]] == [
        target.to_dict() for target in targets
    ]
    assert report["summary"]["status"] == "passed"
    assert report["summary"]["complete"] is True
    assert results == original


def test_severity_is_not_an_acceptance_policy(check):
    report = audit_endmember_export([check], [finding(check, severity="warning")])

    assert report["summary"]["status"] == "passed"
    assert report["summary"]["severity_counts"] == {"info": 0, "warning": 1, "error": 0}
    assert "approved" not in report["summary"]


def test_duplicate_check_plan_is_rejected(check):
    with pytest.raises(ValueError, match="duplicate check_id/target"):
        audit_endmember_export([check, AuditCheck(check.check_id, check.target)])


def test_duplicate_or_undeclared_findings_are_rejected(check):
    with pytest.raises(ValueError, match="Duplicate finding"):
        audit_endmember_export([check], [finding(check), finding(check)])
    with pytest.raises(ValueError, match="undeclared check"):
        audit_endmember_export([], [finding(check)])


@pytest.mark.parametrize("value", ["", None, 1])
def test_check_requires_named_identity(check, value):
    with pytest.raises(ValueError, match="check_id"):
        AuditCheck(value, check.target)


def test_typed_scope_is_required(check):
    with pytest.raises(TypeError, match="EndmemberTarget"):
        AuditCheck("native.THETA", {})
    with pytest.raises(TypeError, match="planned_checks"):
        audit_endmember_export(["native.THETA"])
    with pytest.raises(TypeError, match="findings"):
        audit_endmember_export([check], [{}])
    with pytest.raises(TypeError, match="check must be an AuditCheck"):
        finding("native.THETA")


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "unknown"), ("severity", "critical"), ("code", ""),
        ("message", ""), ("model", ""), ("parameter", ""),
    ],
)
def test_finding_rejects_invalid_fields(check, field, value):
    values = {
        "check": check, "code": "PARAMETER_COMPARED", "status": "passed",
        "message": "Parameter matched.", field: value,
    }
    with pytest.raises(ValueError, match=field):
        AuditFinding(**values)


@pytest.mark.parametrize("field", ["expected", "observed", "evidence"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), object()])
def test_report_values_must_be_finite_json(check, field, value):
    with pytest.raises(ValueError, match=f"{field}.*finite JSON-compatible"):
        finding(check, **{field: {"nested": [value]}})


def test_report_payload_keys_and_evidence_shape_are_checked(check):
    with pytest.raises(ValueError, match="keys must be strings"):
        finding(check, observed={"nested": {1: "would be silently converted"}})
    with pytest.raises(TypeError, match="evidence must be a dictionary"):
        finding(check, evidence=[])
    with pytest.raises(ValueError, match="provenance.*keys must be strings"):
        audit_endmember_export([check], provenance={1: "invalid"})
    with pytest.raises(TypeError, match="provenance must be a mapping"):
        audit_endmember_export([check], provenance=[])
    with pytest.raises(ValueError, match="provenance.*finite JSON-compatible"):
        audit_endmember_export([check], provenance={"objective": float("inf")})


def test_cyclic_evidence_is_rejected(check):
    evidence = []
    evidence.append(evidence)
    with pytest.raises(ValueError, match="expected.*finite JSON-compatible"):
        finding(check, expected=evidence)


def test_report_is_a_detached_strict_json_snapshot(check):
    expected = {"theta": [300.0], "symbol": "log(300)"}
    evidence = {"source_hashes": {"fit": "example-hash"}}
    provenance = {"versions": {"example": "1.0"}}
    result = finding(
        check, expected=expected, observed=300.0, evidence=evidence,
        model="einstein", parameter="theta",
    )
    expected["theta"].append(999)
    evidence["source_hashes"]["fit"] = "changed"
    report = audit_endmember_export([check], [result], provenance=provenance)
    provenance["versions"]["example"] = "changed"

    assert report["findings"][0]["expected"]["theta"] == [300.0]
    assert report["findings"][0]["evidence"]["source_hashes"]["fit"] == "example-hash"
    assert report["provenance"]["versions"]["example"] == "1.0"
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    report["findings"][0]["expected"]["theta"].append(555)
    assert result.expected["theta"] == [300.0]
    assert audit_endmember_export([check], [result]) == audit_endmember_export(
        [check], [result]
    )


def test_import_and_empty_report_need_no_third_party_packages(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    script = f"""
import sys
sys.path.insert(0, {str(repository)!r})
from libreCalphad.databases.endmember_audit import audit_endmember_export
report = audit_endmember_export([])
assert report['summary']['status'] == 'not_checked'
assert 'pycalphad' not in sys.modules
assert 'espei' not in sys.modules
assert 'numpy' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-S", "-c", script], cwd=tmp_path,
        capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []

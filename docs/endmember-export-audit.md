# Endmember export audit report architecture

The reporting API lives in `libreCalphad/databases/endmember_audit.py`:

```text
audit_endmember_export(planned_checks, findings=(), *, provenance=None) -> dict
```

**This first implementation assembles reports from supplied findings. It does not
yet compare fitted models, registries, loaded plugins, or databases.** It reads no
assessment files, imports no ESPEI/pycalphad modules, writes no results, and makes
no fit-acceptance or promotion decision. File loading, comparison adapters,
serialization to files, and command-line presentation are separate work.

FE unary endmembers are the first intended scientific test case, not an
architectural restriction. Compound endmembers use the same report contract.
The existing [unary reference-state workflow](unary-reference-state-export.md)
describes the specific LCRefState adapter that will need to supply checks later;
its unary key convention is not imposed on all endmember reports.

## Identities and findings

- `EndmemberTarget(phase, constituent_array)` identifies an endmember using its
  phase and ordered sublattices. `(("FE",), ("VA",))` and `(("AL",), ("NI",))`
  are supported, as are opaque compound-species labels. Each sublattice must
  contain exactly one species; mixed site occupancy is not an endmember.
- Sublattice order matters: `AL:NI` and `NI:AL` in the same phase are distinct.
  Species names and vacancies are preserved, not interpreted or uppercased.
  List-based configurations are copied into tuples for a stable identity.
- `AuditCheck(check_id, target)` declares a check in the report scope. Check IDs
  should be stable machine-readable names, e.g. `native.THETA`. The same ID may
  appear for different targets, but each ID/target pair must be unique.
- `AuditFinding(check, code, status, message, ...)` supplies one result for a
  declared check. It includes severity, optional model/parameter labels,
  expected/observed values, and evidence. Use stable issue codes and descriptive
  messages rather than requiring downstream tools to parse prose.

There is currently one finding per planned check. Plan parameter-level checks
when separate results are needed for multiple parameters. Undeclared findings
and duplicate results raise errors rather than silently extending scope or
discarding evidence. Output follows plan order, not finding arrival order.

Reference phases, site ratios, species compositions, normalization, plugin names,
and paths are not guessed. Future comparisons must receive the relevant data
explicitly; their reference/normalization basis can be recorded in evidence.

## Outcome is not approval

| Finding status | Meaning |
|---|---|
| `passed` | The supplied check result satisfied its comparison. |
| `failed` | The supplied check result did not satisfy its comparison. |
| `not_checked` | No result is available; the comparison has not established an outcome. |
| `unsupported` | The declared check cannot currently be performed by its adapter. |
| `not_applicable` | The check explicitly does not apply to this target; explain why in its message. |

Severity is separately `info`, `warning`, or `error`; it does not override a
finding's outcome or implement an acceptance threshold. Every finding requires a
nonempty message. The API inserts `AUDIT_CHECK_NOT_RUN`/`not_checked` findings for
planned checks without supplied results.

The report summary has independent status and completeness:

- An empty plan is `not_checked` and incomplete, never a pass.
- Any failed check makes the summary `failed`, even if other checks are pending.
- A nonempty plan is complete only if it contains no `not_checked` or
  `unsupported` outcomes. A complete report can still fail.
- Without failures, pending/unsupported checks produce `incomplete`.
- A complete plan containing only `not_applicable` outcomes is
  `not_applicable`, not a pass.
- Otherwise a complete plan with passed checks is `passed`.

`passed` means only that the **declared checks** passed. It does not establish
that every necessary check was declared, that a model is physically adequate,
or that a fit/database has been approved. No `approved` field is generated.

## Example: record results without assuming a unary target

```python
from libreCalphad.databases.endmember_audit import (
    AuditCheck, AuditFinding, EndmemberTarget, audit_endmember_export,
)

target = EndmemberTarget("B2", (("AL",), ("NI",)))
coefficient_check = AuditCheck("model.coefficient", target)
database_check = AuditCheck("database.reference", target)

# Illustrative evidence from a caller, not a comparison performed by this API.
finding = AuditFinding(
    check=coefficient_check,
    code="COEFFICIENT_MATCH",
    status="passed",
    message="The supplied coefficient values agree.",
    model="example",
    parameter="a",
    expected=1000.0,
    observed=1000.0,
    evidence={"comparison_basis": "example only"},
)
report = audit_endmember_export(
    [coefficient_check, database_check],
    [finding],
    provenance={"input_hashes": {"fit": "example-content-hash"}},
)
assert report["schema_version"] == "1.0"
assert report["summary"]["status"] == "incomplete"
assert report["summary"]["counts"]["passed"] == 1
assert report["summary"]["counts"]["not_checked"] == 1
assert report["findings"][1]["code"] == "AUDIT_CHECK_NOT_RUN"
```

## Serialization and provenance

The returned dictionary has five top-level fields:

| Field | Contents |
|---|---|
| `schema_version` | Currently `"1.0"`, identifying this report contract. |
| `scope` | Ordered list of declared check IDs and targets. |
| `summary` | Overall status, completeness, planned count, and counts by finding status/severity. |
| `findings` | Ordered supplied or generated results with detached JSON evidence. |
| `provenance` | Caller-supplied source hashes, revisions, versions, or other explicit metadata. |

Expected/observed values, evidence, and provenance must be JSON-compatible, with
string dictionary keys and finite numbers. Convert arrays/scientific scalar
objects explicitly and represent symbolic expressions as strings with their
comparison convention recorded separately. Report nonfinite observations with
an explicit representation, such as `{"kind": "nonfinite", "value": "Infinity"}`,
rather than a nonstandard JSON numeric literal. Unsupported objects, cycles, and
nonfinite numbers raise errors; they are not silently stringified or discarded.

The builder returns a detached snapshot and does not mutate inputs. Identities
are frozen; findings copy their JSON payloads on construction and again when
building reports. Payload dictionaries themselves are not deeply immutable.
Callers may format or serialize the returned dictionary without changing the
original findings. No timestamp, machine path, installed dependency version, or
repository state is collected automatically. With the same plan and findings,
the builder produces the same content.

## Next implementation stages

1. Add read-only adapters that compare fitted models, registries, runtime plugin
   dictionaries, and seed database entries. They must declare scope and provide
   findings under this contract; they must not apply unary registry rules to
   compound endmembers by assumption.
2. Add a thin FE driver that handles explicit paths, provenance collection, and
   output formatting. Keep FE-specific choices outside this library module.
3. Add separate thermodynamic comparisons, generation, acceptance policies, and
   promotion operations only as authorized. The report architecture implements
   none of these operations by itself.

For future integration, audit fitted-model diagnostics when only a fit exists,
audit export once registry/database inputs are available, and validate
thermodynamics after generation. Missing later-stage checks must stay explicit,
not be interpreted as successful comparisons.

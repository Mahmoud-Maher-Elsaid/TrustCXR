# EXT-4F.0 Semantic Contract Design

## Finding

The frozen `GroundedOutputEnvelope` schema constrains types, required fields,
literal/enumerated domains, lexical bounds, and object structure. It does not
fully express the executable cross-field semantics in the Pydantic validators.
The 22-entry catalog records 6 schema-enforced invariants, 1 partially
schema-enforced invariant, and 15 validator-only invariants; 15 are
cross-field rules.

The EXT-4E Candidate #3 result is the concrete proof: the output was valid JSON
under constrained decoding, but `NOT_AVAILABLE` uncertainty carried fields
forbidden by `UncertaintyState.validate_uncertainty`.

## New contract boundary

EXT-4F must introduce `EXT4F_SEMANTIC_GENERATION_CONTRACT_V1`. It must not
rename or alter `EXT4_OUTPUT_CONTRACT` v1. The existing validator remains the
authoritative compatibility check until a separately reviewed EXT-4F contract
is implemented.

The proposed pipeline is:

```text
governed structured evidence
  -> deterministic semantic planner
  -> EXT4F semantic plan validation
  -> constrained natural-language realization
  -> frozen validator / audit record
```

The planner owns evidence states, identifiers, references, provenance,
uncertainty availability, DEFER, withholding, contradiction relationships, and
the no-hallucination boundary. The LLM may only realize text already permitted
by the plan: wording, bounded ordering, concise explanation, and expert-review
questions. It cannot invent evidence, claims, states, references, provenance,
or clinical facts.

## State variants

Explicit variants such as `AvailableUncertainty` and `UnavailableUncertainty`
are recommended for EXT-4F where they reduce illegal combinations. They should
be introduced incrementally and composed by a planner rather than by taking a
cartesian product of every envelope state. A discriminated state plus
deterministic reference validation is more auditable than relying on a single
large grammar to discover existing IDs.

## Architecture comparison

| Architecture | Safety | Auditability | Decision |
| --- | --- | --- | --- |
| Stronger declarative variants | High for local state rules | High | Use as a contract representation |
| Grammar/state machine | High for token-local transitions | Medium | Use where backend supports it |
| Deterministic skeleton + realization | Highest for semantic authority | Highest | Recommended |
| Deterministic planner + LLM wording | Highest boundary control; meaningful LLM role | Highest | Recommended implementation |
| LLM then rejection | Low under failure/retry pressure | Medium | Not sufficient as primary control |

EXT-4F does not authorize model generation or benchmark design. It only defines
the next governed implementation gate.

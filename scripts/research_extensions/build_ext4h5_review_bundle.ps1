param(
  [string]$RunId = '20260820T070353Z_fc35947a',
  [string]$RunRoot = 'artifacts/research_extensions/ext4h4/20260820T070353Z_fc35947a',
  [string]$OutputRoot = 'artifacts/research_extensions/ext4h5/20260820T070353Z_fc35947a'
)
$ErrorActionPreference = 'Stop'
function Hash-Text([string]$Text) { $h=[Security.Cryptography.SHA256]::Create(); try { return (($h.ComputeHash([Text.Encoding]::UTF8.GetBytes($Text)) | ForEach-Object { $_.ToString('x2') }) -join '') } finally { $h.Dispose() } }
function Write-Utf8NoBom([string]$Path, [string]$Text) { [IO.File]::WriteAllText($Path, $Text, (New-Object Text.UTF8Encoding($false))) }
$manifest = Get-Content (Join-Path $RunRoot 'run_manifest.json') -Raw | ConvertFrom-Json
if ($manifest.run_id -ne $RunId -or $manifest.benchmark_sha256 -ne '1c34ce622fbf68af9b5114ddbf0f73fcfabffabd36dfc7536ad0c01e5402d324') { throw 'EXT4H5_H4_IDENTITY_MISMATCH' }
if ($manifest.cases_attempted -ne 24 -or $manifest.slots_attempted -ne 80 -or $manifest.model_generate_calls -ne 80) { throw 'EXT4H5_H4_AUTOMATIC_EVIDENCE_MISMATCH' }
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$dims = @('meaning_preservation','polarity_preservation','uncertainty_preservation','evidence_state_preservation','provenance_preservation','reference_fidelity','no_unsupported_addition','no_forbidden_clinical_inference','defer_fidelity','contradiction_fidelity','topic_boundary_fidelity','appropriate_limitation_expression')
$units = [System.Collections.Generic.List[object]]::new(); $map = [System.Collections.Generic.List[object]]::new(); $caseOrdinal = 0
foreach ($case in $manifest.case_ledgers) {
  $caseOrdinal++
  $blindCase = ('CASE-B{0:D3}' -f $caseOrdinal)
  foreach ($slot in $case.slot_ledgers) {
    $blindSlot = ('SLOT-B{0:D3}-{1:D2}' -f $caseOrdinal, [int]$slot.ordinal)
    $slotType = [string]$slot.slot_type; $slotId = [string]$slot.slot_id
    $app = [ordered]@{}; foreach ($d in $dims) { $app[$d] = 'APPLICABLE' }
    if (($slotId -notlike '*DEFER*') -and $slotType -ne 'DEFER_EXPLANATION') { $app.defer_fidelity = 'NOT_APPLICABLE' }
    if (($slotId -notlike '*CONTRADICTION*') -and $slotType -ne 'CONTRADICTION_EXPLANATION') { $app.contradiction_fidelity = 'NOT_APPLICABLE' }
    if ($slotType -notin @('LIMITATION_EXPLANATION','DEFER_EXPLANATION')) { $app.appropriate_limitation_expression = 'NOT_APPLICABLE' }
    $raw = Get-ChildItem $RunRoot -Filter ('{0}_{1:D2}_*_raw.txt' -f $case.case_id,[int]$slot.ordinal) | Select-Object -First 1
    if (-not $raw) { throw "EXT4H5_RAW_MISSING:$($case.case_id):$($slot.ordinal)" }
    $slotText = (Get-Content $raw.FullName -Raw | ConvertFrom-Json).slot_text
    $unit = [ordered]@{
      blind_case_id=$blindCase; blind_slot_id=$blindSlot; slot_type=$slotType
      authorized_semantic_facts=[ordered]@{ semantic_plan_sha256=$case.semantic_plan_sha256; manifest_sha256=$case.manifest_sha256; slot_id=$slotId; required_authority='The deterministic plan and manifest are authoritative; only the supplied slot meaning may be expressed.' }
      authorized_evidence_state='Plan-bound state; do not infer a negative from WITHHELD, NOT_AVAILABLE, or NOT_APPLICABLE.'
      authorized_uncertainty_state='Plan-bound uncertainty state; preserve availability and polarity.'
      authorized_provenance_reference_context='Only plan/request-bound provenance and references are authorized; no fabrication.'
      defer_state=if ($slotId -like '*DEFER*' -or $slotType -eq 'DEFER_EXPLANATION') {'ACTIVE'} else {'NOT_APPLICABLE'}
      contradiction_context=if ($slotId -like '*CONTRADICTION*' -or $slotType -eq 'CONTRADICTION_EXPLANATION') {'PRESENT'} else {'NOT_APPLICABLE'}
      authorized_limitation_topic=if ($slotType -in @('LIMITATION_EXPLANATION','DEFER_EXPLANATION')) {$slotId} else {'NOT_APPLICABLE'}
      model_authored_slot_text=$slotText; applicability=$app
    }
    $units.Add($unit)
    $map.Add([ordered]@{blind_case_id=$blindCase; blind_slot_id=$blindSlot; case_id=$case.case_id; case_sha256=$case.case_sha256; slot_id=$slotId; ordinal=[int]$slot.ordinal})
  }
}
if ($units.Count -ne 80) { throw "EXT4H5_UNIT_COUNT:$($units.Count)" }
$protocol = [ordered]@{protocol_id='EXT4H5_BLINDED_SEMANTIC_REVIEW_PROTOCOL_V1'; version='1'; primary_unit='generated_slot'; ratings=@('PASS','FAIL','NOT_APPLICABLE'); dimensions=$dims; unresolved_internal_state='UNRESOLVED'; unresolved_final_selection='FAIL'; slot_rule='all applicable dimensions PASS and none FAIL'; case_rule='all generated slots in the case PASS'; thresholds=[ordered]@{semantic_faithfulness=0.95; overall_case_pass=0.95; minimum_passing_cases=23; case_count=24}; non_generative_policy='DETERMINISTIC_NON_GENERATIVE_NOT_REVIEWED_AS_LLM_OUTPUT'; preparation_scoring='NO_AUTOMATIC_SEMANTIC_SCORING'}
$form = [ordered]@{bundle_id='EXT4H5_BLINDED_REVIEW_BUNDLE_V1'; instructions='Rate only semantic fidelity. Do not score style. Record PASS, FAIL, or NOT_APPLICABLE for each applicable dimension; unresolved is retained internally and cannot silently pass.'; fields=@('blind_case_id','blind_slot_id','ratings','adjudication_metadata')}
$summary = [ordered]@{review_status='NOT_STARTED'; reviewed_slots=0; semantic_dimension_pass_rate=$null; case_semantic_pass_rate=$null; unresolved_count=0}
$bundleCore = [ordered]@{bundle_id='EXT4H5_BLINDED_REVIEW_BUNDLE_V1'; source=[ordered]@{h4_run_id=$RunId; benchmark_id='EXT4H_FRESH_DEVELOPMENT_BENCHMARK_V1'; benchmark_sha256=$manifest.benchmark_sha256; generated_review_units=80; deterministic_non_generative_topic_count=4; deterministic_non_generative_policy='DETERMINISTIC_NON_GENERATIVE_NOT_REVIEWED_AS_LLM_OUTPUT'}; protocol=$protocol; review_units=@($units); review_form=$form; review_summary_template=$summary}
$json = $bundleCore | ConvertTo-Json -Depth 30 -Compress
$bundleSha = Hash-Text $json
$integrity = [ordered]@{bundle_id='EXT4H5_BLINDED_REVIEW_BUNDLE_V1'; protocol_id='EXT4H5_BLINDED_SEMANTIC_REVIEW_PROTOCOL_V1'; bundle_sha256=$bundleSha; review_unit_count=80; case_count=24; generated_model_outputs_in_preparation=0; candidate_identity_in_reviewer_bundle=$false}
Write-Utf8NoBom (Join-Path $OutputRoot 'review_bundle.json') ($bundleCore | ConvertTo-Json -Depth 30)
Write-Utf8NoBom (Join-Path $OutputRoot 'review_protocol.json') ($protocol | ConvertTo-Json -Depth 20)
Write-Utf8NoBom (Join-Path $OutputRoot 'review_units.json') (@($units) | ConvertTo-Json -Depth 30)
Write-Utf8NoBom (Join-Path $OutputRoot 'review_form.json') ($form | ConvertTo-Json -Depth 20)
Write-Utf8NoBom (Join-Path $OutputRoot 'review_summary_template.json') ($summary | ConvertTo-Json -Depth 20)
Write-Utf8NoBom (Join-Path $OutputRoot 'integrity_manifest.json') ($integrity | ConvertTo-Json -Depth 20)
$mapJson = @($map) | ConvertTo-Json -Depth 20 -Compress
$mapSha = Hash-Text $mapJson
[ordered]@{map_id='EXT4H5_INTERNAL_BLIND_MAP_V1'; map_sha256=$mapSha; entries=@($map)} | ConvertTo-Json -Depth 20 | ForEach-Object { Write-Utf8NoBom (Join-Path $OutputRoot 'internal_blind_map.json') $_ }
Write-Utf8NoBom (Join-Path $OutputRoot 'review_instructions.md') "# EXT-4H.5 blinded semantic review`r`n`r`nThis bundle contains 80 generated slot review units. The four reviewer-question topics are deterministic non-generative obligations and are excluded from LLM semantic scoring.`r`n`r`nRate each applicable dimension only as PASS, FAIL, or NOT_APPLICABLE. Do not score style or wording similarity. Unresolved decisions remain unresolved until adjudicated.`r`n"
Write-Output ([ordered]@{bundle_sha256=$bundleSha; blind_map_sha256=$mapSha; review_units=80; output_root=(Resolve-Path $OutputRoot).Path} | ConvertTo-Json -Compress)

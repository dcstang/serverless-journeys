# Boomerang: Synthetic NHS Patient Journeys That Check Themselves on the Way Back

*Built for the Nebius hackathon*

![Boomerang cover](assets/boomerang-cover.png)

Throw a boomerang and it doesn't just fly away — it comes back, and if
you threw it well, it comes back to your hand. That's the shape of this
project: every patient record it generates goes out as a **forward
pass** — patient, admission, journey, notes — and comes back as a
**backward pass** that checks the thing you actually asked for is in
there, fixing it if it isn't. Out and back. That's the name, and it's
also the whole design.

## The problem

Say you're building a discharge-summary classifier, or testing a
clinical NLP pipeline, or training reviewers on chart abstraction. You
need patient records that look and read like the real thing — messy
NHS demographics, admission notes with a working diagnosis, a clinical
note written the way an actual doctor writes one at 2am. What you
can't have is real patient data, for all the reasons you'd expect.

Most synthetic-data tools solve half of this. Either they hardcode a
single coding standard — good luck if your team works in SNOMED CT
instead of ICD-10 — or they generate plausible-sounding text without
checking it actually reflects the code you asked for. That second gap
is the dangerous one: you ask for `I21.0`, a STEMI (a heart attack),
and get back a discharge summary that never once mentions a heart
attack. It *looks* like data. It isn't the data you asked for.

Boomerang (formerly `serverless-journeys`) is built to close both gaps,
and to do it as a single serverless GPU job on Nebius — no standing
inference endpoint, no separate model-serving infrastructure to babysit.
One container, one job, a batch of verified patients out the other end.

## What it does

Point it at `I21.0` and ask for a handful of patients, and here's what
comes back for each one:

- **A patient** — NHS-style demographics, allergies, past medical
  history, medications, social history.
- **An admission** — specialty, ward, chief complaint, working
  diagnosis, management plan, length of stay, all consistent with a
  STEMI presentation.
- **A journey** — an ordered clinical timeline: ED review, post-take
  ward round, cath lab, therapy review, sized to a configurable target.
- **Clinical notes** — one per journey event, written in the style
  clinicians actually use for that note type: SBAR/NEWS2 bullet points
  for the ED note, ASGBI-standard prose for an operation note, formal
  consultant prose for the ward round.

Every one of those is a real LLM call, prompted with the clinical
context of `I21.0` — not a template with the diagnosis name swapped in.
That's the easy 80%. The interesting part is what happens next.

### It checks its own work

After that patient is generated, a **backward pass** asks a blunt
question: does `I21.0` actually show up anywhere in the admission
record or the notes — not just stamped on as a `diagnostic_codes`
field, but genuinely reflected in the narrative? Sometimes the LLM
drifts. It writes a technically-plausible cardiology admission that
never says "STEMI," "myocardial infarction," or anything close. Ask for
enough patients and this happens more often than you'd like.

When it does, the pipeline doesn't throw the patient away and doesn't
regenerate them from scratch. It makes a **targeted corrective call** —
rewrite just the admission's narrative fields, or just the one note
that should have mentioned it — and checks again. If it still hasn't
landed after the configured number of attempts, that gets logged and
surfaced in the run summary instead of silently shipping. You always
know exactly what you got, not just what you asked for.

### Any coding standard, no code changes

`I21.0` works because ICD-10 is a JSON file, not a code path. Diagnostic
and procedure codes are pluggable: ICD-10 and OPCS-4 ship built-in, but
adding ICD-11, SNOMED CT, CPT, or a private classification means
dropping a JSON file into `code_systems/` — nothing in the pipeline is
hardwired to ICD-10 or OPCS-4. Codes without a curated entry still work:
an optional research mode looks them up via Google Custom Search first,
so generation is grounded in what the code actually means rather than
the model guessing from the bare string.

### Runs as a single Nebius GPU job

Here's the part built specifically for Nebius. `Dockerfile.gpu` packages
a vLLM server *and* the generation pipeline into one image. On startup
the container launches vLLM locally, waits for the model to finish
loading onto the GPU, then points the pipeline at
`http://127.0.0.1:8000/v1` — Nebius's serverless endpoints and vLLM's
own server both speak the OpenAI wire format, so the same client code
just works against either. It runs the full batch and shuts vLLM down
cleanly on exit. Launch one Nebius GPU job — `gpu-l40s-a` for a 7-8B
model, an H100/multi-GPU preset for 70B-class — and you get model
serving and generation as a single ephemeral unit, with nothing left
running afterward. A second, GPU-free image exists for when you'd
rather call an already-running Nebius AI Studio endpoint (or
Anthropic/OpenAI) instead of self-hosting the model.

---

That's what it feels like from the outside: ask for a code, get back a
verified patient. Here's what actually happens between those two
moments — still following `I21.0` all the way through.

## Technical breakdown

![Pipeline overview: Phase 1 code-driven generation, Phase 2 verification and quality control](assets/Serverless_Clinical_Narrative_Pipeline_Overview.png)

### Pipeline architecture

`main.py` runs a fixed, linear pipeline per patient (`run_pipeline`):
parse codes → generate patient → generate admission → generate journey
→ generate notes → verify/correct → optionally evaluate → save. Every
LLM call in every stage goes through one dispatch function,
`processing.call_llm(prompt, model, ...)`, which routes to
`_call_anthropic`, `_call_openai`, or `_call_nebius` based on
`LLM_PROVIDER`. There's no abstract provider interface — the interface
*is* the shared function signature. Anthropic uses
`messages.create(...)`; OpenAI and Nebius both use
`chat.completions.create(...)` through the same `openai` SDK, since
Nebius's endpoint is OpenAI-wire-compatible — Nebius support is just
`_get_nebius_client()` pointing `openai.OpenAI(base_url=...)` at
`NEBIUS_BASE_URL` instead of OpenAI's default. All three paths retry
with exponential backoff and raise after `MAX_LLM_ATTEMPTS`. If a
patient throws at any stage, it's logged and skipped — the batch keeps
going rather than aborting the whole run.

### Where I21.0 actually lives

Before any of that runs, `I21.0` gets looked up. `src/codes/registry.py`
defines a generic `CodeSystem` dataclass (`key`, `name`, `kind`,
`codes`, `specialty_field`, `type_field`, `chapter_map`,
`default_specialty`), and every function that touches a code —
`parse_codes`, `lookup_code`, `infer_specialty`, `get_clinical_context`
— operates on a `CodeSystem` value, never on a hardcoded standard.
`I21.0`'s curated entry sits in `code_systems/icd10.json` as plain data:
a description, a specialty, a typical length of stay — exactly the
shape a JSON file for ICD-11 or SNOMED CT would take, which is how
`tests/test_code_registry.py` proves the point by registering a third,
entirely made-up code system and running it through the same code
paths. `src/codes/loader.py` discovers and validates every file under
`code_systems/`, plus an optional `$EXTRA_CODE_SYSTEMS_DIR` — designed
for mounting a private code-system volume into a Nebius job without
rebuilding the image. A malformed file logs a warning and is skipped
rather than crashing the run.

Worth being honest about: `code_systems/opcs4.json` currently ships
with an empty `codes` dict. An NHS classbrowser audit (commit
`7c6ddaa`) found wrong specialty mappings in the curated OPCS-4 data,
and it was pulled pending re-curation rather than shipped known-wrong.
Every OPCS-4 code currently falls through the generic/researched path.
That's the project auditing its own data quality, not a gap that
quietly slipped through.

### The backward pass, in detail

This is the part that turns "STEMI never mentioned" from a silent
failure into a caught and corrected one. `processing.check_code_
reflected(code, code_system, patient, admission, journey, notes)`
builds a "needles" set for `I21.0` — the code itself, plus significant
words (4+ letters) from its curated description, things like
"myocardial" and "infarction" — and substring-searches that set across
four JSON-serialized artifacts. The admission check deliberately
excludes bookkeeping fields (`diagnostic_codes`, `code_reflection_
check` itself) via `_ADMISSION_CODE_BOOKKEEPING_FIELDS`, so a code
can't trivially "pass" just because it's echoed back as metadata — it
has to actually appear in the narrative.

When it doesn't, here's the correction loop, in
`main.step_verify_code_reflection`:

```python
result = processing.check_code_reflected(code, code_system, patient, admission, journey, notes)
attempts = 0
while not (result["admission"] or result["notes"]) and attempts < max_correction_attempts:
    attempts += 1
    code_context = processing.get_code_context(code_system, code, enable_research=..., model=model)
    admission.update(processing.correct_admission_for_code(admission, code, code_context, model=model))
    target_note = _select_note_for_correction(notes, role)  # operation note for procedures, else earliest
    if target_note is not None:
        corrected_text = processing.correct_note_for_code(target_note["clean_note_text"], code, code_context, model=model)
        target_note["clean_note_text"] = corrected_text
        target_note["raw_blob_content"] = corrected_text
    result = processing.check_code_reflected(code, code_system, patient, admission, journey, notes)
report[f"{role}:{code}"] = result
```

Each correction is a single, targeted LLM call — rewrite the
admission's narrative fields, or rewrite one note — never a full
patient re-run. Results are keyed `role:code`, not just `code`, because
the same code string can appear as both a diagnostic and a procedure
code across different systems; an earlier version collided on that
(fixed in commit `70b2333`, alongside a single-note correction bias
where every correction landed on the same note regardless of the
code's role). A code still unreflected after `max_correction_attempts`
(default 1; `0` disables correction entirely) is logged as a warning
and surfaced in `generation_summary.json`'s `code_reflection_check.
unreflected_codes` — the run tells you exactly what didn't land instead
of quietly shipping it as if it had.

### Two Docker images, one purpose split

By the time `I21.0` has been checked and, if needed, corrected, the
question of *where* all this ran matters. `Dockerfile` is
`python:3.12-slim` plus the pipeline, for calling an already-running
endpoint — Anthropic, OpenAI, or Nebius AI Studio — no GPU needed.
`Dockerfile.gpu` is built on `vllm/vllm-openai:latest`, adds the
pipeline on top, and swaps the entrypoint for `entrypoint.sh`:

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --port "${VLLM_PORT}" \
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}" ${VLLM_EXTRA_ARGS:-} &
# poll /v1/models until ready or VLLM_STARTUP_TIMEOUT elapses,
# trap EXIT to kill vLLM on the way out
export LLM_PROVIDER=nebius
export NEBIUS_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
python3 main.py "$@"
```

Because the pipeline's Nebius provider is just the OpenAI SDK against a
configurable `base_url`, pointing it at a local vLLM server instead of
Nebius's hosted endpoint required zero pipeline code changes — only the
entrypoint script changed. That's the same OpenAI-wire-compatibility
that makes Nebius AI Studio a drop-in provider in the first place.
`tests/test_fake_endpoint.py` exercises this at the wire level: a real
local HTTP server speaking the OpenAI chat-completions format, with the
Nebius provider pointed at it, so the integration is tested without
needing real credentials in CI. CI itself
(`.github/workflows/docker-publish.yml`) builds and pushes both images
to GHCR on every push to `main`, tagged `:latest`/`:latest-gpu` plus
sha-tags, from a build matrix over the two Dockerfiles.

### What actually lands on disk

All of this — patient, admission, journey, notes, the `I21.0` check and
whatever correction it triggered — ends up in four CSVs
(`synthetic_patients`, `synthetic_admissions`, `synthetic_journeys`,
`synthetic_clinical_notes`) plus a `generation_summary.json` with run
stats, the code-reflection report, and, with `--evaluate-notes`,
readability and LLM-judged quality scores per note. There's no
pydantic model pinning the output shape — the LLM's JSON response *is*
the schema, enriched with bookkeeping IDs and serialized with pandas.
That's a deliberate trade-off: it keeps the pipeline flexible as
prompts evolve, at the cost of the output shape being defined by the
prompts rather than a strict contract. For a synthetic-data generator
whose whole value proposition is "as realistic and varied as the
underlying LLM can produce," that felt like the right side to be
loose on.

### Testing philosophy

None of the above needs a real API key to validate, which matters for
a project whose core value is a live LLM integration. 13 test files
cover forward-pass (codes actually reach the prompts), backward-pass
(the reflection check itself), full pipeline correction/reflection
wiring end-to-end with `call_llm` mocked, registry genericity, code
research/search clients, quality metrics, and the Nebius wire format
against a fake local endpoint. A `--test-mode` flag bypasses all LLM
calls with stub data for cheap pipeline smoke-testing — you can prove
the plumbing works before spending a single token on it.

---

## What's next

The most honest thing in this codebase is what's *missing*. OPCS-4
curated data was pulled after an audit found it wrong, rather than
shipped anyway — that's the immediate next job, re-curating it against
the NHS classbrowser. After that: SNOMED CT as a second built-in coding
standard, to prove the pluggable registry design beyond ICD-10/OPCS-4
in practice, not just in a test that registers a made-up system.

Which is, in a way, the same shape as everything else here. Send the
data out, check what comes back, fix what didn't land, throw it again.
That's a boomerang. That's also just good engineering.

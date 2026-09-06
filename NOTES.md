# Porting notes

Working notes for the TypeScript port of the FSRS optimizer. Everything here is
meant to survive into the standalone package: findings, deliberate divergences,
assumptions that have not been proven, and the places where the upstream test
suite turned out to be weaker than it looks.

Written in English because this file ships with the code when the module becomes
its own repository, not because the surrounding project is.

## Provenance and licence

Ported from [fsrs-rs](https://github.com/open-spaced-repetition/fsrs-rs),
Copyright (c) Open Spaced Repetition, **BSD-3-Clause**.

A line-by-line port is a derivative work, so the notice travels with it. Note
that `ts-fsrs` itself is MIT — the two licences are not interchangeable here,
and the package that comes out of this must be BSD-3-Clause (or dual-licensed
with the notice retained).

Reference revision: `fsrs` crate **6.6.2** (`Cargo.toml`, `main`, September
2026). Re-check this before claiming parity with a later release.

## Why the port exists at all

Every prebuilt optimizer was ruled out before writing a line:

| Option                                 | Why not                                                                              |
| -------------------------------------- | ------------------------------------------------------------------------------------ |
| `@open-spaced-repetition/binding` napi | React Native cannot load napi modules                                                |
| Same package, wasm flavour             | `typeof WebAssembly === "undefined"` under Hermes (RN 0.86.3) — measured on a device |
| Threadless wasm in a Cloudflare Worker | Works, but sends the review log off-device                                           |
| Any server-side optimizer              | Same objection: the review log leaves the device                                     |

The threadless build is real and worth knowing about:
`@open-spaced-repetition/binding-wasm32-wasip1` (from 0.6.0-beta) exposes a
`./workerd` entry, has no `SharedArrayBuffer`/`Worker`/`pthread` imports, and is
928 KB (335 KB gzipped). It is a valid option for anyone whose product does not
promise on-device-only processing.

Upstream has no plans for a TypeScript optimizer. From
[ts-fsrs#189](https://github.com/open-spaced-repetition/ts-fsrs/issues/189),
L-M-Sherlock: _"We don't have enough resources to implement the optimizer in
TypeScript natively..."_ — and `CONTRIBUTING.md` now codifies the split:
`packages/fsrs` is the TS scheduler, `packages/binding` is the Rust optimizer.

## File map

| fsrs-rs                           | here                    |
| --------------------------------- | ----------------------- |
| `parameter_initialization.rs`     | `initial-stability.ts`  |
| `analytic.rs`                     | `analytic.ts`           |
| `cosine_annealing.rs`             | `cosine-annealing.ts`   |
| `parameter_clipper.rs`            | `parameter-clipper.ts`  |
| `training.rs`                     | `training.ts`           |
| `training.rs::compute_parameters` | `compute-parameters.ts` |
| `dataset.rs`                      | `dataset.ts`            |
| `simulation.rs` (bounds only)     | `bounds.ts`             |
| `inference.rs` (defaults)         | `constants.ts`          |
| —                                 | `f32.ts` (see below)    |

The FSRS defaults and the stability/difficulty bounds are **copied into
`constants.ts` and `bounds.ts`**, not imported from a scheduler package. During
the port they were read from `ts-fsrs` and verified identical, then the
dependency was dropped on purpose: the optimizer has to agree with the fsrs-rs
revision it was ported from, and inheriting a scheduler's defaults would turn a
version mismatch into silent behaviour drift. The golden tests run against these
constants, which is what would catch a divergence.

## Finding: f32 versus f64 is the whole ballgame

fsrs-rs computes the forward pass and the parameters in `f32`; gradients are
`f64`. JavaScript has only `f64`. Transcribing the arithmetic literally produces
code that runs, converges, and is wrong:

```
loss, naive f64      279.92069611335165
loss, f32-rounded    279.9206961069712   ← fsrs-rs reference
```

The divergence starts in the tenth digit. No test that checks "did it train"
would catch it; it would surface months later as intervals that are slightly
off, with nothing to point at.

`f32.ts` exists for this. The rule when porting: **wherever the Rust type is
`f32`, round with `Math.fround` after every operation** — `mul32`, `div32`,
`add32`, `sub32`, `pow32`, `exp32`, `ln32`, `sqrt32`. Where Rust casts to `f64`
(`x as f64`) and stays there, plain JavaScript arithmetic is correct.

Watch for `f32` constants read as `f64`: `DEFAULT_PARAMETERS[0] as f64` is
`0.21199999749660492`, not `0.212`. Same for `S_MIN as f64`.

### `mul_add` is FMA

`w1.mul_add(-w2, w1 + w2)` rounds **once**. JavaScript has no FMA. `fma32`
emulates it as `f32(a * b + c)`, which is exact when the `f64` intermediate is
exact — the product of two `f32` values always is (24 + 24 ≤ 53 bits); only the
addition can round twice.

Verified against exact rational arithmetic for the `w1`/`w2` pair in
`smooth_and_fill`, the only place fsrs-rs calls it, where both give
`0.728600025177002`. **Not verified in general.** Any new `mul_add` call site
needs the same check.

`CosineAnnealingLR` also calls `mul_add`, but with `eta_min = 0` the addend is
zero and the operation degenerates to a multiply, so no emulation is needed
there. If `eta_min` ever becomes configurable upstream, this stops being true.

## Finding: the upstream gradient tests have holes

`analytic.rs` is tested by comparing analytic gradients against finite
differences — a good design, and stronger than golden constants, because it
validates the mathematics rather than the digits.

The port passed those tests on the first run, which was suspicious enough to
check by mutation. Flipping the sign of one gradient term at a time:

| Mutation                   | Upstream test data | With the added batch |
| -------------------------- | ------------------ | -------------------- |
| `w[9]` (success branch)    | caught             | caught               |
| `w[20]` (forgetting curve) | caught             | caught               |
| `w[19]` (same-day branch)  | **survived**       | caught               |
| `w[12]` (lapse branch)     | **survived**       | caught               |
| `w[17]` (same-day, scaled) | not tried          | caught               |

Two reasons, both in the fixture rather than in the code:

- its histories have `delta_t == 0` only at the first review, which is the
  initialisation branch, so **the same-day branch is never executed at all**;
- in the lapse branch, difficulty sits at `D_MIN = 1`, and the `w[12]` term is
  `g * raw * -ln(d)` — `ln(1) = 0` erases it regardless of sign.

A sign error in either place would ship through the upstream suite. The second
batch in `analytic.test.ts` exercises both branches and asserts that
`w[12]`, `w[17]`, `w[18]`, `w[19]` are actually reached. **Worth reporting
upstream** when the port is offered.

## Trap: sort descending, then walk backwards

`compute_outlier_analysis` and `filter_outlier` both sort sub-groups descending
by size (then by delta_t) and iterate with `.rev()`. That is **not** the same as
sorting ascending: both Rust and JavaScript sort stably, so ties come out in the
opposite order, and which sub-group gets dropped changes with them. The port
keeps the descending sort and the backwards loop.

The quota is `max(20, total / 20)` with integer division. Below about 20 items
per rating, every group looks like an outlier and the initialisation set comes
back empty — worth knowing before wondering why a small deck trains on nothing.

## Deliberate divergences

1. **Batch shuffling order.** fsrs-rs seeds ChaCha12 via `rand`
   (`StdRng::seed_from_u64`) and shuffles with `rand`'s range sampling.
   Reproducing that bit for bit means porting the cipher plus a version-pinned
   sampling algorithm. `training.ts` uses a mulberry32 PRNG instead. Same seed
   gives the same parameters here; results will not equal fsrs-rs on the same
   input. If exact parity ever matters, this is the first thing to fix.

2. **Card grouping order.** fsrs-rs groups into a `BTreeMap<i64, _>`, so cards
   arrive sorted by id. A JavaScript `Map` preserves insertion order, so the
   keys are sorted explicitly. Without that, batching depends on input order.

3. **Reused scratch buffers dropped.** `prefix_loss_and_grad` and
   `card_column_loss_and_grad` take `&mut Vec<StepCache>` (and `g_r_losses`)
   purely to avoid reallocation. The port allocates locally. No behavioural
   difference; revisit only if profiling says so.

4. **Dead binding removed.** `backward_failure` computes `let es = p - 1.0;` and
   discards it via `let _ = es;`. Not carried over.

5. **`batch_loss` is `#[cfg(test)]` upstream**, exported here. It is the
   forward-only counterpart the finite-difference test needs, and it is useful
   to a library consumer who wants to score parameters without training.

6. **Naming.** `snake_case` → `camelCase`; everything else (function boundaries,
   argument order, branch structure) deliberately mirrors the Rust so the two
   can be read side by side. Do not "clean up" the shape of these functions.

7. **`FSRS_NO_OUTLIER` is an option, not an environment variable.**
   `prepare_training_data` reads the environment to skip outlier filtering. A
   library has no business doing that, so `prepareTrainingData` takes
   `{ skipOutlierFilter }` instead.

8. **The two `prepare_training_data` variants are one function.** Rust needs a
   separate `_with_card_ids` because of its return type; here `cardIds` is a
   nullable argument and the result carries `trainCardIds: number[] | null`.

## Unverified assumptions

- **Transcendental functions.** `f32(Math.exp(x))` is assumed to equal Rust's
  `f32::exp(x)`. Computing in `f64` and rounding down can differ by 1 ulp from a
  correctly-rounded `f32` operation. Same for `ln`, `powf`, `sqrt`. This has not
  bitten any test, and gradient descent is self-correcting, but a golden test
  that fails in the last digit should suspect this first.
- **Summation order.** ndarray's `sum()` uses an unrolled fold with several
  accumulators for contiguous arrays; the port sums sequentially. For the
  5-element arrays in the reference tests the results agreed exactly. Longer
  arrays may differ in the last bits. Only affects loss comparisons, never the
  search outcome at `f32` precision.
- **`clip_parameters` on short arrays.** Rust zips parameters with the clamp
  table, so a 9-element input clips only its first 9 entries. The port matches
  this by bounding the loop with both lengths. It is only exercised by a test;
  production always passes 21.

## Additions beyond the upstream suite

- A second finite-difference batch covering the same-day and lapse branches
  (see above), with an assertion that those gradients are non-zero.
- An end-to-end `train` test on synthetic data. fsrs-rs tests this against an
  Anki collection fixture that cannot be carried over, so the port asserts the
  property the fixture is there to demonstrate: the loop lowers the loss it
  started from.
- A reproducibility test: same seed, same input, identical parameters.

## Measurements

Node on a desktop, default config (5 epochs, batch size 512). Synthetic data, so
these say something about speed and nothing about quality of fit.

`train` alone, 480 prefix items: **14 ms**, loss 334.19 → 296.43 (−11%).

Full `computeParameters`, including outlier filtering and weighting:

| Cards | Prefix items | Time   |
| ----- | ------------ | ------ |
| 300   | 2 100        | 47 ms  |
| 1 000 | 9 000        | 146 ms |
| 3 000 | 33 000       | 639 ms |

Roughly linear. Even allowing several times that on a phone, the heaviest case
stays in the low seconds — this is a button, not a background job. **Not a
device measurement.** Re-measure on hardware before quoting it anywhere.

## Still to port

- `evaluate` / `evaluate_with_time_series_splits` — optional; the binding
  refuses it on the threadless target, so it is not table stakes.
- `benchmark` — a thin `compute_parameters` that panics instead of returning a
  result. Nothing to gain from porting the panicking variant.
- `simulation.rs` — a different feature (deck simulation), not the optimizer.

Everything the optimizer itself needs is ported.

## For the standalone package

Resolved:

- ~~Decide whether to keep the `ts-fsrs` dependency~~ — inlined; the package has
  no runtime dependencies at all.
- ~~Keep BSD-3-Clause and the copyright notice~~ — `LICENSE` carries both
  copyright lines and states outright that this is a derivative work.

Still open:

- **Report the two test-coverage holes upstream**, regardless of whether the
  port itself is offered. They apply to fsrs-rs as it stands.
- **A converter helper.** Turning a card's review history into prefix items is
  six lines every consumer has to write, and getting the ordering wrong fails
  silently. fsrs-rs keeps this in its Anki converter rather than in the core, so
  the port has no equivalent. Candidate for 0.2 — it needs a name and a test,
  not a design.
- **Verify on a device.** Every timing so far is Node on a desktop.
- **Real data.** Every end-to-end test is synthetic, which shows the loop runs
  and converges but says nothing about the quality of the fit. The obvious check
  is an Anki collection through both this and `@open-spaced-repetition/binding`.
  They will not match bit for bit (see the PRNG divergence), so the comparison
  has to be on scheduling behaviour rather than on digits.

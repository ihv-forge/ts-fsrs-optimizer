# ts-fsrs-optimizer

Fits the 21 [FSRS](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm)
parameters to one person's review history, in pure TypeScript.

No native module, no WebAssembly, no server. It runs wherever JavaScript runs —
Node, browsers, Cloudflare Workers, and React Native, where the existing
optimizers cannot go.

```bash
npm install ts-fsrs-optimizer
```

## Why this exists

FSRS ships a scheduler in many languages, but only one optimizer: the Rust
implementation in [fsrs-rs](https://github.com/open-spaced-repetition/fsrs-rs),
distributed as `@open-spaced-repetition/binding` in napi and WebAssembly
flavours. Both are excellent and both are unavailable in some places:

- **React Native** cannot load napi modules, and Hermes has no `WebAssembly`
  object at all — measured on RN 0.86.3, where `typeof WebAssembly` is
  `"undefined"`.
- Anywhere the review log must not leave the device, a server-side optimizer is
  not an option regardless of how well it runs.

This package is a line-by-line port of the optimizer half of fsrs-rs, kept close
enough to the original that the two can be read side by side.

It does **not** schedule reviews. Feed the parameters it returns to
[ts-fsrs](https://github.com/open-spaced-repetition/ts-fsrs) or any other FSRS
scheduler. There is no dependency between the two packages.

## Usage

```ts
import { computeParameters } from "ts-fsrs-optimizer";

const parameters = computeParameters({ trainSet });
// → 21 numbers, ready to hand to a scheduler
```

### Shaping the input

`trainSet` is **not** a list of cards. It is a list of prediction problems: each
item is a prefix of one card's history, and the last review in it is the one the
model is asked to predict from everything before it.

So a card reviewed four times contributes three items:

```ts
import type { FsrsItem, FsrsReview } from "ts-fsrs-optimizer";

// One card. deltaT is days since the previous review; 0 means the same day.
const reviews: FsrsReview[] = [
  { rating: 3, deltaT: 0 },
  { rating: 3, deltaT: 2 },
  { rating: 1, deltaT: 5 },
  { rating: 4, deltaT: 1 },
];

const items: FsrsItem[] = [];

for (let length = 2; length <= reviews.length; length += 1) {
  items.push({ reviews: reviews.slice(0, length) });
}
```

Two rules that are easy to get wrong:

- **Order the whole set by review time.** Recent reviews are weighted far more
  heavily than old ones, and nothing in the library can detect the wrong order.
- **Ratings are 1–4** (again, hard, good, easy). Anything else is rejected.

### Grouping by card

Passing `cardIds` alongside `trainSet` lets the optimizer score a whole card in
one pass instead of replaying each prefix separately. Same answer, less work:

```ts
computeParameters({ trainSet: items, cardIds });
```

### Options

| Option               | Default | Meaning                                                                |
| -------------------- | ------- | ---------------------------------------------------------------------- |
| `cardIds`            | `null`  | Card id per item; enables windowed batching                            |
| `enableShortTerm`    | `true`  | Fit same-day parameters. `false` also freezes initial stability        |
| `numRelearningSteps` | `1`     | Relearning steps, which caps `w17 · w18`                               |
| `trainingConfig`     | —       | `numEpochs`, `batchSize`, `seed`, `learningRate`, `maxSeqLen`, `gamma` |
| `skipOutlierFilter`  | `false` | Keep anomalous intervals instead of dropping them                      |

### When it declines to train

Too little history is a result, not an error:

- fewer than 8 items — the defaults come back untouched;
- fewer than 64, or nothing beyond the initialisation set — only initial
  stability is fitted.

`InvalidInput` is thrown only for input that cannot mean anything: an empty
review list, a rating outside 1–4, `cardIds` that do not line up, or a training
config that cannot run.

## How much to trust it

A numerical port is easy to get subtly, invisibly wrong: gradient descent that
converges to the wrong place looks exactly like gradient descent that works.

- **Golden values from fsrs-rs.** Initial stability, the LR schedule, the
  parameter clipper and two Adam steps are checked against the reference
  implementation's own test expectations, bit for bit where Rust asserts
  equality.
- **Finite differences.** Every one of the 21 analytic gradients is checked
  against a numerical derivative of the loss it claims to differentiate. A
  mistranscribed derivative fails without needing any reference number.
- **Mutation-checked.** Those tests were themselves verified by deliberately
  flipping gradient signs to confirm they fail. Two mutations survived the
  upstream fixture — the port ships an extra batch that catches them (see
  [NOTES.md](./NOTES.md)).

## Known divergences from fsrs-rs

Results are **not** bit-identical to fsrs-rs, by one deliberate choice: batch
shuffling uses a small PRNG rather than a port of `rand`'s ChaCha12. The same
seed and input give the same parameters here; they will not equal the Rust
output. Everything else is transcribed faithfully.

[NOTES.md](./NOTES.md) records every divergence, the assumptions that have not
been proven, and the `f32`/`f64` trap that dominates a port like this.

## Performance

Node on a desktop, default config, synthetic data:

| Cards | Prefix items | Time   |
| ----- | ------------ | ------ |
| 300   | 2 100        | 47 ms  |
| 1 000 | 9 000        | 146 ms |
| 3 000 | 33 000       | 639 ms |

Roughly linear. Not a device measurement.

## Licence

BSD-3-Clause, inherited from fsrs-rs. Copyright (c) 2023 Open Spaced Repetition
and (c) 2026 Ihor Vashchenko. See [LICENSE](./LICENSE).

Thanks to [Jarrett Ye](https://github.com/L-M-Sherlock) and the Open Spaced
Repetition contributors, whose work this is a translation of.

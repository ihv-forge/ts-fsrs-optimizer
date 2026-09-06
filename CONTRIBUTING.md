# Contributing

Thanks for looking. This package is a port, which makes contributing to it a
little unusual: most of the rules below exist to keep it readable against the
Rust it came from.

## Getting set up

```bash
npm install
npm run verify   # typecheck, format check, tests
npm run build    # ESM + CJS + types into dist/
```

`npm test` alone runs the suite; `npm run test:watch` while you work.

No linter beyond `tsc` and Prettier. The package has no runtime dependencies and
that is a feature — please do not add one without a discussion first.

## The rules that matter

### 1. Mirror fsrs-rs

Function boundaries, argument order, branch structure and names all deliberately
follow [fsrs-rs](https://github.com/open-spaced-repetition/fsrs-rs). The only
systematic change is `snake_case` → `camelCase`.

This code is trustworthy exactly as long as someone can put it side by side with
the Rust and check it. Refactoring that improves the TypeScript but breaks that
correspondence makes the package worse, not better. If a function looks clumsy,
it probably looks clumsy in Rust too.

[PORTING.md](./PORTING.md) has the file-by-file map.

### 2. `f32` discipline

fsrs-rs computes the forward pass and the parameters in `f32`. JavaScript has
only `f64`. Every operation that is `f32` in Rust must be rounded back through
`Math.fround` — that is what `f32.ts` is for: `mul32`, `div32`, `add32`,
`sub32`, `pow32`, `exp32`, `ln32`, `sqrt32`, `fma32`.

Skipping this does not produce an obviously broken result. It produces one that
is wrong in the tenth digit, converges fine, and shows up months later as
slightly wrong intervals. Where Rust casts to `f64` and stays there, plain
JavaScript arithmetic is correct.

Read the `f32` section of [PORTING.md](./PORTING.md) before touching numerics.

### 3. Tests come from upstream where they can

Golden values are lifted from the `#[cfg(test)]` blocks of the corresponding
Rust file, so the port is checked against the reference implementation rather
than against the expectations of whoever wrote the port. Say in a comment which
Rust test a case came from.

Where a test cannot be carried over — several upstream tests read an Anki
collection fixture — write one for the property that fixture demonstrates, and
mark it as ours rather than as a port.

### 4. Prove your test can fail

Especially for gradients. Every analytic gradient here is checked against finite
differences, which sounds airtight and is not: two upstream mutations survived
the reference fixture because the branches they touch were never executed. See
the "upstream gradient tests have holes" section of
[PORTING.md](./PORTING.md).

Before trusting a new numerical test, break the code it covers on purpose and
confirm it goes red.

### 5. Record divergences

Anything that knowingly differs from fsrs-rs goes in the divergences list in
[PORTING.md](./PORTING.md), with the reason. Same for assumptions you make but
do not prove. A future reader chasing a discrepancy should find it documented
rather than have to rediscover it.

## Pull requests

- One concern per PR.
- `npm run verify` green.
- If behaviour changes, say what it means for someone whose parameters were
  fitted by the previous version.
- Commit messages in the imperative mood, describing what changes and why.

## Reporting a problem

For a wrong result, the useful report includes the input that produced it — a
review history, the options passed, and what you expected instead. "The
parameters look off" is very hard to act on; a failing case is not.

If you can reproduce a difference against `@open-spaced-repetition/binding` on
the same data, that is the most valuable bug report this project can get.

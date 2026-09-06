# Changelog

All notable changes to this package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because parameters fitted by one version can differ from another, any change
that moves them is treated as breaking for the purposes of this file, even when
the API is untouched.

## [Unreleased]

## [0.1.0] - 2026-09-06

First release. A port of the optimizer half of
[fsrs-rs](https://github.com/open-spaced-repetition/fsrs-rs) 6.6.2.

### Added

- `computeParameters` — fits the 21 FSRS parameters to a review history.
- `train`, `initializeStabilityParameters`, `clipParameters` and the dataset
  helpers, for callers who want the steps separately.
- Golden tests against the reference implementation's own expectations, and
  finite-difference tests for all 21 analytic gradients.
- [PORTING.md](./PORTING.md), recording every divergence from fsrs-rs and every
  assumption that has not been proven.

### Known limitations

- Results are not bit-identical to fsrs-rs: batch shuffling uses a different
  PRNG. Reproducible from a seed here, but not equal to the Rust output.
- Verified against synthetic data and upstream test vectors only. It has not yet
  been run against a real collection alongside
  `@open-spaced-repetition/binding`.

[unreleased]: https://github.com/ihv-forge/ts-fsrs-optimizer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ihv-forge/ts-fsrs-optimizer/releases/tag/v0.1.0

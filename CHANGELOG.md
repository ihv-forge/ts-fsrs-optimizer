# Changelog

All notable changes to this package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because parameters fitted by one version can differ from another, any change
that moves them is treated as breaking for the purposes of this file, even when
the API is untouched.

## [Unreleased]

## [0.1.2] - 2026-09-06

### Fixed

- An item whose reviews all fall on one day is now rejected with `InvalidInput`
  and a message that says what to do, instead of a bare `Error` thrown from
  inside the outlier filter. The precondition was always there — inherited from
  `first_long_term_review()` in fsrs-rs, which unwraps — but nothing said so,
  and the README's own example produced such items for any card answered twice
  within a day. That is ordinary in an app with same-day repetition, so this was
  hit on the first run against a real review log.

### Changed

- README documents the rule and builds its example items from the first review
  that reached a later day.

## [0.1.1] - 2026-09-06

No change to the published code: `dist` is identical to 0.1.0. This release
exists to prove that publishing works without a registry token, now that the
repository is configured as a trusted publisher on npm — a thing worth learning
while nothing depends on the answer.

### Changed

- Release workflow publishes through OIDC instead of an `NPM_TOKEN` secret. The
  token has been revoked and the secret deleted, so the repository now holds no
  publishing credential at all.

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

[unreleased]: https://github.com/ihv-forge/ts-fsrs-optimizer/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/ihv-forge/ts-fsrs-optimizer/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/ihv-forge/ts-fsrs-optimizer/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ihv-forge/ts-fsrs-optimizer/releases/tag/v0.1.0

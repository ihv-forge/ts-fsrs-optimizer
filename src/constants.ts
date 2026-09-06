import { f32 } from "./f32";

export const PARAMETER_COUNT = 21;

/**
 * FSRS-6 defaults, copied from fsrs-rs `src/inference.rs`.
 *
 * Deliberately not read from a scheduler package. The optimizer has to agree
 * with the reference implementation it was ported from; if a scheduler ships
 * different defaults one day, that is a mismatch worth noticing rather than
 * inheriting silently.
 *
 * Stored already rounded to `f32`, because that is how fsrs-rs stores them and
 * the difference is load-bearing — see NOTES.md.
 */
export const DEFAULT_PARAMETERS: readonly number[] = Object.freeze(
  [
    0.212, 1.2931, 2.3065, 8.2956, 6.4133, 0.8334, 3.0194, 0.001, 1.8722,
    0.1666, 0.796, 1.4835, 0.0614, 0.2629, 1.6483, 0.6014, 1.8729, 0.5425,
    0.0912, 0.0658, 0.1542,
  ].map(f32),
);

export const FSRS6_DEFAULT_DECAY = f32(0.1542);

import { f32 } from "./f32";

/** fsrs-rs `src/simulation.rs`. */
export const STABILITY_MIN = f32(0.001);
export const STABILITY_MAX = f32(36500);
export const DIFFICULTY_MIN = 1;
export const DIFFICULTY_MAX = 10;

/** fsrs-rs `src/parameter_initialization.rs`. */
export const INIT_STABILITY_MAX = 100;

export function clamp(x: number, min: number, max: number) {
  return Math.min(Math.max(x, min), max);
}

/**
 * Burn and PyTorch let gradients through values sitting exactly on a clamp
 * boundary and stop them only outside the interval. Rounding this the obvious
 * way — stopping at the boundary too — silently freezes parameters that train
 * against their limit.
 */
export function clampGrad(x: number, min: number, max: number) {
  return x >= min && x <= max ? 1 : 0;
}

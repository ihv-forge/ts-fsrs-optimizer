/**
 * fsrs-rs computes large parts of the optimizer in `f32`, and JavaScript has
 * only `f64`. Rounding each of those steps back through `Math.fround` is what
 * makes this port reproduce the reference numbers instead of merely resembling
 * them: reading the parameters as plain `f64` already moves the loss in its
 * tenth digit, which no test would notice and no user could report — it would
 * surface months later as intervals that are subtly wrong.
 */
export const f32 = Math.fround;

export function mul32(a: number, b: number) {
  return f32(a * b);
}

export function div32(a: number, b: number) {
  return f32(a / b);
}

export function sub32(a: number, b: number) {
  return f32(a - b);
}

export function pow32(base: number, exponent: number) {
  return f32(base ** exponent);
}

/**
 * Rust's `mul_add` rounds once, which JavaScript cannot express. The `f64`
 * product of two `f32` values is exact, so rounding the sum back to `f32`
 * lands on the same number — verified against exact rational arithmetic for
 * the `w1`/`w2` pair, the only place fsrs-rs calls it.
 */
export function fma32(a: number, b: number, c: number) {
  return f32(a * b + c);
}

export function add32(a: number, b: number) {
  return f32(a + b);
}

export function exp32(x: number) {
  return f32(Math.exp(x));
}

export function ln32(x: number) {
  return f32(Math.log(x));
}

export function sqrt32(x: number) {
  return f32(Math.sqrt(x));
}

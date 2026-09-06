import {
  DIFFICULTY_MAX,
  DIFFICULTY_MIN,
  INIT_STABILITY_MAX,
  STABILITY_MIN,
} from "./bounds";
import { add32, div32, f32, ln32, mul32, pow32, sqrt32, sub32 } from "./f32";

/**
 * Post-loss stability after `numRelearningSteps` relearning steps must not
 * exceed the stability it started from, and that bound is what caps w17·w18.
 * With `D = 1`, `R = 0.7` and `S = 1` it reduces to
 * `w17 * w18 <= -[ln(w11) + ln(2^w13 - 1) + w14 * 0.3] / numRelearningSteps`.
 */
function shortTermCeiling(parameters: number[], numRelearningSteps: number) {
  if (numRelearningSteps <= 1) {
    return 2;
  }

  const bound = add32(
    add32(ln32(parameters[11]), ln32(sub32(pow32(2, parameters[13]), 1))),
    mul32(parameters[14], f32(0.3)),
  );

  return Math.min(
    sqrt32(Math.max(div32(-bound, numRelearningSteps), f32(0.01))),
    2,
  );
}

export function clipParametersInPlace(
  parameters: number[],
  numRelearningSteps: number,
  enableShortTerm: boolean,
) {
  const ceiling = shortTermCeiling(parameters, numRelearningSteps);
  const w19Floor = enableShortTerm ? f32(0.01) : 0;

  const clamps: [number, number][] = [
    [STABILITY_MIN, INIT_STABILITY_MAX],
    [STABILITY_MIN, INIT_STABILITY_MAX],
    [STABILITY_MIN, INIT_STABILITY_MAX],
    [STABILITY_MIN, INIT_STABILITY_MAX],
    [DIFFICULTY_MIN, DIFFICULTY_MAX],
    [f32(0.001), 4],
    [f32(0.001), 4],
    [f32(0.001), f32(0.75)],
    [0, f32(4.5)],
    [0, f32(0.8)],
    [f32(0.001), f32(3.5)],
    [f32(0.001), 5],
    [f32(0.001), f32(0.25)],
    [f32(0.001), f32(0.9)],
    [0, 4],
    [0, 1],
    [1, 6],
    [0, ceiling],
    [0, ceiling],
    [w19Floor, f32(0.8)],
    [f32(0.1), f32(0.8)],
  ];

  for (let i = 0; i < parameters.length && i < clamps.length; i += 1) {
    const [low, high] = clamps[i];

    parameters[i] = Math.min(Math.max(parameters[i], low), high);
  }
}

export function clipParameters(
  parameters: number[],
  numRelearningSteps: number,
  enableShortTerm: boolean,
) {
  const clipped = [...parameters];

  clipParametersInPlace(clipped, numRelearningSteps, enableShortTerm);

  return clipped;
}

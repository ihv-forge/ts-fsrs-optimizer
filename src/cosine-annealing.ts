/**
 * PyTorch's CosineAnnealingLR, in its recursive form: each rate is derived from
 * the previous one rather than from the step count alone. The closed form would
 * be simpler but would not reproduce the same sequence, and the parameters that
 * come out of training depend on it.
 */
export function cosineAnnealingLR(tMax: number, initialLr: number) {
  const etaMin = 0;

  let stepCount = -1;
  let currentLr = initialLr;

  return function step() {
    stepCount += 1;

    if (stepCount === 0) {
      currentLr = initialLr;
    } else if ((stepCount - 1 - tMax) % (2 * tMax) === 0) {
      currentLr = ((initialLr - etaMin) * (1 - Math.cos(Math.PI / tMax))) / 2;
    } else {
      const ratio =
        (1 + Math.cos((Math.PI * stepCount) / tMax)) /
        (1 + Math.cos((Math.PI * (stepCount - 1)) / tMax));

      currentLr = ratio * (currentLr - etaMin) + etaMin;
    }

    return currentLr;
  };
}

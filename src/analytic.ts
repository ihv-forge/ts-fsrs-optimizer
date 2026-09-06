import {
  clamp,
  clampGrad,
  DIFFICULTY_MAX,
  DIFFICULTY_MIN,
  STABILITY_MAX,
  STABILITY_MIN,
} from "./bounds";
import { add32, div32, exp32, f32, ln32, mul32, pow32, sub32 } from "./f32";

const MIN_PROB = f32(1e-7);
const MAX_PROB = f32(1 - f32(1e-7));

type State = {
  s: number;
  d: number;
};

type CurveCache = {
  t: number;
  s: number;
  decay: number;
  factor: number;
  base: number;
  retrievability: number;
};

type StepCache = {
  stateS: number;
  stateD: number;
  lastS: number;
  lastD: number;
  deltaT: number;
  rating: number;
  retrievability: number;
  failureRaw: number;
  failureFloor: number;
  failureUsedFloor: boolean;
  shortRaw: number;
  shortValue: number;
  shortRawActive: boolean;
  useShort: boolean;
  useFailure: boolean;
  initSelected: boolean;
  padding: boolean;
  preClampS: number;
  meanPreClampD: number;
  initRating: number;
};

type RuntimeParams = {
  w: number[];
  curveDecay: number;
  curveFactor: number;
  curveDFactorDDecay: number;
  expW8: number;
  expW8F64: number;
  failureFloorDivisor: number;
  easyD: number;
  easyDF64: number;
  exp3W5F64: number;
};

function runtimeParams(w: number[]): RuntimeParams {
  const curveDecay = -w[20];
  const decay = curveDecay;
  const c = Math.log(0.9);
  const expTerm = Math.exp(c / decay);
  const easyD = initDifficulty(w, 4);

  return {
    w,
    curveDecay,
    curveFactor: sub32(exp32(div32(ln32(0.9), curveDecay)), 1),
    curveDFactorDDecay: expTerm * (-c / (decay * decay)),
    expW8: exp32(w[8]),
    expW8F64: Math.exp(w[8]),
    failureFloorDivisor: exp32(mul32(w[17], w[18])),
    easyD,
    easyDF64: easyD,
    exp3W5F64: Math.exp(3 * w[5]),
  };
}

function add(gw: number[], index: number, value: number) {
  if (index < gw.length) {
    gw[index] += value;
  }
}

/**
 * Weighted binary cross entropy and its derivative with respect to
 * retrievability. Outside the clamp interval the clamp counts as active and the
 * gradient is zero.
 */
function bceLossAndGradR(
  rawR: number,
  label: number,
  weight: number,
): [number, number] {
  if (weight === 0) {
    return [0, 0];
  }

  const r = clamp(rawR, MIN_PROB, MAX_PROB);
  const loss = mul32(
    -add32(mul32(label, ln32(r)), mul32(sub32(1, label), ln32(sub32(1, r)))),
    weight,
  );
  const grad = mul32(
    -weight,
    sub32(div32(label, r), div32(sub32(1, label), sub32(1, r))),
  );

  return [loss, rawR > MIN_PROB && rawR < MAX_PROB ? grad : 0];
}

/**
 * The FSRS power forgetting curve, keeping the intermediates the backward pass
 * needs. `factor` is chosen so that `R(s, s) = 0.9`, which is what makes
 * stability mean "days until 90% recall".
 */
function curveForward(p: RuntimeParams, rawT: number, s: number): CurveCache {
  const t = Math.max(rawT, 0);
  const base = add32(mul32(div32(t, s), p.curveFactor), 1);

  return {
    t,
    s,
    decay: p.curveDecay,
    factor: p.curveFactor,
    base,
    retrievability: pow32(base, p.curveDecay),
  };
}

/**
 * Accumulates `dL/dw[20]` and returns `dL/ds`. The forward curve uses
 * `decay = -w[20]`, so the parameter derivative is the negated decay one.
 */
function curveBackward(
  p: RuntimeParams,
  cache: CurveCache,
  gR: number,
  gw: number[],
) {
  if (gR === 0) {
    return 0;
  }

  const { retrievability: r, base, decay, t, s, factor } = cache;

  const dbDs = (-t * factor) / (s * s);
  const gS = ((gR * r * decay) / base) * dbDs;

  const dbDdecay = (t / s) * p.curveDFactorDDecay;
  const drDdecay = r * (Math.log(base) + (decay / base) * dbDdecay);

  add(gw, 20, -gR * drDdecay);

  return gS;
}

function initDifficulty(w: number[], rating: number) {
  const offset = Math.max(rating - 1, 0);

  return add32(sub32(w[4], exp32(mul32(w[5], offset))), 1);
}

/**
 * One scalar FSRS state transition, recording every branch decision so the
 * backward pass can follow the same non-smooth path rather than a smoothed
 * approximation of it.
 */
function stepForward(
  p: RuntimeParams,
  state: State,
  deltaT: number,
  rating: number,
  nth: number,
): [State, StepCache] {
  const w = p.w;
  const lastS = clamp(state.s, STABILITY_MIN, STABILITY_MAX);
  const lastD = clamp(state.d, DIFFICULTY_MIN, DIFFICULTY_MAX);

  const blank = {
    stateS: state.s,
    stateD: state.d,
    lastS,
    lastD,
    deltaT,
    rating,
    retrievability: 0,
    failureRaw: 0,
    failureFloor: 0,
    failureUsedFloor: false,
    shortRaw: 0,
    shortValue: 0,
    shortRawActive: false,
    useShort: false,
    useFailure: false,
    initSelected: false,
    padding: false,
  };

  if (rating === 0) {
    return [
      { s: lastS, d: lastD },
      {
        ...blank,
        padding: true,
        preClampS: lastS,
        meanPreClampD: lastD,
        initRating: 1,
      },
    ];
  }

  const initSelected = nth === 0 && state.s === 0;
  const initRating = Math.trunc(clamp(rating, 1, 4));

  if (initSelected) {
    const newS = w[initRating - 1];
    const rawD = initDifficulty(w, initRating);

    return [
      {
        s: clamp(newS, STABILITY_MIN, STABILITY_MAX),
        d: clamp(rawD, DIFFICULTY_MIN, DIFFICULTY_MAX),
      },
      {
        ...blank,
        initSelected: true,
        preClampS: newS,
        meanPreClampD: rawD,
        initRating,
      },
    ];
  }

  const curve = curveForward(p, deltaT, lastS);
  const r = curve.retrievability;

  const useShort = deltaT === 0;
  const useFailure = rating === 1;

  let failureRaw = 0;
  let failureFloor = 0;
  let failureUsedFloor = false;
  let shortRaw = 0;
  let shortValue = 0;
  let shortRawActive = false;
  let newS: number;

  if (useShort) {
    shortRaw = mul32(
      exp32(mul32(w[17], add32(sub32(rating, 3), w[18]))),
      pow32(lastS, -w[19]),
    );
    shortRawActive = !(rating >= 2 && shortRaw < 1);
    shortValue = rating >= 2 ? Math.max(shortRaw, 1) : shortRaw;
    newS = mul32(lastS, shortValue);
  } else if (useFailure) {
    failureRaw = mul32(
      mul32(
        mul32(w[11], pow32(lastD, -w[12])),
        sub32(pow32(add32(lastS, 1), w[13]), 1),
      ),
      exp32(mul32(sub32(1, r), w[14])),
    );
    failureFloor = div32(lastS, p.failureFloorDivisor);
    failureUsedFloor = failureFloor < failureRaw;
    newS = failureUsedFloor ? failureFloor : failureRaw;
  } else {
    const hardPenalty = rating === 2 ? w[15] : 1;
    const easyBonus = rating === 4 ? w[16] : 1;
    const successInc = mul32(
      mul32(
        mul32(
          mul32(mul32(p.expW8, sub32(11, lastD)), pow32(lastS, -w[9])),
          sub32(exp32(mul32(sub32(1, r), w[10])), 1),
        ),
        hardPenalty,
      ),
      easyBonus,
    );

    newS = mul32(lastS, add32(successInc, 1));
  }

  const deltaD = mul32(-w[6], sub32(rating, 3));
  const nextD = add32(lastD, div32(mul32(sub32(10, lastD), deltaD), 9));
  const meanD = add32(mul32(w[7], sub32(p.easyD, nextD)), nextD);

  return [
    {
      s: clamp(newS, STABILITY_MIN, STABILITY_MAX),
      d: clamp(meanD, DIFFICULTY_MIN, DIFFICULTY_MAX),
    },
    {
      ...blank,
      retrievability: r,
      failureRaw,
      failureFloor,
      failureUsedFloor,
      shortRaw,
      shortValue,
      shortRawActive,
      useShort,
      useFailure,
      initSelected,
      preClampS: newS,
      meanPreClampD: meanD,
      initRating,
    },
  ];
}

/** Differentiates `s' = s * (1 + inc)`, the long-term success branch. */
function backwardSuccess(
  p: RuntimeParams,
  c: StepCache,
  g: number,
  gw: number[],
): [number, number, number] {
  if (g === 0) {
    return [0, 0, 0];
  }

  const w = p.w;
  const s = c.lastS;
  const d = c.lastD;
  const r = c.retrievability;
  const a = p.expW8F64;
  const b = 11 - d;
  const cS = s ** -w[9];
  const e = Math.exp((1 - r) * w[10]) - 1;
  const expE = e + 1;
  const hp = c.rating === 2 ? w[15] : 1;
  const eb = c.rating === 4 ? w[16] : 1;
  const inc = a * b * cS * e * hp * eb;
  const gInc = g * s;

  const gS = g * (inc + 1) + gInc * inc * (-w[9] / s);
  const gD = -(gInc * a * cS * e * hp * eb);
  const gR = gInc * a * b * cS * hp * eb * (-w[10] * expE);

  add(gw, 8, gInc * inc);
  add(gw, 9, gInc * inc * -Math.log(s));
  add(gw, 10, gInc * a * b * cS * hp * eb * ((1 - r) * expE));

  if (c.rating === 2) {
    add(gw, 15, gInc * a * b * cS * e * eb);
  }

  if (c.rating === 4) {
    add(gw, 16, gInc * a * b * cS * e * hp);
  }

  return [gS, gD, gR];
}

/**
 * Differentiates the lapse branch. Only the branch `stepForward` actually took
 * is followed, so the gradient at the floor boundary is one-sided on purpose.
 */
function backwardFailure(
  p: RuntimeParams,
  c: StepCache,
  g: number,
  gw: number[],
): [number, number, number] {
  if (g === 0) {
    return [0, 0, 0];
  }

  const w = p.w;
  const s = c.lastS;
  const d = c.lastD;
  const r = c.retrievability;

  if (c.failureUsedFloor) {
    const floor = c.failureFloor;

    add(gw, 17, g * floor * -w[18]);
    add(gw, 18, g * floor * -w[17]);

    return [g * (floor / s), 0, 0];
  }

  const raw = c.failureRaw;
  const base = s + 1;
  const power = base ** w[13];
  const lastDPow = d ** -w[12];
  const er = Math.exp((1 - r) * w[14]);

  add(gw, 11, (g * raw) / w[11]);
  add(gw, 12, g * raw * -Math.log(d));
  add(gw, 13, g * w[11] * lastDPow * er * power * Math.log(base));
  add(gw, 14, g * raw * (1 - r));

  return [
    (g * w[11] * lastDPow * er * w[13] * power) / base,
    g * raw * (-w[12] / d),
    g * raw * -w[14],
  ];
}

/**
 * Differentiates the same-day branch. When the floor at 1 is active only the
 * direct derivative of `s' = s * short_value` with respect to `s` survives.
 */
function backwardShort(
  p: RuntimeParams,
  c: StepCache,
  g: number,
  gw: number[],
) {
  if (g === 0) {
    return 0;
  }

  const w = p.w;
  const s = c.lastS;

  let gS = g * c.shortValue;

  if (c.shortRawActive) {
    const gRaw = g * s;
    const raw = c.shortRaw;
    const q = c.rating - 3 + w[18];

    add(gw, 17, gRaw * raw * q);
    add(gw, 18, gRaw * raw * w[17]);
    add(gw, 19, gRaw * raw * -Math.log(s));

    gS += gRaw * raw * (-w[19] / s);
  }

  return gS;
}

/** Differentiates the linear damping plus mean reversion of difficulty. */
function backwardDifficulty(
  p: RuntimeParams,
  c: StepCache,
  gOut: number,
  gw: number[],
) {
  if (gOut === 0) {
    return 0;
  }

  const w = p.w;
  const gMean =
    gOut * clampGrad(c.meanPreClampD, DIFFICULTY_MIN, DIFFICULTY_MAX);

  if (gMean === 0) {
    return 0;
  }

  const ratingMinus3 = c.rating - 3;
  const lastD = c.lastD;
  const deltaD = -w[6] * ratingMinus3;
  const nextD = lastD + ((10 - lastD) * deltaD) / 9;

  add(gw, 7, gMean * (p.easyDF64 - nextD));
  add(gw, 4, gMean * w[7]);
  add(gw, 5, gMean * w[7] * -3 * p.exp3W5F64);

  const gNext = gMean * (1 - w[7]);

  add(gw, 6, (gNext * (10 - lastD) * -ratingMinus3) / 9);

  return gNext * (1 - deltaD / 9);
}

/** Differentiates the first-review initialisation into `w[0..4]`, `w[4]`, `w[5]`. */
function backwardInit(
  p: RuntimeParams,
  rating: number,
  gS: number,
  gD: number,
  gw: number[],
) {
  const w = p.w;

  add(gw, rating - 1, gS);

  const rawD = initDifficulty(w, rating);
  const gRawD = gD * clampGrad(rawD, DIFFICULTY_MIN, DIFFICULTY_MAX);

  if (gRawD !== 0) {
    const offset = rating - 1;

    add(gw, 4, gRawD);
    add(gw, 5, gRawD * -offset * Math.exp(offset * w[5]));
  }
}

/**
 * Backpropagates one cached transition. `gRExtra` is retrievability gradient
 * arriving directly at this step, which per-card losses use to score every
 * historical review rather than only the last one.
 */
function stepBackward(
  p: RuntimeParams,
  c: StepCache,
  gOutS: number,
  gOutD: number,
  gRExtra: number,
  gw: number[],
): [number, number] {
  const gPreS = gOutS * clampGrad(c.preClampS, STABILITY_MIN, STABILITY_MAX);

  let gLastS = 0;
  let gLastD = 0;
  let gR = gRExtra;

  if (c.padding) {
    gLastS += gPreS;
    gLastD += gOutD;
  } else if (c.initSelected) {
    backwardInit(p, c.initRating, gPreS, gOutD, gw);
  } else {
    if (c.useShort) {
      gLastS += backwardShort(p, c, gPreS, gw);
    } else {
      const [gs, gd, gr] = c.useFailure
        ? backwardFailure(p, c, gPreS, gw)
        : backwardSuccess(p, c, gPreS, gw);

      gLastS += gs;
      gLastD += gd;
      gR += gr;
    }

    gLastD += backwardDifficulty(p, c, gOutD, gw);
  }

  const t = Math.max(c.deltaT, 0);

  gLastS += curveBackward(
    p,
    {
      t,
      s: c.lastS,
      decay: p.curveDecay,
      factor: p.curveFactor,
      base: add32(mul32(div32(t, c.lastS), p.curveFactor), 1),
      retrievability: c.retrievability,
    },
    gR,
    gw,
  );

  return [
    gLastS * clampGrad(c.stateS, STABILITY_MIN, STABILITY_MAX),
    gLastD * clampGrad(c.stateD, DIFFICULTY_MIN, DIFFICULTY_MAX),
  ];
}

/**
 * Scores only the review following a fixed history prefix. Histories are
 * time-major: `idx = time * batch + column`.
 */
function prefixLossAndGrad(
  p: RuntimeParams,
  tHist: number[],
  rHist: number[],
  seqLen: number,
  batch: number,
  column: number,
  deltaT: number,
  label: number,
  weight: number,
  gw: number[] | null,
) {
  const caches: StepCache[] = [];

  let state: State = { s: 0, d: 0 };

  for (let t = 0; t < seqLen; t += 1) {
    const index = t * batch + column;
    const [next, cache] = stepForward(p, state, tHist[index], rHist[index], t);

    state = next;

    if (gw) {
      caches.push(cache);
    }
  }

  const curve = curveForward(p, deltaT, state.s);
  const [loss, gR] = bceLossAndGradR(curve.retrievability, label, weight);

  if (gw) {
    let gS = curveBackward(p, curve, gR, gw);
    let gD = 0;

    for (let t = caches.length - 1; t >= 0; t -= 1) {
      [gS, gD] = stepBackward(p, caches[t], gS, gD, 0, gw);
    }
  }

  return loss;
}

/**
 * Summed next-review loss over a batch of prefixes. `gw` accumulates the 21
 * parameter gradients and is never cleared here.
 */
export function batchLossAndGrad(
  w: number[],
  tHist: number[],
  rHist: number[],
  seqLen: number,
  batch: number,
  seqLens: number[],
  deltaTs: number[],
  labels: number[],
  weights: number[],
  gw: number[],
) {
  const p = runtimeParams(w);

  let loss = 0;

  for (let column = 0; column < batch; column += 1) {
    loss += prefixLossAndGrad(
      p,
      tHist,
      rHist,
      Math.min(seqLens[column], seqLen),
      batch,
      column,
      deltaTs[column],
      labels[column],
      weights[column],
      gw,
    );
  }

  return loss;
}

/** The forward-only counterpart of {@link batchLossAndGrad}. */
export function batchLoss(
  w: number[],
  tHist: number[],
  rHist: number[],
  seqLen: number,
  batch: number,
  seqLens: number[],
  deltaTs: number[],
  labels: number[],
  weights: number[],
) {
  const p = runtimeParams(w);

  let loss = 0;

  for (let column = 0; column < batch; column += 1) {
    loss += prefixLossAndGrad(
      p,
      tHist,
      rHist,
      Math.min(seqLens[column], seqLen),
      batch,
      column,
      deltaTs[column],
      labels[column],
      weights[column],
      null,
    );
  }

  return loss;
}

/**
 * Scores every review of one card after the first, predicting each from the
 * state that preceded it.
 */
function cardColumnLossAndGrad(
  p: RuntimeParams,
  tHist: number[],
  rHist: number[],
  seqLen: number,
  batch: number,
  column: number,
  labels: number[],
  weights: number[],
  gw: number[] | null,
) {
  const caches: StepCache[] = [];
  const gRLosses: number[] = gw ? new Array<number>(seqLen).fill(0) : [];

  let state: State = { s: 0, d: 0 };
  let loss = 0;

  for (let t = 0; t < seqLen; t += 1) {
    const index = t * batch + column;

    if (t + 1 === seqLen) {
      let finalGS = 0;

      if (t !== 0 && weights[index] !== 0) {
        const curve = curveForward(
          p,
          tHist[index],
          clamp(state.s, STABILITY_MIN, STABILITY_MAX),
        );
        const [stepLoss, gR] = bceLossAndGradR(
          curve.retrievability,
          labels[index],
          weights[index],
        );

        loss += stepLoss;

        if (gw) {
          finalGS =
            curveBackward(p, curve, gR, gw) *
            clampGrad(state.s, STABILITY_MIN, STABILITY_MAX);
        }
      }

      if (gw) {
        let gS = finalGS;
        let gD = 0;

        for (let step = caches.length - 1; step >= 0; step -= 1) {
          [gS, gD] = stepBackward(p, caches[step], gS, gD, gRLosses[step], gw);
        }
      }

      return loss;
    }

    const [next, cache] = stepForward(p, state, tHist[index], rHist[index], t);
    const [stepLoss, gR] =
      t === 0
        ? [0, 0]
        : bceLossAndGradR(cache.retrievability, labels[index], weights[index]);

    loss += stepLoss;

    if (gw) {
      gRLosses[t] = gR;
      caches.push(cache);
    }

    state = next;
  }

  return loss;
}

/**
 * Summed in-card review loss over a batch of full card trajectories. Entries
 * for the first review are ignored: there is no prior state to score it from.
 */
export function cardLossAndGrad(
  w: number[],
  tHist: number[],
  rHist: number[],
  seqLen: number,
  batch: number,
  seqLens: number[],
  labels: number[],
  weights: number[],
  gw: number[],
) {
  const p = runtimeParams(w);

  let loss = 0;

  for (let column = 0; column < batch; column += 1) {
    loss += cardColumnLossAndGrad(
      p,
      tHist,
      rHist,
      Math.min(seqLens[column], seqLen),
      batch,
      column,
      labels,
      weights,
      gw,
    );
  }

  return loss;
}

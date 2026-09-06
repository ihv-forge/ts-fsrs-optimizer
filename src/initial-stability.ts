import { INIT_STABILITY_MAX, STABILITY_MIN } from "./bounds";
import { DEFAULT_PARAMETERS } from "./constants";
import { div32, f32, fma32, mul32, pow32, sub32 } from "./f32";
import {
  firstLongTermReview,
  longTermReviewCount,
  type FsrsItem,
  type FsrsRating,
} from "./item";

const RATINGS: FsrsRating[] = [1, 2, 3, 4];

const DEFAULT_STABILITY: Record<FsrsRating, number> = {
  1: DEFAULT_PARAMETERS[0],
  2: DEFAULT_PARAMETERS[1],
  3: DEFAULT_PARAMETERS[2],
  4: DEFAULT_PARAMETERS[3],
};

const DECAY = -DEFAULT_PARAMETERS[20];

const MAX_SEARCH_STEPS = 1000;

const W1 = f32(0.41);
const W2 = f32(0.54);

/** The average pass rate and its sample size for one delta_t under one rating. */
type AverageRecall = {
  deltaT: number;
  recall: number;
  count: number;
};

export type InitialStability = [number, number, number, number];

export class NotEnoughData extends Error {
  constructor() {
    super("fsrs-optimizer: not enough reviews to initialise stability");
    this.name = "NotEnoughData";
  }
}

export function initializeStabilityParameters(
  items: FsrsItem[],
  averageRecall: number,
) {
  const dataset = prepareDataset(items);
  const ratingCount = totalRatingCount(dataset);
  const stability = searchParameters(dataset, averageRecall);

  return { stability: smoothAndFill(stability, ratingCount), ratingCount };
}

export function powerForgettingCurve(deltaT: number[], stability: number) {
  const factor = 0.9 ** (1 / DECAY) - 1;

  return deltaT.map((t) => ((t / stability) * factor + 1) ** DECAY);
}

export function loss(
  deltaT: number[],
  recall: number[],
  count: number[],
  initialS0: number,
  defaultS0: number,
) {
  const predicted = powerForgettingCurve(deltaT, initialS0);

  let logLoss = 0;

  for (let index = 0; index < deltaT.length; index += 1) {
    const observed = recall[index];
    const p = predicted[index];

    logLoss +=
      -(observed * Math.log(p) + (1 - observed) * Math.log(1 - p)) *
      count[index];
  }

  return logLoss + Math.abs(initialS0 - defaultS0) / 16;
}

function prepareDataset(items: FsrsItem[]) {
  const groups = new Map<FsrsRating, Map<number, number[]>>();

  for (const item of items) {
    if (longTermReviewCount(item) !== 1) {
      continue;
    }

    const firstRating = item.reviews[0].rating;
    const review = firstLongTermReview(item);
    const byDelta = groups.get(firstRating) ?? new Map<number, number[]>();
    const labels = byDelta.get(review.deltaT) ?? [];

    labels.push(review.rating > 1 ? 1 : 0);
    byDelta.set(review.deltaT, labels);
    groups.set(firstRating, byDelta);
  }

  const dataset = new Map<FsrsRating, AverageRecall[]>();

  for (const [firstRating, byDelta] of groups) {
    const data = [...byDelta].map(([deltaT, labels]) => ({
      deltaT,
      recall: labels.reduce((total, label) => total + label, 0) / labels.length,
      count: labels.length,
    }));

    data.sort((a, b) => a.deltaT - b.deltaT);
    dataset.set(firstRating, data);
  }

  return dataset;
}

function totalRatingCount(dataset: Map<FsrsRating, AverageRecall[]>) {
  const counts = new Map<FsrsRating, number>();

  for (const [firstRating, data] of dataset) {
    counts.set(
      firstRating,
      Math.trunc(data.reduce((total, entry) => total + entry.count, 0)),
    );
  }

  return counts;
}

export function searchParameters(
  dataset: Map<FsrsRating, AverageRecall[]>,
  averageRecall: number,
) {
  const optimal = new Map<FsrsRating, number>();
  const prior = f32(averageRecall);

  for (const [firstRating, data] of dataset) {
    const defaultS0 = DEFAULT_STABILITY[firstRating];
    const deltaT = data.map((entry) => entry.deltaT);
    const count = data.map((entry) => entry.count);
    // Laplace smoothing: (real_recall * n + average_recall) / (n + 1)
    const recall = data.map(
      (entry) => (entry.recall * entry.count + prior) / (entry.count + 1),
    );

    let low = STABILITY_MIN;
    let high = INIT_STABILITY_MAX;
    let best = defaultS0;

    for (
      let step = 0;
      step < MAX_SEARCH_STEPS && high - low > Number.EPSILON;
      step += 1
    ) {
      const lower = low + (high - low) / 3;
      const upper = high - (high - low) / 3;

      if (
        loss(deltaT, recall, count, lower, defaultS0) <
        loss(deltaT, recall, count, upper, defaultS0)
      ) {
        high = upper;
      } else {
        low = lower;
      }

      best = (high + low) / 2;
    }

    optimal.set(firstRating, f32(best));
  }

  return optimal;
}

const MONOTONIC_PAIRS: [FsrsRating, FsrsRating][] = [
  [1, 2],
  [2, 3],
  [3, 4],
  [1, 3],
  [2, 4],
  [1, 4],
];

export function smoothAndFill(
  ratingStability: Map<FsrsRating, number>,
  ratingCount: Map<FsrsRating, number>,
): InitialStability {
  const known = new Map(ratingStability);

  for (const rating of [...known.keys()]) {
    if (!ratingCount.has(rating)) {
      known.delete(rating);
    }
  }

  for (const [small, big] of MONOTONIC_PAIRS) {
    const smallValue = known.get(small);
    const bigValue = known.get(big);

    if (smallValue === undefined || bigValue === undefined) {
      continue;
    }

    if (smallValue > bigValue) {
      if ((ratingCount.get(small) ?? 0) > (ratingCount.get(big) ?? 0)) {
        known.set(big, smallValue);
      } else {
        known.set(small, bigValue);
      }
    }
  }

  if (known.size === 0) {
    throw new NotEnoughData();
  }

  if (known.size === 1) {
    const [rating] = [...known.keys()];
    const factor = div32(known.get(rating) ?? 0, DEFAULT_STABILITY[rating]);

    return clampToFour(
      RATINGS.map((entry) => mul32(DEFAULT_STABILITY[entry], factor)).sort(
        (a, b) => a - b,
      ),
    );
  }

  const filled = RATINGS.map((rating) => known.get(rating) ?? null);

  if (known.size === 2) {
    fillFromTwo(filled);
  }

  if (known.size === 3) {
    fillFromThree(filled);
  }

  return clampToFour(filled.filter((value) => value !== null));
}

function blend(base: number, baseExp: number, other: number, otherExp: number) {
  return mul32(pow32(base, baseExp), pow32(other, otherExp));
}

function fillFromTwo(filled: (number | null)[]) {
  const [s1, s2, s3, s4] = filled;
  const spread = fma32(W1, -W2, f32(W1 + W2));

  if (s1 === null && s2 === null && s3 !== null && s4 !== null) {
    const r2 = blend(
      s3,
      div32(1, sub32(1, W2)),
      s4,
      sub32(1, div32(1, sub32(1, W2))),
    );

    filled[1] = r2;
    filled[0] = blend(r2, div32(1, W1), s3, sub32(1, div32(1, W1)));
  } else if (s1 === null && s2 !== null && s3 === null && s4 !== null) {
    const r3 = blend(s2, sub32(1, W2), s4, W2);

    filled[2] = r3;
    filled[0] = blend(s2, div32(1, W1), r3, sub32(1, div32(1, W1)));
  } else if (s1 === null && s2 !== null && s3 !== null && s4 === null) {
    filled[3] = blend(s2, sub32(1, div32(1, W2)), s3, div32(1, W2));
    filled[0] = blend(s2, div32(1, W1), s3, sub32(1, div32(1, W1)));
  } else if (s1 !== null && s2 === null && s3 === null && s4 !== null) {
    filled[1] = blend(s1, div32(W1, spread), s4, sub32(1, div32(W1, spread)));
    filled[2] = blend(s1, sub32(1, div32(W2, spread)), s4, div32(W2, spread));
  } else if (s1 !== null && s2 === null && s3 !== null && s4 === null) {
    const r2 = blend(s1, W1, s3, sub32(1, W1));

    filled[1] = r2;
    filled[3] = blend(r2, sub32(1, div32(1, W2)), s3, div32(1, W2));
  } else if (s1 !== null && s2 !== null && s3 === null && s4 === null) {
    const r3 = blend(
      s1,
      sub32(1, div32(1, sub32(1, W1))),
      s2,
      div32(1, sub32(1, W1)),
    );

    filled[2] = r3;
    filled[3] = blend(s2, sub32(1, div32(1, W2)), r3, div32(1, W2));
  }
}

function fillFromThree(filled: (number | null)[]) {
  const [s1, s2, s3, s4] = filled;

  if (s1 === null && s2 !== null && s3 !== null) {
    filled[0] = blend(s2, div32(1, W1), s3, sub32(1, div32(1, W1)));
  } else if (s1 !== null && s2 === null && s3 !== null) {
    filled[1] = blend(s1, W1, s3, sub32(1, W1));
  } else if (s2 !== null && s3 === null && s4 !== null) {
    filled[2] = blend(s2, sub32(1, W2), s4, W2);
  } else if (s2 !== null && s3 !== null && s4 === null) {
    filled[3] = blend(s2, sub32(1, div32(1, W2)), s3, div32(1, W2));
  }
}

function clampToFour(values: number[]): InitialStability {
  const [s1, s2, s3, s4] = values.map((value) =>
    Math.min(Math.max(value, STABILITY_MIN), INIT_STABILITY_MAX),
  );

  if (s4 === undefined) {
    throw new NotEnoughData();
  }

  return [s1, s2, s3, s4];
}

import { add32, div32, f32, mul32, sub32 } from "./f32";
import {
  current,
  firstLongTermReview,
  longTermReviewCount,
  type FsrsItem,
  type FsrsRating,
} from "./item";
import { NO_CARD, type WeightedFsrsItem } from "./training";

const RATINGS: FsrsRating[] = [1, 2, 3, 4];

const MIN_KEPT_GROUP = 6;
const MIN_REMOVED = 20;
const REMOVED_FRACTION = 20;

/** Delta_t values dropped as outliers, indexed by the first review's rating. */
type RemovedPairs = Set<number>[];

type Grouped<T> = Map<FsrsRating, Map<number, T[]>>;

function group<T>(
  entries: { rating: FsrsRating; deltaT: number; payload: T }[],
): Grouped<T> {
  const groups: Grouped<T> = new Map();

  for (const { rating, deltaT, payload } of entries) {
    const byDelta = groups.get(rating) ?? new Map<number, T[]>();
    const bucket = byDelta.get(deltaT) ?? [];

    bucket.push(payload);
    byDelta.set(deltaT, bucket);
    groups.set(rating, byDelta);
  }

  return groups;
}

/**
 * Drops the rarest (first rating, delta_t) pairs — 5% of each rating, at least
 * 20 items — and keeps the rest only where the group is big enough to mean
 * something and the interval is not absurd. Both callers in fsrs-rs run this
 * same selection; only what they carry through it differs.
 */
function selectSurvivors<T>(groups: Grouped<T>) {
  // Indexed by rating, so slot 0 is never used.
  const removedPairs: RemovedPairs = Array.from(
    { length: RATINGS.length + 1 },
    () => new Set<number>(),
  );
  const kept: T[] = [];

  for (const rating of [...groups.keys()].sort((a, b) => a - b)) {
    // Descending by size, then by delta_t, and walked backwards — the smallest
    // groups are the candidates for removal. Sorting ascending instead would
    // reorder ties, because both languages sort stably.
    const subGroups = [...(groups.get(rating) ?? new Map<number, T[]>())].sort(
      ([deltaTA, a], [deltaTB, b]) => b.length - a.length || deltaTB - deltaTA,
    );

    const total = subGroups.reduce((sum, [, bucket]) => sum + bucket.length, 0);
    const quota = Math.max(MIN_REMOVED, Math.floor(total / REMOVED_FRACTION));
    const maxDeltaT = rating === 4 ? 365 : 100;

    let removed = 0;

    for (let i = subGroups.length - 1; i >= 0; i -= 1) {
      const [deltaT, bucket] = subGroups[i];

      if (removed + bucket.length >= quota) {
        if (bucket.length >= MIN_KEPT_GROUP && deltaT <= maxDeltaT) {
          kept.push(...bucket);
        } else {
          removedPairs[rating].add(deltaT);
        }
      } else {
        removed += bucket.length;
        removedPairs[rating].add(deltaT);
      }
    }
  }

  return { removedPairs, kept };
}

function survivesOutlier(item: FsrsItem, removedPairs: RemovedPairs) {
  return !removedPairs[item.reviews[0].rating].has(
    firstLongTermReview(item).deltaT,
  );
}

function computeOutlierAnalysis(items: FsrsItem[]) {
  const entries = items
    .map((item, index) => ({ item, index }))
    .filter(({ item }) => longTermReviewCount(item) === 1)
    .map(({ item, index }) => ({
      rating: item.reviews[0].rating,
      deltaT: current(item).deltaT,
      payload: index,
    }));

  const { removedPairs, kept } = selectSurvivors(group(entries));

  return { removedPairs, keptIndices: kept };
}

/**
 * Filters anomalous reviews out of both sets, using the initialisation set as
 * the reference for what a normal review pattern looks like. Every item in
 * `datasetForInitialization` must have at least two reviews.
 */
export function filterOutlier(
  datasetForInitialization: FsrsItem[],
  trainSet: FsrsItem[],
): [FsrsItem[], FsrsItem[]] {
  const { removedPairs, kept } = selectSurvivors(
    group(
      datasetForInitialization.map((item) => ({
        rating: item.reviews[0].rating,
        deltaT: current(item).deltaT,
        payload: item,
      })),
    ),
  );

  return [kept, trainSet.filter((item) => survivesOutlier(item, removedPairs))];
}

export type PreparedTrainingData = {
  datasetForInitialization: FsrsItem[];
  trainSet: FsrsItem[];
  trainCardIds: number[] | null;
};

/**
 * Splits review histories into the set that seeds initial stability and the set
 * that trains everything else, dropping outliers on the way.
 *
 * fsrs-rs reads `FSRS_NO_OUTLIER` from the environment to skip the filter;
 * a library has no business reading env vars, so it is an option here.
 */
export function prepareTrainingData(
  items: FsrsItem[],
  cardIds: number[] | null = null,
  { skipOutlierFilter = false } = {},
): PreparedTrainingData {
  if (skipOutlierFilter) {
    const datasetForInitialization: FsrsItem[] = [];
    const trainSet: FsrsItem[] = [];
    const trainCardIds: number[] = [];

    items.forEach((item, index) => {
      if (longTermReviewCount(item) === 1) {
        datasetForInitialization.push(item);

        return;
      }

      trainSet.push(item);

      if (cardIds) {
        trainCardIds.push(cardIds[index]);
      }
    });

    return {
      datasetForInitialization,
      trainSet,
      trainCardIds: cardIds ? trainCardIds : null,
    };
  }

  const { removedPairs, keptIndices } = computeOutlierAnalysis(items);
  const trainSet: FsrsItem[] = [];
  const trainCardIds: number[] = [];

  items.forEach((item, index) => {
    if (!survivesOutlier(item, removedPairs)) {
      return;
    }

    trainSet.push(item);

    if (cardIds) {
      trainCardIds.push(cardIds[index]);
    }
  });

  return {
    datasetForInitialization: keptIndices.map((index) => items[index]),
    trainSet,
    trainCardIds: cardIds ? trainCardIds : null,
  };
}

export function constantWeightedItems(items: FsrsItem[]): WeightedFsrsItem[] {
  return items.map((item) => ({ weight: 1, cardId: NO_CARD, item }));
}

/**
 * Weights recent reviews far above old ones — 0.25 rising to 1.0 along a cubic.
 * The input must already be ordered by review time; nothing here can check it.
 */
export function recencyWeightedItems(
  items: FsrsItem[],
  cardIds: number[] | null = null,
): WeightedFsrsItem[] {
  const length = Math.max(sub32(items.length, 1), 1);

  return items.map((item, index) => {
    const position = div32(index, length);

    return {
      weight: add32(
        f32(0.25),
        mul32(f32(0.75), mul32(mul32(position, position), position)),
      ),
      cardId: cardIds ? cardIds[index] : NO_CARD,
      item,
    };
  });
}

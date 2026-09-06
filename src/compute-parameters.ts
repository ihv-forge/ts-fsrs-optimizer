import { DEFAULT_PARAMETERS } from "./constants";
import { prepareTrainingData, recencyWeightedItems } from "./dataset";
import { div32 } from "./f32";
import {
  initializeStabilityParameters,
  smoothAndFill,
  type InitialStability,
} from "./initial-stability";
import { current, type FsrsItem, type FsrsRating } from "./item";
import {
  DEFAULT_TRAINING_CONFIG,
  train,
  type TrainingConfig,
} from "./training";

/** Below this many items nothing is fitted; the defaults are already better. */
const MIN_ITEMS_TO_INITIALISE = 8;

/** Below this, initial stability is fitted but the rest is left alone. */
const MIN_ITEMS_TO_TRAIN = 64;

export class InvalidInput extends Error {
  constructor(reason: string) {
    super(`fsrs-optimizer: ${reason}`);
    this.name = "InvalidInput";
  }
}

export type ComputeParametersInput = {
  /** Review histories, ordered by review time, each one a prefix of a card. */
  trainSet: FsrsItem[];
  /** Card ids aligned with `trainSet`, which switches on windowed batching. */
  cardIds?: number[] | null;
  enableShortTerm?: boolean;
  numRelearningSteps?: number;
  trainingConfig?: Partial<TrainingConfig>;
  skipOutlierFilter?: boolean;
};

export function calculateAverageRecall(items: FsrsItem[]) {
  let recalled = 0;

  for (const item of items) {
    if (current(item).rating > 1) {
      recalled += 1;
    }
  }

  return items.length === 0 ? 0 : div32(recalled, items.length);
}

function initialParameters(
  initialStability: InitialStability,
  freezeShortTermStability: boolean,
) {
  const parameters = [...initialStability, ...DEFAULT_PARAMETERS.slice(4)];

  if (freezeShortTermStability) {
    parameters[17] = 0;
    parameters[18] = 0;
    parameters[19] = 0;
  }

  return parameters;
}

/**
 * Fits FSRS parameters to one person's review history.
 *
 * Returns the defaults untouched when there is too little to learn from, and
 * initial stability only when there is enough to seed it but not enough to
 * train on — both are results, not failures.
 */
export function computeParameters({
  trainSet: originalTrainSet,
  cardIds = null,
  enableShortTerm = true,
  numRelearningSteps = 1,
  trainingConfig,
  skipOutlierFilter = false,
}: ComputeParametersInput) {
  const config: TrainingConfig = {
    ...DEFAULT_TRAINING_CONFIG,
    ...trainingConfig,
  };

  if (
    config.batchSize <= 0 ||
    !Number.isFinite(config.learningRate) ||
    !Number.isFinite(config.gamma)
  ) {
    throw new InvalidInput("training config is out of range");
  }

  if (cardIds && cardIds.length !== originalTrainSet.length) {
    throw new InvalidInput("cardIds must line up with trainSet");
  }

  for (const item of originalTrainSet) {
    if (
      item.reviews.length === 0 ||
      item.reviews.some((review) => review.rating < 1 || review.rating > 4)
    ) {
      throw new InvalidInput("every item needs reviews rated 1 to 4");
    }
  }

  const { datasetForInitialization, trainSet, trainCardIds } =
    prepareTrainingData(originalTrainSet, cardIds, { skipOutlierFilter });

  const averageRecall = calculateAverageRecall(trainSet);

  if (trainSet.length < MIN_ITEMS_TO_INITIALISE) {
    return [...DEFAULT_PARAMETERS];
  }

  const { stability, ratingCount } = initializeStabilityParameters(
    datasetForInitialization,
    averageRecall,
  );

  const freezeShortTermStability = !enableShortTerm;
  const seeded = initialParameters(stability, freezeShortTermStability);

  if (
    trainSet.length === datasetForInitialization.length ||
    trainSet.length < MIN_ITEMS_TO_TRAIN
  ) {
    return seeded;
  }

  const weighted = recencyWeightedItems(trainSet, trainCardIds).filter(
    (entry) => entry.item.reviews.length <= config.maxSeqLen,
  );

  const optimized = train(weighted, seeded, config, {
    freezeInitialStability: freezeShortTermStability,
    freezeShortTermStability,
    numRelearningSteps,
  });

  if (!optimized.every(Number.isFinite)) {
    throw new InvalidInput("training produced non-finite parameters");
  }

  // The trained stabilities are four independent numbers again, so the same
  // monotonicity pass that built them has to run once more.
  const trainedStability = new Map<FsrsRating, number>(
    ([1, 2, 3, 4] as FsrsRating[]).map((rating) => [
      rating,
      optimized[rating - 1],
    ]),
  );

  return [
    ...smoothAndFill(trainedStability, ratingCount),
    ...optimized.slice(4),
  ];
}

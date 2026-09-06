/**
 * A TypeScript port of the FSRS optimizer: fits the 21 FSRS parameters to one
 * person's review history, with no native module, no WebAssembly and no server.
 *
 * Derived from fsrs-rs (https://github.com/open-spaced-repetition/fsrs-rs),
 * Copyright (c) 2023 Open Spaced Repetition, licensed BSD-3-Clause. Structure
 * and naming follow the Rust source on purpose: this code is only trustworthy
 * for as long as it can be read side by side with the original. PORTING.md
 * records where it knowingly departs from it.
 */

export {
  calculateAverageRecall,
  computeParameters,
  InvalidInput,
  type ComputeParametersInput,
} from "./compute-parameters";

export {
  DEFAULT_PARAMETERS,
  FSRS6_DEFAULT_DECAY,
  PARAMETER_COUNT,
} from "./constants";

export {
  constantWeightedItems,
  filterOutlier,
  prepareTrainingData,
  recencyWeightedItems,
  type PreparedTrainingData,
} from "./dataset";

export {
  initializeStabilityParameters,
  NotEnoughData,
  type InitialStability,
} from "./initial-stability";

export { type FsrsItem, type FsrsRating, type FsrsReview } from "./item";

export { clipParameters } from "./parameter-clipper";

export {
  DEFAULT_TRAINING_CONFIG,
  NO_CARD,
  train,
  type ModelConfig,
  type TrainingConfig,
  type WeightedFsrsItem,
} from "./training";

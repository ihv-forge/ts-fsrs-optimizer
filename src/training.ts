import { batchLossAndGrad, cardLossAndGrad } from "./analytic";
import { PARAMETER_COUNT } from "./constants";
import { cosineAnnealingLR } from "./cosine-annealing";
import { f32, mul32, sub32 } from "./f32";
import { current, history, type FsrsItem } from "./item";
import { clipParametersInPlace } from "./parameter-clipper";

const PARAMS_STDDEV = [
  6.43, 9.66, 17.58, 27.85, 0.57, 0.28, 0.6, 0.12, 0.39, 0.18, 0.33, 0.3, 0.09,
  0.16, 0.57, 0.25, 1.03, 0.31, 0.32, 0.14, 0.27,
].map(f32);

const BETA1 = 0.9;
const BETA2 = 0.999;
const EPSILON = 1e-8;

/** Marks an item as not belonging to any card, which selects plain batching. */
export const NO_CARD = -1;

export type WeightedFsrsItem = {
  weight: number;
  cardId: number;
  item: FsrsItem;
};

export type TrainingConfig = {
  numEpochs: number;
  batchSize: number;
  seed: number;
  learningRate: number;
  maxSeqLen: number;
  gamma: number;
};

export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  numEpochs: 5,
  batchSize: 512,
  seed: 2023,
  learningRate: 4e-2,
  maxSeqLen: 256,
  gamma: 1,
};

export type ModelConfig = {
  freezeInitialStability: boolean;
  freezeShortTermStability: boolean;
  numRelearningSteps: number;
};

type BatchHost = {
  seqLen: number;
  batchSize: number;
  realBatchSize: number;
  columnLengths: number[];
  tHistorys: number[];
  rHistorys: number[];
  deltaTs: number[];
  labels: number[];
  weights: number[];
  windowed: boolean;
};

export function adam() {
  const m = new Array<number>(PARAMETER_COUNT).fill(0);
  const v = new Array<number>(PARAMETER_COUNT).fill(0);

  let t = 0;

  return function step(parameters: number[], grad: number[], lr: number) {
    t += 1;

    const bias1 = 1 - BETA1 ** t;
    const bias2 = 1 - BETA2 ** t;

    for (let i = 0; i < PARAMETER_COUNT; i += 1) {
      m[i] = BETA1 * m[i] + (1 - BETA1) * grad[i];
      v[i] = BETA2 * v[i] + (1 - BETA2) * grad[i] * grad[i];

      const mHat = m[i] / bias1;
      const vHat = v[i] / bias2;

      parameters[i] = sub32(
        parameters[i],
        f32((lr * mHat) / (Math.sqrt(vHat) + EPSILON)),
      );
    }
  };
}

/**
 * Pulls parameters back toward where they started, scaled by how far each one
 * usually moves across users. Without it a small deck drags the parameters it
 * has evidence for and leaves the rest wherever the noise pushed them.
 */
export function addL2Gradient(
  parameters: number[],
  initialParameters: number[],
  batchSize: number,
  totalSize: number,
  gamma: number,
  grad: number[],
) {
  const scale = (gamma * batchSize) / totalSize;

  for (let i = 0; i < grad.length && i < PARAMS_STDDEV.length; i += 1) {
    const delta = sub32(parameters[i], initialParameters[i]);
    const stddev = PARAMS_STDDEV[i];

    grad[i] += ((2 * delta) / mul32(stddev, stddev)) * scale;
  }
}

function buildPlainBatch(items: WeightedFsrsItem[]): BatchHost {
  const batchSize = items.length;
  const seqLen = Math.max(...items.map(({ item }) => item.reviews.length - 1));

  const tHistorys = new Array<number>(seqLen * batchSize).fill(0);
  const rHistorys = new Array<number>(seqLen * batchSize).fill(0);
  const columnLengths: number[] = [];
  const deltaTs: number[] = [];
  const labels: number[] = [];
  const weights: number[] = [];

  items.forEach((weighted, column) => {
    columnLengths.push(weighted.item.reviews.length - 1);

    history(weighted.item).forEach((review, t) => {
      const index = t * batchSize + column;

      tHistorys[index] = review.deltaT;
      rHistorys[index] = review.rating;
    });

    const last = current(weighted.item);

    deltaTs.push(last.deltaT);
    labels.push(last.rating > 1 ? 1 : 0);
    weights.push(weighted.weight);
  });

  return {
    seqLen,
    batchSize,
    realBatchSize: batchSize,
    columnLengths,
    tHistorys,
    rHistorys,
    deltaTs,
    labels,
    weights,
    windowed: false,
  };
}

function buildWindowedBatch(cards: WeightedFsrsItem[][]): BatchHost {
  const batchSize = cards.length;
  const seqLen = Math.max(
    ...cards.map((card) => card[card.length - 1].item.reviews.length),
  );

  const tHistorys = new Array<number>(seqLen * batchSize).fill(0);
  const rHistorys = new Array<number>(seqLen * batchSize).fill(0);
  const labels = new Array<number>(seqLen * batchSize).fill(0);
  const weights = new Array<number>(seqLen * batchSize).fill(0);
  const columnLengths: number[] = [];

  cards.forEach((card, column) => {
    const full = card[card.length - 1].item.reviews;

    columnLengths.push(full.length);

    full.forEach((review, t) => {
      const index = t * batchSize + column;

      tHistorys[index] = review.deltaT;
      rHistorys[index] = review.rating;
    });

    for (const weighted of card) {
      const index = (weighted.item.reviews.length - 1) * batchSize + column;
      const last = current(weighted.item);

      labels[index] = last.rating > 1 ? 1 : 0;
      weights[index] = weighted.weight;
    }
  });

  return {
    seqLen,
    batchSize,
    realBatchSize: cards.reduce((total, card) => total + card.length, 0),
    columnLengths,
    tHistorys,
    rHistorys,
    deltaTs: [],
    labels,
    weights,
    windowed: true,
  };
}

function buildPlainBatches(items: WeightedFsrsItem[], batchSize: number) {
  const sorted = [...items].sort(
    (a, b) => a.item.reviews.length - b.item.reviews.length,
  );
  const batches: BatchHost[] = [];

  for (let start = 0; start < sorted.length; start += batchSize) {
    batches.push(buildPlainBatch(sorted.slice(start, start + batchSize)));
  }

  return batches;
}

function buildWindowedBatches(items: WeightedFsrsItem[], batchSize: number) {
  const grouped = new Map<number, WeightedFsrsItem[]>();

  for (const item of items) {
    grouped.set(item.cardId, [...(grouped.get(item.cardId) ?? []), item]);
  }

  // fsrs-rs groups into a BTreeMap, so cards arrive ordered by id and batching
  // is reproducible. A JavaScript Map keeps insertion order instead.
  const cards = [...grouped.keys()]
    .sort((a, b) => a - b)
    .map((cardId) =>
      [...(grouped.get(cardId) ?? [])].sort(
        (a, b) => a.item.reviews.length - b.item.reviews.length,
      ),
    )
    .sort(
      (a, b) =>
        a[a.length - 1].item.reviews.length -
        b[b.length - 1].item.reviews.length,
    );

  const batches: BatchHost[] = [];

  let currentCards: WeightedFsrsItem[][] = [];
  let currentPredictions = 0;

  for (const card of cards) {
    if (
      currentCards.length > 0 &&
      currentPredictions + card.length > batchSize
    ) {
      batches.push(buildWindowedBatch(currentCards));
      currentCards = [];
      currentPredictions = 0;
    }

    currentPredictions += card.length;
    currentCards.push(card);
  }

  if (currentCards.length > 0) {
    batches.push(buildWindowedBatch(currentCards));
  }

  return batches;
}

export function buildHostBatches(items: WeightedFsrsItem[], batchSize: number) {
  return items.every((item) => item.cardId === NO_CARD)
    ? buildPlainBatches(items, batchSize)
    : buildWindowedBatches(items, batchSize);
}

export function batchLossAndGradFor(
  batch: BatchHost,
  parameters: number[],
  grad: number[],
) {
  if (batch.windowed) {
    return cardLossAndGrad(
      parameters,
      batch.tHistorys,
      batch.rHistorys,
      batch.seqLen,
      batch.batchSize,
      batch.columnLengths,
      batch.labels,
      batch.weights,
      grad,
    );
  }

  return batchLossAndGrad(
    parameters,
    batch.tHistorys,
    batch.rHistorys,
    batch.seqLen,
    batch.batchSize,
    batch.columnLengths,
    batch.deltaTs,
    batch.labels,
    batch.weights,
    grad,
  );
}

/**
 * ⚠️ Batch order will not match fsrs-rs. It seeds ChaCha12 through the `rand`
 * crate, and reproducing that bit for bit would mean porting the cipher and
 * `rand`'s own range sampling. What matters for a user is that the same input
 * and seed give the same parameters twice, and that holds here.
 */
function seededRandom(seed: number) {
  let state = seed >>> 0;

  return function next() {
    state = (state + 0x6d2b79f5) >>> 0;

    let t = state;

    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);

    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle(order: number[], random: () => number) {
  for (let i = order.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));

    [order[i], order[j]] = [order[j], order[i]];
  }
}

export function train(
  trainSet: WeightedFsrsItem[],
  initialParameters: number[],
  training: TrainingConfig,
  model: ModelConfig,
) {
  const totalSize = trainSet.length;
  const iterations =
    (Math.floor(totalSize / training.batchSize) + 1) * training.numEpochs;
  const batches = buildHostBatches(trainSet, training.batchSize);
  const nextLr = cosineAnnealingLR(iterations, training.learningRate);
  const random = seededRandom(training.seed);
  const step = adam();

  const parameters = [...initialParameters];

  for (let epoch = 0; epoch < training.numEpochs; epoch += 1) {
    const order = batches.map((_, index) => index);

    shuffle(order, random);

    for (const index of order) {
      const batch = batches[index];
      const lr = nextLr();
      const grad = new Array<number>(PARAMETER_COUNT).fill(0);

      batchLossAndGradFor(batch, parameters, grad);
      addL2Gradient(
        parameters,
        initialParameters,
        batch.realBatchSize,
        totalSize,
        training.gamma,
        grad,
      );

      if (model.freezeInitialStability) {
        grad.fill(0, 0, 4);
      }

      if (model.freezeShortTermStability) {
        grad.fill(0, 17, 20);
      }

      step(parameters, grad, lr);
      clipParametersInPlace(
        parameters,
        model.numRelearningSteps,
        !model.freezeShortTermStability,
      );
    }
  }

  return parameters;
}

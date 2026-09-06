import { describe, expect, it } from "vitest";

import { DEFAULT_PARAMETERS, PARAMETER_COUNT } from "./constants";
import { cosineAnnealingLR } from "./cosine-annealing";
import { f32 } from "./f32";
import type { FsrsReview } from "./item";
import { clipParameters } from "./parameter-clipper";
import {
  adam,
  batchLossAndGradFor,
  buildHostBatches,
  DEFAULT_TRAINING_CONFIG,
  NO_CARD,
  train,
  type WeightedFsrsItem,
} from "./training";

// Ported from the #[cfg(test)] blocks of fsrs-rs `src/cosine_annealing.rs`,
// `src/parameter_clipper.rs` and `src/training.rs`.

const W = [...DEFAULT_PARAMETERS];

function zeros() {
  return new Array<number>(PARAMETER_COUNT).fill(0);
}

describe("cosineAnnealingLR", () => {
  it("matches fsrs-rs test_lr_scheduler", () => {
    const step = cosineAnnealingLR(5, 4e-2);
    const rates = Array.from({ length: 11 }, () => step());

    for (const [index, expected] of [
      0.04, 0.03618033988749895, 0.026180339887498946, 0.013819660112501051,
      0.0038196601125010526, 0.0, 0.003819660112501051, 0.013819660112501048,
      0.026180339887498943, 0.03618033988749895, 0.039999999999999994,
    ].entries()) {
      expect(rates[index]).toBeCloseTo(expected, 10);
    }
  });
});

describe("clipParameters", () => {
  it("matches fsrs-rs test_parameter_clipper_works", () => {
    const clipped = clipParameters(
      [0, -1000, 1000, 0, 1000, -1000, 1, 0.25, -0.1],
      1,
      true,
    );

    expect(clipped).toEqual(
      [0.001, 0.001, 100, 0.001, 10, 0.001, 1, 0.25, 0].map(f32),
    );
  });

  it("leaves the defaults alone at two relearning steps", () => {
    const clipped = clipParameters(W, 2, true);

    for (const [index, expected] of [0.5425, 0.0912, 0.0658].entries()) {
      expect(clipped[17 + index]).toBeCloseTo(expected, 4);
    }
  });
});

describe("adam", () => {
  it("matches fsrs-rs test_host_adam_matches_burn_adam_steps", () => {
    const step = adam();

    let parameters = [...W];

    step(
      parameters,
      [
        -0.095688485, -0.0051607806, -0.0012249565, 0.007462064, 0.03650761,
        -0.082112335, 0.0593964, -2.1474836, 0.57626534, -2.8751316, 0.7154875,
        -0.028993709, 0.0099172965, -0.2189217, -0.0017800558, -0.089381434,
        0.299141, 0.068104014, -0.011605468, -0.25398168, 0.27700496,
      ],
      0.04,
    );
    parameters = clipParameters(parameters, 1, true);

    expectClose(
      parameters,
      [
        0.252, 1.3331, 2.3464994, 8.2556, 6.3733, 0.87340003, 2.9794,
        0.040999997, 1.8322, 0.20660001, 0.756, 1.5235, 0.021400042, 0.3029,
        1.6882998, 0.64140004, 1.8329, 0.5025, 0.13119997, 0.1058, 0.1142,
      ],
    );

    step(
      parameters,
      [
        -0.040530164, -0.0041278866, -0.0010157757, 0.007239434, 0.009321215,
        -0.120117955, 0.039143264, -0.8628009, 0.5794302, -2.5713828, 0.7669307,
        -0.024242667, 0.0, -0.16912507, -0.0017008218, -0.061857328, 0.28093633,
        0.064058185, 0.0063592787, -0.1903223, 0.6257775,
      ],
      0.04,
    );
    parameters = clipParameters(parameters, 1, true);

    expectClose(
      parameters,
      [
        0.2882918, 1.3726242, 2.3861322, 8.215636, 6.339965, 0.9130969,
        2.940639, 0.07696985, 1.7921946, 0.2464217, 0.71595186, 1.5631561,
        0.001, 0.34230903, 1.7282416, 0.68038, 1.7929853, 0.46258268,
        0.14039303, 0.14509967, 0.1,
      ],
    );
  });

  function expectClose(actual: number[], expected: number[]) {
    for (const [index, value] of expected.entries()) {
      expect(actual[index], `parameter ${index}`).toBeCloseTo(value, 4);
    }
  }
});

describe("buildHostBatches", () => {
  it("gives a windowed batch the same loss and gradient as plain prefixes", () => {
    const reviews: FsrsReview[] = [
      { rating: 4, deltaT: 0 },
      { rating: 3, deltaT: 2 },
      { rating: 1, deltaT: 5 },
      { rating: 3, deltaT: 8 },
    ];

    const weighted: WeightedFsrsItem[] = [2, 3, 4].map((length, index) => ({
      weight: 0.5 + index * 0.25,
      cardId: 42,
      item: { reviews: reviews.slice(0, length) },
    }));

    const plain = buildHostBatches(
      weighted.map((entry) => ({ ...entry, cardId: NO_CARD })),
      32,
    );
    const windowed = buildHostBatches(weighted, 32);

    expect(plain).toHaveLength(1);
    expect(windowed).toHaveLength(1);

    const plainGrad = zeros();
    const windowedGrad = zeros();
    const plainLoss = batchLossAndGradFor(plain[0], W, plainGrad);
    const windowedLoss = batchLossAndGradFor(windowed[0], W, windowedGrad);

    expect(Math.abs(plainLoss - windowedLoss)).toBeLessThan(1e-9);

    for (let i = 0; i < PARAMETER_COUNT; i += 1) {
      expect(
        Math.abs(plainGrad[i] - windowedGrad[i]),
        `parameter ${i}: plain=${plainGrad[i]} windowed=${windowedGrad[i]}`,
      ).toBeLessThan(1e-9);
    }
  });
});

// Not from fsrs-rs: its end-to-end test reads an Anki collection fixture. This
// checks the same thing the fixture is there to check — that the loop as a
// whole moves the parameters somewhere better than where it started.
describe("train", () => {
  function syntheticSet() {
    const items: WeightedFsrsItem[] = [];

    for (let card = 0; card < 96; card += 1) {
      const reviews: FsrsReview[] = [{ rating: 3, deltaT: 0 }];

      let gap = 1;

      for (let step = 1; step < 6; step += 1) {
        const lapsed = (card + step) % 5 === 0;

        reviews.push({
          rating: lapsed ? 1 : step % 3 === 0 ? 4 : 3,
          deltaT: gap,
        });
        gap = lapsed ? 1 : gap * 2 + (card % 3);
      }

      for (let length = 2; length <= reviews.length; length += 1) {
        items.push({
          weight: 1,
          cardId: NO_CARD,
          item: { reviews: reviews.slice(0, length) },
        });
      }
    }

    return items;
  }

  function lossOf(items: WeightedFsrsItem[], parameters: number[]) {
    return buildHostBatches(items, 512).reduce(
      (total, batch) => total + batchLossAndGradFor(batch, parameters, zeros()),
      0,
    );
  }

  it("lowers the loss it started from", () => {
    const items = syntheticSet();
    const trained = train(items, W, DEFAULT_TRAINING_CONFIG, {
      freezeInitialStability: false,
      freezeShortTermStability: false,
      numRelearningSteps: 1,
    });

    expect(trained).toHaveLength(PARAMETER_COUNT);
    expect(trained.every(Number.isFinite)).toBe(true);
    expect(lossOf(items, trained)).toBeLessThan(lossOf(items, W));
  });

  it("is reproducible for the same seed", () => {
    const items = syntheticSet();
    const config = { ...DEFAULT_TRAINING_CONFIG, numEpochs: 2 };
    const model = {
      freezeInitialStability: false,
      freezeShortTermStability: false,
      numRelearningSteps: 1,
    };

    expect(train(items, W, config, model)).toEqual(
      train(items, W, config, model),
    );
  });
});

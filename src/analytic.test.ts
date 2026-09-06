import { describe, expect, it } from "vitest";

import { batchLoss, batchLossAndGrad, cardLossAndGrad } from "./analytic";
import { clampGrad } from "./bounds";
import { DEFAULT_PARAMETERS, PARAMETER_COUNT } from "./constants";
import { add32, f32, mul32, sub32 } from "./f32";

// Ported from the #[cfg(test)] block of fsrs-rs `src/analytic.rs`. These check
// the gradients against finite differences of the loss they claim to
// differentiate, so a mistranscribed derivative fails here without needing any
// reference number from Rust.

const W = [...DEFAULT_PARAMETERS];
const PARAMETERS = PARAMETER_COUNT;

type Batch = {
  seqLen: number;
  batch: number;
  tHist: number[];
  rHist: number[];
  deltaTs: number[];
  labels: number[];
  weights: number[];
};

function zeros() {
  return new Array<number>(PARAMETERS).fill(0);
}

function expectGradientMatchesFiniteDifferences(data: Batch) {
  const seqLens = new Array<number>(data.batch).fill(data.seqLen);
  const grad = zeros();

  const lossAt = (w: number[]) =>
    batchLoss(
      w,
      data.tHist,
      data.rHist,
      data.seqLen,
      data.batch,
      seqLens,
      data.deltaTs,
      data.labels,
      data.weights,
    );

  batchLossAndGrad(
    W,
    data.tHist,
    data.rHist,
    data.seqLen,
    data.batch,
    seqLens,
    data.deltaTs,
    data.labels,
    data.weights,
    grad,
  );

  for (let i = 0; i < PARAMETERS; i += 1) {
    const eps = Math.max(f32(1e-3), mul32(Math.abs(W[i]), f32(1e-3)));
    const plus = [...W];
    const minus = [...W];

    plus[i] = add32(plus[i], eps);
    minus[i] = sub32(minus[i], eps);

    const numeric = (lossAt(plus) - lossAt(minus)) / (2 * eps);
    const scale = Math.max(Math.abs(numeric), Math.abs(grad[i]), 1);

    expect(
      Math.abs(numeric - grad[i]) / scale,
      `parameter ${i}: analytic=${grad[i]} numeric=${numeric}`,
    ).toBeLessThan(2e-2);
  }

  return grad;
}

describe("clampGrad", () => {
  it("lets gradients through the boundary and stops outside it", () => {
    expect(clampGrad(0, 0.001, 100)).toBe(0);
    expect(clampGrad(0.001, 0.001, 100)).toBe(1);
    expect(clampGrad(0.5, 0.001, 100)).toBe(1);
    expect(clampGrad(100, 0.001, 100)).toBe(1);
    expect(clampGrad(101, 0.001, 100)).toBe(0);
  });
});

describe("batchLossAndGrad", () => {
  it("matches finite differences of the loss", () => {
    expectGradientMatchesFiniteDifferences({
      seqLen: 3,
      batch: 2,
      tHist: [0, 0, 3, 2, 5, 6],
      rHist: [4, 1, 3, 2, 1, 3],
      deltaTs: [7, 9],
      labels: [1, 0],
      weights: [1, 0.7],
    });
  });

  // The batch above never reaches the same-day branch and touches the lapse one
  // only where its w[12] term vanishes, so a sign error in either survives it.
  // Histories are time-major (idx = time * batch + column): column 0 goes
  // init, same-day easy, lapse, same-day again, success; column 1 goes init,
  // success, same-day, lapse, easy.
  it("matches finite differences through the same-day and lapse branches", () => {
    const grad = expectGradientMatchesFiniteDifferences({
      seqLen: 5,
      batch: 2,
      tHist: [0, 0, 0, 3, 5, 0, 0, 7, 2, 4],
      rHist: [3, 1, 4, 2, 1, 2, 1, 1, 3, 4],
      deltaTs: [6, 9],
      labels: [1, 0],
      weights: [1, 0.7],
    });

    for (const index of [12, 17, 18, 19]) {
      expect(grad[index], `parameter ${index} was never reached`).not.toBe(0);
    }
  });
});

describe("cardLossAndGrad", () => {
  it("equals the sum over every prefix of the same card", () => {
    const seq = 4;
    const tHistCard = [0, 2, 5, 8];
    const rHistCard = [4, 3, 1, 3];
    const labelsCard = [0, 1, 0, 1];
    const weightsCard = [0, 0.5, 0.7, 1.1];

    const cardGrad = zeros();
    const cardLoss = cardLossAndGrad(
      W,
      tHistCard,
      rHistCard,
      seq,
      1,
      [seq],
      labelsCard,
      weightsCard,
      cardGrad,
    );

    const prefixGrad = zeros();

    let prefixLoss = 0;

    for (let prefix = 2; prefix <= seq; prefix += 1) {
      const historyLength = prefix - 1;

      prefixLoss += batchLossAndGrad(
        W,
        tHistCard.slice(0, historyLength),
        rHistCard.slice(0, historyLength),
        historyLength,
        1,
        [historyLength],
        [tHistCard[prefix - 1]],
        [labelsCard[prefix - 1]],
        [weightsCard[prefix - 1]],
        prefixGrad,
      );
    }

    expect(Math.abs(cardLoss - prefixLoss)).toBeLessThan(1e-9);

    for (let i = 0; i < PARAMETERS; i += 1) {
      expect(
        Math.abs(cardGrad[i] - prefixGrad[i]),
        `parameter ${i}: card=${cardGrad[i]} prefix=${prefixGrad[i]}`,
      ).toBeLessThan(1e-9);
    }
  });
});

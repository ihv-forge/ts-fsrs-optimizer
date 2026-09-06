import { describe, expect, it } from "vitest";

import {
  calculateAverageRecall,
  computeParameters,
  InvalidInput,
} from "./compute-parameters";
import { DEFAULT_PARAMETERS, PARAMETER_COUNT } from "./constants";
import { f32 } from "./f32";
import type { FsrsItem, FsrsRating, FsrsReview } from "./item";
import { clipParameters } from "./parameter-clipper";

const W = [...DEFAULT_PARAMETERS];

function card(index: number): FsrsReview[] {
  const reviews: FsrsReview[] = [
    { rating: ((index % 4) + 1) as FsrsRating, deltaT: 0 },
  ];

  let gap = 1;

  for (let step = 1; step < 6; step += 1) {
    const lapsed = (index + step) % 5 === 0;

    reviews.push({
      rating: lapsed ? 1 : step % 3 === 0 ? 4 : 3,
      deltaT: gap,
    });
    gap = lapsed ? 1 : gap * 2 + (index % 3);
  }

  return reviews;
}

function prefixes(cards: number) {
  const items: FsrsItem[] = [];

  for (let index = 0; index < cards; index += 1) {
    const reviews = card(index);

    for (let length = 2; length <= reviews.length; length += 1) {
      items.push({ reviews: reviews.slice(0, length) });
    }
  }

  return items;
}

describe("calculateAverageRecall", () => {
  it("counts the last review of each item", () => {
    expect(
      calculateAverageRecall([
        { reviews: [{ rating: 3, deltaT: 0 }] },
        { reviews: [{ rating: 1, deltaT: 2 }] },
        { reviews: [{ rating: 4, deltaT: 5 }] },
        { reviews: [{ rating: 2, deltaT: 7 }] },
      ]),
    ).toBe(f32(0.75));
  });

  it("is zero for no items rather than NaN", () => {
    expect(calculateAverageRecall([])).toBe(0);
  });
});

describe("computeParameters", () => {
  it("returns the defaults when there is nothing to learn from", () => {
    expect(computeParameters({ trainSet: prefixes(1) })).toEqual(W);
  });

  it("rejects a mismatched cardIds list", () => {
    expect(() =>
      computeParameters({ trainSet: prefixes(20), cardIds: [1, 2] }),
    ).toThrow(InvalidInput);
  });

  it("rejects items with no reviews or an impossible rating", () => {
    expect(() => computeParameters({ trainSet: [{ reviews: [] }] })).toThrow(
      InvalidInput,
    );
    expect(() =>
      computeParameters({
        trainSet: [{ reviews: [{ rating: 7 as FsrsRating, deltaT: 0 }] }],
      }),
    ).toThrow(InvalidInput);
  });

  it("rejects a training config that cannot run", () => {
    expect(() =>
      computeParameters({
        trainSet: prefixes(20),
        trainingConfig: { batchSize: 0 },
      }),
    ).toThrow(InvalidInput);
  });

  it("fits parameters that stay inside the clipper's bounds", () => {
    const parameters = computeParameters({ trainSet: prefixes(96) });

    expect(parameters).toHaveLength(PARAMETER_COUNT);
    expect(parameters.every(Number.isFinite)).toBe(true);
    expect(parameters).toEqual(clipParameters(parameters, 1, true));
    expect(parameters).not.toEqual(W);
  });

  it("gives the same answer twice", () => {
    const items = prefixes(96);

    expect(computeParameters({ trainSet: items })).toEqual(
      computeParameters({ trainSet: items }),
    );
  });

  it("takes the windowed path when card ids are supplied", () => {
    const cards = 96;
    const items = prefixes(cards);
    const cardIds = items.map((_, index) => Math.floor(index / 5));

    const parameters = computeParameters({ trainSet: items, cardIds });

    expect(parameters).toHaveLength(PARAMETER_COUNT);
    expect(parameters.every(Number.isFinite)).toBe(true);
  });

  it("zeroes the short-term parameters when short term is disabled", () => {
    const parameters = computeParameters({
      trainSet: prefixes(96),
      enableShortTerm: false,
    });

    expect(parameters.slice(17, 20)).toEqual([0, 0, 0]);
  });
});

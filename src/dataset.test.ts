import { describe, expect, it } from "vitest";

import {
  filterOutlier,
  prepareTrainingData,
  recencyWeightedItems,
} from "./dataset";
import { f32 } from "./f32";
import type { FsrsItem, FsrsRating } from "./item";

// fsrs-rs tests this file against an Anki collection fixture, which cannot come
// along. These check the behaviour the fixture demonstrates instead.

function pair(rating: FsrsRating, deltaT: number): FsrsItem {
  return {
    reviews: [
      { rating: 3, deltaT: 0 },
      { rating, deltaT },
    ],
  };
}

function repeat(count: number, make: (index: number) => FsrsItem) {
  return Array.from({ length: count }, (_, index) => make(index));
}

describe("filterOutlier", () => {
  // Quota is max(20, total / 20), so a rating group needs more than 20 items
  // before anything is kept at all — with less, everything looks like an
  // outlier. This shape has one interval that occurs once against two that are
  // common.
  const initialisation = [
    ...repeat(30, () => pair(3, 1)),
    ...repeat(25, () => pair(3, 2)),
    pair(3, 999),
  ];

  it("drops the rare interval and keeps the common ones", () => {
    const [kept] = filterOutlier(initialisation, []);

    expect(kept).toHaveLength(55);
    expect(kept.some((item) => item.reviews[1].deltaT === 999)).toBe(false);
  });

  it("removes training items that share the dropped pair", () => {
    const doomed: FsrsItem = {
      reviews: [
        { rating: 3, deltaT: 0 },
        { rating: 3, deltaT: 999 },
        { rating: 3, deltaT: 4 },
      ],
    };
    const survivor: FsrsItem = {
      reviews: [
        { rating: 3, deltaT: 0 },
        { rating: 3, deltaT: 1 },
        { rating: 3, deltaT: 4 },
      ],
    };

    const [, trainSet] = filterOutlier(initialisation, [doomed, survivor]);

    expect(trainSet).toEqual([survivor]);
  });
});

describe("prepareTrainingData", () => {
  it("splits on the long-term review count when the filter is skipped", () => {
    const single = pair(3, 2);
    const many: FsrsItem = {
      reviews: [
        { rating: 3, deltaT: 0 },
        { rating: 3, deltaT: 2 },
        { rating: 3, deltaT: 5 },
      ],
    };

    const prepared = prepareTrainingData([single, many], null, {
      skipOutlierFilter: true,
    });

    expect(prepared.datasetForInitialization).toEqual([single]);
    expect(prepared.trainSet).toEqual([many]);
    expect(prepared.trainCardIds).toBeNull();
  });

  it("keeps card ids aligned with the items that survive", () => {
    const items = [
      ...repeat(30, () => pair(3, 1)),
      pair(3, 999),
      {
        reviews: [
          { rating: 3, deltaT: 0 },
          { rating: 3, deltaT: 999 },
          { rating: 3, deltaT: 4 },
        ],
      } satisfies FsrsItem,
    ];
    const cardIds = items.map((_, index) => index);

    const prepared = prepareTrainingData(items, cardIds);

    expect(prepared.trainCardIds).toHaveLength(prepared.trainSet.length);
    expect(prepared.trainCardIds).not.toContain(30);
    expect(prepared.trainCardIds).not.toContain(31);
  });
});

describe("recencyWeightedItems", () => {
  it("rises from a quarter to one along a cubic", () => {
    const weighted = recencyWeightedItems(repeat(5, () => pair(3, 1)));
    const weights = weighted.map((entry) => entry.weight);

    expect(weights[0]).toBe(f32(0.25));
    expect(weights[weights.length - 1]).toBe(1);

    for (let i = 1; i < weights.length; i += 1) {
      expect(weights[i]).toBeGreaterThan(weights[i - 1]);
    }

    // Cubic, so the midpoint sits far below the halfway mark.
    expect(weights[2]).toBeLessThan(0.4);
  });

  it("survives a single item without dividing by zero", () => {
    expect(recencyWeightedItems([pair(3, 1)])[0].weight).toBe(f32(0.25));
  });

  it("carries card ids through when given", () => {
    const weighted = recencyWeightedItems(
      repeat(3, () => pair(3, 1)),
      [7, 8, 9],
    );

    expect(weighted.map((entry) => entry.cardId)).toEqual([7, 8, 9]);
  });
});

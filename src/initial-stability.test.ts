import { describe, expect, it } from "vitest";

import { f32 } from "./f32";
import {
  loss,
  powerForgettingCurve,
  searchParameters,
  smoothAndFill,
} from "./initial-stability";
import type { FsrsRating } from "./item";

// Ported from the #[cfg(test)] block of fsrs-rs `src/parameter_initialization.rs`.
// The expected numbers are the reference implementation's own, so a port that
// drifts fails here rather than months later on someone's schedule. Values Rust
// prints as f32 are compared through f32(): 1.1227008 is how Rust writes the
// same bits JavaScript writes as 1.122700810432434.

const DEFAULT_S0 = f32(0.212);

describe("powerForgettingCurve", () => {
  it("matches fsrs-rs test_power_forgetting_curve", () => {
    expect(powerForgettingCurve([0, 1, 2, 3], 1)).toEqual([
      1, 0.9, 0.8458846447796301, 0.8093881028681906,
    ]);
  });
});

describe("loss", () => {
  const deltaT = [1, 2, 3, 4, 5];
  const recall = [0.86666667, 0.90721649, 0.73015873, 0.76315789, 0.67857143];
  const count = [435, 97, 63, 38, 28];

  it("matches fsrs-rs test_loss", () => {
    expect(loss(deltaT, recall, count, 0.7840586, DEFAULT_S0)).toBe(
      279.9206961069712,
    );
    expect(loss(deltaT, recall, count, 0.7840590622451964, DEFAULT_S0)).toBe(
      279.9206977311556,
    );
  });
});

describe("searchParameters", () => {
  const data = [
    { deltaT: 1, recall: 0.86666667, count: 435 },
    { deltaT: 2, recall: 0.90721649, count: 97 },
    { deltaT: 3, recall: 0.73015873, count: 63 },
    { deltaT: 4, recall: 0.76315789, count: 38 },
    { deltaT: 5, recall: 0.67857143, count: 28 },
  ];

  it("matches fsrs-rs test_search_parameters", () => {
    const dataset = new Map<FsrsRating, typeof data>([[1, data]]);

    expect(searchParameters(dataset, 0.94302857).get(1)).toBeCloseTo(
      0.7355089,
      4,
    );
  });
});

describe("smoothAndFill", () => {
  it("interpolates the one missing rating", () => {
    const stability = ratings([
      [1, 0.4],
      [3, 2.3],
      [4, 10.9],
    ]);

    expect(smoothAndFill(stability, counts([1, 2, 3, 4]))).toEqual(
      [0.4, 1.1227008, 2.3, 10.9].map(f32),
    );
  });

  it("scales the defaults when only one rating is known", () => {
    expect(smoothAndFill(ratings([[2, 0.35]]), counts([2]))).toEqual(
      [0.05738148, 0.35, 0.6242943, 2.2453482].map(f32),
    );
  });

  function ratings(entries: [FsrsRating, number][]) {
    return new Map(entries.map(([rating, value]) => [rating, f32(value)]));
  }

  function counts(present: FsrsRating[]) {
    return new Map(present.map((rating) => [rating, 1]));
  }
});

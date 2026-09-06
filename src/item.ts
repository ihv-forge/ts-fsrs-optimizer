export type FsrsRating = 1 | 2 | 3 | 4;

export type FsrsReview = {
  rating: FsrsRating;
  /** Days since the previous review. Zero for a repeat on the same day. */
  deltaT: number;
};

export type FsrsItem = {
  reviews: FsrsReview[];
};

export function longTermReviewCount(item: FsrsItem) {
  return item.reviews.filter((review) => review.deltaT > 0).length;
}

export function firstLongTermReview(item: FsrsItem) {
  const review = item.reviews.find((entry) => entry.deltaT > 0);

  if (!review) {
    throw new Error("fsrs item has no review with deltaT > 0");
  }

  return review;
}

/** Every review before the one being predicted. */
export function history(item: FsrsItem) {
  return item.reviews.slice(0, -1);
}

export function current(item: FsrsItem) {
  return item.reviews[item.reviews.length - 1];
}

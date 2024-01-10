We are now able to compress a targeted linear layer, and figure out the performance lost from doing so. We can also do a search over all the linear layers in the model, and figure out which are the easiest to remove without damaging the model's brain.

However, this doesn't capture complexities or search over space of possible compressions. (Maybe A and B are both ok to remove, but if you remove A and B it's bad.) Also no way to trade off performance vs. compression (maybe removing A leads to worse performance than B, but A is a bigger layer so it's more "worth it".)

Future directions:
- Instead of treating layers as independent, do search (possibly random sample) over sets of linear to try to compress, and see which group compromises performance the least.
- Then make this iterative: compress the lowest-cost group, then start search from there for the next lowest-cost group, etc.
- This algorithm is greedy (no backtracking) but it's possible that the greedy choice at step 1 leads to worse choices at step 2, so could maintain some "beams" or set of candidate models
- Way to score compression % vs. score decline and combine them to an overall scoring metric for how "good" the result is, or keep one fixed (like compression %) and find the model that compresses that much but keeps the score the best.
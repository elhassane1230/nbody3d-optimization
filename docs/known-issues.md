# Known issues

This file documents correctness problems that exist in the **last two**
optimization steps. They are kept **as originally benchmarked** so the results
in `benchmarks/` correspond exactly to the code in this repository. They are
listed openly here because, in a performance study, a fast result is only
meaningful if it computes the right answer.

Steps `00` through `06` are believed correct (modulo floating-point reordering
from `-ffast-math`).

---

## Step 07 — `src/07_cache_blocking.c`

1. **Broken unrolling.** The four unrolled distance computations all read
   `p->x[j]` (and `y`, `z`) instead of `p->x[j+0..3]`. The same neighbour is
   therefore accumulated four times rather than four distinct neighbours.
   Step 06 (`06_unrolling.c`) does this correctly and can be used as the
   reference.

2. **Manual tiling drops work.** Tiles are distributed across threads with

   ```c
   tiles_per_thread = (n + tile_size - 1) / tile_size / num_threads;
   ```

   Integer division truncates, so when the tile count is not divisible by the
   thread count the remaining tiles are never processed and the corresponding
   particles are skipped. A guarded `#pragma omp for` over the tiles would
   distribute them safely.

3. **Missing position update.** The second loop that advances positions
   (`p->x[i] += dt * p->vx[i]`, present in every other step) is absent, so
   positions never move.

## Step 08 — `src/08_intrinsics.c`

1. **Load vs broadcast.** For each scalar `j`, the value should be broadcast
   across the vector with `_mm256_set1_ps(p->x[j])`. The current code uses
   `_mm256_loadu_ps(p->x + j)`, which loads eight consecutive `j` values while
   holding eight `i` values, pairing `i+k` with `j+k` instead of every `i` with
   the single `j`.

2. **Multiply instead of divide.** The force accumulation uses

   ```c
   fx = _mm256_fmadd_ps(dx, d_3_over_2, fx);   // fx += dx * d^3
   ```

   but Newtonian gravity needs `dx / d^3`. The reciprocal is missing; the fix is
   to divide (`_mm256_div_ps`) or multiply by `_mm256_rcp_ps(d_3_over_2)`.

---

### How to proceed

Two reasonable options:

* **Keep as-is** and treat steps 07–08 as a record of what was measured, with
  this file as the honest caveat.
* **Fix and re-measure.** Apply the corrections above and re-run
  `scripts/run_benchmarks.sh` so the gain curves reflect correct code. Expect
  the corrected versions to be somewhat slower, since they perform the full,
  correct amount of work.

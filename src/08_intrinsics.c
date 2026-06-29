// ============================================================================
// N-body 3D - Step 08: manual AVX/FMA intrinsics
// ----------------------------------------------------------------------------
// Vectorizes the outer loop by hand: 8 i-particles are processed per __m256.
//
// !!! KNOWN ISSUES (see docs/known-issues.md) - kept as originally benchmarked:
//   1. The j-particle should be broadcast (_mm256_set1_ps(p->x[j])) rather
//      than loaded as 8 consecutive values.
//   2. fmadd uses dx * d^3 instead of dx / d^3 (missing reciprocal/division).
// These do not affect compilation but make the result physically incorrect.
// ============================================================================
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

typedef float              f32;
typedef double             f64;
typedef unsigned long long u64;

typedef struct particles_s {
  f32 *x;
  f32 *y;
  f32 *z;
  f32 *vx;
  f32 *vy;
  f32 *vz;
} particles_t;

void init_particles(particles_t *p, u64 n) {
  for (u64 i = 0; i < n; i++) {
    u64 r1 = (u64)rand();
    u64 r2 = (u64)rand();
    f32 sign = (r1 > r2) ? 1 : -1;

    p->x[i] = sign * (f32)rand() / (f32)RAND_MAX;
    p->y[i] = (f32)rand() / (f32)RAND_MAX;
    p->z[i] = sign * (f32)rand() / (f32)RAND_MAX;

    p->vx[i] = (f32)rand() / (f32)RAND_MAX;
    p->vy[i] = sign * (f32)rand() / (f32)RAND_MAX;
    p->vz[i] = (f32)rand() / (f32)RAND_MAX;
  }
}

void move_particles(particles_t *p, const f32 dt, u64 n) {
    const f32 softening = 1e-20;

    __m256 sof, dtvec;
    sof = _mm256_set1_ps(softening);
    dtvec = _mm256_set1_ps(dt);

    for (u64 i = 0; i < n; i += 8) {
        __m256 fx, fy, fz;
        fx = _mm256_setzero_ps();
        fy = _mm256_setzero_ps();
        fz = _mm256_setzero_ps();

        __m256 x1, y1, z1, vx, vy, vz;
        x1 = _mm256_loadu_ps(p->x + i);
        y1 = _mm256_loadu_ps(p->y + i);
        z1 = _mm256_loadu_ps(p->z + i);
        vx = _mm256_loadu_ps(p->vx + i);
        vy = _mm256_loadu_ps(p->vy + i);
        vz = _mm256_loadu_ps(p->vz + i);

        for (u64 j = 0; j < n; j++) {
            __m256 x2, y2, z2, dx, dy, dz, dx2, dy2, dz2, d_2, d_3_over_2;

            x2 = _mm256_loadu_ps(p->x + j);
            y2 = _mm256_loadu_ps(p->y + j);
            z2 = _mm256_loadu_ps(p->z + j);

            dx = _mm256_sub_ps(x2, x1);
            dy = _mm256_sub_ps(y2, y1);
            dz = _mm256_sub_ps(z2, z1);

            dx2 = _mm256_mul_ps(dx, dx);
            dy2 = _mm256_mul_ps(dy, dy);
            dz2 = _mm256_mul_ps(dz, dz);

            d_2 = _mm256_add_ps(dx2, dy2);
            d_2 = _mm256_add_ps(d_2, dz2);
            d_2 = _mm256_add_ps(d_2, sof);

            d_2 = _mm256_sqrt_ps(d_2);

            d_3_over_2 = _mm256_mul_ps(d_2, d_2);
            d_3_over_2 = _mm256_mul_ps(d_3_over_2, d_2);

            fx = _mm256_fmadd_ps(dx, d_3_over_2, fx);
            fy = _mm256_fmadd_ps(dy, d_3_over_2, fy);
            fz = _mm256_fmadd_ps(dz, d_3_over_2, fz);
        }

        vx = _mm256_fmadd_ps(dtvec, fx, vx);
        vy = _mm256_fmadd_ps(dtvec, fy, vy);
        vz = _mm256_fmadd_ps(dtvec, fz, vz);

        _mm256_storeu_ps(p->vx + i, vx);
        _mm256_storeu_ps(p->vy + i, vy);
        _mm256_storeu_ps(p->vz + i, vz);
    
}

  for (u64 i = 0; i < n; i++) {
    p->x[i] += dt * p->vx[i];
    p->y[i] += dt * p->vy[i];
    p->z[i] += dt * p->vz[i];
  }
}

int main(int argc, char **argv) {
  const u64 n = (argc > 1) ? atoll(argv[1]) : 16384;
  const u64 steps = 10;
  const f32 dt = 0.01;

  f64 rate = 0.0, drate = 0.0;
  const u64 warmup = 3;

  particles_t *particles = malloc(sizeof(*particles));
  particles->x = aligned_alloc(64, sizeof(f32) * n);
  particles->y = aligned_alloc(64, sizeof(f32) * n);
  particles->z = aligned_alloc(64, sizeof(f32) * n);
  particles->vx = aligned_alloc(64, sizeof(f32) * n);
  particles->vy = aligned_alloc(64, sizeof(f32) * n);
  particles->vz = aligned_alloc(64, sizeof(f32) * n);
  init_particles(particles, n);
 const u64 s = sizeof(particles_t) * n;
  printf("\n\033[1mTotal memory size:\033[0m %llu B, %llu KiB, %llu MiB\n\n", sizeof(f32) * 6 * n, sizeof(f32) * 6 * n >> 10, sizeof(f32) * 6 * n >> 20);
  
  printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s", "GFLOP/s"); fflush(stdout);

  for (u64 i = 0; i < steps; i++) {
    const f64 start = omp_get_wtime();

    move_particles(particles, dt, n);

    const f64 end = omp_get_wtime();
    const f32 h1 = (f32)(n) * (f32)(n - 1);
    const f32 h2 = (23.0 * h1 + 3.0 * (f32)n) * 1e-9;
      
    if (i >= warmup) {
      rate += h2 / (end - start);
      drate += (h2 * h2) / ((end - start) * (end - start));
    }

    printf("%5llu %10.3e %10.3e %8.1f %s\n",
      i,
      (end - start),
      h1 / (end - start),
      h2 / (end - start),
      (i < warmup) ? "*" : "");

    fflush(stdout);
  }

  rate /= (f64)(steps - warmup);
  drate = sqrt(drate / (f64)(steps - warmup) - (rate * rate));

  printf("-----------------------------------------------------\n");
  printf("\033[1m%s %4s \033[42m%10.1lf +- %.1lf GFLOP/s\033[0m\n",
    "Average performance:", "", rate, drate);
  printf("-----------------------------------------------------\n");

  free(particles->x);
  free(particles->y);
  free(particles->z);
  free(particles->vx);
  free(particles->vy);
  free(particles->vz);
  free(particles);
  return 0;
}

// ============================================================================
// N-body 3D - Step 01: SoA + aligned allocation
// ----------------------------------------------------------------------------
// Switch from Array-of-Structures to Structure-of-Arrays and align each
// coordinate array on a 64-byte (cache-line) boundary. Enables contiguous,
// vectorization-friendly memory access in the inner loop.
// ============================================================================
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

  for (u64 i = 0; i < n; i++) {
    f32 fx = 0.0;
    f32 fy = 0.0;
    f32 fz = 0.0;

    for (u64 j = 0; j < n; j++) {
      const f32 dx = p->x[j] - p->x[i];
      const f32 dy = p->y[j] - p->y[i];
      const f32 dz = p->z[j] - p->z[i];
      const f32 d_2 = (dx * dx) + (dy * dy) + (dz * dz) + softening;
      const f32 d_3_over_2 = pow(d_2, 3.0 / 2.0);

      fx += dx / d_3_over_2;
      fy += dy / d_3_over_2;
      fz += dz / d_3_over_2;
    }

    p->vx[i] += dt * fx;
    p->vy[i] += dt * fy;
    p->vz[i] += dt * fz;
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
  particles->x  = aligned_alloc(64, sizeof(f32) * n);
  particles->y  = aligned_alloc(64, sizeof(f32) * n);
  particles->z  = aligned_alloc(64, sizeof(f32) * n);
  particles->vx = aligned_alloc(64, sizeof(f32) * n);
  particles->vy = aligned_alloc(64, sizeof(f32) * n);
  particles->vz = aligned_alloc(64, sizeof(f32) * n);
  init_particles(particles, n);

  printf("\n\033[1mTotal memory size:\033[0m %llu B, %llu KiB, %llu MiB\n\n",
    sizeof(f32) * 6 * n, sizeof(f32) * 6 * n >> 10, sizeof(f32) * 6 * n >> 20);
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
      i, (end - start), h1 / (end - start), h2 / (end - start),
      (i < warmup) ? "*" : "");
    fflush(stdout);
  }

  rate /= (f64)(steps - warmup);
  drate = sqrt(drate / (f64)(steps - warmup) - (rate * rate));

  printf("-----------------------------------------------------\n");
  printf("\033[1m%s %4s \033[42m%10.1lf +- %.1lf GFLOP/s\033[0m\n",
    "Average performance:", "", rate, drate);
  printf("-----------------------------------------------------\n");

  free(particles->x);  free(particles->y);  free(particles->z);
  free(particles->vx); free(particles->vy); free(particles->vz);
  free(particles);
  return 0;
}

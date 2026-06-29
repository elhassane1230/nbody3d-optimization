// ============================================================================
// N-body 3D - Step 07: cache blocking / tiling
// ----------------------------------------------------------------------------
// Tiles the i-loop and distributes tiles across threads manually.
//
// !!! KNOWN ISSUES (see docs/known-issues.md) - kept as originally benchmarked:
//   1. The unrolled reads use x[j] for all 4 lanes instead of x[j+0..3].
//   2. Integer division in the manual tiling can drop the last tiles.
//   3. The position-update loop is missing, so positions never advance.
// These do not affect compilation but make the result physically incorrect.
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
  const int tile_size= 64 ;
  // Using Parallelization for i loop
    #pragma omp parallel
{

    const int num_threads = omp_get_num_threads();
    const int thread_id = omp_get_thread_num();
    const int tiles_per_thread = (n + tile_size - 1) / tile_size / num_threads;
    const int start_tile = thread_id * tiles_per_thread;
    const int end_tile = (thread_id + 1) * tiles_per_thread;

   for (int tile = start_tile; tile < end_tile; tile++) {
      const int tile_start_index = tile * tile_size;
      const int tile_end_index = fmin((tile + 1) * tile_size, n);
	    // Travailler avec les particules dans la tuile
      for (u64 i = tile_start_index; i < tile_end_index; i++) {
    		f32 fx = 0.0;
   		f32 fy = 0.0;
    		f32 fz = 0.0;
    		 // Using Parallelization SIMD for j loop
    		#pragma omp simd reduction(+:fx,fy,fz) 
    		// Loop unrolling with factor 4
    		for (u64 j = 0; j < n-3; j+=4) {

      			const f32 dx1 = p->x[j] - p->x[i];
      			const f32 dy1 = p->y[j] - p->y[i];
      			const f32 dz1 = p->z[j] - p->z[i];
      			const f32 dx2 = p->x[j] - p->x[i];
      			const f32 dy2 = p->y[j] - p->y[i];
      			const f32 dz2 = p->z[j] - p->z[i];
      			const f32 dx3 = p->x[j] - p->x[i];
      			const f32 dy3 = p->y[j] - p->y[i];
      			const f32 dz3 = p->z[j] - p->z[i];
      			const f32 dx4 = p->x[j] - p->x[i];
      			const f32 dy4 = p->y[j] - p->y[i];
      			const f32 dz4 = p->z[j] - p->z[i];

      const f32 d_2_1 = (dx1 * dx1) + (dy1 * dy1) + (dz1 * dz1) + softening;
      const f32 d_2_2 = (dx2 * dx2) + (dy2 * dy2) + (dz2 * dz2) + softening;
      const f32 d_2_3 = (dx3 * dx3) + (dy3 * dy3) + (dz3 * dz3) + softening;
      const f32 d_2_4 = (dx4 * dx4) + (dy4 * dy4) + (dz4 * dz4) + softening;
      const f32 sqrt_d_2_1 = sqrt(d_2_1);
      const f32 sqrt_d_2_2 = sqrt(d_2_2);
      const f32 sqrt_d_2_3 = sqrt(d_2_3);
      const f32 sqrt_d_2_4 = sqrt(d_2_4);

      const f32 d_3_over_2_1 = sqrt_d_2_1 * sqrt_d_2_1 * sqrt_d_2_1;
      const f32 d_3_over_2_2 = sqrt_d_2_2 * sqrt_d_2_2 * sqrt_d_2_2;
      const f32 d_3_over_2_3 = sqrt_d_2_3 * sqrt_d_2_3 * sqrt_d_2_3;
      const f32 d_3_over_2_4 = sqrt_d_2_4 * sqrt_d_2_4 * sqrt_d_2_4;

      fx += dx1 / d_3_over_2_1;
      fy += dy1 / d_3_over_2_1;
      fz += dz1 / d_3_over_2_1;

      fx += dx2 / d_3_over_2_2;
      fy += dy2 / d_3_over_2_2;
      fz += dz2 / d_3_over_2_2;

      fx += dx3 / d_3_over_2_3;
      fy += dy3 / d_3_over_2_3;
      fz += dz3 / d_3_over_2_3;

      fx += dx4 / d_3_over_2_4;
      fy += dy4 / d_3_over_2_4;
      fz += dz4 / d_3_over_2_4;
    }

    p->vx[i] += dt * fx;
    p->vy[i] += dt * fy;
    p->vz[i] += dt * fz;
   }
  }
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

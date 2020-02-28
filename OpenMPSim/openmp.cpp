#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <list>
#include <vector>
#include "common.h"
#include "omp.h"

extern double size;
#define cutoff 0.01

typedef std::list<particle_t *> container;

struct ExtraParticleInfo {
	struct {
		unsigned i, j;
	} bin;
};

//
//  benchmarking program
//
int main(int argc, char **argv) {
	int navg, nabsavg   = 0, numthreads;
	double dmin, absmin = 1.0, davg, absavg = 0.0;

	if (find_option(argc, argv, "-h") >= 0) {
		printf("Options:\n");
		printf("-h to see this help\n");
		printf("-n <int> to set number of particles\n");
		printf("-o <filename> to specify the output file name\n");
		printf("-s <filename> to specify a summary file name\n");
		printf("-no turns off all correctness checks and particle output\n");
		return 0;
	}

	int n          = read_int(argc, argv, "-n", 1000);
	char *savename = read_string(argc, argv, "-o", NULL);
	char *sumname  = read_string(argc, argv, "-s", NULL);

	FILE *fsave = savename ? fopen(savename, "w") : NULL;
	FILE *fsum  = sumname ? fopen(sumname, "a") : NULL;

	particle_t *particles = (particle_t *) malloc(n * sizeof(particle_t));
	set_size(n);
	init_particles(n, particles);

	unsigned numGrids         = size / cutoff;
	double gridSize           = size / numGrids;
	ExtraParticleInfo *extras = new ExtraParticleInfo[n];
	container *buckets        = new container[numGrids * numGrids];
	omp_lock_t *locks         = new omp_lock_t[numGrids * numGrids];
	printf("NumGrids: %d, ", numGrids);

	for (unsigned p = 0; p < n; p++) {
		ExtraParticleInfo &extra = extras[p];
		extra.bin.i              = particles[p].x / gridSize;
		extra.bin.j              = particles[p].y / gridSize;

		buckets[extra.bin.j + extra.bin.i * numGrids].push_back(particles + p);
	}

#pragma omp parallel for
	for (unsigned i = 0; i < numGrids * numGrids; i++) { omp_init_lock(locks + i); }
	numthreads = omp_get_max_threads();

	//
	//  simulate a number of time steps
	//
	double simulation_time = read_timer();

	for (int step = 0; step < 1000; step++) {
#pragma omp parallel private(dmin)
		{
			navg = 0;
			davg = 0.0;
			dmin = 1.0;
			//
			//  compute all forces
			//
#pragma omp for collapse(2) reduction(+ : navg) reduction(+ : davg) schedule(static)
			for (unsigned i = 0; i < numGrids; i++) {
				for (unsigned j = 0; j < numGrids; j++) {
					container &bucket = buckets[j + i * numGrids];

					for (particle_t *p1 : bucket) {
						p1->ax = p1->ay = 0;
						// Loop through all neighboring buckets
						for (unsigned i2 = max(i, 1) - 1; i2 < i + 2 && i2 < numGrids; i2++) {
							for (unsigned j2 = max(j, 1) - 1; j2 < j + 2 && j2 < numGrids; j2++) {
								container &bucket2 = buckets[j2 + i2 * numGrids];

								for (particle_t *p2 : bucket2) {
									apply_force(*p1, *p2, &dmin, &davg, &navg);
								}
							}
						}
					}
				}
			}

//
//  move particles
//
#pragma omp for schedule(guided)
			for (int p = 0; p < n; p++) {
				move(particles[p]);

				unsigned newBinI         = particles[p].x / gridSize;
				unsigned newBinJ         = particles[p].y / gridSize;
				ExtraParticleInfo &extra = extras[p];

				if (newBinI != extra.bin.i || newBinJ != extra.bin.j) {
					omp_set_lock(locks + extra.bin.j + extra.bin.i * numGrids);
					buckets[extra.bin.j + extra.bin.i * numGrids].remove(particles + p);
					omp_unset_lock(locks + extra.bin.j + extra.bin.i * numGrids);

					extra.bin.i = particles[p].x / gridSize;
					extra.bin.j = particles[p].y / gridSize;

					omp_set_lock(locks + extra.bin.j + extra.bin.i * numGrids);
					buckets[extra.bin.j + extra.bin.i * numGrids].push_back(particles + p);
					omp_unset_lock(locks + extra.bin.j + extra.bin.i * numGrids);
				}
			}

			if (find_option(argc, argv, "-no") == -1) {
//
//  compute statistical data
//
#pragma omp master
				if (navg) {
					absavg += davg / navg;
					nabsavg++;
				}

#pragma omp critical
				if (dmin < absmin) absmin = dmin;

//
//  save if necessary
//
#pragma omp master
				if (fsave && (step % SAVEFREQ) == 0) save(fsave, n, particles);
			}
		}
	}
	// }
	simulation_time = read_timer() - simulation_time;

	printf("n = %d,threads = %d, simulation time = %g seconds", n, numthreads, simulation_time);

#pragma omp parallel for
	for (unsigned i = 0; i < numGrids * numGrids; i++) { omp_destroy_lock(locks + i); }

	if (find_option(argc, argv, "-no") == -1) {
		if (nabsavg) absavg /= nabsavg;
		//
		//  -the minimum distance absmin between 2 particles during the run of the simulation
		//  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with
		//  typical values between .7-.8 -A simulation were particles don't interact correctly will
		//  be less than 0.4 (of cutoff) with typical values between .01-.05
		//
		//  -The average distance absavg is ~.95 when most particles are interacting correctly and
		//  ~.66 when no particles are interacting
		//
		printf(", absmin = %lf, absavg = %lf", absmin, absavg);
		if (absmin < 0.4)
			printf(
			    "\nThe minimum distance is below 0.4 meaning that some particle is not "
			    "interacting");
		if (absavg < 0.8)
			printf(
			    "\nThe average distance is below 0.8 meaning that most particles are not "
			    "interacting");
	}
	printf("\n");

	//
	// Printing summary data
	//
	if (fsum) fprintf(fsum, "%d %d %g\n", n, numthreads, simulation_time);

	//
	// Clearing space
	//
	if (fsum) fclose(fsum);

	free(particles);
	if (fsave) fclose(fsave);

	delete[] buckets;
	delete[] extras;

	return 0;
}

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <list>
#include <vector>
#include "common.h"

extern double size;
#define cutoff 0.01

typedef std::list<particle_t *> container;

//
//  benchmarking program
//
int main(int argc, char **argv) {
	int navg, nabsavg = 0;
	double davg, dmin, absmin = 1.0, absavg = 0.0;

	if (find_option(argc, argv, "-h") >= 0) {
		printf("Options:\n");
		printf("-h to see this help\n");
		printf("-n <int> to set the number of particles\n");
		printf("-o <filename> to specify the output file name\n");
		printf("-s <filename> to specify a summary file name\n");
		printf("-no turns off all correctness checks and particle output\n");
		return 0;
	}

	int n = read_int(argc, argv, "-n", 1000);

	char *savename = read_string(argc, argv, "-o", NULL);
	char *sumname  = read_string(argc, argv, "-s", NULL);

	FILE *fsave = savename ? fopen(savename, "w") : NULL;
	FILE *fsum  = sumname ? fopen(sumname, "a") : NULL;

	particle_t *particles = (particle_t *) malloc(n * sizeof(particle_t));
	set_size(n);
	init_particles(n, particles);

	unsigned numGrids  = size / cutoff / 3;
	double gridSize    = size / numGrids;
	container *buckets = new container[numGrids * numGrids];

	for (unsigned p = 0; p < n; p++) {
		unsigned i = particles[p].x / gridSize;
		unsigned j = particles[p].y / gridSize;

		buckets[j + i * numGrids].push_back(particles + p);
	}

	//
	//  simulate a number of time steps
	//

	double simulation_time = read_timer();

	for (int step = 0; step < NSTEPS; step++) {
		navg = 0;
		davg = 0.0;
		dmin = 1.0;
		//
		//  compute forces
		//

		// Loop through all buckets
		for (unsigned i = 0; i < numGrids; i++) {
			for (unsigned j = 0; j < numGrids; j++) {
				container &bucket = buckets[j + i * numGrids];

				for (particle_t *p1 : bucket) {
					p1->ax = p1->ay = 0;
					// Loop through all neighboring buckets, excluding those we will check in the
					// future
					for (unsigned i2 = max(i, 1) - 1; i2 < i + 2 && i2 < numGrids; i2++) {
						for (unsigned j2 = max(j, 1) - 1; j2 < j + 2 && j2 < numGrids; j2++) {
							container &bucket2 = buckets[j2 + i2 * numGrids];

							// We know bucket isn't empty, but bucket2 might be, so use it as the
							// outer loop
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
		for (int i = 0; i < n; i++) move(particles[i]);

		// Now swap buckets
		for (unsigned i = 0; i < numGrids; i++) {
			for (unsigned j = 0; j < numGrids; j++) {
				container &bucket     = buckets[j + i * numGrids];
				container::iterator p = bucket.begin();

				while (p != bucket.end()) {
					// Calculate which bucket this particle should belong to
					unsigned iReal = (*p)->x / gridSize;
					unsigned jReal = (*p)->y / gridSize;

					// If it's a different bucket, move it to that bucket and delete from this one
					if (iReal != i || jReal != j) {
						buckets[jReal + iReal * numGrids].push_back(*p);
						p = bucket.erase(p);
					} else {
						p++;
					}
				}
			}
		}

		if (find_option(argc, argv, "-no") == -1) {
			//
			// Computing statistical data
			//
			if (navg) {
				absavg += davg / navg;
				nabsavg++;
			}
			if (dmin < absmin) absmin = dmin;

			//
			//  save if necessary
			//
			if (fsave && (step % SAVEFREQ) == 0) save(fsave, n, particles);
		}
	}
	simulation_time = read_timer() - simulation_time;

	printf("n = %d, simulation time = %g seconds", n, simulation_time);

	if (find_option(argc, argv, "-no") == -1) {
		if (nabsavg) absavg /= nabsavg;
		//
		//  -the minimum distance absmin between 2 particles during the run of the
		//  simulation -A Correct simulation will have particles stay at greater
		//  than 0.4 (of cutoff) with typical values between .7-.8 -A simulation
		//  were particles don't interact correctly will be less than 0.4 (of
		//  cutoff) with typical values between .01-.05
		//
		//  -The average distance absavg is ~.95 when most particles are interacting
		//  correctly and ~.66 when no particles are interacting
		//
		printf(", absmin = %lf, absavg = %lf", absmin, absavg);
		if (absmin < 0.4)
			printf(
			    "\nThe minimum distance is below 0.4 meaning that some particle is "
			    "not interacting");
		if (absavg < 0.8)
			printf(
			    "\nThe average distance is below 0.8 meaning that most particles are "
			    "not interacting");
	}
	printf("\n");

	//
	// Printing summary data
	//
	if (fsum) fprintf(fsum, "%d %g\n", n, simulation_time);

	//
	// Clearing space
	//
	if (fsum) fclose(fsum);
	free(particles);
	if (fsave) fclose(fsave);

	delete[] buckets;

	return 0;
}

#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <list>
#include <vector>
#include "common.h"

extern double size;
#define cutoff 0.01
#define NUM_PARTICLES_PER_BUCKET 1.0

struct ExtraParticleInfo {
	struct {
		unsigned i, j;
	} bin;

	ExtraParticleInfo() {}

	ExtraParticleInfo(unsigned i, unsigned j) {
		bin.i = i;
		bin.j = j;
	}
};

struct PackedParticle {
	struct {
		double x, y;
	} position;

	struct {
		double x, y;
	} velocity;

	PackedParticle(particle_t &p) {
		position.x = p.x;
		position.y = p.y;
		velocity.x = p.vx;
		velocity.y = p.vy;
	}

	PackedParticle() {}

	operator particle_t() const {
		particle_t re;
		re.x  = position.x;
		re.y  = position.y;
		re.vx = velocity.x;
		re.vy = velocity.y;

		return re;
	}
};

struct Dimensions {
	unsigned x, y;
};

void putParticleIntoAppropriateList(
    const PackedParticle &p, std::list<std::pair<particle_t, ExtraParticleInfo>> &localParticles,
    std::vector<std::pair<particle_t, ExtraParticleInfo>> &localGhosts,
    std::list<particle_t *> *buckets, Dimensions gridSize, Dimensions numGrids, unsigned rank) {
	// Now sift out ghost region particles
	if (p.position.y / gridSize.y < rank * numGrids.y ||
	    p.position.y / gridSize.y >= (rank + 1) * numGrids.y) {
		localGhosts.push_back(std::make_pair(
		    p, ExtraParticleInfo(p.position.x / gridSize.x,
		                         p.position.y / gridSize.y - rank * numGrids.y + 1)));

		buckets[localGhosts.back().second.bin.j + localGhosts.back().second.bin.i * numGrids.x]
		    .push_back(&localGhosts.back().first);
	} else {
		// Add all particles which actually belong to us to our master list
		localParticles.push_back(std::make_pair(
		    p, ExtraParticleInfo(p.position.x / gridSize.x,
		                         p.position.y / gridSize.y - rank * numGrids.y + 1)));

		buckets[localParticles.back().second.bin.j +
		        localParticles.back().second.bin.i * numGrids.x]
		    .push_back(&localParticles.back().first);
	}
}

//
//  benchmarking program
//
int main(int argc, char **argv) {
	int navg, nabsavg   = 0;
	double dmin, absmin = 1.0, davg, absavg = 0.0;
	double rdavg, rdmin;
	int rnavg;

	//
	//  process command line parameters
	//
	if (find_option(argc, argv, "-h") >= 0) {
		printf("Options:\n");
		printf("-h to see this help\n");
		printf("-n <int> to set the number of particles\n");
		printf("-o <filename> to specify the output file name\n");
		printf("-s <filename> to specify a summary file name\n");
		printf("-no turns off all correctness checks and particle output\n");
		return 0;
	}

	int n          = read_int(argc, argv, "-n", 1000);
	char *savename = read_string(argc, argv, "-o", NULL);
	char *sumname  = read_string(argc, argv, "-s", NULL);

	//
	//  set up MPI
	//
	int n_proc, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//
	//  allocate generic resources
	//
	FILE *fsave = savename && rank == 0 ? fopen(savename, "w") : NULL;
	FILE *fsum  = sumname && rank == 0 ? fopen(sumname, "a") : NULL;

	particle_t *particles  = (particle_t *) malloc(n * sizeof(particle_t));
	PackedParticle *packed = new PackedParticle[n];

	MPI_Datatype PARTICLE;
	MPI_Type_contiguous(4, MPI_DOUBLE, &PARTICLE);
	MPI_Type_commit(&PARTICLE);

	//
	//  initialize and distribute the particles (that's fine to leave it unoptimized)
	//
	set_size(n);

	double procSize = size / n_proc;
	if (rank == 0) {
		init_particles(n, particles);
		for (unsigned i = 0; i < n; i++) { packed[i] = PackedParticle(particles[i]); }
	}

	delete[] particles;
	MPI_Bcast(packed, n, PARTICLE, 0, MPI_COMM_WORLD);

	std::list<std::pair<particle_t, ExtraParticleInfo>> localParticles;
	std::vector<std::pair<particle_t, ExtraParticleInfo>> localGhosts;

	Dimensions numGrids;  // # of grids on this processor
	Dimensions gridSize;  // size of each grid
	numGrids.x                       = size / cutoff;
	numGrids.y                       = size / n_proc / cutoff;
	gridSize.x                       = size / numGrids.x;
	gridSize.y                       = size / n_proc / numGrids.y;
	std::list<particle_t *> *buckets = new std::list<particle_t *>[numGrids.x * (numGrids.y + 2)];
	// omp_lock_t *locks                = new omp_lock_t[numGrids.x * numGrids.y];

	unsigned bufferSize           = numGrids.x * NUM_PARTICLES_PER_BUCKET;
	PackedParticle *sendBuffer    = new PackedParticle[2 * bufferSize];
	PackedParticle *receiveBuffer = new PackedParticle[2 * bufferSize];

	localGhosts.reserve(bufferSize * 2);

	unsigned sendSize[2];
	MPI_Request requests[4];
	MPI_Status statuses[4];

	for (unsigned i = 0; i < 2; i++) { sendSize[i] = 0; }

	for (unsigned i = 0; i < n; i++) {
		// Discard any particles which don't belong to our process (or our ghost region)
		// +/- 1 to account for ghost regions
		if (packed[i].position.y / gridSize.y < rank * numGrids.y - 1 ||
		    packed[i].position.y / gridSize.y >= (rank + 1) * numGrids.y + 1) {
			continue;
		}

		putParticleIntoAppropriateList(packed[i], localParticles, localGhosts, buckets, gridSize,
		                               numGrids, rank);
	}

	delete[] packed;

	//
	//  simulate a number of time steps
	//
	double simulation_time = read_timer();
	for (int step = 0; step < NSTEPS; step++) {
		navg = 0;
		dmin = 1.0;
		davg = 0.0;
		//
		//  collect all global data locally (not good idea to do)
		//

		//
		//  save current step if necessary (slightly different semantics than in other codes)
		//
		if (find_option(argc, argv, "-no") == -1)
			if (fsave && (step % SAVEFREQ) == 0) save(fsave, n, particles);

		//
		//  compute all forces
		//

		for (unsigned i = 1; i <= numGrids.y; i++) {
			for (unsigned j = 0; j < numGrids.x; j++) {
				std::list<particle_t *> &bucket = buckets[j + i * numGrids.x];

				for (particle_t *p1 : bucket) {
					p1->ax = p1->ay = 0;
					// Loop through all neighboring buckets
					for (unsigned i2 = i - 1; i2 <= i + 1; i2++) {
						for (unsigned j2 = max(j, 1) - 1; j2 < j + 2 && j2 < numGrids.x; j2++) {
							std::list<particle_t *> &bucket2 = buckets[j2 + i2 * numGrids.x];

							for (particle_t *p2 : bucket2) {
								apply_force(*p1, *p2, &dmin, &davg, &navg);
							}
						}
					}
				}
			}
		}

		if (find_option(argc, argv, "-no") == -1) {
			MPI_Reduce(&davg, &rdavg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(&navg, &rnavg, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(&dmin, &rdmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

			if (rank == 0) {
				//
				// Computing statistical data
				//
				if (rnavg) {
					absavg += rdavg / rnavg;
					nabsavg++;
				}
				if (rdmin < absmin) absmin = rdmin;
			}
		}

		//
		//  move particles
		//
		for (auto i = localParticles.begin(); i != localParticles.end();) {
			particle_t &p            = i->first;
			ExtraParticleInfo &extra = i->second;
			move(p);
			unsigned newBinIGlobal = p.x / gridSize.x;
			unsigned newBinJGlobal = p.y / gridSize.y;

			// If the particle has moved off the process, remove from master list and bin and add to
			// send buffer
			if (newBinIGlobal < rank * numGrids.y) {
				buckets[extra.bin.j + extra.bin.i * numGrids.x].remove(&p);

				sendBuffer[sendSize[0]] = p;
				++sendSize[0];

				i = localParticles.erase(i);
			} else if (newBinIGlobal >= (rank + 1) * numGrids.y) {
				buckets[extra.bin.j + extra.bin.i * numGrids.x].remove(&p);

				sendBuffer[bufferSize + sendSize[1]] = p;
				++sendSize[1];

				i = localParticles.erase(i);
			} else {
				// Otherwise, we belong to a valid bin, so calculate that
				unsigned newBinI = newBinIGlobal - rank * numGrids.y;
				unsigned newBinJ = newBinJGlobal - (rank + 1) * numGrids.y + 1;

				// Now, move to a new bin if neccesary
				if (newBinI != extra.bin.i || newBinJ != extra.bin.j) {
					buckets[extra.bin.j + extra.bin.i * numGrids.x].remove(&p);

					extra.bin.i = newBinI;
					extra.bin.j = newBinJ;

					buckets[extra.bin.j + extra.bin.i * numGrids.x].push_back(&p);
				}

				++i;
			}
		}

		unsigned numRequests = 0;
		// Upper Neighbor
		if (rank > 0) {
			MPI_Isend(sendBuffer, sendSize[0], PARTICLE, rank - 1, 0, MPI_COMM_WORLD, requests);
			MPI_Irecv(receiveBuffer, bufferSize, PARTICLE, rank - 1, 0, MPI_COMM_WORLD,
			          requests + 1);

			numRequests += 2;
		}

		// Lower neighbor
		if (rank < (n_proc - 1)) {
			MPI_Isend(sendBuffer + bufferSize, sendSize[1], PARTICLE, rank + 1, 0, MPI_COMM_WORLD,
			          requests + 2);
			MPI_Irecv(receiveBuffer + bufferSize, bufferSize, PARTICLE, rank + 1, 0, MPI_COMM_WORLD,
			          requests + 3);

			numRequests += 2;
		}

		// Now that we have sent all of our stuff and are waiting to receive our stuff,
		// clear buffers, ghost particle list, and ghost region buckets.
		for (unsigned i = 0; i < 2; i++) { sendSize[i] = 0; }
		localGhosts.clear();

		for (unsigned j = 0; j < numGrids.x; j++) {
			buckets[j].clear();
			buckets[j + (numGrids.y + 1) * numGrids.x].clear();
		}

		MPI_Waitall(numRequests, requests + (rank > 0 ? 2 : 0), statuses);

		int numReceived;

		// Receive ghost particles from our upper neighbor
		if (rank > 0) {
			MPI_Get_count(statuses + 1, PARTICLE, &numReceived);

			for (unsigned i = 0; i < numReceived; i++) {
				putParticleIntoAppropriateList(receiveBuffer[i], localParticles, localGhosts,
				                               buckets, gridSize, numGrids, rank);
			}
		}

		// Receive ghost particles from our lower neighbor
		if (rank < n_proc - 1) {
			MPI_Get_count(statuses + 3, PARTICLE, &numReceived);

			for (unsigned i = 0; i < numReceived; i++) {
				putParticleIntoAppropriateList(receiveBuffer[i + bufferSize], localParticles,
				                               localGhosts, buckets, gridSize, numGrids, rank);
			}
		}
	}
	simulation_time = read_timer() - simulation_time;

	if (rank == 0) {
		printf("n = %d, simulation time = %g seconds", n, simulation_time);

		if (find_option(argc, argv, "-no") == -1) {
			if (nabsavg) absavg /= nabsavg;
			//
			//  -the minimum distance absmin between 2 particles during the run of the
			//  simulation -A Correct simulation will have particles stay at greater than 0.4
			//  (of cutoff) with typical values between .7-.8 -A simulation were particles don't
			//  interact correctly will be less than 0.4 (of cutoff) with typical values between
			//  .01-.05
			//
			//  -The average distance absavg is ~.95 when most particles are interacting
			//  correctly and ~.66 when no particles are interacting
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
		if (fsum) fprintf(fsum, "%d %d %g\n", n, n_proc, simulation_time);
	}

	//
	//  release resources
	//
	if (fsum) fclose(fsum);
	if (fsave) fclose(fsave);

	delete[] buckets;
	delete[] sendBuffer;
	delete[] receiveBuffer;

	MPI_Finalize();

	return 0;
}

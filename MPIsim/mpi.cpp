#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <list>
#include "common.h"

extern double size;
#define cutoff 0.01
#define NUM_PARTICLES_PER_BUCKET 1.0

typedef std::list<particle_t *> container;

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

	// unsigned procHeight =
	//     (sqrt(n_proc) == floor(sqrt(n_proc))) ? sqrt(n_proc) : sqrt(n_proc / 2) * 2;
	// unsigned procWidth = n_proc / procHeight;
	const unsigned procHeight = n_proc;
	const unsigned procWidth  = 1;
	// unsigned numNeighbors     = rank % (n_proc - 1) == 0) ? 1 : 2;
	const unsigned numNeighbors = 2;

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
	struct {
		unsigned x, y;
	} numGrids;  // # of grids on this processor
	numGrids.x = size / procWidth / cutoff;
	numGrids.y = size / procHeight / cutoff;
	struct {
		unsigned x, y;
	} gridSize;  // size of each grid
	gridSize.x                       = size / procWidth / numGrids.x;
	gridSize.y                       = size / procHeight / numGrids.y;
	std::list<particle_t *> *buckets = new std::list<particle_t *>[numGrids.x * (numGrids.y + 2)];
	// omp_lock_t *locks                = new omp_lock_t[numGrids.x * numGrids.y];

	unsigned bufferSize           = numNeighbors * numGrids.x * NUM_PARTICLES_PER_BUCKET;
	PackedParticle *sendBuffer    = new PackedParticle[bufferSize];
	PackedParticle *receiveBuffer = new PackedParticle[bufferSize];

	unsigned *sendSize    = new unsigned[numNeighbors];
	MPI_Request *requests = new MPI_Request[numNeighbors];

	for (unsigned i = 0; i < numNeighbors; i++) { sendSize[i] = 0; }

	for (unsigned i = 0; i < n; i++) {
		if (packed[i].position.x / gridSize.x < (rank % procWidth) * numGrids.x ||
		    packed[i].position.x / gridSize.x >= ((rank % procWidth) + 1) * numGrids.x ||
		    packed[i].position.y / gridSize.y < (rank / procWidth) * numGrids.y ||
		    packed[i].position.y / gridSize.y >= ((rank / procWidth) + 1) * numGrids.y) {
			continue;
		}

		localParticles.push_back(std::make_pair(
		    packed[i],
		    ExtraParticleInfo(
		        packed[i].position.x / gridSize.x - (rank % procWidth) * numGrids.x,
		        packed[i].position.y / gridSize.y - (rank / procWidth) * numGrids.y + 1)));

		buckets[localParticles.back().second.bin.i +
		        localParticles.back().second.bin.j * numGrids.x]
		    .push_back(&localParticles.back().first);
	}

	delete[] packed;

	// Upper neighbor
	if (rank / procWidth > 0) {
		for (unsigned i = 0; i < numGrids.x; i++) {
			for (particle_t *p : buckets[i]) {
				sendBuffer[sendSize[0]] = *p;
				++sendSize[0];
			}
		}
	}

	// Lower neighbor
	if (rank / procWidth < (procHeight - 1)) {
		for (unsigned i = 0; i < numGrids.x; i++) {
			for (particle_t *p : buckets[i + numGrids.x * (numGrids.y - 1)]) {
				// unsigned index = sendSize[numNeighbors - 1] +
				//                  (numNeighbors - 1) * numGrids.x * NUM_PARTICLES_PER_BUCKET;
				sendBuffer[sendSize[1]] = *p;
				++sendSize[numNeighbors - 1];
			}
		}
	}

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
		// Upper Neighbor
		if (rank / procWidth > 0) {
			MPI_Isend(sendBuffer, sendSize[0], PARTICLE, rank - procWidth, 0, MPI_COMM_WORLD,
			          requests);
		}

		// Lower neighbor
		if (rank / procWidth < (procHeight - 1)) {
			// unsigned index = (numNeighbors - 1) * numGrids.x * NUM_PARTICLES_PER_BUCKET;
			unsigned index = numGrids.x * NUM_PARTICLES_PER_BUCKET;
			MPI_Isend(sendBuffer + index, sendSize[1], PARTICLE, rank + procWidth, 0,
			          MPI_COMM_WORLD, requests + 1);
		}

		//
		//  save current step if necessary (slightly different semantics than in other codes)
		//
		if (find_option(argc, argv, "-no") == -1)
			if (fsave && (step % SAVEFREQ) == 0) save(fsave, n, particles);

		//
		//  compute all forces
		//
		// for (int i = 0; i < nlocal; i++) {
		// 	local[i].ax = local[i].ay = 0;
		// 	for (int j = 0; j < n; j++) apply_force(local[i], particles[j], &dmin, &davg, &navg);
		// }

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
		// for (int i = 0; i < nlocal; i++) move(local[i]);
	}
	simulation_time = read_timer() - simulation_time;

	if (rank == 0) {
		printf("n = %d, simulation time = %g seconds", n, simulation_time);

		if (find_option(argc, argv, "-no") == -1) {
			if (nabsavg) absavg /= nabsavg;
			//
			//  -the minimum distance absmin between 2 particles during the run of the simulation
			//  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with
			//  typical values between .7-.8 -A simulation were particles don't interact correctly
			//  will be less than 0.4 (of cutoff) with typical values between .01-.05
			//
			//  -The average distance absavg is ~.95 when most particles are interacting correctly
			//  and ~.66 when no particles are interacting
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
	free(particles);
	if (fsave) fclose(fsave);

	delete[] buckets;
	delete[] sendBuffer;
	delete[] receiveBuffer;
	delete[] sendSize;
	delete[] requests;

	MPI_Finalize();

	return 0;
}

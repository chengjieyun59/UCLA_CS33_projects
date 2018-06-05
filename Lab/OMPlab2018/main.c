#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "ompDeblur.h"
#include "seqDeblur.h"

int CompareResults(double* a, double* b, int xM, int yM, int zM)
{
	int i;
	for(i = 0; i < xM * yM * zM; i++)
	{
		if(a[i] != b[i])
		{
			printf("bad result at %d", i);
			return 0;
		}
	}
	return 1;
}

double* LoadMatrix(char* fname, int* xM, int* yM, int* zM)
{
	FILE* in;
	int i = 0;
	double* data;
	char line[256];
	assert(xM);
	assert(yM);
	assert(zM);
	assert(in = fopen(fname, "r"));
	assert(fgets(line, 256, in) != NULL && sscanf(line, "%d", xM) == 1);
	assert(fgets(line, 256, in) != NULL && sscanf(line, "%d", yM) == 1);
	assert(fgets(line, 256, in) != NULL && sscanf(line, "%d", zM) == 1);
	assert(*xM > 2 && *yM > 2 && *zM > 2);
	assert(data = (double*)malloc(sizeof(double) * *xM * *yM * *zM));
	for(i = 0; i < *xM * *yM * *zM && fgets(line, 256, in) != NULL; i++)
	{
		assert(sscanf(line, "%lf", data + i) == 1);
	}
	assert(i == *xM * *yM * *zM);
	assert(fclose(in) == 0);
	return data;
}

double DiffTime(struct timeval t1, struct timeval t2)
{
	return (double)(t2.tv_sec - t1.tv_sec) + ((double)(t2.tv_usec - t1.tv_usec)) * 1e-6;
}

int main()
{
	double seqDelay;
	double ompDelay;
	struct timeval start, end;
	int xMu, yMu, zMu, xMf, yMf, zMf;

	double* uSeq;
	double* uOmp;
	double* f;

	printf("Loading data...");

	uSeq = LoadMatrix("u.file", &xMu, &yMu, &zMu);
	uOmp = LoadMatrix("u.file", &xMu, &yMu, &zMu);
	f = LoadMatrix("f.file", &xMf, &yMf, &zMf);
	assert(xMu == xMf && yMu == yMf && zMu == zMf);
	printf("Verifying load integrity...\n");

	printf("Starting the deblur parallelization test\n");

	SEQ_Initialize(xMu, yMu, zMu);
	gettimeofday(&start, 0);
	SEQ_Deblur(uSeq, f, 10, 0.001, 0.085 / (0.008 * 0.008), 0.008, 1.8);
	gettimeofday(&end, 0);
	seqDelay = DiffTime(start, end);
	SEQ_Finish();

	printf("Finished Sequential test, starting parallel test...\n");

	OMP_Initialize(xMu, yMu, zMu);
	gettimeofday(&start, 0);
	OMP_Deblur(uOmp, f, 10, 0.001, 0.085 / (0.008 * 0.008), 0.008, 1.8);
	gettimeofday(&end, 0);
	ompDelay = DiffTime(start, end);
	OMP_Finish();

	assert(CompareResults(uSeq, uOmp, xMu, yMu, zMu));

	printf("Sequential took %lf time units\nOpenMP took %lf time units\nThis resulted in a %fx speed-up\n", seqDelay, ompDelay, (float)(seqDelay / ompDelay));
	printf("Ending the deblur parallelization test\n");

	fprintf(stderr, "%lf", seqDelay / ompDelay);
	return 0;
}

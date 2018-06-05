//Sequential version, treat this as your golden copy
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

int SEQ_xMax;
#define xMax SEQ_xMax
int SEQ_yMax;
#define yMax SEQ_yMax
int SEQ_zMax;
#define zMax SEQ_zMax

int SEQ_Index(int x, int y, int z)
{
	return ((z * yMax + y) * xMax + x);
}
#define Index(x, y, z) SEQ_Index(x, y, z)

double SEQ_SQR(double x)
{
	return pow(x, 2.0);
}
#define SQR(x) SEQ_SQR(x)

double* SEQ_conv;
double* SEQ_g;

void SEQ_Initialize(int xM, int yM, int zM)
{
	xMax = xM;
	yMax = yM;
	zMax = zM;
	assert(SEQ_conv = (double*)malloc(sizeof(double) * xMax * yMax * zMax));
	assert(SEQ_g = (double*)malloc(sizeof(double) * xMax * yMax * zMax));
}
void SEQ_Finish()
{
	free(SEQ_conv);
	free(SEQ_g);
}
void SEQ_GaussianBlur(double *u, double Ksigma, int stepCount)
{
	double lambda = (Ksigma * Ksigma) / (double)(2 * stepCount);
	double nu = (1.0 + 2.0*lambda - sqrt(1.0 + 4.0*lambda))/(2.0*lambda);
	int x, y, z, step;
	double boundryScale = 1.0 / (1.0 - nu);
	double postScale = pow(nu / lambda, (double)(3 * stepCount));

	for(step = 0; step < stepCount; step++)
	{
		for(y = 0; y < yMax; y++)
		{
			for(z = 0; z < zMax; z++)
			{
				u[Index(0, y, z)] *= boundryScale;
			}
		}
		for(x = 1; x < xMax; x++)
		{
			for(y = 0; y < yMax; y++)
			{
				for( z = 0; z < zMax; z++)
				{
					u[Index(x, y, z)] += u[Index(x - 1, y, z)] * nu;
				}
			}
		}
		for(y = 0; y < yMax; y++)
		{
			for(z = 0; z < zMax; z++)
			{
				u[Index(0, y, z)] *= boundryScale;
			}
		}
		for(x = xMax - 2; x >= 0; x--)
		{
			for(y = 0; y < yMax; y++)
			{
				for(z = 0; z < zMax; z++)
				{
					u[Index(x, y, z)] += u[Index(x + 1, y, z)] * nu;
				}
			}
		}
		for(x = 0; x < xMax; x++)
		{
			for(z = 0; z < zMax; z++)
			{
				u[Index(x, 0, z)] *= boundryScale;
			}
		}
		for(x = 0; x < xMax; x++)
		{
			for(y = 1; y < yMax; y++)
			{
				for(z = 0; z < zMax; z++)
				{
					u[Index(x, y, z)] += u[Index(x, y - 1, z)] * nu;
				}
			}
		}
		for(x = 0; x < xMax; x++)
		{
			for(z = 0; z < zMax; z++)
			{
				u[Index(x, yMax - 1, z)] *= boundryScale;
			}
		}
		for(x = 0; x < xMax; x++)
		{
			for(y = yMax - 2; y >= 0; y--)
			{
				for(z = 0; z < zMax; z++)
				{
					u[Index(x, y, z)] += u[Index(x, y + 1, z)] * nu;
				}
			}
		}
		for(x = 0; x < xMax; x++)
		{
			for(y = 0; y < yMax; y++)
			{
				u[Index(x, y, 0)] *= boundryScale;
			}
		}
		for(x = 0; x < xMax; x++)
		{
			for(y = 0; y < yMax; y++)
			{
				for(z = 1; z < zMax; z++)
				{
					u[Index(x, y, z)] = u[Index(x, y, z - 1)] * nu;
				}
			}
		}
		for(x = 0; x < xMax; x++)
		{
			for(y = 0; y < yMax; y++)
			{
				u[Index(x, y, zMax - 1)] *= boundryScale;
			}
		}
		for(x = 0; x < xMax; x++)
		{
			for(y = 0; y < yMax; y++)
			{
				for(z = zMax - 2; z >= 0; z--)
				{
					u[Index(x, y, z)] += u[Index(x, y, z + 1)] * nu;
				}
			}
		}
	}
	for(x = 0; x < xMax; x++)
	{
		for(y = 0; y < yMax; y++)
		{
			for(z = 0; z < zMax; z++)
			{
				u[Index(x, y, z)] *= postScale;
			}
		}
	}
}
void SEQ_Deblur(double* u, const double* f, int maxIterations, double dt, double gamma, double sigma, double Ksigma)
{
	double epsilon = 1.0e-7;
	double sigma2 = SQR(sigma);
	int x, y, z, iteration;
	int converged = 0;
	int lastConverged = 0;
	int fullyConverged = (xMax - 1) * (yMax - 1) * (zMax - 1);
	double* conv = SEQ_conv;
	double* g = SEQ_g;

	for(iteration = 0; iteration < maxIterations && converged != fullyConverged; iteration++)
	{
		for(x = 1; x < xMax - 1; x++)
		{
			for(y = 1; y < yMax - 1; y++)
			{
				for(z = 1; z < zMax - 1; z++)
				{
					g[Index(x, y, z)] = 1.0 / sqrt(epsilon + 
						SQR(u[Index(x, y, z)] - u[Index(x + 1, y, z)]) + 
						SQR(u[Index(x, y, z)] - u[Index(x - 1, y, z)]) + 
						SQR(u[Index(x, y, z)] - u[Index(x, y + 1, z)]) + 
						SQR(u[Index(x, y, z)] - u[Index(x, y - 1, z)]) + 
						SQR(u[Index(x, y, z)] - u[Index(x, y, z + 1)]) + 
						SQR(u[Index(x, y, z)] - u[Index(x, y, z - 1)]));
				}
			}
		}
		memcpy(conv, u, sizeof(double) * xMax * yMax * zMax);
		SEQ_GaussianBlur(conv, Ksigma, 3);
		for(x = 0; x < xMax; x++)
		{
			for(y = 0; y < yMax; y++)
			{
				for(z = 0; z < zMax; z++)
				{
					double r = conv[Index(x, y, z)] * f[Index(x, y, z)] / sigma2;
					r = (r * (2.38944 + r * (0.950037 + r))) / (4.65314 + r * (2.57541 + r * (1.48937 + r)));
					conv[Index(x, y, z)] -= f[Index(x, y, z)] * r;
				}
			}
		}
		SEQ_GaussianBlur(conv, Ksigma, 3);
		converged = 0;
		for(x = 1; x < xMax - 1; x++)
		{
			for(y = 1; y < yMax - 1; y++)
			{
				for(z = 1; z < zMax - 1; z++)
				{
					double oldVal = u[Index(x, y, z)];
					double newVal = (u[Index(x, y, z)] + dt * ( 
						u[Index(x - 1, y, z)] * g[Index(x - 1, y, z)] + 
						u[Index(x + 1, y, z)] * g[Index(x + 1, y, z)] + 
						u[Index(x, y - 1, z)] * g[Index(x, y - 1, z)] + 
						u[Index(x, y + 1, z)] * g[Index(x, y + 1, z)] + 
						u[Index(x, y, z - 1)] * g[Index(x, y, z - 1)] + 
						u[Index(x, y, z + 1)] * g[Index(x, y, z + 1)] - gamma * conv[Index(x, y, z)])) /
						(1.0 + dt * (g[Index(x + 1, y, z)] + g[Index(x - 1, y, z)] + g[Index(x, y + 1, z)] + g[Index(x, y - 1, z)] + g[Index(x, y, z + 1)] + g[Index(x, y, z - 1)]));
					if(fabs(oldVal - newVal) < epsilon)
					{
						converged++;
					}
					u[Index(x, y, z)] = newVal;
				}
			}
		}
		if(converged > lastConverged)
		{
			printf("%d pixels have converged on iteration %d\n", converged, iteration);
			lastConverged = converged;
		}
	}
}


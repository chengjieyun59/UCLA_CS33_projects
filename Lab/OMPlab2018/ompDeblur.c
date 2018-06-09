//OpenMP version.  Edit and submit only this file.
/* Enter your details below
 * Name : Jie Yun Cheng
 * UCLA ID: 004460366
 * Email id: jycheng@ucla.edu
 * Input: Old files
 */

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

int OMP_xMax;
#define xMax OMP_xMax
int OMP_yMax;
#define yMax OMP_yMax
int OMP_zMax;
#define zMax OMP_zMax

// declare some short functions as inline
int inline OMP_Index(int x, int y, int z)
{
    return ((z * yMax + y) * xMax + x);
}
#define Index(x, y, z) OMP_Index(x, y, z)

double inline OMP_SQR(double x)
{
    return pow(x, 2.0);
}
#define SQR(x) OMP_SQR(x)

double* OMP_conv;
double* OMP_g;

void OMP_Initialize(int xM, int yM, int zM)
{
    xMax = xM;
    yMax = yM;
    zMax = zM;
    assert(OMP_conv = (double*)malloc(sizeof(double) * xMax * yMax * zMax));
    assert(OMP_g = (double*)malloc(sizeof(double) * xMax * yMax * zMax));
}
void OMP_Finish()
{
    free(OMP_conv);
    free(OMP_g);
}
void OMP_GaussianBlur(double *u, double Ksigma, int stepCount)
{
    double lambda = (Ksigma * Ksigma) / (double)(2 * stepCount);
    double nu = (1.0 + 2.0*lambda - sqrt(1.0 + 4.0*lambda))/(2.0*lambda);
    int x, y, z, step;
    double boundryScale = 1.0 / (1.0 - nu);
    double postScale = pow(nu / lambda, (double)(3 * stepCount));
    
    for(step = 0; step < stepCount; step++)
    {
#pragma omp parallel for shared(zMax, yMax, xMax) private(z,y,x) num_threads(16)
        for(z = 0; z < zMax; z++) // combine all groups that have "z = 0; z < zMax; z++"
        {
            for(y = 0; y < yMax; y++) // combine 4 groups that all have "y = 0; y < yMax; y++"
            {
                u[Index(0, y, z)] *= boundryScale;
                for(x = 1; x < xMax; x++)
                {
                    u[Index(x, y, z)] += u[Index(x - 1, y, z)] * nu;
                }
                u[Index(0, y, z)] *= boundryScale;
                for(x = xMax - 2; x >= 0; x--)
                {
                    u[Index(x, y, z)] += u[Index(x + 1, y, z)] * nu;
                }
            }
            for(x = 0; x < xMax; x++) // combine 4 groups that all have "x = 0; x < xMax; x++"
            {
                u[Index(x, 0, z)] *= boundryScale;
                for(y = 1; y < yMax; y++)
                {
                    u[Index(x, y, z)] += u[Index(x, y - 1, z)] * nu;
                }
                u[Index(x, yMax - 1, z)] *= boundryScale;
                for(y = yMax - 2; y >= 0; y--)
                {
                    u[Index(x, y, z)] += u[Index(x, y + 1, z)] * nu;
                }
            }
        }
        
        // below does not iterate through z = 0 to zMax in the inner loop anymore
        #pragma omp parallel for shared(zMax, yMax, xMax) private(z,y,x) num_threads(16)
        for(y = 0; y < yMax; y++)
        {
            for(x = 0; x < xMax; x++)
            {
                u[Index(x, y, 0)] *= boundryScale;
                for(z = 1; z < zMax; z++)
                {
                    u[Index(x, y, z)] = u[Index(x, y, z - 1)] * nu;
                }
            }
        }
        
        #pragma omp parallel for shared(zMax, yMax, xMax) private(y,x) num_threads(16)
        for(y = 0; y < yMax; y++)
        {
            for(x = 0; x < xMax; x++)
            {
                u[Index(x, y, zMax - 1)] *= boundryScale;
            }
        }
        for(z = zMax - 2; z >= 0; z--)
        {
            for(y = 0; y < yMax; y++)
            {
                for(x = 0; x < xMax; x++)
                {
                    u[Index(x, y, z)] += u[Index(x, y, z + 1)] * nu;
                }
            }
        }
    }
    
#pragma omp parallel for private(z,y,x) shared(zMax, yMax, xMax) num_threads(16)
    for(z = 0; z < zMax; z++)
    {
        for(y = 0; y < yMax; y++)
        {
            for(x = 0; x < xMax; x++)
            {
                u[Index(x, y, z)] *= postScale;
            }
        }
    }
}
void OMP_Deblur(double* u, const double* f, int maxIterations, double dt, double gamma, double sigma, double Ksigma)
{
    double epsilon = 1.0e-7;
    double sigma2 = SQR(sigma);
    int x, y, z, iteration;
    int converged = 0;
    int lastConverged = 0;
    int fullyConverged = (xMax - 1) * (yMax - 1) * (zMax - 1);
    double* conv = OMP_conv;
    double* g = OMP_g;
    
    int xMaxless = xMax - 1;
    int yMaxless = yMax - 1;
    int zMaxless = zMax - 1;
    int val0, val1, val2, val3, val4, val5, val6;
    int temp;
    double r, rsquare;
    double oldVal, newVal;
    
    for(iteration = 0; iteration < maxIterations && converged != fullyConverged; iteration++)
    {
        
        
#pragma omp parallel for shared(zMaxless, yMaxless, xMaxless) private(z,y,x, temp, val0, val1, val2, val3, val4, val5, val6) num_threads(16)
        for(z = 1; z < zMaxless; z++)
        {
            for(y = 1; y < yMaxless; y++)
            {
                for(x = 1; x < xMaxless; x++)
                {
                    temp = Index(x,y,z);
                    
                    g[temp] = 1.0 / sqrt(epsilon +
                                         SQR(u[temp] - u[Index(x + 1, y, z)]) +
                                         SQR(u[temp] - u[Index(x - 1, y, z)]) +
                                         SQR(u[temp] - u[Index(x, y + 1, z)]) +
                                         SQR(u[temp] - u[Index(x, y - 1, z)]) +
                                         SQR(u[temp] - u[Index(x, y, z + 1)]) +
                                         SQR(u[temp] - u[Index(x, y, z - 1)]));
                }
            }
        }
        memcpy(conv, u, sizeof(double) * xMax * yMax * zMax);
        OMP_GaussianBlur(conv, Ksigma, 3);
        for(z = 0; z < zMax; z++)
        {
            for(y = 0; y < yMax; y++)
            {
                for(x = 0; x < xMax; x++)
                {
                    temp = Index(x,y,z);
                    r = conv[temp] * f[temp] / sigma2;
                    rsquare = r * r;
                    r = (r * (2.38944 + r * (0.950037 + r))) / (4.65314 + r * (2.57541 + r * (1.48937 + r)));
                    conv[temp] -= f[temp] * r;
                }
            }
        }
        OMP_GaussianBlur(conv, Ksigma, 3);
        converged = 0;
        for(z = 1; z < zMaxless; z++)
        {
            for(y = 1; y < yMaxless; y++)
            {
                for(x = 1; x < xMaxless; x++)
                {
                    val0 = Index(x, y, z);
                    val1 = Index(x - 1, y, z);
                    val2 = Index(x + 1, y, z);
                    val3 = Index(x, y - 1, z);
                    val4 = Index(x, y + 1, z);
                    val5 = Index(x, y, z - 1);
                    val6 = Index(x, y, z + 1);
                    
                    oldVal = u[val0];
                    newVal = (oldVal + dt * (
                                             u[val1] * g[val1] +
                                             u[val2] * g[val2] +
                                             u[val3] * g[val3] +
                                             u[val4] * g[val4] +
                                             u[val5] * g[val5] +
                                             u[val6] * g[val6] - gamma * conv[val0])) /
                    (1.0 + dt * (g[val2] + g[val1] + g[val4] + g[val3] + g[val6] + g[val5]));
                    if(fabs(oldVal - newVal) < epsilon)
                    {
                        converged++;
                    }
                    u[val0] = newVal;
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

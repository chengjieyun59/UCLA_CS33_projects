#ifndef OMP_DEBLUR_H
#define OMP_DEBLUR_H

//OpenMP version INTERFACE, do not change anything here.  You must adhere to this interface.

void OMP_Initialize(int xM, int yM, int zM);
void OMP_Finish();
void OMP_GaussianBlur(double *u, double nu, int stepCount);
void OMP_Deblur(double* u, const double* f, int maxIterations, double dt, double gamma, double sigma, double Ksigma);

#endif

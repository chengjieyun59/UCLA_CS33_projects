#ifndef SEQ_DEBLUR_H
#define SEQ_DEBLUR_H

//Sequential version, treat this as your golden copy

void SEQ_Initialize(int xM, int yM, int zM);
void SEQ_Finish();
void SEQ_GaussianBlur(double *u, double nu, int stepCount);
void SEQ_Deblur(double* u, const double* f, int maxIterations, double dt, double gamma, double sigma, double Ksigma);

#endif

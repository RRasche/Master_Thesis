#ifndef SIMULATION1D_H
#define SIMULATION1D_H

struct Simulation_boundaryVtbl;

typedef struct
{
    size_t size;
    int timesteps;
    int currtime;
	double imp0;
    
    struct  Simulation_boundaryVtbl const * bVtblp;
    
    double* Ezdata;
    double* Hydata;
	
	double * EzCoeff;
	double * EzCoeffH;
    
    
    
} Simulation;

struct Simulation_boundaryVtbl
{
    void (*BoundaryEz)(Simulation* const sim);
    void (*BoundaryHy)(Simulation* const sim);
};


void Simulation_ctor(Simulation* sim, int timesteps, size_t size, double imp0, double loss);
//
//void Simulation_set_Ez(Simulation* sim, int x, double data);
//void Simulation_set_Hy(Simulation* sim, int x, double data);

void Simulation_updateEz(Simulation* sim);
void Simulation_updateHy(Simulation* sim);

int Simulation_run(Simulation* sim);
double* Simulation_get_Ezdata(Simulation* sim);
double* Simulation_get_Hydata(Simulation* sim);

void Simulation_reflecting_boundaryEz_(Simulation* sim);
void Simulation_reflecting_boundaryHy_(Simulation* sim);

void Simulation_absorbing_boundaryEz_(Simulation* sim);
void Simulation_absorbing_boundaryHy_(Simulation* sim);

void Simulation_TFSF_Boundary(Simulation* sim, double* field, int place, double fieldValue);

void Simulation_dtor(Simulation*);

#endif


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "Simulation1D.h"

//Access the component comp of the array Arr in the Simulation sim at the position (x, y, z)
#define SimulationArr2D(sim, Arr, time, x) Arr[(time) * sim->size + (x)]

//Memory allocation with error handling
#define Calloc_check(out_pointer, size, type) \
out_pointer = (type*) calloc(size, sizeof(type));\
if(out_pointer == NULL)\
{\
	perror("Error: ");\
	printf("Memory allocation (calloc) has failed for array of size %lu in line %d \n", size, __LINE__);\
	exit(-1);\
}

#define Malloc_check(out_pointer, size, type) \
out_pointer = (type*) malloc(size * sizeof(type));\
if(out_pointer == NULL)\
{\
	perror("Error: ");\
	printf("Memory allocation (malloc) has failed for array of size %lu in line %d \n", size, __LINE__);\
	exit(-1);\
}

static inline void Simulation_boundaryEz(Simulation* const sim)
{
    (*sim->bVtblp->BoundaryEz)(sim);
    return;
}

static inline void Simulation_boundaryHy(Simulation* const sim)
{
    (*sim->bVtblp->BoundaryHy)(sim);
    return;
}

void Simulation_ctor(Simulation* sim, int timesteps, size_t size, double imp0, double loss)
{
    static struct Simulation_boundaryVtbl bvtbl = {&Simulation_reflecting_boundaryEz_, &Simulation_reflecting_boundaryHy_};
    
    sim->bVtblp = &bvtbl;
    
    sim->timesteps  = timesteps;
    sim->size       = size;
    sim->imp0       = imp0;
    sim->currtime   = 0;
	
	printf("s = %lu\n", size);
	printf("t * s = %lu\n", timesteps*size);
    Calloc_check(sim->Ezdata,timesteps*size, double);
	Calloc_check(sim->Hydata,timesteps*size, double);
	

	Malloc_check(sim->EzCoeff, size, double);
	Malloc_check(sim->EzCoeffH, size, double);
	
	double coeff = (1 - loss)/(1 + loss); // loss = sigma * deltaT / (2 * epsilon)
	
	
	for(int i = 0; i < 100; ++i)
	{
			sim->EzCoeff[i]  = 1.0;
			sim->EzCoeffH[i] = imp0;
	}
	for(int i = 100; i < size; ++i)
	{
		sim->EzCoeff[i]  = coeff;
		sim->EzCoeffH[i] = (imp0 )/(1 + loss);
	}

    return;
}

void Simulation_updateEz(Simulation* sim)
{
    double* Ez  = sim->Ezdata;
    double* Hy  = sim->Hydata;
    int time    = sim->currtime;
    
    sim->bVtblp->BoundaryEz(sim);
    
    for(int x = 1; x < sim->size; ++x)
    {
        double updateZ = 0.0;
        updateZ +=  (SimulationArr2D(sim, Hy, time + 1, x) - SimulationArr2D(sim, Hy, time + 1, x - 1));
		updateZ *= sim->EzCoeffH[x];
        SimulationArr2D(sim, Ez, time + 1, x) = sim->EzCoeff[x] * SimulationArr2D(sim, Ez, time, x) + updateZ;
    }
    return;
}

void Simulation_updateHy(Simulation* sim)
{
    double* Ez  = sim->Ezdata;
    double* Hy  = sim->Hydata;
    int time    = sim->currtime;

    sim->bVtblp->BoundaryHy(sim);
    
    for(int x = 0; x < sim->size - 1; ++x)
    {
        double updateY = 0.0;
        updateY += (SimulationArr2D(sim, Ez, time, x + 1) - SimulationArr2D(sim, Ez, time, x));

        SimulationArr2D(sim, Hy, time + 1, x) = SimulationArr2D(sim, Hy, time, x) + updateY/sim->imp0;
    }

    return;
}


int Simulation_run(Simulation* sim)
{
    sim->currtime = 0;
	double fieldValueEz = 0.0;
	double fieldValueHy = 0.0;
	
    while(sim->currtime < sim->timesteps - 1)
    {	
		//fieldValueEz = exp(-(sim->currtime + 0.5 - (-0.5) - 30) * (sim->currtime + 0.5 - (-0.5) - 30)/100.0);	// The scattering field Hy[49 + 1/2] has to be converted into a total field 
																												// by adding the new field at [t + 1/2, -0.5] 
		//fieldValueHy = -exp(-(sim->currtime - 30) * (sim->currtime - 30)/100.0) / sim->imp0;								// The one Ez in the update for Hy has to be turned into a scattering field
																												// by subtracting the new field from Ez[t, 0]
		SimulationArr2D(sim, sim->Ezdata, sim->currtime, 100) = exp(-(sim->currtime - 10.0) * (sim->currtime - 10.0) / 20.);

        Simulation_updateHy(sim);
		//Simulation_TFSF_Boundary(sim, sim->Hydata, 49, fieldValueHy);
	
        Simulation_updateEz(sim);
        //Simulation_TFSF_Boundary(sim, sim->Ezdata, 50, fieldValueEz);
        ++(sim->currtime);

    }
    return 0;
}

double* Simulation_get_Ezdata(Simulation* sim)
{
    return sim->Ezdata;
}

double* Simulation_get_Hydata(Simulation* sim)
{
    return sim->Hydata;
}


void Simulation_reflecting_boundaryEz_(Simulation* const sim)
{
    return;
}

void Simulation_reflecting_boundaryHy_(Simulation* const sim)
{
    return;
}

void Simulation_absorbing_boundaryEz_(Simulation* const sim)
{
    SimulationArr2D(sim, sim->Ezdata, sim->currtime + 1, 0) = SimulationArr2D(sim, sim->Ezdata, sim->currtime, 1);
    return;
}

void Simulation_absorbing_boundaryHy_(Simulation* const sim)
{
    SimulationArr2D(sim, sim->Hydata, sim->currtime + 1, (sim->size) - 1) = SimulationArr2D(sim, sim->Hydata, sim->currtime, (sim->size) - 2);
    return;
}


void Simulation_TFSF_Boundary(Simulation* sim, double* field, int place, double fieldValue)
{
	SimulationArr2D(sim, field, sim->currtime + 1, place) += fieldValue;
	return;
}

void Simulation_dtor(Simulation* sim)
{
    free(sim->Ezdata);
    free(sim->Hydata);
    
    return;
}



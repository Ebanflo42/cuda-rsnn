#include <stdio>
#include <string>
#include <stdlib>
#include <math>
#include <iostream.h>
#include <vector.h>

#include "cuda-kernel.cu"

using namespace std;

float uniform(float low, float high) {
    return low + (static_cast<float>(rand())/RAND_MAX)*(high - low);
}

class Simulation {
    
    public:

        int nRec;
        int nIn;
       
        int nRef;
        float threshold;
        float tauVoltage;
        float voltageCoeff;

        bool deviceMemoryAllocated;

        int weightUpdatePeriod;

        int netStateSize;
        int wRecSize;
        int wInSize;

        vector<float> wRec;
        vector<float> wIn;
        vector<float> voltages;
        vector<float> spikes;
        vector<float> refractoryBuffer;

        Simulation(int nrec,
                   int nin,
                   int nref,
                   float thr,
                   float tau_v,
                   int update_period);

        void allocate();

        void copyToDevice();

        void simulate(int timesteps);

        void copyFromDevice();

        void free();

    protected:

        float* wInDevice;
        float* wRecDevice;

        float* voltageDevice;
        float* spikesDevice;
        int* refractoryBufferDevice;
};

Simulation::Simulation(int nrec,
                       int nin,
                       int nref,
                       float thr,
                       float tauv,
                       int updatePeriod) {

    nRec = nrec;
    nIn = nin;
    nRef = nref;

    threshold = thr;
    tauVoltage = tauv;
    voltageCoeff = exp(-1.0/tauv);

    deviceMemoryAllocated = false;

    weightUpdatePeriod = updatePeriod;

    netStateSize = updatePeriod*nrec;
    wRecSize = nrec*nrec;
    wInSize = nin*nrec;

    // first index is presynaptic, second index is postsynaptic
    for(size_t i = 0; i < nIn; ++i) {
        for(size_t j = 0; j < nRec; ++j) {
            //wIn[nRec*i + j] = j == 4 ? 1.0 : 0.0; //all the current will go to the fifth postsynaptic neuron
            wIn[nRec*i + j] = uniform(0.0, 1.0);
        }
    }

    for(size_t i = 0; i < nRec; ++i) {
        for(size_t j = 0; j < nRec; ++j) {
        /*
            if(i == j - 1) {
                wRec[i*nRec + j] = 1.0;
            }
            else {
                wRec[i*nRec + j] = 0.0;
            }
        //printf("%f   ", wRec[i*nRec + j]);
        //*/
        //wRec[nRec*i + j] = i == j ? 1.0 : 0.0;
        wRec[nRec*i + j] = uniform(-1.0, 1.0);
        }
    }    
    //wRec[(nRec - 1)*nRec] = 1.0;
    
    for(size_t i = 0; i < weightUpdatePeriod; ++i) {
        for(size_t j = 0; j < nRec; ++j) {
            voltages[nRec*i + j] = 0.0;
        }
    }

    for(size_t i = 0; i < weightUpdatePeriod; ++i) {
        for(size_t j = 0; j < nRec; ++j) {
            spikes[nRec*i + j] = 0.0;
        }
    }

    for(size_t i = 0; i < weightUpdatePeriod; ++i) {
        for(size_t j = 0; j < nRec; ++j) {
            refractoryBuffer[nRec*i + j] = 0;
        }
    }
}

void Simulation::allocate() {
    
    cudaMalloc(&wInDevice, wInSize*sizeof(float));
    cudaMalloc(&wRecDevice, wRecSize*sizeof(float));
    cudaMalloc(&voltageDevice, netStateSize*sizeof(float));
    cudaMalloc(&spikesDevice, netStateSize*sizeof(float));
    cudaMalloc(&refractoryBufferDevice, netStateSize*sizeof(int));

    deviceMemoryAllocated = true;
}

void Simulation::copyToDevice() {

    cudaMemcpy(wInDevice, &(wIn[0]), wInSize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(wRecDevice, &(wRec[0]), wRecSize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(voltageDevice, &(voltages[0]), netStateSize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(spikesDevice, &(spikes[0]), netStateSize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(refractoryBufferDevice, &(refractoryBuffer[0]), netStateSize*sizeof(int), cudaMemcpyHostToDevice);
}

void Simulation::simulate(int timesteps) {

    float inCurrents[weightUpdatePeriod*nIn];
    float* inCurrentsDevice;
    cudaMalloc(&inCurrentsDevice, weightUpdatePeriod*nIn*sizeof(float));

    for(int time = 1; time < timesteps; ++time) {

        if(time%weightUpdatePeriod == 0) {
            for(size_t i = 0; i < weightUpdatePeriod; ++i) {
                for(size_t j = 0; j < nRec; ++j) {
                    inCurrents[nRec*i + j] = uniform(0.0, 1.0);
                }
            }
            cudaMemcpy(inCurrentsDevice, inCurrents, weightUpdatePeriod*nIn*sizeof(float), cudaMemcpyHostToDevice);
        }

        int numBlocks = 4;
        dim3 threadsPerBlock(nRec, nRec);
        cudakernel::neuronStep<<<numBlocks, threadsPerBlock>>>(voltageDevice,
                                                               spikesDevice,
                                                               refractoryBufferDevice,
                                                               inCurrentsDevice,
                                                               wInDevice,
                                                               wRecDevice,
                                                               nRec,
                                                               nIn,
                                                               wInSize,
                                                               wRecSize,
                                                               weightUpdatePeriod,
                                                               nRef,
                                                               voltageCoeff,
                                                               threshold, 
                                                               time);
    }

    // just in case leaving the scope doesn't destroy the allocation(?)
    cudaFree(inCurrentsDevice);
}

void Simulation::copyFromDevice() {

    cudaMemcpy(wIn, wInDevice, wInSize*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(wRec, wRecDevice, wRecSize*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(voltages, voltageDevice, netStateSize*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(spikes, spikesDevice, netStateSize*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(refractoryBuffer, refractoryBufferDevice, netStateSize*sizeof(float), cudaMemcpyDeviceToHost);
}

void Simulation::free() {

    cudaFree(wInDevice);
    cudaFree(wRecDevice);
    cudaFree(voltageDevice);
    cudaFree(spikesDevice);
    cudaFree(refractoryBufferDevice);

    deviceMemoryAllocated = false;
}

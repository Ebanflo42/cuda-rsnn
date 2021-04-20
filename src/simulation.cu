#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>

#include "cuda-kernel.cu"

using namespace std;

float uniform(float low, float high) {
    return low + (static_cast<float>(rand())/RAND_MAX)*(high - low);
}

class Simulation {
    
    public:

        int n_rec;
        int n_in;
       
        int n_ref;
        float threshold;
        float tau_volt;
        float volt_coeff;

        bool device_memory_allocated;

        int buffer_length;

        int net_state_size;
        int weights_rec_size;
        int weights_in_size;

        vector<float> weights_rec;
        vector<float> weights_in;
        vector<float> voltages;
        vector<float> spikes;
        vector<float> refractory_buffer;

        Simulation(int nrec,
                   int nin,
                   int nref,
                   float thr,
                   float tau_v,
                   int buffer_len);

        void allocate();

        void copyToDevice();

        void simulate(int timesteps);

        void copyFromDevice();

        void free();

    protected:

        float* weights_in_device;
        float* weights_rec_device;

        float* voltage_device;
        float* spikes_device;
        int* refractory_buffer_device;
};

Simulation::Simulation(int nrec,
                       int nin,
                       int nref,
                       float thr,
                       float tauv,
                       int buffer_len) {

    n_rec = nrec;
    n_in = nin;
    n_ref = nref;

    threshold = thr;
    tau_volt = tauv;
    volt_coeff = exp(-1.0/tauv);

    device_memory_allocated = false;

    buffer_length = buffer_len;

    net_state_size = buffer_len*nrec;
    weights_rec_size = nrec*nrec;
    weights_in_size = nin*nrec;

    // first index is presynaptic, second index is postsynaptic
    for(size_t i = 0; i < n_in; ++i) {
        for(size_t j = 0; j < n_rec; ++j) {
            weights_in.push_back(uniform(0.0, 1.0));
        }
    }

    for(size_t i = 0; i < n_rec; ++i) {
        for(size_t j = 0; j < n_rec; ++j) {
            weights_rec.push_back(uniform(-1.0, 1.0));
        }
    }
    
    for(size_t i = 0; i < buffer_length; ++i) {
        for(size_t j = 0; j < n_rec; ++j) {
            voltages.push_back(0.0);
        }
    }

    for(size_t i = 0; i < buffer_length; ++i) {
        for(size_t j = 0; j < n_rec; ++j) {
            spikes.push_back(0.0);
        }
    }

    for(size_t i = 0; i < buffer_length; ++i) {
        for(size_t j = 0; j < n_rec; ++j) {
            refractory_buffer.push_back(0);
        }
    }
}

void Simulation::allocate() {

    printf("got here\n");
    cudaMalloc(&weights_in_device, weights_in_size*sizeof(float));
    printf("got here\n");
    cudaMalloc(&weights_rec_device, weights_rec_size*sizeof(float));
    printf("got here\n");
    cudaMalloc(&voltage_device, net_state_size*sizeof(float));
    cudaMalloc(&spikes_device, net_state_size*sizeof(float));
    cudaMalloc(&refractory_buffer_device, net_state_size*sizeof(int));

    device_memory_allocated = true;
}

void Simulation::copyToDevice() {

    cudaMemcpy(weights_in_device, weights_in.data(), weights_in_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weights_rec_device, weights_rec.data(), weights_rec_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(voltage_device, voltages.data(), net_state_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(spikes_device, spikes.data(), net_state_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(refractory_buffer_device, refractory_buffer.data(), net_state_size*sizeof(int), cudaMemcpyHostToDevice);
}

void Simulation::simulate(int timesteps) {

    float in_currents[buffer_length*n_in];
    float* in_currents_device;
    cudaMalloc(&in_currents_device, buffer_length*n_in*sizeof(float));

    for(int time = 0; time < timesteps; ++time) {

        if(time%buffer_length == 0) {
            for(size_t i = 0; i < buffer_length; ++i) {
                for(size_t j = 0; j < n_in; ++j) {
                    in_currents[n_in*i + j] = uniform(0.0, 0.3);
                }
            }
            cudaMemcpy(in_currents_device, in_currents, buffer_length*n_in*sizeof(float), cudaMemcpyHostToDevice);
        }

        cudakernel::stepLIF<<<4, n_rec/3>>>(voltage_device,
                                            spikes_device,
                                            refractory_buffer_device,
                                            in_currents_device,
                                            weights_in_device,
                                            weights_rec_device,
                                            n_in,
                                            n_rec,
                                            weights_in_size,
                                            weights_rec_size,
                                            buffer_length,
                                            n_ref,
                                            volt_coeff,
                                            threshold, 
                                            time);
    }

    // just in case leaving the scope doesn't destroy the allocation(?)
    cudaFree(in_currents_device);
}

void Simulation::copyFromDevice() {

    cudaMemcpy(weights_in.data(), weights_in_device, weights_in_size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(weights_rec.data(), weights_rec_device, weights_rec_size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(voltages.data(), voltage_device, net_state_size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(spikes.data(), spikes_device, net_state_size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(refractory_buffer.data(), refractory_buffer_device, net_state_size*sizeof(int), cudaMemcpyDeviceToHost);
}

void Simulation::free() {

    cudaFree(weights_in_device);
    cudaFree(weights_rec_device);
    cudaFree(voltage_device);
    cudaFree(spikes_device);
    cudaFree(refractory_buffer_device);

    device_memory_allocated = false;
}

extern "C" {
    Simulation* new_simulation(int nrec, int nin, int nref, float thr, float tauv, int buffer_len) {
        return new Simulation(nrec, nin, nref, thr, tauv, buffer_len);
    }
    void allocate_simulation(Simulation* sim) { sim->allocate(); }
    void copy_to_device(Simulation* sim) { sim->copyToDevice(); }
    void run_simulation(Simulation* sim, int steps) { sim->simulate(steps); }
    void copy_from_device(Simulation* sim) { sim->copyFromDevice(); }
    void free_simulation(Simulation* sim) { sim->free(); }
}

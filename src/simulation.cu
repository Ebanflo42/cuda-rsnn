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
        float* voltages;
        float* spikes;
        vector<float> refractory_buffer;

        Simulation(int nrec,
                   int nin,
                   int nref,
                   float thr,
                   float tau_v,
                   int buffer_len);

        Simulation(int nrec,
                   int nin,
                   int nref,
                   float thr,
                   float tau_v,
                   int buffer_len,
                   float* voltage_array,
                   float* spike_array);

        void allocate();

        void copyToDevice();

        void simulate(int timesteps);

        void copyFromDevice();

        void free();

    protected:

        float* weights_in_device;
        float* weights_rec_device;

        float* voltages_device;
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
    
    voltages = (float*) malloc(net_state_size*sizeof(float));

    spikes = (float*) malloc(net_state_size*sizeof(float));

    for(size_t i = 0; i < buffer_length; ++i) {
        for(size_t j = 0; j < n_rec; ++j) {
            refractory_buffer.push_back(0);
        }
    }

    weights_in_device = NULL;
    weights_rec_device = NULL;
    voltages_device = NULL;
    spikes_device = NULL;
    refractory_buffer_device = NULL;

}

Simulation::Simulation(int nrec,
                       int nin,
                       int nref,
                       float thr,
                       float tauv,
                       int buffer_len,
                       float* volt_arr,
                       float* spike_arr) {

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

    voltages = volt_arr;
    spikes = spike_arr;

    for(size_t i = 0; i < buffer_length; ++i) {
        for(size_t j = 0; j < n_rec; ++j) {
            refractory_buffer.push_back(0);
        }
    }

    weights_in_device = NULL;
    weights_rec_device = NULL;
    voltages_device = NULL;
    spikes_device = NULL;
    refractory_buffer_device = NULL;
}

void Simulation::allocate() {

    cudaMalloc(&weights_in_device, weights_in_size*sizeof(float));
    cudaMalloc(&weights_rec_device, weights_rec_size*sizeof(float));
    cudaMalloc(&voltages_device, net_state_size*sizeof(float));
    cudaMalloc(&spikes_device, net_state_size*sizeof(float));
    cudaMalloc(&refractory_buffer_device, net_state_size*sizeof(int));

    device_memory_allocated = true;
}

void Simulation::copyToDevice() {

    cudaMemcpy(weights_in_device, weights_in.data(), weights_in_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weights_rec_device, weights_rec.data(), weights_rec_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(voltages_device, voltages, net_state_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(spikes_device, spikes, net_state_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(refractory_buffer_device, refractory_buffer.data(), net_state_size*sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
}

void Simulation::simulate(int timesteps) {

    float in_currents[buffer_length*n_in];
    float* in_currents_device;
    cudaMalloc(&in_currents_device, buffer_length*n_in*sizeof(float));

    for(size_t i = 0; i < buffer_length; ++i) {
        for(size_t j = 0; j < n_in; ++j) {
            in_currents[n_in*i + j] = uniform(0.0, 0.3);
        }
    }
    cudaMemcpy(in_currents_device, in_currents, buffer_length*n_in*sizeof(float), cudaMemcpyHostToDevice);

    cudakernel::stepLIF<<<1, n_rec>>>(voltages_device,
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
                                      0,
                                      timesteps);
    cudaDeviceSynchronize();

    // just in case leaving the scope doesn't destroy the allocation(?)
    cudaFree(in_currents_device);
}

void Simulation::copyFromDevice() {

    cudaMemcpy(weights_in.data(), weights_in_device, weights_in_size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(weights_rec.data(), weights_rec_device, weights_rec_size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(voltages, voltages_device, net_state_size*sizeof(float), cudaMemcpyDeviceToHost);

    /*
    for(size_t i = 0; i < 20; ++i) {
        for(size_t j = 0; j < 5; ++j) {
            printf("%f  ", voltages[n_rec*i + j]);
        }
        printf("\n");
    }
    */

    cudaMemcpy(spikes, spikes_device, net_state_size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(refractory_buffer.data(), refractory_buffer_device, net_state_size*sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
}

void Simulation::free() {

    cudaFree(weights_in_device);
    cudaFree(weights_rec_device);
    cudaFree(voltages_device);
    cudaFree(spikes_device);
    cudaFree(refractory_buffer_device);

    device_memory_allocated = false;
}

extern "C" {

    Simulation* new_simulation(int nrec, int nin, int nref, float thr, float tauv, int buffer_len, float* volt_arr, float* spike_arr) {
        return new Simulation(nrec, nin, nref, thr, tauv, buffer_len, volt_arr, spike_arr);
    }    

    void allocate_simulation(Simulation* sim) { sim->allocate(); }
    void copy_to_device(Simulation* sim) { sim->copyToDevice(); }
    void run_simulation(Simulation* sim, int steps) { sim->simulate(steps); }
    void copy_from_device(Simulation* sim) { sim->copyFromDevice(); }
    void free_simulation(Simulation* sim) { sim->free(); }
}

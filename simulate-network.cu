#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define N_IN 5
#define N_REC 10
#define N_WINDOW 20
#define TIMESTEPS 20

#define W_IN_SIZE N_IN*N_REC
#define W_REC_SIZE N_REC*N_REC
#define NET_STATE_SIZE N_WINDOW*N_REC

float uniform(float low, float high) {
    return low + (static_cast<float>(rand())/RAND_MAX)*(high - low);
}

__global__ void network_step(float voltages[NET_STATE_SIZE], 
			     float in_currents[N_WINDOW*N_IN],
			     float weight_in[W_IN_SIZE],
		             float weight_rec[W_REC_SIZE],
			     int t) {

    int pre = blockIdx.x*blockDim.x + threadIdx.x;
    int post = blockIdx.y*blockDim.y + threadIdx.y;

    int last_t = (t - 1)%N_WINDOW;

    //compute recurrent synapses
    __shared__ float rec_synapses[W_REC_SIZE];
    if(pre < N_REC && post < N_REC) {
        float last_volts = voltages[N_REC*last_t + pre];
	rec_synapses[N_REC*pre + post] = last_volts*weight_rec[N_REC*pre + post];
    }
    __syncthreads();

    //compute input synapses
    __shared__ float in_synapses[W_IN_SIZE];
    if(pre < N_IN && post < N_REC) {
	int tn = t%N_WINDOW;
	float current_current = in_currents[N_IN*tn + pre];
	in_synapses[N_REC*pre + post] = current_current*weight_in[N_REC*pre + post];
    }
    __syncthreads();

    //printf("%d  %d  %f\n", row, col, elmntwise_mul[N_REC*row + col]);

    //take the sum of all presynaptic potentials
    if(pre == 0 && post < N_REC) {

        int tm = t % N_WINDOW;
	voltages[N_REC*tm + post] = 0.0;

	for(size_t pre_ = 0; pre_ < N_REC; ++pre_) {
            voltages[N_REC*tm + post] += rec_synapses[N_REC*pre_ + post];
	}

	for(size_t pre_ = 0; pre_ < N_IN; ++pre_) {
	    voltages[N_REC*tm + post] += in_synapses[N_REC*pre_ + post];
	}
    }
}

int main() {

    float w_in[W_IN_SIZE];

    // first index is presynaptic, second index is postsynaptic
    for(size_t i = 0; i < N_IN; ++i) {
	for(size_t j = 0; j < N_REC; ++j) {
            //w_in[i*N_IN + j] = uniform(-0.1, 0.1);
	    w_in[N_REC*i + j] = j == 4 ? 1.0 : 0.0; //all the current will go to the fifth postsynaptic neuron
            //printf("%f   ", w_in[N_REC*i + j]);
	}
	//printf("\n");
    }

    float w_rec[W_REC_SIZE];
    
    for(size_t i = 0; i < N_REC; ++i) {
	for(size_t j = 0; j < N_REC; ++j) {
	    //*
	    if(i == j - 1) {
		w_rec[i*N_REC + j] = 1.0;
	    }
	    else {
		w_rec[i*N_REC + j] = 0.0;
	    }
	    //printf("%f   ", w_rec[i*N_REC + j]);
	    //*/
	    //w_rec[N_REC*i + j] = i == j ? 1.0 : 0.0;
	}
	//printf("\n");
    }	

    w_rec[(N_REC - 1)*N_REC] = 1.0;
    
    float volts[NET_STATE_SIZE];

    for(size_t i = 0; i < N_WINDOW; ++i) {
	for(size_t j = 0; j < N_REC; ++j) {
            volts[N_REC*i + j] = uniform(-1.0, 1.0);
	}
    }

    float in_currents[N_WINDOW*N_IN];

    for(size_t i = 0; i < N_WINDOW; ++i) {
	for(size_t j = 0; j < N_REC; ++j) {
            in_currents[N_REC*i + j] = uniform(0.0, 1.0);
	}
    }

    //printf("\n");
    float* w_in_gpu;
    float* w_rec_gpu;
    float* volts_gpu;
    float* in_currents_gpu; //pre-determined current so that we don't have to genereate it every time step

    cudaMalloc(&w_in_gpu, W_IN_SIZE*sizeof(float));
    cudaMalloc(&w_rec_gpu, W_REC_SIZE*sizeof(float));
    cudaMalloc(&volts_gpu, NET_STATE_SIZE*sizeof(float));
    cudaMalloc(&in_currents_gpu, N_IN*N_WINDOW*sizeof(float));

    cudaMemcpy(w_in_gpu, w_in, W_IN_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(w_rec_gpu, w_rec, W_REC_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(volts_gpu, volts, NET_STATE_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(in_currents_gpu, in_currents, N_WINDOW*N_IN*sizeof(float), cudaMemcpyHostToDevice);

    for(int time = 1; time < TIMESTEPS; ++time) {

	//*
	if(time%N_WINDOW == 0) {
            for(size_t i = 0; i < N_WINDOW; ++i) {
                for(size_t j = 0; j < N_REC; ++j) {
		    in_currents[N_REC*i + j] = uniform(0.0, 1.0);
		}
	    }
	    cudaMemcpy(in_currents_gpu, in_currents, N_WINDOW*N_IN*sizeof(float), cudaMemcpyHostToDevice);
	}
	//*/

	int numBlocks = 1;
	dim3 threadsPerBlock(N_REC, N_REC);
	network_step<<<numBlocks, threadsPerBlock>>>(volts_gpu, in_currents_gpu, w_in_gpu, w_rec_gpu, time);
    }

    cudaMemcpy(w_in, w_in_gpu, W_IN_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(w_rec, w_rec_gpu, W_REC_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(volts, volts_gpu, NET_STATE_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(w_in_gpu);
    cudaFree(w_rec_gpu);
    cudaFree(volts_gpu);
    cudaFree(in_currents_gpu);

    for(size_t i = 0; i < N_WINDOW; ++i) {
	printf("%d    ", i);
	for(size_t j = 0; j < N_REC; ++j) {
            printf("%f   ", volts[N_REC*i + j]);
	}
	printf("\n");
    }

    return 0;
}

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define N_IN 5
#define N_REC 10
#define N_WINDOW 20
#define TIMESTEPS 40

#define W_IN_SIZE N_IN*N_REC
#define W_REC_SIZE N_REC*N_REC
#define NET_STATE_SIZE N_WINDOW*N_REC

float uniform(float low, float high) {
    return low + (static_cast<float>(rand())/RAND_MAX)*(high - low);
}

__global__ void network_step(float network_state[N_WINDOW*N_REC], float weight_rec[N_REC*N_REC], int t) {

    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;

    int last_t = (t - 1) % N_WINDOW;
    float last_state = network_state[last_t*N_REC + row];

    __shared__ float elmntwise_mul[N_REC*N_REC];

    if(row < N_REC && col < N_REC) {
	//assuming network state is column vector being left-multiplied by weights with row-major order
	elmntwise_mul[N_REC*row + col] = last_state*weight_rec[row*N_REC + col];
    }

    __syncthreads();

    //printf("%d  %d  %f\n", row, col, elmntwise_mul[N_REC*row + col]);

    if(row < N_REC && col == 0) {
        int tm = t % N_WINDOW;
	network_state[tm*N_REC + row] = 0.0;
	for(size_t k = 0; k < N_REC; ++k) {
            network_state[tm*N_REC + row] += elmntwise_mul[k*N_REC + row];
	}
    }
}

int main() {

    float w_in[W_IN_SIZE];

    for(size_t i = 0; i < N_IN; ++i) {
	for(size_t j = 0; j < N_REC; ++j) {
            w_in[i*N_IN + j] = uniform(-0.1, 0.1);
	}
    }

    float w_rec[W_REC_SIZE];
    
    for(size_t i = 0; i < N_REC; ++i) {
	for(size_t j = 0; j < N_REC; ++j) {
	    if(i == j - 1) {
		w_rec[i*N_REC + j] = 1.0;
	    }
	    else {
		w_rec[i*N_REC + j] = 0.0;
	    }
	    printf("%f   ", w_rec[i*N_REC + j]);
	}
	printf("\n");
    }	

    w_rec[(N_REC - 1)*N_REC] = 1.0;
    
    float net_state[NET_STATE_SIZE];

    for(size_t i = 0; i < N_WINDOW; ++i) {
	for(size_t j = 0; j < N_REC; ++j) {
            net_state[i*N_REC + j] = uniform(-1.0, 1.0);
	}
    }

    printf("\n");
    float* w_in_gpu;
    float* w_rec_gpu;
    float* net_state_gpu;

    cudaMalloc(&w_in_gpu, W_IN_SIZE*sizeof(float));
    cudaMalloc(&w_rec_gpu, W_REC_SIZE*sizeof(float));
    cudaMalloc(&net_state_gpu, NET_STATE_SIZE*sizeof(float));

    cudaMemcpy(w_in_gpu, w_in, W_IN_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(w_rec_gpu, w_rec, W_REC_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(net_state_gpu, net_state, NET_STATE_SIZE*sizeof(float), cudaMemcpyHostToDevice);

    for(int time = 1; time < TIMESTEPS; ++time) {
	int numBlocks = 1;
	dim3 threadsPerBlock(N_REC, N_REC);
	network_step<<<numBlocks, threadsPerBlock>>>(net_state_gpu, w_rec_gpu, time);
    }

    cudaMemcpy(w_in, w_in_gpu, W_IN_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(w_rec, w_rec_gpu, W_REC_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(net_state, net_state_gpu, NET_STATE_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(w_in_gpu);
    cudaFree(w_rec_gpu);
    cudaFree(net_state_gpu);

    for(size_t i = 0; i < N_WINDOW; ++i) {
	printf("%d    ", i);
	for(size_t j = 0; j < N_REC; ++j) {
            printf("%f   ", net_state[i*N_REC + j]);
	}
	printf("\n");
    }

    return 0;
}

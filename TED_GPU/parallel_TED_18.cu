#include "TED_C++.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void task_GPU(int* x_orl, int* x_kr, int* y_orl, int* y_kr, int* Delta, int* D, int* D_tree, int thread_in_number, int table_in_number, int L, int m, int n){
    int row = table_in_number / L;
    int column = table_in_number % L;
    int num_table_offset = thread_in_number*(m+1)*(n+1);
    int width = n+1;
    int index;
    int i;
    int j;

    int i_0 = x_kr[row];
    int j_0 = y_kr[column];
    int i_max = x_orl[i_0] + 1;
    int j_max = y_orl[j_0] + 1;

    D[j_max + i_max*width + num_table_offset] = 0;
//    D[i_max][j_max] = 0;

    for (i = i_max - 1; i > i_0 - 1; i--) {
        index = j_max + i*width + num_table_offset;
        D[index] = 1 + D[index + width];

//        single_unit(x_orl, y_orl, i_0, i_max, j_0, j_max, i, j_max,
//                    Delta, D, D_tree, thread_in_number, m, n);
//        D[i][j_max] = 1 + D[i + 1][j_max];
    }

    for (j = j_max - 1; j > j_0 - 1; j--) {
        index = j + i_max*width + num_table_offset;
        D[index] = 1 + D[index + 1];

//        single_unit(x_orl, y_orl, i_0, i_max, j_0, j_max, i_max, j,
//                    Delta, D, D_tree, thread_in_number, m, n);
//        D[i_max][j] = 1 + D[i_max][j + 1];
    }

    for (i = i_max - 1; i > i_0 - 1; i--) {
        for (j = j_max - 1; j > j_0 - 1; j--) {

//            single_unit(x_orl, y_orl, i_0, i_max, j_0, j_max, i, j,
//                        Delta, D, D_tree, thread_in_number, m, n);

            if ((x_orl[i] == x_orl[i_0]) & (y_orl[j] == y_orl[j_0])) {

                index = j + i*width + num_table_offset;

                D[index] = min3_D(Delta[j+i*n] + D[index + 1 + width],
                                                           1 + D[index + width],
                                                           1 + D[index + 1]);
                D_tree[j + i*n] = D[index];

//                D[i][j] = min3(Delta[i][j] + D[i + 1][j + 1], 1 + D[i + 1][j], 1 + D[i][j + 1]);
//                D_tree[i][j] = D[i][j];


            } else {

                index = j + i*width + num_table_offset;
                D[index] = min3_D(D_tree[j + i*n] + D[y_orl[j] + 1 + (x_orl[i] + 1) * width + num_table_offset],
                                                           1 + D[index + width],
                                                           1 + D[index + 1]);


//                D[i][j] = min3(D_tree[i][j] + D[x_orl[i] + 1][y_orl[j] + 1], 1 + D[i + 1][j],
//                               1 + D[i][j + 1]);

            }
        }
    }
}


__global__ void parallel_task_stage(int* x_orl, int* x_kr, int* y_orl, int* y_kr, int* Delta, int* D, int* D_tree, int n, int m, int L, dev::Queue<int, uint32_t> d_queue, int batch_size, int batch_num, int end){
    for (int i=TID_1D; i<end; i+=TOTAL_THREADS_1D){
//        printf("The task table is %u \n", d_queue[i+batch_num*batch_size]);
//        if(i<=2142){
//            task_GPU(x_orl,x_kr,y_orl,y_kr,Delta,D,D_tree,i,d_queue[i+batch_num*batch_size],L,m,n);
//        } else{
//            task_GPU(x_orl,x_kr,y_orl,y_kr,Delta,D_2,D_tree,(i-2143),d_queue[i+batch_num*batch_size],L,m,n);
//        }
        task_GPU(x_orl,x_kr,y_orl,y_kr,Delta,D,D_tree,i,d_queue[i+batch_num*batch_size],L,m,n);

    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void simple_parallel(int* x_orl, int* x_kr, int* y_orl, int* y_kr, int* Delta, int* D, int* D_2, int* D_tree, int n, int m, int L, dev::Queue<int, uint32_t> d_queue, int limitation){

    for (int i=threadIdx.x+blockIdx.x * blockDim.x; i<d_queue.size(); i += gridDim.x*blockDim.x){
        int thread_in_number = i%(2*limitation);
        if(thread_in_number<=(limitation-1)){
            task_GPU(x_orl,x_kr,y_orl,y_kr,Delta,D,D_tree,thread_in_number,d_queue[i],L,m,n);
        }else{
            task_GPU(x_orl,x_kr,y_orl,y_kr,Delta,D_2,D_tree,(thread_in_number-limitation),d_queue[i],L,m,n);
        }

    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void simple_parallel_new(int* x_orl, int* x_kr, int* y_orl, int* y_kr, int* Delta, int* D, int* D_tree, int n, int m, int L, dev::Queue<int, uint32_t> d_queue, int limitation){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if(x < limitation) {
        for (int i = (threadIdx.x + blockIdx.x * blockDim.x); i < d_queue.size(); i += limitation) {
            int thread_in_number = i % (limitation);
            task_GPU(x_orl, x_kr, y_orl, y_kr, Delta, D, D_tree, thread_in_number, d_queue[i], L, m, n);
        }
    }
}


__device__ volatile int Array_in[82];
__device__ volatile int Array_out[82];

__global__ void zhang_parallel(int* x_orl, int* x_kr, int* y_orl, int* y_kr, int* Delta, int* D, int* D_tree, int n, int m, int L, dev::Queue<int, uint32_t> d_queue, int limitation){

    int index = (threadIdx.x+blockIdx.x*1024);
    int x_loc = 199 - index / (n+1);
    int y_loc = 199 - index % (n+1);
    if(x_loc<0 || y_loc <0){
        x_loc = -1;
        y_loc = -1;
    }

    if(blockIdx.x ==0 && threadIdx.x == 0){
        printf("m = %d, n = %d\n", m, n);
    }


    for (int step = 0; step < m + n + 1; step++) {
        if ((x_loc + y_loc == m + n - step) & (x_loc>=0) & (y_loc>=0)) {
            for (int i = 0; i < 6889; i++) {
                int row = i / 83;
                int column = i % 83;
                int i_0 = x_kr[row];
                int j_0 = y_kr[column];
                int i_max = x_orl[i_0] + 1;
                int j_max = y_orl[j_0] + 1;
                single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc, Delta, D, D_tree, i, m, n);
            }
        }
        __syncthreads();
        __gpu_sync_range_fence(0, 63, step + 1, Array_in, Array_out);
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void single_unit(int* x_orl, int* y_orl, int i_0, int i_max, int j_0, int j_max, int i, int j, int* Delta, int* D, int* D_tree, int thread_in_number, int m, int n){
    int num_table_offset = thread_in_number*(m+1)*(n+1);
    int width = n+1;
    int index = j + i*width + num_table_offset;
    if ((i == i_max) & (j == j_max)){
        D[index] = 0;
    }else if ((i <= i_max-1) & (i >= i_0) & (j == j_max)){
        D[index] = 1 + D[index+width];
    }else if ((j <= j_max-1) & (j >= j_0) & (i == i_max)){
        D[index] = 1 + D[1+index];
    }else if ((i <= i_max-1) & (i >= i_0) & (j <= j_max-1) & (j >= j_0)){
        if ((x_orl[i] == x_orl[i_0]) & (y_orl[j] == y_orl[j_0])) {

            D[index] = min3_D(Delta[j+i*n] + D[1 + width + index],
                              1 + D[width+index],
                              1 + D[1 + index]);
            D_tree[j + i*n] = D[index];

        }else{
            D[index] = min3_D(D_tree[j + i*n] + D[y_orl[j] + 1 + (x_orl[i] + 1) * width + num_table_offset],
                                                       1 + D[index+width],
                                                       1 + D[1 + index]);
        }
    }
}

__device__ void single_unit_2(int* x_orl, int* y_orl, int i_0, int i_max, int j_0, int j_max, int i, int j, int* Delta, int* D, int* D_tree, int thread_in_number, int m, int n){
    int num_table_offset = 0;
    int width = 1001;
    int index = j + i*width + num_table_offset;

    if ((x_orl[i] == x_orl[i_0]) & (y_orl[j] == y_orl[j_0])) {

        D[index] = min3_D(Delta[j+i*n] + D[1 + width + index],
                          1 + D[width+index],
                          1 + D[1 + index]);
        D_tree[j + i*n] = D[index];

    }else{
        D[index] = min3_D(D_tree[j + i*n] + D[y_orl[j] + 1 + (x_orl[i] + 1) * width + num_table_offset],
                          1 + D[index+width],
                          1 + D[1 + index]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void parallel_single_table_1(int thread_in_number, int* x_orl, int* y_orl, int i_0, int i_max, int j_0, int j_max, int* Delta, int* D, int* D_tree, int n, int m, int t, int k){
    for (int w=TID_1D; w<=t; w+=TOTAL_THREADS_1D){
        single_unit(x_orl,y_orl,i_0,i_max,j_0,j_max,i_max-w,j_max+1-k+w,Delta,D,D_tree,thread_in_number,m,n);
    }
}

__global__ void parallel_single_table_2(int thread_in_number, int* x_orl, int* y_orl, int i_0, int i_max, int j_0, int j_max, int* Delta, int* D, int* D_tree, int n, int m, int t, int k){
    for (int w=TID_1D; w<=t; w+=TOTAL_THREADS_1D){
        single_unit(x_orl,y_orl,i_0,i_max,j_0,j_max,i_max-k+j_max - j_0 + 1-w,j_0+w,Delta,D,D_tree,thread_in_number,m,n);
    }
}


void single_table(Stream& stream_sing, int i, int thread_in_number, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, int L, int blockSize, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){

    int row = i / L;
    int column = i % L;

    int i_0;
    int j_0;
    int i_max;
    int j_max;
    int numBlocks;

    i_0 = x_kr[row];
    j_0 = y_kr[column];
    i_max = x_orl[i_0] + 1;
    j_max = y_orl[j_0] + 1;


    int x = j_max - j_0 + 1;
    int y = i_max - i_0 + 1;
    int t;


    for (int k=1; k<=x+y-1; k++){
        if(k<=x){
            t = min(k-1,i_max-i_0);
            numBlocks = (t+1 + blockSize) / blockSize;
//            printf("numBlock = %u, blockSize = %u\n", numBlocks, blockSize);
            parallel_single_table_1<<<numBlocks, blockSize, 0, stream_sing.cuda_stream()>>>(thread_in_number, p_x_orl_d, p_y_orl_d, i_0, i_max, j_0, j_max, p_Delta_d, p_D_d, p_D_tree_d,
                                                                                       n, m, t, k);
        }else{
            t = min(j_max, i_max-i_0-k+x);
            numBlocks = (t+1 + blockSize) / blockSize;
//            printf("numBlock = %u, blockSize = %u\n", numBlocks, blockSize);
            parallel_single_table_2<<<numBlocks, blockSize, 0, stream_sing.cuda_stream()>>>(thread_in_number, p_x_orl_d, p_y_orl_d, i_0, i_max, j_0, j_max, p_Delta_d, p_D_d, p_D_tree_d,
                                                                                       n, m, t, k);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void single_table_parallel_new(int thread_in_number, int* x_orl, int* y_orl, int i_0, int i_max, int j_0, int j_max, int table_row_size, int table_column_size, int tile_size, int* Delta, int* D, int* D_tree, int n, int m){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    __shared__ int row_size, column_size;

    row_size = (i_max-i_0+1 + tile_size - 1) / tile_size;
    column_size = (j_max-j_0+1 + tile_size - 1) / tile_size;


    cooperative_groups::grid_group grid = cooperative_groups::this_grid();


    for ( int i = x; i<row_size; i+=gridDim.x * blockDim.x){
        for ( int j=y; j<column_size; j+=gridDim.y * blockDim.y){

            for (int step = 0; step < row_size + column_size - 1; step++) {
                int index_i = (i_max - x * tile_size);
                int index_j = j_max - y * tile_size;
                for (int number = 0; number < 2 * tile_size - 1; number++) {
                    if (y == step - x) {
                        for (int j = max(0, number - tile_size + 1); j <= min(number, tile_size - 1); j++) {
                            single_unit(x_orl, y_orl, i_0, i_max, j_0, j_max, index_i - number + j, index_j - j,
                                        Delta, D, D_tree, thread_in_number, m, n);
                        }
                    }
                }
                grid.sync();
            }
        }
    }
}


void single_table_new(Stream& stream_sing, int i, int thread_in_number, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, int L, int blockSize, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){
    int row = i / L;
    int column = i % L;

    int i_0 = x_kr[row];
    int j_0 = y_kr[column];
    int i_max = x_orl[i_0] + 1;
    int j_max = y_orl[j_0] + 1;
    int tile = 4;

    dim3 block(32,32);
    dim3 grid((m+tile*block.x)/(tile*block.x),(n+tile*block.y)/(tile*block.y));

    printf("test100\n");
    void *kernel_args[] = { (void*)&thread_in_number, (void*)&p_x_orl_d, (void*)&p_y_orl_d,  (void *)&i_0, (void *)&i_max, (void *)&j_0, (void *)&j_max, (void *)&m, (void *)&n, (void *)&tile, (void *)&p_Delta_d, (void *)&p_D_d, (void *)&p_D_tree_d, (void *)&n, (void *)&m};
    cudaLaunchCooperativeKernel((void *) single_table_parallel_new, grid, block, kernel_args, 0, stream_sing.cuda_stream());

    CHECK_CUDA(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//__shared__ int sharedD[32][32];

__device__ void single_unit_5(int x_edge, int y_edge, int* x_orl, int* y_orl, int i_0, int i_max, int j_0, int j_max, int i, int j, int* Delta, int* D, int* D_tree, int thread_in_number, int m, int n){
    int num_table_offset = 0;
    int width = 1001;
    int index = j + i*width + num_table_offset;

    if ((x_orl[i] == x_edge) & (y_orl[j] == y_edge)) {

        D[index] = min3_D(Delta[j+i*n] + D[1 + width + index],
                          1 + D[width+index],
                          1 + D[1 + index]);
        D_tree[j + i*n] = D[index];

    }else{
        D[index] = min3_D(D_tree[j + i*n] + D[y_orl[j] + 1 + (x_orl[i] + 1) * width + num_table_offset],
                          1 + D[index+width],
                          1 + D[1 + index]);


    }
}

__global__ void single_table_parallel_new_5(int thread_in_number, int* __restrict__ x_orl, int* __restrict__ y_orl, int i_0, int i_max, int j_0, int j_max, int table_row_size, int table_column_size, int tile_size, int* Delta, int* D, int* D_tree, int n, int m){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    __shared__ int row_size, column_size, x_edge, y_edge;

//    row_size = (i_max-i_0+1 + tile_size - 1) / tile_size;
//    column_size = (j_max-j_0+1 + tile_size - 1) / tile_size;
    if(threadIdx.x == 0){
        x_edge = x_orl[i_0];
        y_edge = y_orl[j_0];
        row_size = (i_max-i_0+ tile_size) / tile_size;
        column_size = (j_max-j_0+ tile_size) / tile_size;
    }

    __syncthreads();


    cooperative_groups::grid_group grid = cooperative_groups::this_grid();


    for ( int i = x; i<row_size; i+=gridDim.x * blockDim.x){
        for ( int j=y; j<column_size; j+=gridDim.y * blockDim.y){

            for (int step = 0; step < row_size + column_size - 1; step++) {
                int index_i = (i_max-1 - x * tile_size);
                int index_j = j_max-1 - y * tile_size;
                if (y == step - x) {
                    for (int number = 0; number < 2 * tile_size - 1; number++) {
//                        if (y == step - x) {
                            for (int j = max(0, number - tile_size + 1); j <= min(number, tile_size - 1); j++) {
                                single_unit_5(x_edge, y_edge, x_orl, y_orl, i_0, i_max, j_0, j_max, index_i - number + j, index_j - j,
                                              Delta, D, D_tree, thread_in_number, m, n);
                            }
//                        }

                    }
                }
                grid.sync();
//                __syncthreads();
            }
        }
    }
}

__global__ void initial_row_column(int thread_in_number, int* __restrict__ x_orl, int* __restrict__ y_orl, int i_0, int i_max, int j_0, int j_max, int table_row_size, int table_column_size, int tile_size, int* Delta, int* D, int* D_tree, int n, int m){
    int num_table_offset = 0;
    int width = 1001;
    for (int i=TID_1D; i<=m;i+=TOTAL_THREADS_1D){
        D[(m-i)*width+n] = i;
        D[m*width+n-i] = i;
    }
}


void single_table_new_5(Stream& stream_sing, int i, int thread_in_number, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, int L, int blockSize, int n, int m, int * __restrict__ p_x_orl_d, int* __restrict__ p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){
    int row = i / L;
    int column = i % L;

    int i_0 = x_kr[row];
    int j_0 = y_kr[column];
    int i_max = x_orl[i_0] + 1;
    int j_max = y_orl[j_0] + 1;
    int tile = 4;

    int one_block = 256;
    int one_grid = 4;

    initial_row_column<<<one_grid, one_block, 0, stream_sing.cuda_stream()>>>(thread_in_number, p_x_orl_d, p_y_orl_d, i_0, i_max, j_0, j_max, m, n, tile, p_Delta_d, p_D_d, p_D_tree_d, n, m);


    dim3 block(32,32);
    dim3 grid((m+tile*block.x)/(tile*block.x),(n+tile*block.y)/(tile*block.y));

    printf("test15\n");
    void *kernel_args[] = { (void*)&thread_in_number, (void*)&p_x_orl_d, (void*)&p_y_orl_d,  (void *)&i_0, (void *)&i_max, (void *)&j_0, (void *)&j_max, (void *)&m, (void *)&n, (void *)&tile, (void *)&p_Delta_d, (void *)&p_D_d, (void *)&p_D_tree_d, (void *)&n, (void *)&m};
    cudaLaunchCooperativeKernel((void *) single_table_parallel_new_5, grid, block, kernel_args, 0, stream_sing.cuda_stream());

    CHECK_CUDA(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}






////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__device__ void single_unit_6(int* x_orl, int* y_orl, int i_0, int i_max, int j_0, int j_max, int i, int j, int* Delta, int* D, int* D_tree, int thread_in_number, int m, int n){
    int num_table_offset = 0;
    int width = 1001;
    int index = j + i*width + num_table_offset;

    if ((x_orl[i] == x_orl[i_0]) & (y_orl[j] == y_orl[j_0])) {

        D[index] = min3_D(Delta[j+i*n] + D[1 + width + index],
                          1 + D[width+index],
                          1 + D[1 + index]);
        D_tree[j + i*n] = D[index];


    }else{
        D[index] = min3_D(D_tree[j + i*n] + D[y_orl[j] + 1 + (x_orl[i] + 1) * width + num_table_offset],
                          1 + D[index+width],
                          1 + D[1 + index]);


    }
}

__device__ void single_tile(int index_i, int index_j, int tile_size, int* x_orl, int* y_orl, int i_0, int i_max, int j_0, int j_max, int* Delta, int* D, int* D_tree, int thread_in_number, int m, int n){
    for (int number = 0; number < 2 * tile_size - 1; number++) {
        for (int j = max(0, number - tile_size + 1); j <= min(number, tile_size - 1); j++) {
            single_unit_6(x_orl, y_orl, i_0, i_max, j_0, j_max, index_i - number + j, index_j - j,
                          Delta, D, D_tree, thread_in_number, m, n);
        }
    }
}


__device__ void single_tile_column(int index_i, int index_j, int tile_size, int* x_orl, int* y_orl, int i_0, int i_max, int j_0, int j_max, int* Delta, int* D, int* D_tree, int thread_in_number, int m, int n){
    for (int w=0; w<tile_size*tile_size; w++){
        single_unit_6(x_orl, y_orl, i_0, i_max, j_0, j_max, index_i - (w/tile_size), index_j - (w % tile_size),
                          Delta, D, D_tree, thread_in_number, m, n);
    }
}


__global__ void single_table_parallel_new_6(int loc_x, int loc_y, int thread_in_number, int* x_orl, int* y_orl, int i_0, int i_max, int j_0, int j_max, int table_row_size, int table_column_size, int tile_size, int* Delta, int* D, int* D_tree, int n, int m){
    int x = threadIdx.x;
    int y = threadIdx.y;

//    if (x<blockDim.x & y < blockDim.y){
    for (int step = 0; step < blockDim.x + blockDim.y - 1; step++) {
        int index_i = i_max-1 - x * tile_size -(loc_x + blockIdx.x)*tile_size*blockDim.x;
        int index_j = j_max-1 - y * tile_size - (loc_y - blockIdx.x)*tile_size*blockDim.y;
//        int index_i = (i_max-1 - x * tile_size)-(loc_x + blockIdx.x)*tile_size*blockDim.x;
//        int index_j = j_max-1 - y * tile_size - (loc_y - blockIdx.x)*tile_size*blockDim.y;
        if (y == step - x) {
            single_tile(index_i, index_j,tile_size, x_orl, y_orl, i_0, i_max, j_0, j_max, Delta, D, D_tree, thread_in_number, m, n);
//            single_tile_column(index_i, index_j,tile_size, x_orl, y_orl, i_0, i_max, j_0, j_max, Delta, D, D_tree, thread_in_number, m, n);
        }
        __syncthreads();
    }
//    }
}

void single_table_new_6(Stream& stream_sing, int i, int thread_in_number, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, int L, int blockSize, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){
    int row = i / L;
    int column = i % L;

    int i_0 = x_kr[row];
    int j_0 = y_kr[column];
    int i_max = x_orl[i_0] + 1;
    int j_max = y_orl[j_0] + 1;
    int tile = 2;

    int one_block = 256;
    int one_grid = 4;
    initial_row_column<<<one_grid, one_block, 0, stream_sing.cuda_stream()>>>(thread_in_number, p_x_orl_d, p_y_orl_d, i_0, i_max, j_0, j_max, m, n, tile, p_Delta_d, p_D_d, p_D_tree_d, n, m);

    dim3 block(32,32);
//    dim3 grid((m+tile*block.x)/(tile*block.x),(n+tile*block.y)/(tile*block.y));
    dim3 grid(16,16);

//    printf("test28\n");
//    printf("grid.x = %u, grid.y = %u\n", grid.x, grid.y);

    int row_block = grid.x;
    int column_block = grid.y;

    for (int step = 0; step < grid.x+grid.y-1; step++){
        for (int j = max(0,step-column_block+1); j<=min(step, row_block-1); j++){
            int num_blocks = min(step, row_block-1) - max(0, step-column_block+1) + 1;
            int begin_x = max(0, step-column_block+1);
            int begin_y = step - begin_x;
            single_table_parallel_new_6<<<num_blocks, block, 0, stream_sing.cuda_stream()>>>(begin_x, begin_y, thread_in_number, p_x_orl_d, p_y_orl_d, i_0,  i_max, j_0,  j_max, m, n, tile, p_Delta_d, p_D_d, p_D_tree_d, n, m);
        }
    }


    CHECK_CUDA(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}






////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void single_unit_7(int* x_orl, int* y_orl, int i_0, int i_max, int j_0, int j_max, int i, int j, int* Delta, int* D, int* D_tree, int thread_in_number, int m, int n){
    int width = 1001;
    int index = j + i*width;
    if ((i == i_max) & (j == j_max)){
        D[index] = 0;
    }else if ((i <= i_max-1) & (i >= i_0) & (j == j_max)){
        D[index] = 1 + D[index+width];
    }else if ((j <= j_max-1) & (j >= j_0) & (i == i_max)){
        D[index] = 1 + D[1+index];
    }else if ((i <= i_max-1) & (i >= i_0) & (j <= j_max-1) & (j >= j_0)){
        if ((x_orl[i] == x_orl[i_0]) & (y_orl[j] == y_orl[j_0])) {

            D[index] = min3_D(Delta[j+i*n] + D[1 + width + index],
                              1 + D[width+index],
                              1 + D[1 + index]);
            D_tree[j + i*n] = D[index];

        }else{
            D[index] = min3_D(D_tree[j + i*n] + D[y_orl[j] + 1 + (x_orl[i] + 1) * width],
                              1 + D[index+width],
                              1 + D[1 + index]);
        }
    }
}

__device__ void single_tile_7(int index_i, int index_j, int tile_size, int* x_orl, int* y_orl, int i_0, int i_max, int j_0, int j_max, int* Delta, int* D, int* D_tree, int thread_in_number, int m, int n){
    for (int number = 0; number < 2 * tile_size - 1; number++) {
        for (int j = max(0, number - tile_size + 1); j <= min(number, tile_size - 1); j++) {
            single_unit_7(x_orl, y_orl, i_0, i_max, j_0, j_max, index_i - number + j, index_j - j,
                          Delta, D, D_tree, thread_in_number, m, n);
        }
    }
}


__device__ void single_tile_column_7(int index_i, int index_j, int tile_size, int* x_orl, int* y_orl, int i_0, int i_max, int j_0, int j_max, int* Delta, int* D, int* D_tree, int thread_in_number, int m, int n){
    for (int w=0; w<tile_size*tile_size; w++){
        single_unit_7(x_orl, y_orl, i_0, i_max, j_0, j_max, index_i - (w/tile_size), index_j - (w % tile_size),
                      Delta, D, D_tree, thread_in_number, m, n);
    }
}


__global__ void single_table_parallel_new_7(int number_per_thread, int thread_in_number, int* x_orl, int* y_orl, int i_0, int i_max, int j_0, int j_max,int tile_size, int* Delta, int* D, int* D_tree, int n, int m){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;;

    int x_loc = i_max - x*4;
    int y_loc = j_max - y*4;

    if(x <= i_max/4 & y <= j_max/4) {

        cooperative_groups::grid_group grid = cooperative_groups::this_grid();

        for (int step = 0; step < i_max+1+j_max; step++) {

//            if(D[y_loc-3+(x_loc-3)*1001] == 0) {
                for (int k = 0; k < 4 + 4 - 1; k++) {
                    for (int w = max(0, k - 4 + 1); w <= min(k, 4 - 1); w++) {
                        if ((x_loc - w + y_loc - k + w == i_max+j_max - step)){
                            single_unit_7(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc - w, y_loc - k + w,
                                          Delta, D, D_tree, thread_in_number, m, n);
                        }
                    }
                }

//            }


            grid.sync();
        }

    }
}

void single_table_new_7(Stream& stream_sing, int i, int thread_in_number, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, int L, int blockSize, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){
    int row = i / L;
    int column = i % L;

    int i_0 = x_kr[row];
    int j_0 = y_kr[column];
    int i_max = x_orl[i_0] + 1;
    int j_max = y_orl[j_0] + 1;
    int tile = 1;

    int one_block = 256;
    int one_grid = 4;
    initial_row_column<<<one_grid, one_block, 0, stream_sing.cuda_stream()>>>(thread_in_number, p_x_orl_d, p_y_orl_d, i_0, i_max, j_0, j_max, m, n, tile, p_Delta_d, p_D_d, p_D_tree_d, n, m);

    dim3 block = {32,32,1};
    dim3 grid = {8,8,1};

//    int test_block = 256;
//    int test_grid = 64;

    int number_per_thread = (1001 + (32*8-1)) / (32*8);

    printf("%u\n", number_per_thread);
    printf("%d, %d, %d, %d\n", i_0, i_max, j_0, j_max);

    void *kernel_args[] = { (void*)&number_per_thread,(void*)&thread_in_number, (void*)&p_x_orl_d, (void*)&p_y_orl_d,  (void *)&i_0, (void *)&i_max, (void *)&j_0, (void *)&j_max, (void *)&tile, (void *)&p_Delta_d, (void *)&p_D_d, (void *)&p_D_tree_d, (void *)&n, (void *)&m};
    cudaLaunchCooperativeKernel((void *) single_table_parallel_new_7, grid, block, kernel_args, 0, stream_sing.cuda_stream());
//    single_table_parallel_new_7<<<grid, block, 0, stream_sing.cuda_stream()>>>(number_per_thread, thread_in_number, p_x_orl_d, p_y_orl_d, i_0, i_max, j_0, j_max, m, n, tile, p_Delta_d, p_D_d, p_D_tree_d, n, m);


    CHECK_CUDA(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}






////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void single_table_parallel_new_8(int number_per_thread, int thread_in_number, int* x_orl, int* y_orl, int i_0, int i_max, int j_0, int j_max, int table_row_size, int table_column_size, int tile_size, int* Delta, int* D, int* D_tree, int n, int m){

//    int x = threadIdx.x + blockDim.x * blockIdx.x;
//    int y_loc = 1000 - 16 * blockIdx.x;
//    int x_loc = 1000 - threadIdx.x;
//
////    if(blockIdx.x == 62 && threadIdx.x == 1000){
////        printf("%d, %d\n", y_loc, x_loc);
////    }
//
////    if (threadIdx.x<1001 && blockIdx.x <=62){
//        cooperative_groups::grid_group grid = cooperative_groups::this_grid();
//        for (int step = 0; step < 2001; step++) {
//            if((x_loc>=0) & (y_loc>=0) & (x_loc + y_loc == 2000-step) & (y_loc <=j_max - 16 * blockIdx.x-15)){
//                single_unit_7(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
//                                      Delta, D, D_tree, thread_in_number, m, n);
//                y_loc = y_loc - 1;
//            }
//            grid.sync();
//        }
//
////    }

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y_loc = 1000 - 25 * blockIdx.x;
    int x_loc = 1000 - threadIdx.x;
    int k = 0;

    if(blockIdx.x == 40 && threadIdx.x == 1000){
        printf("%d, %d\n", y_loc, x_loc);
    }

//    if (threadIdx.x<1001 && blockIdx.x <=62){
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    for (int step = 0; step < 2001; step++) {
        if((x_loc + y_loc == 2000-step) && (k<25) && y_loc>=0 && x_loc>=0 ){
            if(blockIdx.x == 40 && threadIdx.x == 1000){
                printf("%d, %d\n", x_loc, y_loc);
            }
            single_unit_7(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                          Delta, D, D_tree, thread_in_number, m, n);
            y_loc = y_loc - 1;
            k++;
        }
        grid.sync();
    }

//    }


}

void single_table_new_8(Stream& stream_sing, int i, int thread_in_number, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, int L, int blockSize, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){
    int row = i / L;
    int column = i % L;

    int i_0 = x_kr[row];
    int j_0 = y_kr[column];
    int i_max = x_orl[i_0] + 1;
    int j_max = y_orl[j_0] + 1;
    int tile = 1;

    int one_block = 256;
    int one_grid = 4;
    initial_row_column<<<one_grid, one_block, 0, stream_sing.cuda_stream()>>>(thread_in_number, p_x_orl_d, p_y_orl_d, i_0, i_max, j_0, j_max, m, n, tile, p_Delta_d, p_D_d, p_D_tree_d, n, m);



    int test_block = 1024;
    int test_grid = 41;

    int number_per_thread = 16;

    printf("%u\n", number_per_thread);
    printf("%d, %d, %d, %d\n", i_0, i_max, j_0, j_max);

    void *kernel_args[] = { (void*)&number_per_thread,(void*)&thread_in_number, (void*)&p_x_orl_d, (void*)&p_y_orl_d,  (void *)&i_0, (void *)&i_max, (void *)&j_0, (void *)&j_max, (void *)&m, (void *)&n, (void *)&tile, (void *)&p_Delta_d, (void *)&p_D_d, (void *)&p_D_tree_d, (void *)&n, (void *)&m};
    cudaLaunchCooperativeKernel((void *) single_table_parallel_new_8, 41, 1024, kernel_args, 0, stream_sing.cuda_stream());

    CHECK_CUDA(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__constant__ int units_per_thread, d_step;



__device__ void __gpu_sync(int goalVal, volatile int *Arrayin, volatile int *Arrayout){
    // Thread ID in a block
    int tid_in_blk = threadIdx.x;
    int nBlockNum = gridDim.x;
    int bid = blockIdx.x;

    // Only thread 0 is used for synchronization
    if (tid_in_blk == 0){Arrayin[bid] = goalVal;}

    if (bid == 1){
        if (tid_in_blk < nBlockNum){
            while (Arrayin[tid_in_blk] != goalVal){}
        }
        __syncthreads();
        if (tid_in_blk < nBlockNum){
            Arrayout[tid_in_blk] = goalVal;
        }
    }

    if (tid_in_blk == 0){
        while (Arrayout[bid] != goalVal){}
    }
    __syncthreads();
}




__device__ void single_unit_9(int* x_orl, int* y_orl, int i_0, int i_max, int j_0, int j_max, int i, int j, int* Delta, int* D, int* D_tree, int thread_in_number, int m, int n){
    int width = 1001;
    int index = j + i*width;

    if ((i <= i_max-1) & (i >= i_0) & (j <= j_max-1) & (j >= j_0)){
        if ((x_orl[i] == i_max-1) & (y_orl[j] == j_max-1)) {

            D[index] = min3_D(Delta[j+i*n] + D[1 + width + index],
                              1 + D[width+index],
                              1 + D[1 + index]);
            D_tree[j + i*n] = D[index];

        }else{
            D[index] = min3_D(D_tree[j + i*n] + D[y_orl[j] + 1 + (x_orl[i] + 1) * width],
                              1 + D[index+width],
                              1 + D[1 + index]);
        }
    }
}

__global__ void single_table_parallel_new_9(int number_per_thread, int thread_in_number, int* x_orl, int* y_orl, int i_0, int i_max, int j_0, int j_max, int tile_size, int* Delta, int* D, int* D_tree, int n, int m){

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y_loc = j_max - units_per_thread * blockIdx.x;
    int x_loc = i_max - threadIdx.x;
    int k = 0;

    for (int step = 0; step < i_max+j_max-i_0-j_0+1; step++) {
        if((x_loc + y_loc == i_max+j_max-i_0-j_0 - step) && (k<units_per_thread) && y_loc>=0 && x_loc>=0 ){

            single_unit_9(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                          Delta, D, D_tree, thread_in_number, m, n);
            y_loc--;
            k++;
        }
        __gpu_sync(step+1,Array_in, Array_out);
    }

}


void single_table_new_9(Stream& stream_sing, int i, int thread_in_number, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, int L, int blockSize, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){
    int row = i / L;
    int column = i % L;

    int i_0 = x_kr[row];
    int j_0 = y_kr[column];
    int i_max = x_orl[i_0] + 1;
    int j_max = y_orl[j_0] + 1;
    int tile = 1;

    int one_block = 256;
    int one_grid = 4;
    initial_row_column<<<one_grid, one_block, 0, stream_sing.cuda_stream()>>>(thread_in_number, p_x_orl_d, p_y_orl_d, i_0, i_max, j_0, j_max, m, n, tile, p_Delta_d, p_D_d, p_D_tree_d, n, m);


    int test_block = 1024;
    int test_grid = 64;
    int number_per_thread = (1000+test_grid-1)/test_grid;
    int step = 2001;

    cudaMemcpyToSymbol(units_per_thread, &number_per_thread, sizeof(int));
    cudaMemcpyToSymbol(d_step, &step, sizeof(int));



//    printf("hello90 %u\n", number_per_thread);
//    printf("%d, %d, %d, %d\n", i_0, i_max, j_0, j_max);

//    void *kernel_args[] = { (void*)&number_per_thread,(void*)&thread_in_number, (void*)&p_x_orl_d, (void*)&p_y_orl_d,  (void *)&i_0, (void *)&i_max, (void *)&j_0, (void *)&j_max, (void *)&tile, (void *)&p_Delta_d, (void *)&p_D_d, (void *)&p_D_tree_d, (void *)&n, (void *)&m};
//    cudaLaunchCooperativeKernel((void *) single_table_parallel_new_9, test_grid, test_block, kernel_args, 0, stream_sing.cuda_stream());

    single_table_parallel_new_9<<<test_grid, test_block, 0, stream_sing.cuda_stream()>>>(number_per_thread, thread_in_number, p_x_orl_d, p_y_orl_d, i_0, i_max, j_0, j_max, tile, p_Delta_d, p_D_d, p_D_tree_d, n, m);

    CHECK_CUDA(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void parallel_multi_table_1(dev::Queue<int, uint32_t> d_queue, int* x_orl, int* x_kr, int* y_orl, int* y_kr, int* Delta, int* D, int* D_tree, int n, int m, int t, int k, int L){

    for (int w=TID_1D; w<(t+1)*d_queue.size(); w+= TOTAL_THREADS_1D){

        int new_row = w / (t+1);
        int new_column = w % (t+1);
        int i = d_queue[new_row];

        int row = i / L;
        int column = i % L;

        int i_0;
        int j_0;
        int i_max;
        int j_max;

        i_0 = x_kr[row];
        j_0 = y_kr[column];
        i_max = x_orl[i_0] + 1;
        j_max = y_orl[j_0] + 1;

        single_unit(x_orl, y_orl, i_0, i_max, j_0, j_max, m - new_column, n + 1 - k + new_column, Delta, D, D_tree,
                    new_row, m, n);
//        single_unit_2(x_orl, y_orl, i_0, i_max, j_0, j_max, m - new_column, n + 1 - k + new_column, Delta, D, D_tree,
//                    new_row, m, n, d_queue.size());
    }

}

__global__ void parallel_multi_table_2(dev::Queue<int, uint32_t> d_queue, int* x_orl, int* x_kr, int* y_orl, int* y_kr, int* Delta, int* D, int* D_tree, int n, int m, int t, int k, int L){

    for (int w=TID_1D; w<(t+1)*d_queue.size(); w+= TOTAL_THREADS_1D){

        int new_row = w / (t+1);
        int new_column = w % (t+1);
        int i = d_queue[new_row];

        int row = i / L;
        int column = i % L;

        int i_0;
        int j_0;
        int i_max;
        int j_max;

        i_0 = x_kr[row];
        j_0 = y_kr[column];
        i_max = x_orl[i_0] + 1;
        j_max = y_orl[j_0] + 1;

        single_unit(x_orl, y_orl, i_0, i_max, j_0, j_max, m-k+n + 1-new_column,0+new_column, Delta, D, D_tree,
                    new_row, m, n);
//        single_unit_2(x_orl, y_orl, i_0, i_max, j_0, j_max, m-k+n + 1-new_column,0+new_column, Delta, D, D_tree,
//                    new_row, m, n, d_queue.size());
    }

}

void multi_table(Stream& stream_mul, dev::Queue<int, uint32_t> d_queue, int size, int L, int blockSize, int n, int m, int *p_x_orl_d, int *p_x_kr_d, int *p_y_orl_d, int *p_y_kr_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){

    int numBlocks;

    int x = n + 1;
    int y = m + 1;
    int t;

    for (int k=1; k<=x+y-1; k++){
        if(k<=x){
            t = min(k-1,m);
            numBlocks = ((t+1)*size + blockSize) / blockSize;
            parallel_multi_table_1<<<numBlocks, blockSize, 0, stream_mul.cuda_stream()>>>(d_queue, p_x_orl_d, p_x_kr_d, p_y_orl_d, p_y_kr_d, p_Delta_d, p_D_d, p_D_tree_d, n, m, t, k ,L);
        }else{
            t = min(n, m-k+x);
            numBlocks = ((t+1)*size + blockSize) / blockSize;
            parallel_multi_table_2<<<numBlocks, blockSize, 0, stream_mul.cuda_stream()>>>(d_queue, p_x_orl_d, p_x_kr_d, p_y_orl_d, p_y_kr_d, p_Delta_d, p_D_d, p_D_tree_d, n, m, t, k ,L);
        }
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void single_unit_10(int* x_orl, int* y_orl, int i_0, int i_max, int j_0, int j_max, int i, int j, int* Delta, int* D, int* D_tree, int thread_in_number, int m, int n){
    int num_table_offset = thread_in_number*(m+1)*(n+1);
    int width = n+1;
    int index = j + i*width + num_table_offset;
//    printf("ok\n");
    if ((i == i_max) & (j == j_max)){
        D[index] = 0;
    }else if ((i <= i_max-1) & (i >= i_0) & (j == j_max)){
        D[index] = 1 + D[index+width];
    }else if ((j <= j_max-1) & (j >= j_0) & (i == i_max)){
        D[index] = 1 + D[1+index];
    }else if ((i <= i_max-1) & (i >= i_0) & (j <= j_max-1) & (j >= j_0)){
        if ((x_orl[i] == x_orl[i_0]) & (y_orl[j] == y_orl[j_0])) {

            D[index] = min3_D(Delta[j+i*n] + D[1 + width + index],
                              1 + D[width+index],
                              1 + D[1 + index]);
            D_tree[j + i*n] = D[index];

        }else{
            D[index] = min3_D(D_tree[j + i*n] + D[y_orl[j] + 1 + (x_orl[i] + 1) * width + num_table_offset],
                              1 + D[index+width],
                              1 + D[1 + index]);
        }
    }
}

__device__ void __gpu_sync_range_fence(int startBlockIdx, int endBlockIdx, int goalVal, volatile int *Arrayin, volatile int *Arrayout){
    int tid_in_blk = threadIdx.x;
    int nBlockNum = gridDim.x;
    int bid = blockIdx.x;

    if (tid_in_blk == 0){
        if (bid >= startBlockIdx && bid <= endBlockIdx){
            Arrayin[bid] = goalVal;
            __threadfence();
        }
    }

    __syncthreads();

    if (bid == startBlockIdx){
        if (tid_in_blk >= startBlockIdx & tid_in_blk <= endBlockIdx){
            while (Arrayin[tid_in_blk] != goalVal){}
        }
        __syncthreads();

        if (tid_in_blk >= startBlockIdx & tid_in_blk <= endBlockIdx){
            Arrayout[tid_in_blk] = goalVal;
            __threadfence();
        }
    }

    if (tid_in_blk == 0){
        if (bid >= startBlockIdx && bid <= endBlockIdx){
            while (Arrayout[bid] != goalVal){}
        }
        __threadfence();
    }
    __syncthreads();
}



__device__ void single_unit_11(int* x_orl, int* y_orl, int i_0, int i_max, int j_0, int j_max, int i, int j, int* Delta, int* D, int* D_tree, int thread_in_number, int m, int n){
//    int num_table_offset = 0;
    int width = n+1;
    int index = j + i*width;
//    printf("ok\n");
    if ((i == i_max) & (j == j_max)){
        D[index] = 0;
    }else if ((i <= i_max-1) & (i >= i_0) & (j == j_max)){
        D[index] = 1 + D[index+width];
    }else if ((j <= j_max-1) & (j >= j_0) & (i == i_max)){
        D[index] = 1 + D[1+index];
    }else if ((i <= i_max-1) & (i >= i_0) & (j <= j_max-1) & (j >= j_0)){
        if ((x_orl[i] == x_orl[i_0]) & (y_orl[j] == y_orl[j_0])) {

            D[index] = min3_D(Delta[j+i*n] + D[1 + width + index],
                              1 + D[width+index],
                              1 + D[1 + index]);
            D_tree[j + i*n] = D[index];

        }else{
            D[index] = min3_D(D_tree[j + i*n] + D[y_orl[j] + 1 + (x_orl[i] + 1) * width],
                              1 + D[index+width],
                              1 + D[1 + index]);
        }
    }
}


__global__ void multi_table_parallel_new(int blocks_per_table, int size, int* x_orl, int* y_orl, int* p_i_0, int* p_i_max, int* p_j_0, int* p_j_max, int tile_size, int* Delta, int* D, int* D_tree, int n, int m){


    if(blockIdx.x<1){


        int i_max = p_i_max[0];
        int j_max = p_j_max[0];
        int i_0 = p_i_0[0];
        int j_0 = p_j_0[0];

        j_max = 1099;
        j_0 = 0;
        i_max = 1000;
        i_0 = 0;


        int x_loc = i_max - threadIdx.x;
        int y_loc = j_max;
        int k=0;
        for (int step = 0; step < i_max+j_max-i_0-j_0+1; step++) {
            if((x_loc + y_loc == i_max+j_max - step) & (y_loc>=j_0) & (x_loc>=i_0)){
                single_unit_11(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                              Delta, D, D_tree, 0, m, n);
                y_loc--;
                k++;
            }
            __syncthreads();
        }

    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//    if(blockIdx.x >=36){
//
//        int i_max = p_i_max[1];
//        int j_max = p_j_max[1];
//        int i_0 = p_i_0[1];
//        int j_0 = p_j_0[1];
//
////        int x_loc = i_max - threadIdx.x;
////        int y_loc = j_max - 32*(blockIdx.x);
//
//
//        int x_loc = i_max - 6*(blockIdx.x-36);
//        int y_loc = j_max - threadIdx.x;
//
//
//        int w=0;
//        for (int step = 0; step < 1158; step++) {
//            if((x_loc + y_loc == 1218 - step) && (w<6) && (x_loc>=i_0) && (y_loc>=j_0)){
//
//                single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
//                               Delta, D, D_tree, 1, m, n);
//
//                x_loc--;
////                y_loc--;
//                w = w+1;
//            }
////            __threadfence();
////            __threadfence_block();
////            __gpu_sync_range_update(0,31,step,Array_in,Array_out);
//            __gpu_sync_range_fence(36,63,step+1,Array_in,Array_out);
//        }
//    }
}


__global__ void multi_table_parallel_new_mul_tables(int blocks_per_table, int size, int* x_orl, int* y_orl, int* p_i_0, int* p_i_max, int* p_j_0, int* p_j_max, int tile_size, int* Delta, int* D, int* D_tree, int n, int m){

    int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
    int lane_id = (threadIdx.x + blockIdx.x * blockDim.x) % 32;

    if(warp_id<1){


        int i_max = p_i_max[0];
        int j_max = p_j_max[0];
        int i_0 = p_i_0[0];
        int j_0 = p_j_0[0];

        j_max = 48;
        j_0 = 13;
        i_max = 1000;
        i_0 = 0;

        int x_begin, y_begin, y_end, x_end;

        int iteration = (i_max-i_0+32)/32;

        for (int k=0; k<iteration; k++){

            x_begin = i_max - 32*k;
            y_begin = j_max;
            y_end = j_0;
            if(k<iteration-1){
                x_end = x_begin - 31;
            }else{
                x_end = i_0;
            }

            int x_loc = x_begin - lane_id;
            int y_loc = y_begin;

            for(int step = 0; step < x_begin + y_begin - x_end - y_end+1; step++){
                if((x_loc + y_loc == x_begin + y_begin - step) & (y_loc>=y_end) & (x_loc>=x_end)){
                    single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                                   Delta, D, D_tree, 0, m, n);
                    y_loc--;
                }
                __syncwarp();
            }
        }

    }

}



//__global__ void multi_table_parallel_new_single(int blocks_per_table, int size, int* x_orl, int* y_orl, int* p_i_0, int* p_i_max, int* p_j_0, int* p_j_max, int tile_size, int* Delta, int* D, int* D_tree, int n, int m){
//
//    if(threadIdx.x ==0 && blockIdx.x == 0){
//
//        int num_table_offset = 0*(m+1)*(n+1);
//        int width = n+1;
//        int index;
//        int i;
//        int j;
//
//        int i_max = p_i_max[0];
//        int j_max = p_j_max[0];
//        int i_0 = p_i_0[0];
//        int j_0 = p_j_0[0];
//
//        j_max = 14;
//        j_0 = 13;
//        i_max =125;
//        i_0 = 0;
//
//        D[j_max + i_max*width + num_table_offset] = 0;
//
//        for (i = i_max - 1; i > i_0 - 1; i--) {
//            index = j_max + i*width + num_table_offset;
//            D[index] = 1 + D[index + width];
//        }
//
//        for (j = j_max - 1; j > j_0 - 1; j--) {
//            index = j + i_max*width + num_table_offset;
//            D[index] = 1 + D[index + 1];
//
//        }
//
//        for (i = i_max - 1; i > i_0 - 1; i--) {
//            for (j = j_max - 1; j > j_0 - 1; j--) {
//
//                if ((x_orl[i] == x_orl[i_0]) & (y_orl[j] == y_orl[j_0])) {
//
//                    index = j + i*width + num_table_offset;
//
//                    D[index] = min3_D(Delta[j+i*n] + D[index + 1 + width],
//                                      1 + D[index + width],
//                                      1 + D[index + 1]);
//                    D_tree[j + i*n] = D[index];
//
//                } else {
//
//                    index = j + i*width + num_table_offset;
//                    D[index] = min3_D(D_tree[j + i*n] + D[y_orl[j] + 1 + (x_orl[i] + 1) * width + num_table_offset],
//                                      1 + D[index + width],
//                                      1 + D[index + 1]);
//                }
//            }
//        }
//    }
//}


void multi_table_new (Stream& stream_mul, thrust::host_vector<int>& queue_h, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, dev::Queue<int, uint32_t> d_queue, int size, int L, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){
    vector<int> i_0_array (queue_h.size(),0);
    vector<int> i_max_array (queue_h.size(),0);
    vector<int> j_0_array (queue_h.size(),0);
    vector<int> j_max_array (queue_h.size(),0);

    for (int i=0; i<queue_h.size();i++){
        int row = queue_h[i] / L;
        int column = queue_h[i] % L;
        i_0_array[i] = x_kr[row];
        j_0_array[i] = y_kr[column];
        i_max_array[i] = x_orl[i_0_array[i]] + 1;
        j_max_array[i] = y_orl[j_0_array[i]] + 1;
    }

    printf("Total units = %d\n", (i_max_array[0]-i_0_array[0]+1)*(1+j_max_array[0]-j_0_array[0]));
//    printf("%d\n", (i_max_array[1]-i_0_array[1]+1)*(1+j_max_array[1]-j_0_array[1]));

    thrust::device_vector<int> d_i_0_array(i_0_array);
    thrust::device_vector<int> d_i_max_array(i_max_array);
    thrust::device_vector<int> d_j_0_array(j_0_array);
    thrust::device_vector<int> d_j_max_array(j_max_array);

    int *p_d_i_0_array = thrust::raw_pointer_cast(d_i_0_array.data());
    int *p_d_i_max_array = thrust::raw_pointer_cast(d_i_max_array.data());
    int *p_d_j_0_array = thrust::raw_pointer_cast(d_j_0_array.data());
    int *p_d_j_max_array = thrust::raw_pointer_cast(d_j_max_array.data());

    int block = 1024;
    int grid = 82;
    int tile = 1;
    int blocks_per_table = grid / queue_h.size();

    int units_per_thread_1 = (j_max_array[0] - j_0_array[0] + 12)/ 12;
    int units_per_thread_2 = (j_max_array[1] - j_0_array[1] + 52)/ 52;

    printf("%d, %d hello world TEST 1\n", units_per_thread_1, units_per_thread_2);
//    printf("Hello Cuda \n");

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream_mul.cuda_stream());

    multi_table_parallel_new<<<grid, block, 0, stream_mul.cuda_stream()>>>(blocks_per_table, queue_h.size(), p_x_orl_d, p_y_orl_d, p_d_i_0_array, p_d_i_max_array, p_d_j_0_array, p_d_j_max_array, tile, p_Delta_d, p_D_d, p_D_tree_d, n, m);

//    CHECK_CUDA(cudaDeviceSynchronize());
//    cudaError_t error = cudaGetLastError();
//    if(error != cudaSuccess)
//    {
//        // print the CUDA error message and exit
//        printf("CUDA error: %s\n", cudaGetErrorString(error));
//        exit(-1);
//    }


    cudaEventRecord(stop, stream_mul.cuda_stream());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("BLOCK = %.3fms\n", milliseconds);


//    printf("world %d, %d, %d\n", blocks_per_table, units_per_thread_1, units_per_thread_2);


    cudaEvent_t start_2, stop_2;
    float milliseconds_2 = 0;
    cudaEventCreate(&start_2);
    cudaEventCreate(&stop_2);
    cudaEventRecord(start_2, stream_mul.cuda_stream());

//    multi_table_parallel_new_single<<<grid, block, 0, stream_mul.cuda_stream()>>>(blocks_per_table, queue_h.size(), p_x_orl_d, p_y_orl_d, p_d_i_0_array, p_d_i_max_array, p_d_j_0_array, p_d_j_max_array, tile, p_Delta_d, p_D_d, p_D_tree_d, n, m);
    multi_table_parallel_new_mul_tables<<<grid, block, 0, stream_mul.cuda_stream()>>>(blocks_per_table, queue_h.size(), p_x_orl_d, p_y_orl_d, p_d_i_0_array, p_d_i_max_array, p_d_j_0_array, p_d_j_max_array, tile, p_Delta_d, p_D_d, p_D_tree_d, n, m);

//    CHECK_CUDA(cudaDeviceSynchronize());
//    cudaError_t error = cudaGetLastError();
//    if(error != cudaSuccess)
//    {
//        // print the CUDA error message and exit
//        printf("CUDA error: %s\n", cudaGetErrorString(error));
//        exit(-1);
//    }


    cudaEventRecord(stop_2, stream_mul.cuda_stream());
    cudaEventSynchronize(stop_2);
    cudaEventElapsedTime(&milliseconds, start_2, stop_2);
    cudaEventDestroy(start_2);
    cudaEventDestroy(stop_2);
    printf("MULTI_BLOCKS = %.3fms\n", milliseconds);


}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void __gpu_sync_range_update(int startBlockIdx, int endBlockIdx, int goalVal, volatile int *Arrayin, volatile int *Arrayout){
    int tid_in_blk = threadIdx.x;
    int nBlockNum = gridDim.x;
    int bid = blockIdx.x;

    if (bid >= startBlockIdx && bid <= endBlockIdx){
        if (tid_in_blk == 0){
                Arrayin[bid] = goalVal;
        }
    }

    __syncthreads();

    if (bid == startBlockIdx){
        if (tid_in_blk >= startBlockIdx & tid_in_blk <= endBlockIdx){
            while (Arrayin[tid_in_blk] != goalVal){}
        }
        __syncthreads();

        if (tid_in_blk >= startBlockIdx & tid_in_blk <= endBlockIdx){
            Arrayout[tid_in_blk] = goalVal;
        }
    }

    if (bid >= startBlockIdx && bid <= endBlockIdx){
        if (tid_in_blk == 0){
                while (Arrayout[bid] != goalVal){}
            }
    }
    __syncthreads();
}


//__global__ void multi_table_parallel_new_update(int* blocks_per_table, int size, int unit_per_thread, int* x_orl, int* y_orl, int* p_i_0, int* p_i_max, int* p_j_0, int* p_j_max, int tile_size, int* Delta, int* D, int* D_tree, int n, int m){
//    int block_begin = 0;
//    int block_end = blocks_per_table[0]-1;
//
//
//    for (int i=0; i< size; i++){
//
//        if(blockIdx.x<= block_end & blockIdx.x >= block_begin){
//
//            int i_max = p_i_max[i];
//            int j_max = p_j_max[i];
//            int i_0 = p_i_0[i];
//            int j_0 = p_j_0[i];
//
//            int x_loc = i_max - threadIdx.x;
//            int y_loc = j_max - unit_per_thread*(blockIdx.x-block_begin);
//            int k=0;
//            for (int step = 0; step < i_max+j_max-i_0-j_0+1; step++) {
//                if((x_loc + y_loc == i_max+j_max - step) && (k<unit_per_thread) && y_loc>=j_0 && x_loc>=i_0 ){
//                    single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
//                                   Delta, D, D_tree, i, m, n);
//                    y_loc--;
//                    k++;
//                }
////                __threadfence();
//                __gpu_sync_range_fence(block_begin,block_end,step+1,Array_in,Array_out);
//            }
//        }
//
//        if (block_end + blocks_per_table[i+1] <= 63){
//            block_begin = block_end + 1;
//            block_end = block_end + blocks_per_table[i+1];
//        } else{
//            block_begin = 0;
//            block_end = blocks_per_table[i+1]-1;
//        }
//    }
//
//}

__global__ void initializeArrayToZero() {
    int tid = threadIdx.x;
    if (tid < 64) {
        Array_in[tid] = 0;
        Array_out[tid] = 0;
    }
}

__global__ void multi_table_parallel_new_update_2(int* p_d_column_major, int* blocks_per_table, int size, int unit_per_thread, int* x_orl, int* y_orl, int* p_i_0, int* p_i_max, int* p_j_0, int* p_j_max, int tile_size, int* Delta, int* D, int* D_tree, int n, int m){
    int block_begin = 0;
    int block_end = blocks_per_table[0]-1;

//    int block_begin = 0;
//    int block_end = 63;


    for (int i=0; i< size; i++){

        if(blockIdx.x<= block_end & blockIdx.x >= block_begin){

            int i_max = p_i_max[i];
            int j_max = p_j_max[i];
            int i_0 = p_i_0[i];
            int j_0 = p_j_0[i];
            int x_loc, y_loc;

            if(p_d_column_major[i] == 1){
                x_loc = i_max - threadIdx.x;
                y_loc = j_max - unit_per_thread*(blockIdx.x-block_begin);
            }else{
                x_loc = i_max - unit_per_thread*(blockIdx.x-block_begin);
                y_loc = j_max - threadIdx.x;
            }
            int k=0;
            for (int step = 0; step < i_max+j_max-i_0-j_0+1; step++) {
                if((x_loc + y_loc == i_max+j_max - step) && (k<unit_per_thread) && y_loc>=j_0 && x_loc>=i_0 ){
                    single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                                   Delta, D, D_tree, i, m, n);
                    if(p_d_column_major[i] == 1){
                        y_loc--;
                    }else{
                        x_loc--;
                    }
                    k++;
                }
//                __gpu_sync_range_fence(block_begin,block_end,step+1,Array_in,Array_out);
                __threadfence();
                __gpu_sync_range_update(block_begin,block_end,step+1,Array_in,Array_out);
            }
        }

        if (block_end + blocks_per_table[i+1] <= 63){
            block_begin = block_end + 1;
            block_end = block_end + blocks_per_table[i+1];
        } else{
            block_begin = 0;
            block_end = blocks_per_table[i+1]-1;
        }
    }

}



void multi_table_new_update (Stream& stream_mul, float& total_milliseconds, thrust::host_vector<int>& queue_h, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, dev::Queue<int, uint32_t> d_queue, int size, int L, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){
    vector<int> i_0_array (queue_h.size(),0);
    vector<int> i_max_array (queue_h.size(),0);
    vector<int> j_0_array (queue_h.size(),0);
    vector<int> j_max_array (queue_h.size(),0);
    vector<int> blocks_per_table (queue_h.size(),0);
    vector<int> column_major (queue_h.size(),0);

    int unit_per_thread = 64;

    for (int i=0; i<queue_h.size();i++){
        int row = queue_h[i] / L;
        int column = queue_h[i] % L;
        i_0_array[i] = x_kr[row];
        j_0_array[i] = y_kr[column];
        i_max_array[i] = x_orl[i_0_array[i]] + 1;
        j_max_array[i] = y_orl[j_0_array[i]] + 1;
        blocks_per_table[i] = (j_max_array[i] - j_0_array[i] + unit_per_thread)/unit_per_thread;
        if(i_max_array[i]-i_0_array[i]+1 > j_max_array[i]-j_0_array[i]+1){
            blocks_per_table[i] = (j_max_array[i] - j_0_array[i] + unit_per_thread)/unit_per_thread;
            column_major[i] = 1;
        }else{
            blocks_per_table[i] = (i_max_array[i]-i_0_array[i]+ unit_per_thread)/unit_per_thread;
            column_major[i] = 0;
        }
        printf("--- TableSize = %d NumBlocks = %d row = %d, column = %d \n", (i_max_array[i]-i_0_array[i]+1)*(j_max_array[i]-j_0_array[i]+1), blocks_per_table[i], i_max_array[i]-i_0_array[i]+1, j_max_array[i]-j_0_array[i]+1);
    }

    thrust::device_vector<int> d_i_0_array(i_0_array);
    thrust::device_vector<int> d_i_max_array(i_max_array);
    thrust::device_vector<int> d_j_0_array(j_0_array);
    thrust::device_vector<int> d_j_max_array(j_max_array);
    thrust::device_vector<int> d_blocks_per_table(blocks_per_table);
    thrust::device_vector<int> d_column_major(column_major);

    int *p_d_i_0_array = thrust::raw_pointer_cast(d_i_0_array.data());
    int *p_d_i_max_array = thrust::raw_pointer_cast(d_i_max_array.data());
    int *p_d_j_0_array = thrust::raw_pointer_cast(d_j_0_array.data());
    int *p_d_j_max_array = thrust::raw_pointer_cast(d_j_max_array.data());
    int *p_d_blocks_per_table = thrust::raw_pointer_cast(d_blocks_per_table.data());
    int *p_d_column_major = thrust::raw_pointer_cast(d_column_major.data());

    int block = 1024;
    int grid = 64;
    int tile = 1;
    int block_per_table = grid / queue_h.size();

    printf("AAAAAAAAAAAAAA %u hello \n", queue_h.size());

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream_mul.cuda_stream());

//    multi_table_parallel_new_update<<<grid, block, 0, stream_mul.cuda_stream()>>>(p_d_blocks_per_table, queue_h.size(), unit_per_thread, p_x_orl_d, p_y_orl_d, p_d_i_0_array, p_d_i_max_array, p_d_j_0_array, p_d_j_max_array, tile, p_Delta_d, p_D_d, p_D_tree_d, n, m);
    multi_table_parallel_new_update_2<<<grid, block, 0, stream_mul.cuda_stream()>>>(p_d_column_major, p_d_blocks_per_table, queue_h.size(), unit_per_thread, p_x_orl_d, p_y_orl_d, p_d_i_0_array, p_d_i_max_array, p_d_j_0_array, p_d_j_max_array, tile, p_Delta_d, p_D_d, p_D_tree_d, n, m);

    CHECK_CUDA(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaEventRecord(stop, stream_mul.cuda_stream());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("Measured time for MUTIL_New_Test parallel execution = %.3fms\n", milliseconds);
    total_milliseconds+=milliseconds;

}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void multi_table_parallel_new_update_8(int* p_d_column_major, int* unit_per_threads, int size, int* x_orl, int* y_orl, int* p_i_0, int* p_i_max, int* p_j_0, int* p_j_max, int tile_size, int* Delta, int* D, int* D_tree, int n, int m){

//    if(blockIdx.x == 0 && threadIdx.x == 0){
//        printf("table no. %d is computed\n", 0);
//    }

    for (int i=blockIdx.x;i<size; i+=gridDim.x){
        int i_max = p_i_max[i];
        int j_max = p_j_max[i];
        int i_0 = p_i_0[i];
        int j_0 = p_j_0[i];
        int x_loc, y_loc;
        int num_in_thread = i % 2143;

        if(p_d_column_major[i] == 1){
            x_loc = i_max - threadIdx.x;
            y_loc = j_max;
        }else{
            x_loc = i_max;
            y_loc = j_max - threadIdx.x;
        }

        int k=0;
        for (int step = 0; step < i_max+j_max-i_0-j_0+1; step++) {
            if((x_loc + y_loc == i_max+j_max - step) && (k<unit_per_threads[i]) && y_loc>=j_0 && x_loc>=i_0 ){
                single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                               Delta, D, D_tree, num_in_thread, m, n);
                if(p_d_column_major[i] == 1){
                    y_loc--;
                }else{
                    x_loc--;
                }
                k++;
            }
            __syncthreads();
        }
    }

}

void multi_table_new_update_8 (Stream& stream_mul, float& total_milliseconds, thrust::host_vector<int>& queue_h, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, dev::Queue<int, uint32_t> d_queue, int size, int L, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){
    vector<int> i_0_array (queue_h.size(),0);
    vector<int> i_max_array (queue_h.size(),0);
    vector<int> j_0_array (queue_h.size(),0);
    vector<int> j_max_array (queue_h.size(),0);
    vector<int> blocks_per_table (queue_h.size(),0);
    vector<int> column_major (queue_h.size(),0);
    vector<int> unit_per_threads (queue_h.size(),0);

    for (int i=0; i<queue_h.size();i++){
        int row = queue_h[i] / L;
        int column = queue_h[i] % L;
        i_0_array[i] = x_kr[row];
        j_0_array[i] = y_kr[column];
        i_max_array[i] = x_orl[i_0_array[i]] + 1;
        j_max_array[i] = y_orl[j_0_array[i]] + 1;

        if(i_max_array[i]-i_0_array[i]+1 > j_max_array[i]-j_0_array[i]+1){
            unit_per_threads[i] = j_max_array[i]-j_0_array[i]+1;
            column_major[i] = 1;
        }else{
            unit_per_threads[i] = i_max_array[i]-i_0_array[i]+1;
            column_major[i] = 0;
        }
//        printf("--- TableSize = %d unit_per_thread = %d row = %d, column = %d \n", (i_max_array[i]-i_0_array[i]+1)*(j_max_array[i]-j_0_array[i]+1), unit_per_threads[i], i_max_array[i]-i_0_array[i]+1, j_max_array[i]-j_0_array[i]+1);
    }

    thrust::device_vector<int> d_i_0_array(i_0_array);
    thrust::device_vector<int> d_i_max_array(i_max_array);
    thrust::device_vector<int> d_j_0_array(j_0_array);
    thrust::device_vector<int> d_j_max_array(j_max_array);
    thrust::device_vector<int> d_blocks_per_table(blocks_per_table);
    thrust::device_vector<int> d_column_major(column_major);
    thrust::device_vector<int> d_unit_per_threads(unit_per_threads);

    int *p_d_i_0_array = thrust::raw_pointer_cast(d_i_0_array.data());
    int *p_d_i_max_array = thrust::raw_pointer_cast(d_i_max_array.data());
    int *p_d_j_0_array = thrust::raw_pointer_cast(d_j_0_array.data());
    int *p_d_j_max_array = thrust::raw_pointer_cast(d_j_max_array.data());
    int *p_d_blocks_per_table = thrust::raw_pointer_cast(d_blocks_per_table.data());
    int *p_d_column_major = thrust::raw_pointer_cast(d_column_major.data());
    int *p_d_unit_per_threads = thrust::raw_pointer_cast(d_unit_per_threads.data());


    int block = 1024;
    int grid = 256;
    int tile = 1;
    int block_per_table = grid / queue_h.size();

    printf("--- BLOCK METHOD: %u\n", queue_h.size());

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream_mul.cuda_stream());

//    multi_table_parallel_new_update<<<grid, block, 0, stream_mul.cuda_stream()>>>(p_d_blocks_per_table, queue_h.size(), unit_per_thread, p_x_orl_d, p_y_orl_d, p_d_i_0_array, p_d_i_max_array, p_d_j_0_array, p_d_j_max_array, tile, p_Delta_d, p_D_d, p_D_tree_d, n, m);
    multi_table_parallel_new_update_8<<<grid, block, 0, stream_mul.cuda_stream()>>>(p_d_column_major, p_d_unit_per_threads, queue_h.size(), p_x_orl_d, p_y_orl_d, p_d_i_0_array, p_d_i_max_array, p_d_j_0_array, p_d_j_max_array, tile, p_Delta_d, p_D_d, p_D_tree_d, n, m);

    CHECK_CUDA(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaEventRecord(stop, stream_mul.cuda_stream());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("Measured time for BLOCK = %.3fms\n", milliseconds);
    total_milliseconds+=milliseconds;

}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void multi_table_parallel_warp_18(int* p_d_column_major, int* unit_per_threads, int size, int* x_orl, int* y_orl, int* p_i_0, int* p_i_max, int* p_j_0, int* p_j_max, int tile_size, int* Delta, int* D, int* D_tree, int n, int m){

    int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
    int lane_id = (threadIdx.x + blockIdx.x * blockDim.x) % 32;

    if(warp_id <2143) {
//        for (int i = warp_id; i < size; i += (gridDim.x * blockDim.x) / 32) {
        for (int i = warp_id; i < size; i += 2143) {

            int i_max = p_i_max[i];
            int j_max = p_j_max[i];
            int i_0 = p_i_0[i];
            int j_0 = p_j_0[i];
            int x_begin, y_begin, y_end, x_end;
            int iteration;
            if (p_d_column_major[i] == 1) {
                iteration = (i_max - i_0 + 32) / 32;

                for (int k = 0; k < iteration; k++) {

                    x_begin = i_max - 32 * k;
                    y_begin = j_max;
                    y_end = j_0;
                    if (k < iteration - 1) {
                        x_end = x_begin - 31;
                    } else {
                        x_end = i_0;
                    }

                    int x_loc = x_begin - lane_id;
                    int y_loc = y_begin;

                    for (int step = 0; step < x_begin + y_begin - x_end - y_end + 1; step++) {
                        if ((x_loc + y_loc == x_begin + y_begin - step) & (y_loc >= y_end) & (x_loc >= x_end)) {
                            single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                                           Delta, D, D_tree, warp_id, m, n);
                            y_loc--;
                        }
                        __syncwarp();
                    }

                }


            } else {
                iteration = (j_max - j_0 + 32) / 32;

                for (int k = 0; k < iteration; k++) {

                    x_begin = i_max;
                    x_end = i_0;
                    y_begin = j_max - 32 * k;

                    if (k < iteration - 1) {
                        y_end = y_begin - 31;
                    } else {
                        y_end = j_0;
                    }

                    int x_loc = x_begin;
                    int y_loc = y_begin - lane_id;

                    for (int step = 0; step < x_begin + y_begin - x_end - y_end + 1; step++) {
                        if ((x_loc + y_loc == x_begin + y_begin - step) & (y_loc >= y_end) & (x_loc >= x_end)) {
                            single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                                           Delta, D, D_tree, warp_id, m, n);
                            x_loc--;
                        }
                        __syncwarp();
                    }


                }

            }

        }
    }
}


void multi_table_warp_18 (Stream& stream_mul, float& total_milliseconds, thrust::host_vector<int>& queue_h, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, dev::Queue<int, uint32_t> d_queue, int size, int L, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){
    vector<int> i_0_array (queue_h.size(),0);
    vector<int> i_max_array (queue_h.size(),0);
    vector<int> j_0_array (queue_h.size(),0);
    vector<int> j_max_array (queue_h.size(),0);
    vector<int> blocks_per_table (queue_h.size(),0);
    vector<int> column_major (queue_h.size(),0);
    vector<int> unit_per_threads (queue_h.size(),0);

    for (int i=0; i<queue_h.size();i++){
        int row = queue_h[i] / L;
        int column = queue_h[i] % L;
        i_0_array[i] = x_kr[row];
        j_0_array[i] = y_kr[column];
        i_max_array[i] = x_orl[i_0_array[i]] + 1;
        j_max_array[i] = y_orl[j_0_array[i]] + 1;

        if(i_max_array[i]-i_0_array[i]+1 > j_max_array[i]-j_0_array[i]+1){
            unit_per_threads[i] = j_max_array[i]-j_0_array[i]+1;
            column_major[i] = 1;
        }else{
            unit_per_threads[i] = i_max_array[i]-i_0_array[i]+1;
            column_major[i] = 0;
        }
//        printf("--- TableSize = %d unit_per_thread = %d row = %d, column = %d \n", (i_max_array[i]-i_0_array[i]+1)*(j_max_array[i]-j_0_array[i]+1), unit_per_threads[i], i_max_array[i]-i_0_array[i]+1, j_max_array[i]-j_0_array[i]+1);
    }

    thrust::device_vector<int> d_i_0_array(i_0_array);
    thrust::device_vector<int> d_i_max_array(i_max_array);
    thrust::device_vector<int> d_j_0_array(j_0_array);
    thrust::device_vector<int> d_j_max_array(j_max_array);
    thrust::device_vector<int> d_blocks_per_table(blocks_per_table);
    thrust::device_vector<int> d_column_major(column_major);
    thrust::device_vector<int> d_unit_per_threads(unit_per_threads);

    int *p_d_i_0_array = thrust::raw_pointer_cast(d_i_0_array.data());
    int *p_d_i_max_array = thrust::raw_pointer_cast(d_i_max_array.data());
    int *p_d_j_0_array = thrust::raw_pointer_cast(d_j_0_array.data());
    int *p_d_j_max_array = thrust::raw_pointer_cast(d_j_max_array.data());
    int *p_d_blocks_per_table = thrust::raw_pointer_cast(d_blocks_per_table.data());
    int *p_d_column_major = thrust::raw_pointer_cast(d_column_major.data());
    int *p_d_unit_per_threads = thrust::raw_pointer_cast(d_unit_per_threads.data());


    int block = 1024;
    int grid = 256;
    int tile = 1;
    int block_per_table = grid / queue_h.size();

    printf("--- WARP METHOD: %d \n", queue_h.size());

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream_mul.cuda_stream());

    multi_table_parallel_warp_18<<<grid, block, 0, stream_mul.cuda_stream()>>>(p_d_column_major, p_d_unit_per_threads, queue_h.size(), p_x_orl_d, p_y_orl_d, p_d_i_0_array, p_d_i_max_array, p_d_j_0_array, p_d_j_max_array, tile, p_Delta_d, p_D_d, p_D_tree_d, n, m);
//    multi_table_parallel_new_mul_tables<<<grid, block, 0, stream_mul.cuda_stream()>>>(0, queue_h.size(), p_x_orl_d, p_y_orl_d, p_d_i_0_array, p_d_i_max_array, p_d_j_0_array, p_d_j_max_array, tile, p_Delta_d, p_D_d, p_D_tree_d, n, m);

    CHECK_CUDA(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaEventRecord(stop, stream_mul.cuda_stream());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("Measured time for WARP = %.3fms\n", milliseconds);
    total_milliseconds+=milliseconds;

}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void multi_table_parallel_warp_28(int* p_d_column_major, int* unit_per_threads, int size, int* x_orl, int* y_orl, int* p_i_0, int* p_i_max, int* p_j_0, int* p_j_max, int tile_size, int* Delta, int* D, int* D_tree, int n, int m){

    int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
    int lane_id = (threadIdx.x + blockIdx.x * blockDim.x) % 32;

    if(warp_id <2048) {
//        for (int i = warp_id; i < size; i += (gridDim.x * blockDim.x) / 32) {
        for (int i = warp_id; i < size; i += 2048) {

            int i_max = p_i_max[i];
            int j_max = p_j_max[i];
            int i_0 = p_i_0[i];
            int j_0 = p_j_0[i];
            int x_begin, y_begin, y_end, x_end;
            int iteration;
            if (p_d_column_major[i] == 1) {
                iteration = (i_max - i_0 + 32) / 32;

                for (int k = 0; k < iteration; k++) {

                    x_begin = i_max - 32 * k;
                    y_begin = j_max;
                    y_end = j_0;
                    if (k < iteration - 1) {
                        x_end = x_begin - 31;
                    } else {
                        x_end = i_0;
                    }

                    int x_loc = x_begin - lane_id;
                    int y_loc = y_begin;

                    for (int step = 0; step < x_begin + y_begin - x_end - y_end + 1; step++) {
                        if ((x_loc + y_loc == x_begin + y_begin - step) & (y_loc >= y_end) & (x_loc >= x_end)) {
                            single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                                           Delta, D, D_tree, warp_id, m, n);
                            y_loc--;
                        }
                        __syncwarp();
                    }

                }


            } else {
                iteration = (j_max - j_0 + 32) / 32;

                for (int k = 0; k < iteration; k++) {

                    x_begin = i_max;
                    x_end = i_0;
                    y_begin = j_max - 32 * k;

                    if (k < iteration - 1) {
                        y_end = y_begin - 31;
                    } else {
                        y_end = j_0;
                    }

                    int x_loc = x_begin;
                    int y_loc = y_begin - lane_id;

                    for (int step = 0; step < x_begin + y_begin - x_end - y_end + 1; step++) {
                        if ((x_loc + y_loc == x_begin + y_begin - step) & (y_loc >= y_end) & (x_loc >= x_end)) {
                            single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                                           Delta, D, D_tree, warp_id, m, n);
                            x_loc--;
                        }
                        __syncwarp();
                    }


                }

            }

        }
    }
}



void multi_table_warp_28 (Stream& stream_mul, float& total_milliseconds, thrust::host_vector<int>& queue_h, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, dev::Queue<int, uint32_t> d_queue, int size, int L, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){
    vector<int> i_0_array (queue_h.size(),0);
    vector<int> i_max_array (queue_h.size(),0);
    vector<int> j_0_array (queue_h.size(),0);
    vector<int> j_max_array (queue_h.size(),0);
    vector<int> blocks_per_table (queue_h.size(),0);
    vector<int> column_major (queue_h.size(),0);
    vector<int> unit_per_threads (queue_h.size(),0);

    for (int i=0; i<queue_h.size();i++){
        int row = queue_h[i] / L;
        int column = queue_h[i] % L;
        i_0_array[i] = x_kr[row];
        j_0_array[i] = y_kr[column];
        i_max_array[i] = x_orl[i_0_array[i]] + 1;
        j_max_array[i] = y_orl[j_0_array[i]] + 1;

        if(i_max_array[i]-i_0_array[i]+1 > j_max_array[i]-j_0_array[i]+1){
            unit_per_threads[i] = j_max_array[i]-j_0_array[i]+1;
            column_major[i] = 1;
        }else{
            unit_per_threads[i] = i_max_array[i]-i_0_array[i]+1;
            column_major[i] = 0;
        }
//        printf("--- TableSize = %d unit_per_thread = %d row = %d, column = %d \n", (i_max_array[i]-i_0_array[i]+1)*(j_max_array[i]-j_0_array[i]+1), unit_per_threads[i], i_max_array[i]-i_0_array[i]+1, j_max_array[i]-j_0_array[i]+1);
    }

    thrust::device_vector<int> d_i_0_array(i_0_array);
    thrust::device_vector<int> d_i_max_array(i_max_array);
    thrust::device_vector<int> d_j_0_array(j_0_array);
    thrust::device_vector<int> d_j_max_array(j_max_array);
    thrust::device_vector<int> d_blocks_per_table(blocks_per_table);
    thrust::device_vector<int> d_column_major(column_major);
    thrust::device_vector<int> d_unit_per_threads(unit_per_threads);

    int *p_d_i_0_array = thrust::raw_pointer_cast(d_i_0_array.data());
    int *p_d_i_max_array = thrust::raw_pointer_cast(d_i_max_array.data());
    int *p_d_j_0_array = thrust::raw_pointer_cast(d_j_0_array.data());
    int *p_d_j_max_array = thrust::raw_pointer_cast(d_j_max_array.data());
    int *p_d_blocks_per_table = thrust::raw_pointer_cast(d_blocks_per_table.data());
    int *p_d_column_major = thrust::raw_pointer_cast(d_column_major.data());
    int *p_d_unit_per_threads = thrust::raw_pointer_cast(d_unit_per_threads.data());


    int block = 256;
    int grid = 256;
    int tile = 1;
    int block_per_table = grid / queue_h.size();

    printf("--- New WARP METHOD: %d \n", queue_h.size());

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream_mul.cuda_stream());

    multi_table_parallel_warp_28<<<grid, block, 0, stream_mul.cuda_stream()>>>(p_d_column_major, p_d_unit_per_threads, queue_h.size(), p_x_orl_d, p_y_orl_d, p_d_i_0_array, p_d_i_max_array, p_d_j_0_array, p_d_j_max_array, tile, p_Delta_d, p_D_d, p_D_tree_d, n, m);
//    multi_table_parallel_new_mul_tables<<<grid, block, 0, stream_mul.cuda_stream()>>>(0, queue_h.size(), p_x_orl_d, p_y_orl_d, p_d_i_0_array, p_d_i_max_array, p_d_j_0_array, p_d_j_max_array, tile, p_Delta_d, p_D_d, p_D_tree_d, n, m);

    CHECK_CUDA(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaEventRecord(stop, stream_mul.cuda_stream());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("Measured time for WARP = %.3fms\n", milliseconds);
    total_milliseconds+=milliseconds;

}




vector<vector<int>> parallel_standard_ted_2_28(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threads, vector<vector<int>>& x_adj, vector<vector<int>>& y_adj){

    int K = (int)x_kr.size();
    int L = (int)y_kr.size();


    vector<int> depth(K*L, 0);

    int number_x_nodes = x_adj.size();
    int number_y_nodes = y_adj.size();


    vector<int> x_keyroot_depth_2(K, 0);
    vector<int> y_keyroot_depth_2(L, 0);

// Preprocessing

    for (int i=0;i<K;i++){
        int node = x_kr[i];
        if((node == x_orl[node])||(node == x_orl[node]-1)){
            x_keyroot_depth_2[i] = 0;
        }else{
            for (int j=0;j<i;j++){
                int node_2 = x_kr[j];
                if ((node <= node_2)&&(x_orl[node] >= node_2)){
                    x_keyroot_depth_2[i] = max(x_keyroot_depth_2[i],x_keyroot_depth_2[j]+1);
                }
            }
        }
    }

    for (int i=0;i<L;i++){
        int node = y_kr[i];
        if((node == y_orl[node])||(node == y_orl[node]-1)){
            y_keyroot_depth_2[i] = 0;
        }else{
            for (int j=0;j<i;j++){
                int node_2 = y_kr[j];
                if ((node <= node_2)&&(y_orl[node] >= node_2)){
                    y_keyroot_depth_2[i] = max(y_keyroot_depth_2[i],y_keyroot_depth_2[j]+1);
                }
            }
        }
    }

    int max_depth = 0;
    for (int i=0;i<K*L;i++){
        depth[i] = x_keyroot_depth_2[i/L] + y_keyroot_depth_2[i%L];
        if(depth[i] > max_depth){
            max_depth = depth[i];
        }
    }
//    printf("max_depth = %d\n", max_depth);


    // GPU Programming
    int blockSize = 256;
    int current_depth = 0;
    double limitation = (8*1024*1024*1024.0)/((m+1)*(n+1)*4);
//    printf("limitation is %u\n", int(limitation));
    int batch_size = min(K*L,int(limitation));

    // x_orl, x_kr, y_orl, y_kr
    thrust::device_vector<int> x_orl_d(x_orl);
    thrust::device_vector<int> x_kr_d(x_kr);
    thrust::device_vector<int> y_orl_d(y_orl);
    thrust::device_vector<int> y_kr_d(y_kr);

    // Delta, D_tree
    vector<int> Delta_trans (m*n);
    for (int i=0; i<m;i++){
        for (int j=0; j<n;j++){
            Delta_trans[i*n+j] = Delta[i][j];
        }
    }
    thrust::device_vector<int> Delta_d(Delta_trans);
    vector<int> D_tree_trans(m*n, 0);
    thrust::device_vector<int> D_tree_d(D_tree_trans);

    // depth
    thrust::device_vector<int> depth_d(depth);

    // D
//    vector<int> D_trans((m+1)*(n+1)*batch_size, 0);
//    thrust::device_vector<int> D_d(D_trans);
    thrust::device_vector<int> D_d((m+1)*(n+1)*batch_size, 0);
//    thrust::device_vector<int> D_d_2((m+1)*(n+1)*batch_size, 0);

    // Size
//    vector<int> size(K*L,0);
//    thrust::device_vector<int> size_d(size);

    // Pointer: x_orl, x_kr, y_orl, y_kr, Delta, D_tree, depth, worklist1, worklist2, D
    int *p_x_orl_d = thrust::raw_pointer_cast(x_orl_d.data());
    int *p_x_kr_d = thrust::raw_pointer_cast(x_kr_d.data());
    int *p_y_orl_d = thrust::raw_pointer_cast(y_orl_d.data());
    int *p_y_kr_d = thrust::raw_pointer_cast(y_kr_d.data());
    int *p_Delta_d = thrust::raw_pointer_cast(Delta_d.data());
    int *p_D_tree_d = thrust::raw_pointer_cast(D_tree_d.data());
    int *p_depth_d = thrust::raw_pointer_cast(depth_d.data());
    int *p_D_d = thrust::raw_pointer_cast(D_d.data());
//    int *p_D_d_2 = thrust::raw_pointer_cast(D_d_2.data());

    total_milliseconds = 0;

    Stream stream;

    ArrayView<int> depth_d_view(depth_d);

    Queue<int> queue1;
    queue1.Init(K*L*sizeof(int));
    dev::Queue<int,uint32_t> d_queue1 = queue1.DeviceObject();

    Queue<int> queue2;
    queue2.Init(K*L*sizeof(int));
    dev::Queue<int,uint32_t> d_queue2 = queue2.DeviceObject();

    Queue<int> large_queue;
    large_queue.Init(K*L*sizeof(int));
    dev::Queue<int,uint32_t> d_large_queue = large_queue.DeviceObject();
    large_queue.set_size(stream, 0);

    Queue<int> block_size_queue;
    block_size_queue.Init(K*L*sizeof(int));
    dev::Queue<int,uint32_t> d_block_size_queue = block_size_queue.DeviceObject();
    block_size_queue.set_size(stream, 0);


    // Fetch task

    int numBlocks = (int)(depth_d_view.size() + blockSize - 1) / blockSize;
    fetch_task<<<numBlocks,blockSize, 0, stream.cuda_stream()>>>(depth_d_view, d_queue1, 0);
    stream.Sync();

    initializeArrayToZero<<<1, 64>>>();
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }



    while (queue1.size() !=0 || large_queue.size() != 0 || block_size_queue.size() != 0) {

        printf("\ncurrent depth = %d, batch_size = %d\n", current_depth, batch_size);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        if (current_depth == -1){

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            int num_task = queue1.size();
            numBlocks = (num_task + blockSize - 1) / blockSize;
            numBlocks = 128;

            printf("---------------------Hello\n");


            cudaEventRecord(start, stream.cuda_stream());
            if(queue1.size()>0){
                printf("--- SINGLE METHOD: %d\n", queue1.size());
                simple_parallel_new<<<numBlocks, blockSize, 0, stream.cuda_stream()>>>(p_x_orl_d, p_x_kr_d, p_y_orl_d, p_y_kr_d, p_Delta_d, p_D_d, p_D_tree_d, n, m, L, d_queue1, batch_size);
            }

            cudaEventRecord(stop, stream.cuda_stream());
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            printf("Measured time for SINGLE = %.3fms\n", milliseconds);
            total_milliseconds+=milliseconds;

            if(large_queue.size() != 0){
                thrust::host_vector<int> large_queue_h(large_queue.size());
                cudaMemcpyAsync(thrust::raw_pointer_cast(large_queue_h.data()),large_queue.data(), large_queue.size() * sizeof(int),cudaMemcpyDeviceToHost, stream.cuda_stream());
                stream.Sync();
                multi_table_warp_28(stream, total_milliseconds, large_queue_h, x_orl, x_kr, y_orl, y_kr, d_queue1, large_queue_h.size(), L, n, m, p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d);

            }

            if(block_size_queue.size() != 0){
                thrust::host_vector<int> block_size_queue_h(block_size_queue.size());
                cudaMemcpyAsync(thrust::raw_pointer_cast(block_size_queue_h.data()),block_size_queue.data(), block_size_queue.size() * sizeof(int),cudaMemcpyDeviceToHost, stream.cuda_stream());
                stream.Sync();
                multi_table_new_update_8(stream, total_milliseconds, block_size_queue_h, x_orl, x_kr, y_orl, y_kr, d_queue1, block_size_queue_h.size(), L, n, m, p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d);
//                multi_table_new_update(stream, total_milliseconds, block_size_queue_h, x_orl, x_kr, y_orl, y_kr, d_queue1, block_size_queue_h.size(), L, n, m, p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d);
            }


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        }else if (queue1.size()<=5 && queue1.size()>6){

            thrust::host_vector<int> queue_h(queue1.size());
            cudaMemcpyAsync(thrust::raw_pointer_cast(queue_h.data()),
                            queue1.data(), queue1.size() * sizeof(int),
                            cudaMemcpyDeviceToHost, stream.cuda_stream());
            stream.Sync();


//            multi_table_new(stream, queue_h, x_orl, x_kr, y_orl, y_kr, d_queue1, queue_h.size(), L, n, m, p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d);
//            multi_table_new_update_8(stream, total_milliseconds, queue_h, x_orl, x_kr, y_orl, y_kr, d_queue1, queue_h.size(), L, n, m, p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d);
            multi_table_warp_18(stream, total_milliseconds, queue_h, x_orl, x_kr, y_orl, y_kr, d_queue1, queue_h.size(), L, n, m, p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d);
//


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        }else if (current_depth == -1){

            cudaEvent_t start, stop;
            float milliseconds = 0;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, stream.cuda_stream());


            single_table_new_9(stream, K*L-1, 0, x_orl, x_kr, y_orl, y_kr, L, blockSize, n, m , p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d);
            cudaEventRecord(stop, stream.cuda_stream());
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            printf("Measured time for parallel execution = %.3fms\n", milliseconds);
            total_milliseconds+=milliseconds;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        }else{
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            int num_task = queue1.size();
            numBlocks = (num_task + blockSize - 1) / blockSize;
//            numBlocks = 128;
            cudaEventRecord(start, stream.cuda_stream());

//            if(queue1.size()>0){
//                printf("--- SINGLE METHOD: %d\n", queue1.size());
//                simple_parallel_new<<<numBlocks, blockSize, 0, stream.cuda_stream()>>>(p_x_orl_d, p_x_kr_d, p_y_orl_d, p_y_kr_d, p_Delta_d, p_D_d, p_D_tree_d, n, m, L, d_queue1, batch_size);
//            }


            int num_batch = (num_task + batch_size - 1) / (batch_size);
            int final_end = num_task % batch_size;
            if (final_end == 0) { final_end = batch_size; }


            for (int x = 0; x < num_batch; x++) {
                if (x != num_batch - 1) {
                    numBlocks = (batch_size + blockSize - 1) / blockSize;
//                    printf("begin = %d, end = %d\n", 0, batch_size);
                    parallel_task_stage<<<numBlocks, blockSize, 0, stream.cuda_stream()>>>(p_x_orl_d, p_x_kr_d,
                                                                                           p_y_orl_d, p_y_kr_d,
                                                                                           p_Delta_d, p_D_d, p_D_tree_d,
                                                                                           n, m, L, d_queue1,
                                                                                           batch_size, x, batch_size);
                } else {
                    numBlocks = (final_end + blockSize - 1) / blockSize;
//                    printf("begin = %d, end = %d\n", 0, final_end);
                    parallel_task_stage<<<numBlocks, blockSize, 0, stream.cuda_stream()>>>(p_x_orl_d, p_x_kr_d,
                                                                                           p_y_orl_d, p_y_kr_d,
                                                                                           p_Delta_d, p_D_d, p_D_tree_d,
                                                                                           n, m, L, d_queue1,
                                                                                           batch_size, x, final_end);
                }
            }




            cudaEventRecord(stop, stream.cuda_stream());
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            printf("Measured time for SINGLE = %.3fms\n", milliseconds);
            total_milliseconds+=milliseconds;

//            if(queue1.size()>0) {
//                simple_thread(stream, total_milliseconds, limitation, d_queue1, L, n, m, p_x_orl_d, p_y_orl_d, p_x_kr_d,
//                              p_y_kr_d, p_Delta_d, p_D_d, p_D_tree_d);
//            }


            if(large_queue.size() != 0){
                thrust::host_vector<int> large_queue_h(large_queue.size());
                cudaMemcpyAsync(thrust::raw_pointer_cast(large_queue_h.data()),large_queue.data(), large_queue.size() * sizeof(int),cudaMemcpyDeviceToHost, stream.cuda_stream());
                stream.Sync();
//                multi_table_new_update_8(stream, total_milliseconds, large_queue_h, x_orl, x_kr, y_orl, y_kr, d_queue1, large_queue_h.size(), L, n, m, p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d);
//                multi_table_new_update(stream, total_milliseconds, large_queue_h, x_orl, x_kr, y_orl, y_kr, d_queue1, large_queue_h.size(), L, n, m, p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d);
                multi_table_warp_28(stream, total_milliseconds, large_queue_h, x_orl, x_kr, y_orl, y_kr, d_queue1, large_queue_h.size(), L, n, m, p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d);

            }

            if(block_size_queue.size() != 0){
                thrust::host_vector<int> block_size_queue_h(block_size_queue.size());
                cudaMemcpyAsync(thrust::raw_pointer_cast(block_size_queue_h.data()),block_size_queue.data(), block_size_queue.size() * sizeof(int),cudaMemcpyDeviceToHost, stream.cuda_stream());
                stream.Sync();
                multi_table_new_update_8(stream, total_milliseconds, block_size_queue_h, x_orl, x_kr, y_orl, y_kr, d_queue1, block_size_queue_h.size(), L, n, m, p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d);

            }

        }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        queue1.set_size(stream, 0);
        large_queue.set_size(stream, 0);
        block_size_queue.set_size(stream, 0);
        current_depth++;
        numBlocks = (depth_d_view.size() + blockSize - 1) / blockSize;
        fetch_task<<<numBlocks, blockSize, 0, stream.cuda_stream()>>>(depth_d_view, d_queue1, current_depth);
        stream.Sync();

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Sort by Size
//        queue2.set_size(stream, queue1.size());
//        numBlocks = (queue1.size() + blockSize - 1) / blockSize;
//
//        fetch_size_queue<<<numBlocks, blockSize, 0, stream.cuda_stream()>>>(d_queue2, d_queue1,p_x_orl_d, p_y_orl_d, p_x_kr_d, p_y_kr_d, L);
//        stream.Sync();
//
//        thrust::sort_by_key(queue2.getRawPointer(), queue2.getRawPointer()+queue1.size(), queue1.getRawPointer());
//        cudaStreamSynchronize(stream.cuda_stream());
//        queue2.set_size(stream,0);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//        if(current_depth < 6 && current_depth >4) {
//            printQueueVector<<<1, 1, 0, stream.cuda_stream()>>>(d_queue1);
            numBlocks = (queue1.size() + blockSize - 1) / blockSize;
            filter<<<numBlocks, blockSize, 0, stream.cuda_stream()>>>(d_queue1, d_queue2, d_large_queue, d_block_size_queue, 10000000, 10000000);
//            filter<<<numBlocks, blockSize, 0, stream.cuda_stream()>>>(d_queue1, d_queue2, d_large_queue, d_block_size_queue,15, 10000000);
            stream.Sync();
            int old_size = queue1.size();
            queue1.set_size(stream,old_size - large_queue.size()-block_size_queue.size());

            printf("//////////////////////////////////\n");
            printf("QUEUE1.size = %d\n", queue1.size());
            printf("Large_queue.size = %d\n", large_queue.size());
            printf("block_size_queue.size = %d\n", block_size_queue.size());
//        }
    }


    int final = D_tree_d[0];
    printf("The total final distance is %u\n", final);
    printf("hello\n");
    printf("Measured time for total parallel execution = %.3fms\n", total_milliseconds);

    cudaMemcpy(D_tree_trans.data(), p_D_tree_d, sizeof(int) * m * n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("D[0][13] = %d, D[61][0] = %d\n", D_tree_trans[0*1000+13], D_tree_trans[61*1000+0]);

    vector<vector<int>> final_result(m,vector<int>(n,0));
    return final_result;

}


vector<vector<int>> parallel_standard_ted_Zhang(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threads, vector<vector<int>>& x_adj, vector<vector<int>>& y_adj){

    int K = (int)x_kr.size();
    int L = (int)y_kr.size();


    vector<int> depth(K*L, 0);

    int number_x_nodes = x_adj.size();
    int number_y_nodes = y_adj.size();


    vector<int> x_keyroot_depth_2(K, 0);
    vector<int> y_keyroot_depth_2(L, 0);

// Preprocessing

    for (int i=0;i<K;i++){
        int node = x_kr[i];
        if((node == x_orl[node])||(node == x_orl[node]-1)){
            x_keyroot_depth_2[i] = 0;
        }else{
            for (int j=0;j<i;j++){
                int node_2 = x_kr[j];
                if ((node <= node_2)&&(x_orl[node] >= node_2)){
                    x_keyroot_depth_2[i] = max(x_keyroot_depth_2[i],x_keyroot_depth_2[j]+1);
                }
            }
        }
    }

    for (int i=0;i<L;i++){
        int node = y_kr[i];
        if((node == y_orl[node])||(node == y_orl[node]-1)){
            y_keyroot_depth_2[i] = 0;
        }else{
            for (int j=0;j<i;j++){
                int node_2 = y_kr[j];
                if ((node <= node_2)&&(y_orl[node] >= node_2)){
                    y_keyroot_depth_2[i] = max(y_keyroot_depth_2[i],y_keyroot_depth_2[j]+1);
                }
            }
        }
    }

    int max_depth = 0;
    for (int i=0;i<K*L;i++){
//        depth[i] = x_keyroot_depth_2[i/L] + y_keyroot_depth_2[i%L];
        depth[i] = 0;
        if(depth[i] > max_depth){
            max_depth = depth[i];
        }
    }
    printf("max_depth = %d, k=%d, l=%d\n", max_depth, K,L);


    // GPU Programming
    int blockSize = 256;
    int current_depth = 0;
    double limitation = (8*1024*1024*1024.0)/((m+1)*(n+1)*4);
    printf("ZHANG limitation is %u\n", int(limitation));
    int batch_size = min(K*L,int(limitation));

    // x_orl, x_kr, y_orl, y_kr
    thrust::device_vector<int> x_orl_d(x_orl);
    thrust::device_vector<int> x_kr_d(x_kr);
    thrust::device_vector<int> y_orl_d(y_orl);
    thrust::device_vector<int> y_kr_d(y_kr);

    // Delta, D_tree
    vector<int> Delta_trans (m*n);
    for (int i=0; i<m;i++){
        for (int j=0; j<n;j++){
            Delta_trans[i*n+j] = Delta[i][j];
        }
    }
    thrust::device_vector<int> Delta_d(Delta_trans);
    vector<int> D_tree_trans(m*n, 0);
    thrust::device_vector<int> D_tree_d(D_tree_trans);

    // depth
    thrust::device_vector<int> depth_d(depth);

    // D
//    vector<int> D_trans((m+1)*(n+1)*batch_size, 0);
//    thrust::device_vector<int> D_d(D_trans);
    thrust::device_vector<int> D_d((m+1)*(n+1)*batch_size, 0);
//    thrust::device_vector<int> D_d_2((m+1)*(n+1)*batch_size, 0);

    // Size
//    vector<int> size(K*L,0);
//    thrust::device_vector<int> size_d(size);

    // Pointer: x_orl, x_kr, y_orl, y_kr, Delta, D_tree, depth, worklist1, worklist2, D
    int *p_x_orl_d = thrust::raw_pointer_cast(x_orl_d.data());
    int *p_x_kr_d = thrust::raw_pointer_cast(x_kr_d.data());
    int *p_y_orl_d = thrust::raw_pointer_cast(y_orl_d.data());
    int *p_y_kr_d = thrust::raw_pointer_cast(y_kr_d.data());
    int *p_Delta_d = thrust::raw_pointer_cast(Delta_d.data());
    int *p_D_tree_d = thrust::raw_pointer_cast(D_tree_d.data());
    int *p_depth_d = thrust::raw_pointer_cast(depth_d.data());
    int *p_D_d = thrust::raw_pointer_cast(D_d.data());
//    int *p_D_d_2 = thrust::raw_pointer_cast(D_d_2.data());

    float total_milliseconds = 0;

    Stream stream;

    ArrayView<int> depth_d_view(depth_d);

    Queue<int> queue1;
    queue1.Init(K*L*sizeof(int));
    dev::Queue<int,uint32_t> d_queue1 = queue1.DeviceObject();

    Queue<int> queue2;
    queue2.Init(K*L*sizeof(int));
    dev::Queue<int,uint32_t> d_queue2 = queue2.DeviceObject();

    Queue<int> large_queue;
    large_queue.Init(K*L*sizeof(int));
    dev::Queue<int,uint32_t> d_large_queue = large_queue.DeviceObject();
    large_queue.set_size(stream, 0);

    Queue<int> block_size_queue;
    block_size_queue.Init(K*L*sizeof(int));
    dev::Queue<int,uint32_t> d_block_size_queue = block_size_queue.DeviceObject();
    block_size_queue.set_size(stream, 0);


    // Fetch task

    int numBlocks = (int)(depth_d_view.size() + blockSize - 1) / blockSize;
    fetch_task<<<numBlocks,blockSize, 0, stream.cuda_stream()>>>(depth_d_view, d_queue1, 0);
    stream.Sync();
    printf("TEST SIZE = %d\n", queue1.size());

    initializeArrayToZero<<<1, 64>>>();


    while (queue1.size() !=0 || large_queue.size() != 0 || block_size_queue.size() != 0) {

        printf("\ncurrent depth = %d, batch_size = %d\n", current_depth, batch_size);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        if (current_depth == 10){


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        }else if (queue1.size()<=5 && queue1.size()>6){

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        }else if (current_depth == 1000){


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        }else{
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);


            int num_task = queue1.size();
            numBlocks = (num_task + blockSize - 1) / blockSize;
            numBlocks = 128;
            printf("numBlocks = %d，Hello 28 numTasks = %d\n", numBlocks,num_task);


            cudaEventRecord(start, stream.cuda_stream());
            if(queue1.size()>0){
                printf("--- SINGLE METHOD: %d\n", queue1.size());
//                simple_parallel_new<<<numBlocks, blockSize, 0, stream.cuda_stream()>>>(p_x_orl_d, p_x_kr_d, p_y_orl_d, p_y_kr_d, p_Delta_d, p_D_d, p_D_tree_d, n, m, L, d_queue1, batch_size);
                zhang_parallel<<<64, 1024, 0, stream.cuda_stream()>>>(p_x_orl_d, p_x_kr_d, p_y_orl_d, p_y_kr_d, p_Delta_d, p_D_d, p_D_tree_d, n, m, L, d_queue1, batch_size);
            }
            cudaEventRecord(stop, stream.cuda_stream());
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            printf("------- Measured time for SINGLE = %.3fms\n", milliseconds);
            total_milliseconds+=milliseconds;

        }
        queue1.set_size(stream,0);
    }


    int final = D_tree_d[0];
    printf("The total final distance is %u\n", final);
    printf("hello\n");
    printf("Measured time for total parallel execution = %.3fms\n", total_milliseconds);

    cudaMemcpy(D_tree_trans.data(), p_D_tree_d, sizeof(int) * m * n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("D[0][13] = %d, D[61][0] = %d\n", D_tree_trans[0*1000+13], D_tree_trans[61*1000+0]);

    vector<vector<int>> final_result(m,vector<int>(n,0));
    return final_result;

}


vector<vector<int>> parallel_standard_ted_test(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threads, vector<vector<int>>& x_adj, vector<vector<int>>& y_adj){

    int K = (int)x_kr.size();
    int L = (int)y_kr.size();


    vector<int> depth(K*L, 0);

    int number_x_nodes = x_adj.size();
    int number_y_nodes = y_adj.size();


    vector<int> x_keyroot_depth_2(K, 0);
    vector<int> y_keyroot_depth_2(L, 0);

// Preprocessing
    auto start_time_2 = chrono::steady_clock::now();
    for (int i=0;i<K;i++){
        int node = x_kr[i];
        if((node == x_orl[node])||(node == x_orl[node]-1)){
            x_keyroot_depth_2[i] = 0;
        }else{
            for (int j=0;j<i;j++){
                int node_2 = x_kr[j];
                if ((node <= node_2)&&(x_orl[node] >= node_2)){
                    x_keyroot_depth_2[i] = max(x_keyroot_depth_2[i],x_keyroot_depth_2[j]+1);
                }
            }
        }
    }

    for (int i=0;i<L;i++){
        int node = y_kr[i];
        if((node == y_orl[node])||(node == y_orl[node]-1)){
            y_keyroot_depth_2[i] = 0;
        }else{
            for (int j=0;j<i;j++){
                int node_2 = y_kr[j];
                if ((node <= node_2)&&(y_orl[node] >= node_2)){
                    y_keyroot_depth_2[i] = max(y_keyroot_depth_2[i],y_keyroot_depth_2[j]+1);
                }
            }
        }
    }



    int max_depth = 0;
    for (int i=0;i<K*L;i++){
        depth[i] = x_keyroot_depth_2[i/L] + y_keyroot_depth_2[i%L];
        if(depth[i] > max_depth){
            max_depth = depth[i];
        }
    }
//    printf("max_depth = %d\n", max_depth);

    auto end_time_2 = chrono::steady_clock::now();
    auto ms_2 = chrono::duration_cast<chrono::microseconds>(end_time_2 - start_time_2).count();
    cout << "Preprocessing Method, the time is " << ms_2/1000.0 << " ms consumed" << endl;

    // GPU Programming
    int blockSize = 256;
    int current_depth = 0;
    double limitation = (8*1024*1024*1024.0)/((m+1)*(n+1)*4);
//    printf("limitation is %u\n", int(limitation));
    int batch_size = min(K*L,int(limitation));

    // x_orl, x_kr, y_orl, y_kr
    thrust::device_vector<int> x_orl_d(x_orl);
    thrust::device_vector<int> x_kr_d(x_kr);
    thrust::device_vector<int> y_orl_d(y_orl);
    thrust::device_vector<int> y_kr_d(y_kr);

    // Delta, D_tree
    vector<int> Delta_trans (m*n);
    for (int i=0; i<m;i++){
        for (int j=0; j<n;j++){
            Delta_trans[i*n+j] = Delta[i][j];
        }
    }
    thrust::device_vector<int> Delta_d(Delta_trans);
    vector<int> D_tree_trans(m*n, 0);
    thrust::device_vector<int> D_tree_d(D_tree_trans);

    // depth
    thrust::device_vector<int> depth_d(depth);

    // D
    thrust::device_vector<int> D_d((m+1)*(n+1)*batch_size, 0);

    // Pointer: x_orl, x_kr, y_orl, y_kr, Delta, D_tree, depth, worklist1, worklist2, D
    int *p_x_orl_d = thrust::raw_pointer_cast(x_orl_d.data());
    int *p_x_kr_d = thrust::raw_pointer_cast(x_kr_d.data());
    int *p_y_orl_d = thrust::raw_pointer_cast(y_orl_d.data());
    int *p_y_kr_d = thrust::raw_pointer_cast(y_kr_d.data());
    int *p_Delta_d = thrust::raw_pointer_cast(Delta_d.data());
    int *p_D_tree_d = thrust::raw_pointer_cast(D_tree_d.data());
    int *p_depth_d = thrust::raw_pointer_cast(depth_d.data());
    int *p_D_d = thrust::raw_pointer_cast(D_d.data());

//    float total_milliseconds = 0;
    total_milliseconds = 0;

    Stream stream;

    ArrayView<int> depth_d_view(depth_d);

    Queue<int> queue1;
    queue1.Init(K*L*sizeof(int));
    dev::Queue<int,uint32_t> d_queue1 = queue1.DeviceObject();

    Queue<int> queue2;
    queue2.Init(K*L*sizeof(int));
    dev::Queue<int,uint32_t> d_queue2 = queue2.DeviceObject();

    Queue<int> large_queue;
    large_queue.Init(K*L*sizeof(int));
    dev::Queue<int,uint32_t> d_large_queue = large_queue.DeviceObject();
    large_queue.set_size(stream, 0);

    Queue<int> block_size_queue;
    block_size_queue.Init(K*L*sizeof(int));
    dev::Queue<int,uint32_t> d_block_size_queue = block_size_queue.DeviceObject();
    block_size_queue.set_size(stream, 0);

    Queue<int> multi_block_queue;
    multi_block_queue.Init(K*L*sizeof(int));
    dev::Queue<int,uint32_t> d_multi_block_queue = multi_block_queue.DeviceObject();
    multi_block_queue.set_size(stream, 0);


    // Fetch task

    int numBlocks = (int)(depth_d_view.size() + blockSize - 1) / blockSize;
    fetch_task<<<numBlocks,blockSize, 0, stream.cuda_stream()>>>(depth_d_view, d_queue1, 0);
    stream.Sync();

    initializeArrayToZero<<<1, 64>>>();



    while (queue1.size() !=0 || large_queue.size() != 0 || block_size_queue.size() != 0 || multi_block_queue.size() != 0) {

//        printf("\n-------- current depth = %d, batch_size = %d\n", current_depth, batch_size);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        if (current_depth == 10){


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        }else if (current_depth == max_depth+10){

            last_table(stream, total_milliseconds, K*L-1, 0, x_orl, x_kr, y_orl, y_kr, L, blockSize, n, m , p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        }else{

            if(queue1.size()>0) {
                simple_thread(stream, total_milliseconds, limitation, d_queue1, L, n, m, p_x_orl_d, p_y_orl_d, p_x_kr_d,
                              p_y_kr_d, p_Delta_d, p_D_d, p_D_tree_d);
            }


            if(large_queue.size() > 0){
                thrust::host_vector<int> large_queue_h(large_queue.size());
                cudaMemcpyAsync(thrust::raw_pointer_cast(large_queue_h.data()),large_queue.data(), large_queue.size() * sizeof(int),cudaMemcpyDeviceToHost, stream.cuda_stream());
                stream.Sync();
                simple_warp(stream, total_milliseconds, limitation, large_queue_h, x_orl, x_kr, y_orl, y_kr, L, n, m, p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d);
            }

            if(block_size_queue.size() != 0){
                thrust::host_vector<int> block_size_queue_h(block_size_queue.size());
                cudaMemcpyAsync(thrust::raw_pointer_cast(block_size_queue_h.data()),block_size_queue.data(), block_size_queue.size() * sizeof(int),cudaMemcpyDeviceToHost, stream.cuda_stream());
                stream.Sync();
                simple_block(stream, total_milliseconds, limitation, block_size_queue_h, x_orl, x_kr, y_orl, y_kr, L, n, m, p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d);
            }

            if(multi_block_queue.size() != 0){
                thrust::host_vector<int> multi_block_queue_h(multi_block_queue.size());
                cudaMemcpyAsync(thrust::raw_pointer_cast(multi_block_queue_h.data()),multi_block_queue.data(), multi_block_queue.size() * sizeof(int),cudaMemcpyDeviceToHost, stream.cuda_stream());
                stream.Sync();
                multi_block(stream, total_milliseconds, limitation, multi_block_queue_h, x_orl, x_kr, y_orl, y_kr, L, n, m, p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d);
            }

        }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        queue1.set_size(stream, 0);
        large_queue.set_size(stream, 0);
        block_size_queue.set_size(stream, 0);
        multi_block_queue.set_size(stream, 0);
        current_depth++;
        numBlocks = (depth_d_view.size() + blockSize - 1) / blockSize;
        fetch_task<<<numBlocks, blockSize, 0, stream.cuda_stream()>>>(depth_d_view, d_queue1, current_depth);
        stream.Sync();

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Sort by Size
        queue2.set_size(stream, queue1.size());
        numBlocks = (queue1.size() + blockSize - 1) / blockSize;

        fetch_size_queue<<<numBlocks, blockSize, 0, stream.cuda_stream()>>>(d_queue2, d_queue1,p_x_orl_d, p_y_orl_d, p_x_kr_d, p_y_kr_d, L);
        stream.Sync();

        thrust::sort_by_key(queue2.getRawPointer(), queue2.getRawPointer()+queue1.size(), queue1.getRawPointer());
        cudaStreamSynchronize(stream.cuda_stream());
        queue2.set_size(stream,0);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        numBlocks = (queue1.size() + blockSize - 1) / blockSize;
//        filter_new<<<numBlocks, blockSize, 0, stream.cuda_stream()>>>(d_queue1, d_queue2, d_large_queue, d_block_size_queue, d_multi_block_queue, 15, 30000, 4500000);
//        filter_new<<<numBlocks, blockSize, 0, stream.cuda_stream()>>>(d_queue1, d_queue2, d_large_queue, d_block_size_queue, d_multi_block_queue, 15, 8750, 4500000);
//        filter_new<<<numBlocks, blockSize, 0, stream.cuda_stream()>>>(d_queue1, d_queue2, d_large_queue, d_block_size_queue, d_multi_block_queue, 15, 4500000, 4500000);
        filter_new<<<numBlocks, blockSize, 0, stream.cuda_stream()>>>(d_queue1, d_queue2, d_large_queue, d_block_size_queue, d_multi_block_queue, 4500000, 4500000, 4500000);
        stream.Sync();
        int old_size = queue1.size();
        queue1.set_size(stream,old_size -multi_block_queue.size()- large_queue.size()-block_size_queue.size());

//        printf("Thread = %d\n", queue1.size());
//        printf("Warp = %d\n", large_queue.size());
//        printf("Block = %d\n", block_size_queue.size());
//        printf("Multi_Block = %d\n", multi_block_queue.size());
//
//        printf("//////////////////////////////////\n");
//        printf("//////////////////////////////////\n");

    }


    int final = D_tree_d[0];
    printf("The total final distance is %u\n", final);
//    printf("hello\n");
    printf("Measured time for total parallel execution = %.3fms\n", total_milliseconds);

    cudaMemcpy(D_tree_trans.data(), p_D_tree_d, sizeof(int) * m * n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
//    printf("D[0][13] = %d, D[61][0] = %d\n", D_tree_trans[0*1000+13], D_tree_trans[61*1000+0]);

    vector<vector<int>> final_result(m,vector<int>(n,0));
    return final_result;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void simple_thread_parallel(int* x_orl, int* x_kr, int* y_orl, int* y_kr, int* Delta, int* D, int* D_tree, int n, int m, int L, dev::Queue<int, uint32_t> d_queue, int limitation){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int total = blockDim.x*gridDim.x;

    if(total>=limitation){
        if(x < limitation) {
            for (int i = (threadIdx.x + blockIdx.x * blockDim.x); i < d_queue.size(); i += limitation) {
                int thread_in_number = i % (limitation);
                task_GPU(x_orl, x_kr, y_orl, y_kr, Delta, D, D_tree, thread_in_number, d_queue[i], L, m, n);
            }
        }
    }else{
        for (int i = x; i < d_queue.size(); i += total) {
            task_GPU(x_orl, x_kr, y_orl, y_kr, Delta, D, D_tree, x, d_queue[i], L, m, n);
        }
    }
}

void simple_thread(Stream& stream_single, float& total_milliseconds, int limitation, dev::Queue<int, uint32_t> d_queue1, int L, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int*p_x_kr_d, int*p_y_kr_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){

    int block = 256;
    int grid = 128;

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream_single.cuda_stream());

    simple_thread_parallel<<<grid, block, 0, stream_single.cuda_stream()>>>(p_x_orl_d, p_x_kr_d, p_y_orl_d, p_y_kr_d, p_Delta_d, p_D_d, p_D_tree_d, n, m, L, d_queue1, limitation);

    cudaEventRecord(stop, stream_single.cuda_stream());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
//    printf("Single-Thread = %.3fms\n", milliseconds);
    total_milliseconds+=milliseconds;

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void simple_warp_parallel(int limitation, int* p_d_column_major, int size, int* x_orl, int* y_orl, int* p_i_0, int* p_i_max, int* p_j_0, int* p_j_max, int* Delta, int* D, int* D_tree, int n, int m){

    int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
    int lane_id = (threadIdx.x + blockIdx.x * blockDim.x) % 32;
    int total = (blockDim.x * gridDim.x)/32;

    if(total <= limitation) {

        for (int i = warp_id; i < size; i += total) {

            int i_max = p_i_max[i];
            int j_max = p_j_max[i];
            int i_0 = p_i_0[i];
            int j_0 = p_j_0[i];
            int x_begin, y_begin, y_end, x_end;
            int iteration;

            if (p_d_column_major[i] == 1) {
                iteration = (i_max - i_0 + 32) / 32;

                for (int k = 0; k < iteration; k++) {

                    x_begin = i_max - 32 * k;
                    y_begin = j_max;
                    y_end = j_0;
                    if (k < iteration - 1) {
                        x_end = x_begin - 31;
                    } else {
                        x_end = i_0;
                    }

                    int x_loc = x_begin - lane_id;
                    int y_loc = y_begin;

                    for (int step = 0; step < x_begin + y_begin - x_end - y_end + 1; step++) {
                        if ((x_loc + y_loc == x_begin + y_begin - step) & (y_loc >= y_end) & (x_loc >= x_end)) {
                            single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                                           Delta, D, D_tree, warp_id, m, n);
                            y_loc--;
                        }
                        __syncwarp();
                    }
                }


            } else {
                iteration = (j_max - j_0 + 32) / 32;

                for (int k = 0; k < iteration; k++) {
                    x_begin = i_max;
                    x_end = i_0;
                    y_begin = j_max - 32 * k;

                    if (k < iteration - 1) {
                        y_end = y_begin - 31;
                    } else {
                        y_end = j_0;
                    }

                    int x_loc = x_begin;
                    int y_loc = y_begin - lane_id;

                    for (int step = 0; step < x_begin + y_begin - x_end - y_end + 1; step++) {
                        if ((x_loc + y_loc == x_begin + y_begin - step) & (y_loc >= y_end) & (x_loc >= x_end)) {
                            single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                                           Delta, D, D_tree, warp_id, m, n);
                            x_loc--;
                        }
                        __syncwarp();
                    }

                }

            }

        }

    }else{
        if(warp_id < limitation){
            for (int i = warp_id; i < size; i += limitation) {

                int thread_in_number = i % (limitation);
                int i_max = p_i_max[i];
                int j_max = p_j_max[i];
                int i_0 = p_i_0[i];
                int j_0 = p_j_0[i];
                int x_begin, y_begin, y_end, x_end;
                int iteration;

                if (p_d_column_major[i] == 1) {
                    iteration = (i_max - i_0 + 32) / 32;
                    for (int k = 0; k < iteration; k++) {
                        x_begin = i_max - 32 * k;
                        y_begin = j_max;
                        y_end = j_0;
                        if (k < iteration - 1) {
                            x_end = x_begin - 31;
                        } else {
                            x_end = i_0;
                        }
                        int x_loc = x_begin - lane_id;
                        int y_loc = y_begin;
                        for (int step = 0; step < x_begin + y_begin - x_end - y_end + 1; step++) {
                            if ((x_loc + y_loc == x_begin + y_begin - step) & (y_loc >= y_end) & (x_loc >= x_end)) {
                                single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                                               Delta, D, D_tree, thread_in_number, m, n);
                                y_loc--;
                            }
                            __syncwarp();
                        }
                    }
                } else {
                    iteration = (j_max - j_0 + 32) / 32;
                    for (int k = 0; k < iteration; k++) {
                        x_begin = i_max;
                        x_end = i_0;
                        y_begin = j_max - 32 * k;
                        if (k < iteration - 1) {
                            y_end = y_begin - 31;
                        } else {
                            y_end = j_0;
                        }
                        int x_loc = x_begin;
                        int y_loc = y_begin - lane_id;
                        for (int step = 0; step < x_begin + y_begin - x_end - y_end + 1; step++) {
                            if ((x_loc + y_loc == x_begin + y_begin - step) & (y_loc >= y_end) & (x_loc >= x_end)) {
                                single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                                               Delta, D, D_tree, thread_in_number, m, n);
                                x_loc--;
                            }
                            __syncwarp();
                        }
                    }
                }

            }
        }
    }
}


void simple_warp (Stream& stream, float& total_milliseconds, int limitation, thrust::host_vector<int>& queue_h, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, int L, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){
    vector<int> i_0_array (queue_h.size(),0);
    vector<int> i_max_array (queue_h.size(),0);
    vector<int> j_0_array (queue_h.size(),0);
    vector<int> j_max_array (queue_h.size(),0);
    vector<int> blocks_per_table (queue_h.size(),0);
    vector<int> column_major (queue_h.size(),0);

    for (int i=0; i<queue_h.size();i++){
        int row = queue_h[i] / L;
        int column = queue_h[i] % L;
        i_0_array[i] = x_kr[row];
        j_0_array[i] = y_kr[column];
        i_max_array[i] = x_orl[i_0_array[i]] + 1;
        j_max_array[i] = y_orl[j_0_array[i]] + 1;

        if(i_max_array[i]-i_0_array[i]+1 > j_max_array[i]-j_0_array[i]+1){
            column_major[i] = 1;
        }else{
            column_major[i] = 0;
        }
    }

    thrust::device_vector<int> d_i_0_array(i_0_array);
    thrust::device_vector<int> d_i_max_array(i_max_array);
    thrust::device_vector<int> d_j_0_array(j_0_array);
    thrust::device_vector<int> d_j_max_array(j_max_array);
    thrust::device_vector<int> d_blocks_per_table(blocks_per_table);
    thrust::device_vector<int> d_column_major(column_major);

    int *p_d_i_0_array = thrust::raw_pointer_cast(d_i_0_array.data());
    int *p_d_i_max_array = thrust::raw_pointer_cast(d_i_max_array.data());
    int *p_d_j_0_array = thrust::raw_pointer_cast(d_j_0_array.data());
    int *p_d_j_max_array = thrust::raw_pointer_cast(d_j_max_array.data());
    int *p_d_blocks_per_table = thrust::raw_pointer_cast(d_blocks_per_table.data());
    int *p_d_column_major = thrust::raw_pointer_cast(d_column_major.data());


    int block = 256;
    int grid = 256;
    int block_per_table = grid / queue_h.size();

//    printf("--- Warp =: %u \n", queue_h.size());

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream.cuda_stream());

    simple_warp_parallel<<<grid, block, 0, stream.cuda_stream()>>>(limitation, p_d_column_major, queue_h.size(), p_x_orl_d, p_y_orl_d, p_d_i_0_array, p_d_i_max_array, p_d_j_0_array, p_d_j_max_array, p_Delta_d, p_D_d, p_D_tree_d, n, m);

    cudaEventRecord(stop, stream.cuda_stream());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

//    printf("Single-Warp = %.3fms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    total_milliseconds+=milliseconds;

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void simple_block_parallel(int limitation, int* p_d_column_major, int size, int* x_orl, int* y_orl, int* p_i_0, int* p_i_max, int* p_j_0, int* p_j_max, int* Delta, int* D, int* D_tree, int n, int m){
    int total = gridDim.x;
    int x = blockIdx.x;

    if(total <= limitation) {

        for (int i = x; i < size; i += total) {

            int i_max = p_i_max[i];
            int j_max = p_j_max[i];
            int i_0 = p_i_0[i];
            int j_0 = p_j_0[i];
            int x_begin, y_begin, x_end, y_end, iteration;

            if (p_d_column_major[i] == 1) {
                iteration = (i_max - i_0 + 1024) / 1024;
                for (int k = 0; k < iteration; k++) {
                    x_begin = i_max - k * 1024;
                    y_begin = j_max;
                    y_end = j_0;
                    if (k < iteration - 1) {
                        x_end = x_begin - 1023;
                    } else {
                        x_end = i_0;
                    }
                    int x_loc = x_begin - threadIdx.x;
                    int y_loc = y_begin;
                    for (int step = 0; step < x_begin + y_begin - x_end - y_end + 1; step++) {
                        if ((x_loc + y_loc == x_begin + y_begin - step) & (y_loc >= y_end) & (x_loc >= x_end)) {
                            single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                                           Delta, D, D_tree, x, m, n);
                            y_loc--;
                        }
                        __syncthreads();
                    }
                }
            } else {
                iteration = (j_max - j_0 + 1024) / 1024;
                for (int k = 0; k < iteration; k++) {
                    x_begin = i_max;
                    x_end = i_0;
                    y_begin = j_max - 1024 * k;
                    if (k < iteration - 1) {
                        y_end = y_begin - 1023;
                    } else {
                        y_end = j_0;
                    }
                    int x_loc = x_begin;
                    int y_loc = y_begin - threadIdx.x;
                    for (int step = 0; step < x_begin + y_begin - x_end - y_end + 1; step++) {
                        if ((x_loc + y_loc == x_begin + y_begin - step) & (y_loc >= y_end) & (x_loc >= x_end)) {
                            single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                                           Delta, D, D_tree, x, m, n);
                            x_loc--;
                        }
                        __syncthreads();
                    }
                }
            }

        }
    }else{

        if(blockIdx.x < limitation){
            for (int i = x; i < size; i += limitation) {
                int num_in_thread = i % limitation;
                int i_max = p_i_max[i];
                int j_max = p_j_max[i];
                int i_0 = p_i_0[i];
                int j_0 = p_j_0[i];
                int x_begin, y_begin, x_end, y_end, iteration;

                if (p_d_column_major[i] == 1) {
                    iteration = (i_max - i_0 + 1024) / 1024;
                    for (int k = 0; k < iteration; k++) {
                        x_begin = i_max - k * 1024;
                        y_begin = j_max;
                        y_end = j_0;
                        if (k < iteration - 1) {
                            x_end = x_begin - 1023;
                        } else {
                            x_end = i_0;
                        }
                        int x_loc = x_begin - threadIdx.x;
                        int y_loc = y_begin;
                        for (int step = 0; step < x_begin + y_begin - x_end - y_end + 1; step++) {
                            if ((x_loc + y_loc == x_begin + y_begin - step) & (y_loc >= y_end) & (x_loc >= x_end)) {


                                single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                                               Delta, D, D_tree, num_in_thread, m, n);
                                y_loc--;

                            }
                            __syncthreads();
                        }
                    }
                } else {
                    iteration = (j_max - j_0 + 1024) / 1024;
                    for (int k = 0; k < iteration; k++) {
                        x_begin = i_max;
                        x_end = i_0;
                        y_begin = j_max - 1024 * k;
                        if (k < iteration - 1) {
                            y_end = y_begin - 1023;
                        } else {
                            y_end = j_0;
                        }
                        int x_loc = x_begin;
                        int y_loc = y_begin - threadIdx.x;
                        for (int step = 0; step < x_begin + y_begin - x_end - y_end + 1; step++) {
                            if ((x_loc + y_loc == x_begin + y_begin - step) & (y_loc >= y_end) & (x_loc >= x_end)) {
                                single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                                               Delta, D, D_tree, num_in_thread, m, n);
                                x_loc--;
                            }
                            __syncthreads();
                        }
                    }
                }
            }
        }
    }
}


void simple_block (Stream& stream_mul, float& total_milliseconds, int limitation, thrust::host_vector<int>& queue_h, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, int L, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){
    vector<int> i_0_array (queue_h.size(),0);
    vector<int> i_max_array (queue_h.size(),0);
    vector<int> j_0_array (queue_h.size(),0);
    vector<int> j_max_array (queue_h.size(),0);
    vector<int> blocks_per_table (queue_h.size(),0);
    vector<int> column_major (queue_h.size(),0);

    for (int i=0; i<queue_h.size();i++){
        int row = queue_h[i] / L;
        int column = queue_h[i] % L;
        i_0_array[i] = x_kr[row];
        j_0_array[i] = y_kr[column];
        i_max_array[i] = x_orl[i_0_array[i]] + 1;
        j_max_array[i] = y_orl[j_0_array[i]] + 1;

        if(i_max_array[i]-i_0_array[i]+1 > j_max_array[i]-j_0_array[i]+1){
            column_major[i] = 1;
        }else{
            column_major[i] = 0;
        }
    }

    thrust::device_vector<int> d_i_0_array(i_0_array);
    thrust::device_vector<int> d_i_max_array(i_max_array);
    thrust::device_vector<int> d_j_0_array(j_0_array);
    thrust::device_vector<int> d_j_max_array(j_max_array);
    thrust::device_vector<int> d_blocks_per_table(blocks_per_table);
    thrust::device_vector<int> d_column_major(column_major);

    int *p_d_i_0_array = thrust::raw_pointer_cast(d_i_0_array.data());
    int *p_d_i_max_array = thrust::raw_pointer_cast(d_i_max_array.data());
    int *p_d_j_0_array = thrust::raw_pointer_cast(d_j_0_array.data());
    int *p_d_j_max_array = thrust::raw_pointer_cast(d_j_max_array.data());
    int *p_d_blocks_per_table = thrust::raw_pointer_cast(d_blocks_per_table.data());
    int *p_d_column_major = thrust::raw_pointer_cast(d_column_major.data());


    int block = 1024;
    int grid = 256;

//    printf("--- Block =: %u \n", queue_h.size());

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream_mul.cuda_stream());

    simple_block_parallel<<<grid, block, 0, stream_mul.cuda_stream()>>>(limitation, p_d_column_major, queue_h.size(), p_x_orl_d, p_y_orl_d, p_d_i_0_array, p_d_i_max_array, p_d_j_0_array, p_d_j_max_array, p_Delta_d, p_D_d, p_D_tree_d, n, m);

    cudaEventRecord(stop, stream_mul.cuda_stream());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
//    printf("Single-Block = %.3fms\n", milliseconds);
    total_milliseconds+=milliseconds;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void multi_block_parallel(int limitation, int* p_d_column_major, int* blocks_per_table, int size, int* x_orl, int* y_orl, int* p_i_0, int* p_i_max, int* p_j_0, int* p_j_max, int* Delta, int* D, int* D_tree, int n, int m){
    int block_begin = 0;
    int block_end = blocks_per_table[0]-1;

    for (int i=0; i< size; i++){

        if(blockIdx.x<= block_end & blockIdx.x >= block_begin){

            int t = block_end - block_begin + 1;
            int x = threadIdx.x + (blockIdx.x-block_begin)*blockDim.x;
            int num_in_thread = i % limitation;

            int i_max = p_i_max[i];
            int j_max = p_j_max[i];
            int i_0 = p_i_0[i];
            int j_0 = p_j_0[i];
            int x_begin, y_begin, x_end, y_end, iteration;
            int lock = 0;

            if (p_d_column_major[i] == 1) {
                iteration = (i_max - i_0 + t*1024) / (t*1024);
                for (int k = 0; k < iteration; k++) {
                    x_begin = i_max - k * t*1024;
                    y_begin = j_max;
                    y_end = j_0;
                    if (k < iteration - 1) {
                        x_end = x_begin - t*1024+1;
                    } else {
                        x_end = i_0;
                    }
                    int x_loc = x_begin - x;
                    int y_loc = y_begin;
                    for (int step = 0; step < x_begin + y_begin - x_end - y_end + 1; step++) {
                        if ((x_loc + y_loc == x_begin + y_begin - step) & (y_loc >= y_end) & (x_loc >= x_end)) {
                            single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                                           Delta, D, D_tree, num_in_thread, m, n);
                            y_loc--;
                        }
                        __threadfence();
                        __gpu_sync_range_update(block_begin,block_end,step+1,Array_in,Array_out);
                    }
                }
            } else {
                iteration = (j_max - j_0 + t*1024) / (t*1024);
                for (int k = 0; k < iteration; k++) {
                    x_begin = i_max;
                    x_end = i_0;
                    y_begin = j_max - t*1024 * k;
                    if (k < iteration - 1) {
                        y_end = y_begin - t*1024+1;
                    } else {
                        y_end = j_0;
                    }
                    int x_loc = x_begin;
                    int y_loc = y_begin - x;
                    for (int step = 0; step < x_begin + y_begin - x_end - y_end + 1; step++) {
                        if ((x_loc + y_loc == x_begin + y_begin - step) & (y_loc >= y_end) & (x_loc >= x_end)) {
                            single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                                           Delta, D, D_tree, num_in_thread, m, n);
                            x_loc--;
                        }
                        __threadfence();
                        __gpu_sync_range_update(block_begin,block_end,step+1,Array_in,Array_out);
                    }
                }
            }
        }

        if (block_end + blocks_per_table[i+1] <= 63){
            block_begin = block_end + 1;
            block_end = block_end + blocks_per_table[i+1];
        } else{
            block_begin = 0;
            block_end = blocks_per_table[i+1]-1;
        }
    }

}

void multi_block (Stream& stream_mul, float& total_milliseconds, int limitation, thrust::host_vector<int>& queue_h, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, int L, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){
    vector<int> i_0_array (queue_h.size(),0);
    vector<int> i_max_array (queue_h.size(),0);
    vector<int> j_0_array (queue_h.size(),0);
    vector<int> j_max_array (queue_h.size(),0);
    vector<int> blocks_per_table (queue_h.size(),0);
    vector<int> column_major (queue_h.size(),0);


    for (int i=0; i<queue_h.size();i++){
        int row = queue_h[i] / L;
        int column = queue_h[i] % L;
        i_0_array[i] = x_kr[row];
        j_0_array[i] = y_kr[column];
        i_max_array[i] = x_orl[i_0_array[i]] + 1;
        j_max_array[i] = y_orl[j_0_array[i]] + 1;
        if(i_max_array[i]-i_0_array[i]+1 > j_max_array[i]-j_0_array[i]+1){
            blocks_per_table[i] = 2;
            column_major[i] = 1;
        }else{
            blocks_per_table[i] = 2;
            column_major[i] = 0;
        }
    }

    thrust::device_vector<int> d_i_0_array(i_0_array);
    thrust::device_vector<int> d_i_max_array(i_max_array);
    thrust::device_vector<int> d_j_0_array(j_0_array);
    thrust::device_vector<int> d_j_max_array(j_max_array);
    thrust::device_vector<int> d_blocks_per_table(blocks_per_table);
    thrust::device_vector<int> d_column_major(column_major);

    int *p_d_i_0_array = thrust::raw_pointer_cast(d_i_0_array.data());
    int *p_d_i_max_array = thrust::raw_pointer_cast(d_i_max_array.data());
    int *p_d_j_0_array = thrust::raw_pointer_cast(d_j_0_array.data());
    int *p_d_j_max_array = thrust::raw_pointer_cast(d_j_max_array.data());
    int *p_d_blocks_per_table = thrust::raw_pointer_cast(d_blocks_per_table.data());
    int *p_d_column_major = thrust::raw_pointer_cast(d_column_major.data());

    int block = 1024;
    int grid = 64;

//    printf("--- Blocks =: %u \n", queue_h.size());

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream_mul.cuda_stream());


    multi_block_parallel<<<grid, block, 0, stream_mul.cuda_stream()>>>(limitation, p_d_column_major, p_d_blocks_per_table, queue_h.size(), p_x_orl_d, p_y_orl_d, p_d_i_0_array, p_d_i_max_array, p_d_j_0_array, p_d_j_max_array, p_Delta_d, p_D_d, p_D_tree_d, n, m);


    cudaEventRecord(stop, stream_mul.cuda_stream());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
//    printf("Multi-Blocks = %.3fms\n", milliseconds);
    total_milliseconds+=milliseconds;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void initial_last(int* D, int n, int m){
    int width = n+1;
    for(int i = TID_1D; i<=n;i+=TOTAL_THREADS_1D){
        D[m*width+n-i] = i;
    }
    for(int i = TID_1D; i <= m; i+= TOTAL_THREADS_1D){
        D[(m-i)*width+n] = i;
    }
}

__device__ void last_table_unit(int* x_orl, int* y_orl, int i, int j, int* Delta, int* D, int* D_tree, int m, int n){
    int width = n+1;
    int index = j + i*width;
    if ((i <= m-1) & (i >= 0) & (j <= n-1) & (j >= 0)){
        if ((x_orl[i] == m-1) & (y_orl[j] == n-1)) {
            D[index] = min3_D(Delta[j+i*n] + D[1 + width + index],
                              1 + D[width+index],
                              1 + D[1 + index]);
            D_tree[j + i*n] = D[index];
        }else{
            D[index] = min3_D(D_tree[j + i*n] + D[y_orl[j] + 1 + (x_orl[i] + 1) * width],
                              1 + D[index+width],
                              1 + D[1 + index]);
        }
    }
}



__global__ void last_table_parallel(int number_per_thread, int* x_orl, int* y_orl, int* Delta, int* D, int* D_tree, int n, int m){

    int y_loc = n - number_per_thread * blockIdx.x;
    int x_begin, x_end;
    int iteration = (m + 1024) / 1024;
    for (int f = 0; f < iteration; f++) {
        x_begin = m- f*1024;
        if (f < iteration - 1) {
            x_end = x_begin - 1024+1;
        } else {
            x_end = 0;
        }
        int x_loc = x_begin-threadIdx.x;
        int k = 0;
        for (int step = 0; step < x_begin + n - x_end + 1; step++) {
            if ((x_loc + y_loc == x_begin + n - step) & (k < units_per_thread) & y_loc >= 0 & x_loc >= x_end) {
                last_table_unit(x_orl, y_orl, x_loc, y_loc, Delta, D, D_tree, m, n);
                y_loc--;
                k++;
            }
            __gpu_sync(step + 1, Array_in, Array_out);
        }
    }

//    int y_loc = n - units_per_thread * blockIdx.x;
//    int x_loc = m - threadIdx.x;
//    int k = 0;
//    for (int step = 0; step < m+n-0-0+1; step++) {
//        if((x_loc + y_loc == m+n - step) & (k<units_per_thread) & y_loc>=0 & x_loc>=0 ){
//
//            last_table_unit(x_orl, y_orl, x_loc, y_loc, Delta, D, D_tree, m, n);
//            y_loc--;
//            k++;
//        }
//        __gpu_sync(step+1,Array_in, Array_out);
//    }
}


void last_table(Stream& stream_sing, float& total_milliseconds, int i, int thread_in_number, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, int L, int blockSize, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){
    int row = i / L;
    int column = i % L;

    int i_0 = x_kr[row];
    int j_0 = y_kr[column];
    int i_max = x_orl[i_0] + 1;
    int j_max = y_orl[j_0] + 1;

    int one_block = 256;
    int one_grid = 4;

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream_sing.cuda_stream());
    initial_last<<<one_grid, one_block, 0, stream_sing.cuda_stream()>>>(p_D_d, n, m);
    cudaEventRecord(stop, stream_sing.cuda_stream());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
//    printf("Initialization = %.3fms\n", milliseconds);
    total_milliseconds+=milliseconds;


    int test_block = 1024;
    int test_grid = 64;
    int number_per_thread = (n+test_grid-1)/test_grid;

    cudaMemcpyToSymbol(units_per_thread, &number_per_thread, sizeof(int));



    milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream_sing.cuda_stream());

    last_table_parallel<<<test_grid, test_block, 0, stream_sing.cuda_stream()>>>(number_per_thread, p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d, n, m);

    cudaEventRecord(stop, stream_sing.cuda_stream());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
//    printf("Computing = %.3fms\n", milliseconds);
    total_milliseconds+=milliseconds;

}

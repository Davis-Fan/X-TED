#include "TED_C++.h"

__global__ void mathKernel1(int* c, int matrix_size, int warpSize) {
    for (int i = TID_1D; i<matrix_size; i+= TOTAL_THREADS_1D) {

        int laneIdx = threadIdx.x % warpSize;  // Index of the thread within the warp
        int warpIdx = i / warpSize;         // Index of the warp

        int a, b;
        a = b = 0;

        // Use warp-level operations to reduce branch divergence
        // Each thread in the warp performs the same operation
        if ((warpIdx % 2) == 0) {
            a = 1;
        } else {
            b = 2;
        }

        // Use warp-level synchronization to ensure all threads within a warp have finished
        __syncwarp();

        // Assign the computed values to the output array
        c[i] = a + b;
    }
}

__global__ void tiling(int* matrix, int matrix_size, int tileSize, int m, int n){
    for (int i = TID_1D; i<matrix_size; i+= TOTAL_THREADS_1D) {

        int laneIdx = i % tileSize;
        int tileIdx = i / tileSize;

        for (int stage = 1; stage <= m+n-1 ; stage++){
            if (stage <= n){
                if (tileIdx >= (stage-1)*(stage)/2 && tileIdx < (stage) * (stage+1)/2){
                    matrix[i] = 1;
                }
            }
            if (stage > n){
                int r_stage = m + n - stage;
                int r_i = m*n - i - 1;
                if (r_i >= (r_stage-1)*(r_stage)/2 && r_i < r_stage*(r_stage+1)/2){
                    matrix[i] = 1;
                }
            }
            __syncthreads();
        }

    }
}

__global__ void compute (int* matrix, int m, int n){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    for ( int i = x; i<m; i+=gridDim.x * blockDim.x){
        for ( int j=y; j<n; j+=gridDim.y * blockDim.y){
            matrix[x*n + y] = 1;
        }
    }
    __syncthreads();
}

__global__ void compute2 (int* matrix, int m, int n){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    matrix[x*n + y] = 1;
    __syncthreads();
}


__global__ void diagonal(int* matrix, int m, int n){


    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int k,w;

    for ( int i = x; i<m; i+=gridDim.x * blockDim.x){
        for ( int j=y; j<n; j+=gridDim.y * blockDim.y){
            for (int step = 0; step<m+n-1; step++){
                k = x;
                w = step - x;
                if (y == w){
                    printf("i = %u, j = %u, index=%u\n", k, w, x*n+y);
                    matrix[(m-x-1)*n + n-y-1] = step;
                }
                __syncthreads();
            }
        }
    }
}

__global__ void diagonal2(int* matrix, int table_row_size, int table_column_size, int tile_size){
    int i_0 = 0;
    int i_max = table_row_size-1;
    int j_0 = 0;
    int j_max = table_column_size-1;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int k,w;

    int row_size = (i_max-i_0+1 + tile_size - 1) / tile_size;
    int column_size = (j_max-j_0+1 + tile_size - 1) / tile_size;

    for ( int i = x; i<row_size; i+=gridDim.x * blockDim.x){
        for ( int j=y; j<column_size; j+=gridDim.y * blockDim.y){

            for (int step = 0; step<row_size+column_size-1; step++){
                k = x;
                w = step - x;
                if (y == w){
                    int x_begin = (i_max-x*tile_size);
                    int y_begin = j_max-y*tile_size;
//                    printf("x_begin = %u, y_begin = %u\n", x_begin, y_begin);
                    for (int i=0; i<tile_size; i++){
                        for (int j=0; j< tile_size; j++){
                            if ((x_begin-i)>=i_0 && (y_begin-j)>=j_0){
                                matrix[(x_begin-i)*table_column_size + y_begin-j] = (x_begin-i)*table_column_size + y_begin-j;
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
}

void printDeviceVector2(thrust::device_vector<int> &d_a, int n) {
    int j = 0;
    for (int k = 0; k < d_a.size(); k++) {
        int temp = d_a[k];
        printf("%i ", temp);
        j++;
        if (j == n+1){
            printf("\n");
            j = 0;
        }
    }
}

__device__ void single_unit_test_2(int step, int i, int j, int* D, int m, int n){
    int width = n+1;
    int index = j + i*width;


    if ((i == m) && (j == n)){
        D[index] = step;
//        D[index] = 0;
    }else if ((i <= m-1) && (i >= 0) && (j == n)){
//        D[index] = 1 + D[index+width];
        D[index] = step;
    }else if ((j <= n-1) && (j >= 0) && (i == m)){
//        D[index] = 1 + D[index+1];
        D[index] = step;
    }else if ((i <= m-1) && (i >= 0) && (j <= n-1) && (j >= 0)){
        D[index] = step;
//        D[index] = 0;
    }
}

__global__ void single_table_parallel_test(int* matrix, int m, int n){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
//    int k,w;

    int tile_size = 1;
    int row_size = (m+1 + tile_size - 1) / tile_size;
    int column_size = (n+1 + tile_size - 1) / tile_size;

    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
//    bool s = grid.is_valid();

    printf("hello\n");

//    printf("x = %u, y = %u\n",gridDim.x,gridDim.y);
//    printf("%u, %u, %u, %u \n", row_size, column_size, m, n);

    for ( int i = x; i<row_size; i+=gridDim.x * blockDim.x){
        for ( int j=y; j<column_size; j+=gridDim.y * blockDim.y){

//            for (int s = 0; s < gridDim.x + gridDim.y - 1; s++){
//                printf("%u \n", s);
//                if(blockIdx.y == s - blockIdx.x){

                    for (int step = 0; step<row_size+column_size-1; step++){
                        int index_i = m-x;
                        int index_j = n-y;
                        for (int number=0;number<2*tile_size-1; number++){
                            if (y == step - x){
                                for (int j = max(0,number-tile_size+1); j<=min(number, tile_size-1); j++){
                                    single_unit_test_2(step, index_i-number+j,index_j-j,matrix,m,n);
                                }
                            }
//                            __syncthreads();
                            grid.sync();
                        }

                    }
//                grid.sync();
//            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void single_unit_test_4(int step, int i, int j, int* D, int m, int n){
    int width = n+1;
    int index = j + i*width;


    if ((i == m) && (j == n)){
        D[index] = 0;
//        D[index] = 0;
    }else if ((i <= m-1) && (i >= 0) && (j == n)){
//        D[index] = 1 + D[index+width];
        D[index] = 0;
    }else if ((j <= n-1) && (j >= 0) && (i == m)){
//        D[index] = 1 + D[index+1];
        D[index] = 0;
    }else if ((i <= m-1) && (i >= 0) && (j <= n-1) && (j >= 0)){
        D[index] = D[index+1]+1;
//        D[index] = 0;
    }
}

//__device__ volatile int lock[2500];
//
//__device__ void busy_wait(int target){
//    int tid_in_block = threadIdx.x * blockDim.y + threadIdx.y;
//    while (lock[tid_in_block] != target){}
//    atomicAdd((int*)&(lock[tid_in_block]), 1);
//}


__device__ void busy_wait_1(int value, int i, int j, int* D, int m, int n){
    int width = n+1;
    int index = j + i*width;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    printf("i = %u, j = %u, D[i][j] = %u, thread.x = %u, thread.y = %u\n", i, j, D[index], x, y);

    if (i != m && j != n){

//        while ((atomicCAS(&D[index-1], 0, 0) == 0) || (atomicCAS(&D[index-width], 0, 0) == 0) || (atomicCAS(&D[index-1-width], 0, 0) == 0)){
//            // Do Nothing.
//        }

    } else if (i == m && j != n){
//        while ((atomicCAS(&D[index-1], 0, 0) == 0)){
//            // Do Nothing.
//        }

    } else if (j == n && i != m){
//        while ((atomicCAS(&D[index-width], 0, 0) == 0)){
//            // Do Nothing.
//        }


    }
    single_unit_test_4(value, i,j,D,m,n);
    printf("i = %u, j = %u, D[i][j] = %u\n", i, j, D[index]);
}



__global__ void single_table_parallel_test_4(int* matrix, int m, int n){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int numBlocks = gridDim.x*gridDim.y;

    int tile_size = 1;
//    int row_size = (m+1 + tile_size - 1) / tile_size;
//    int column_size = (n+1 + tile_size - 1) / tile_size;

    int row_size = blockDim.x;
    int column_size = blockDim.y;

    int x_offset = blockDim.x * tile_size;
    int y_offset = blockDim.x * tile_size;


    for ( int i = x; i<row_size; i+=gridDim.x * blockDim.x){
        for ( int j=y; j<column_size; j+=gridDim.y * blockDim.y){

            for (int step = 0; step<row_size+column_size-1; step++){
                int index_i = m-x*tile_size;
                int index_j = n-y*tile_size;

                if (y == step - x){


//                    single_unit_test_4(step, index_i,index_j,matrix,m,n);
//                    single_unit_test_4(step, index_i-x_offset,index_j,matrix,m,n);
//                    single_unit_test_4(step, index_i,index_j-y_offset,matrix,m,n);
//                    single_unit_test_4(step, index_i-x_offset,index_j-y_offset,matrix,m,n);

                    busy_wait_1(step+1, index_i,index_j,matrix,m,n);
                    busy_wait_1(step+1, index_i-x_offset,index_j,matrix,m,n);
                    busy_wait_1(step+1, index_i,index_j-y_offset,matrix,m,n);
                    busy_wait_1(step+1, index_i-x_offset,index_j-y_offset,matrix,m,n);
                }
            }
        }
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//__device__ volatile int g_mutex;
//__device__ void __gpu_sync(int goalVal)
//{
//    int tid_in_block = threadIdx.x * blockDim.y + threadIdx.y;  // Calculate the thread ID within the block
//
//    if (tid_in_block == 0)  // Only thread 0 performs synchronization
//    {
//        atomicAdd((int*)&g_mutex, 1);  // Increment g_mutex using atomicAdd to avoid race conditions
//
//        while (g_mutex != goalVal)  // Wait until g_mutex reaches the goal value
//        {
//            // Do nothing here, waiting for synchronization
//        }
////        printf("g_mutex = %u, Idx.x = %u, Idx.y = %u, blockIdx.x = %u, blockIdx.y = %u\n", g_mutex, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
////        atomicExch((int*)&g_mutex, 0);
////        printf("g_mutex = %u, Idx.x = %u, Idx.y = %u\n", g_mutex, threadIdx.x, threadIdx.y);
//    }
//
//    __syncthreads();  // Synchronize all threads in the block
//}

//__global__ void single_table_parallel_test_2(int* matrix, int m, int n){
//    int x = threadIdx.x + blockIdx.x * blockDim.x;
//    int y = threadIdx.y + blockIdx.y * blockDim.y;
//    int numBlocks = gridDim.x*gridDim.y;
////    int k,w;
//
//    int tile_size = 2;
//    int row_size = (m+1 + tile_size - 1) / tile_size;
//    int column_size = (n+1 + tile_size - 1) / tile_size;
//
////    __gpu_sync(numBlocks);
//
//    for ( int i = x; i<row_size; i+=gridDim.x * blockDim.x){
//        for ( int j=y; j<column_size; j+=gridDim.y * blockDim.y){
//
//            for (int step = 0; step<row_size+column_size-1; step++){
//                int index_i = m-x*tile_size;
//                int index_j = n-y*tile_size;
//                for (int number=0;number<2*tile_size-1; number++){
//                    if (y == step - x){
//                        for (int j = max(0,number-tile_size+1); j<=min(number, tile_size-1); j++){
//                            single_unit_test_2(step, index_i-number+j,index_j-j,matrix,m,n);
//                        }
//                    }
//                }
////                __syncthreads();
//                int lock = (step+1)*numBlocks;
//                __gpu_sync(lock);
//            }
//        }
//    }
//
//}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//__device__ void __gpu_sync_2(int goalVal, volatile int* Arrayin, volatile int* Arrayout)
//{
//    int tid_in_blk = threadIdx.x * blockDim.y + threadIdx.y;  // Calculate the thread ID within the block
//    int nBlockNum = gridDim.x * gridDim.y;
//    int bid = blockIdx.x * gridDim.y + blockIdx.y;  // Calculate the block ID
//
//    if (tid_in_blk == 0)  // Only thread 0 performs synchronization
//    {
//        Arrayin[bid] = goalVal;  // Set the goal value in the input array
//    }
//
//    if (bid == 1)
//    {
//        if (tid_in_blk < nBlockNum)
//        {
//            while (Arrayin[tid_in_blk] != goalVal)  // Wait until all blocks have set the goal value
//            {
//                // Do nothing here, waiting for synchronization
//            }
//        }
//
//        __syncthreads();
//
//        if (tid_in_blk < nBlockNum)
//        {
//            Arrayout[tid_in_blk] = goalVal;  // Set the goal value in the output array
//        }
//    }
//
//    if (tid_in_blk == 0)
//    {
//        while (Arrayout[bid] != goalVal)  // Wait until the current block has set the goal value
//        {
//            // Do nothing here, waiting for synchronization
//        }
//    }
//
//    __syncthreads();  // Synchronize all threads in the block
//}
//
//
//__device__ volatile int Array_in[49];
//__device__ volatile int Array_out[49];
//
//__global__ void single_table_parallel_test_3(int* matrix, int m, int n){
//    int x = threadIdx.x + blockIdx.x * blockDim.x;
//    int y = threadIdx.y + blockIdx.y * blockDim.y;
//    int numBlocks = gridDim.x*gridDim.y;
//
//
//    int tile_size = 2;
//    int row_size = (m+1 + tile_size - 1) / tile_size;
//    int column_size = (n+1 + tile_size - 1) / tile_size;
//
//
////    printf("numBlocks = %u\n", numBlocks);
//
//    for ( int i = x; i<row_size; i+=gridDim.x * blockDim.x){
//        for ( int j=y; j<column_size; j+=gridDim.y * blockDim.y){
//
//            for (int step = 0; step<row_size+column_size-1; step++){
//                int index_i = m-x;
//                int index_j = n-y;
//                for (int number=0;number<2*tile_size-1; number++){
//                    if (y == step - x){
//                        for (int j = max(0,number-tile_size+1); j<=min(number, tile_size-1); j++){
//                            single_unit_test_2(step, index_i-number+j,index_j-j,matrix,m,n);
//                        }
//                    }
//                }
////                __syncthreads();
//                __gpu_sync_2(step+1, Array_in, Array_out);
//            }
//        }
//    }
//}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void single_unit_test_5(int step, int i, int j, int* D, int m, int n){
    int width = n+1;
    int index = j + i*width;

    if ((i == m) && (j == n)){
        D[index] = 0;
//        D[index] = step;
    }else if ((i <= m-1) && (i >= 0) && (j == n)){
        D[index] = 1 + D[index+width];
//        D[index] = step;
    }else if ((j <= n-1) && (j >= 0) && (i == m)){
//        D[index] = 1 + D[index+1];
//        D[index] = step;
    }else if ((i <= m-1) && (i >= 0) && (j <= n-1) && (j >= 0)){
        D[index] = 0;
//        D[index] = step;
    }
}


__global__ void single_table_parallel_test_5(int* matrix, int m, int n){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int column_thread = blockDim.x*gridDim.x;

    int tile_size = 1;
    int row_size = (m+1 + tile_size - 1) / tile_size;
    int column_size = (n+1 + tile_size - 1) / tile_size;

    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    for (int step = 0; step<row_size+column_size-1;step++) {
        for (int j = max(0,step-column_size+1); j<=min(step, row_size-1); j++){
            if (j%column_thread == x){
                single_unit_test_5(step, row_size-1-j,column_size-1-step+j,matrix,m,n);
            }
        }
//        __syncthreads();
        grid.sync();
    }
}





////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void single_unit_test_6(int step, int i, int j, int* D, int m, int n){
    int width = n+1;
    int index = j + i*width;

    if ((i == m) && (j == n)){
        D[index] = 0;
//        D[index] = step;
    }else if ((i <= m-1) && (i >= 0) && (j == n)){
        D[index] = 1 + D[index+width];
//        D[index] = step;
    }else if ((j <= n-1) && (j >= 0) && (i == m)){
//        D[index] = 1 + D[index+1];
//        D[index] = step;
    }else if ((i <= m-1) && (i >= 0) && (j <= n-1) && (j >= 0)){
        D[index] = 0;
//        D[index] = step;
    }
}


__global__ void single_table_parallel_test_6(int* matrix, int m, int n){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int row_thread = gridDim.x * blockDim.x;
    int column_thread = gridDim.y * blockDim.y;


    int tile_size = 1;
    int row_size = (m+1 + tile_size - 1) / tile_size;
    int column_size = (n+1 + tile_size - 1) / tile_size;

    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    for (int step = 0; step<row_size+column_size-1;step++) {
        for (int j = max(0,step-column_size+1); j<=min(step, row_size-1); j++){
            if (j%row_thread == x & (step-j)%column_thread == y){
                single_unit_test_5(step, row_size-1-j,column_size-1-step+j,matrix,m,n);
            }
        }
//        __syncthreads();
        grid.sync();
    }
}





////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void single_unit_test_7(int step, int i, int j, int* D, int m, int n){
    int width = n+1;
    int index = j + i*width;

    if ((i == m) && (j == n)){
        D[index] = 0;
//        D[index] = step;
    }else if ((i <= m-1) && (i >= 0) && (j == n)){
        D[index] = 1 + D[index+width];
//        D[index] = step;
    }else if ((j <= n-1) && (j >= 0) && (i == m)){
//        D[index] = 1 + D[index+1];
//        D[index] = step;
    }else if ((i <= m-1) && (i >= 0) && (j <= n-1) && (j >= 0)){
        D[index] = 0;
//        D[index] = step;
    }
}


__global__ void single_table_parallel_test_7(int* matrix, int m, int n){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int row_thread = gridDim.x * blockDim.x;
    int column_thread = gridDim.y * blockDim.y;


    int tile_size = 1;
    int row_size = (m+1 + tile_size - 1) / tile_size;
    int column_size = (n+1 + tile_size - 1) / tile_size;

    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    for (int step = 0; step<row_size+column_size-1;step++) {
        for (int j = max(0,step-column_size+1); j<=min(step, row_size-1); j++){
            if (j%row_thread == x & (step-j)%column_thread == y){
                single_unit_test_5(step, row_size-1-j,column_size-1-step+j,matrix,m,n);
            }
        }
//        __syncthreads();
        grid.sync();
    }
}

//__device__ volatile int Array_in[64];
//__device__ volatile int Array_out[64];
//
//__device__ void __gpu_sync(int goalVal, volatile int *Arrayin, volatile int *Arrayout)
//{
//    // Thread ID in a block
//    int tid_in_blk = threadIdx.x;
//    int nBlockNum = gridDim.x;
//    int bid = blockIdx.x;
//
//    // Only thread 0 is used for synchronization
//    if (tid_in_blk == 0)
//    {
//        Arrayin[bid] = goalVal;
//    }
//
//    if (bid == 1)
//    {
//        if (tid_in_blk < nBlockNum)
//        {
//            while (Arrayin[tid_in_blk] != goalVal)
//            {
//            }
//        }
//
//        __syncthreads();
//
//        if (tid_in_blk < nBlockNum)
//        {
//            Arrayout[tid_in_blk] = goalVal;
//        }
//    }
//
//    if (tid_in_blk == 0)
//    {
//        while (Arrayout[bid] != goalVal)
//        {
//        }
//    }
//
//    __syncthreads();
//}


__global__ void d_test(){
//    int x = threadIdx.x + blockIdx.x*blockDim.x;
//    int y = threadIdx.y + blockIdx.y*blockDim.y;
//    printf("x = %u, y = %u \n",x,y);
    int c = 1+1;
    for(int i = 0 ; i<2000; i++) {
//        __gpu_sync(i, Array_in, Array_out);
    }
}

void test(){
    Stream stream;
    int grid = 64;
    int block = 1024;

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream.cuda_stream());

    d_test<<<grid, block, 0, stream.cuda_stream()>>>();
    stream.Sync();

    cudaEventRecord(stop, stream.cuda_stream());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("sync_time = %.3fms\n", milliseconds);


    cudaDeviceSynchronize();
    printf("hello\n");
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




__global__ void test_time(){
    int c = 1+1;
}

__global__ void myKernel(int* data, int step){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    bool s = grid.is_valid();

    printf("%d\n", s);

    printf("hello world\n");

    grid.sync();

}

__global__ void warp(){
//    int warpId= (threadIdx.x+blockIdx.x*blockDim.x)/32;
    for (int i=0;i<100000;i++) {
        __syncwarp();
    }
}

__global__ void block(){
    for (int i=0;i<1000000;i++) {
        __syncthreads();
    }
}

__device__ volatile int Array_in_3[82];
__device__ volatile int Array_out_3[82];
__global__ void multi_block(){
    for (int i=0;i<500000;i++) {
        __gpu_sync_range_fence(0, 1, 0, Array_in_3, Array_out_3);
        __gpu_sync_range_fence(0, 1, 1, Array_in_3, Array_out_3);
    }
}


int main(int argc, char* argv[]) {

    int num_nodes = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    int parallel_version = atoi(argv[3]);
    char* file_path_1 = argv[4];
    char* file_path_2 = argv[5];

    num_nodes = 100;
    num_threads = 5;
    parallel_version = 36;

//    test_1(num_nodes,num_threads,parallel_version);
//    test_2(num_threads,parallel_version);
for(int i=0; i <5;i++) {
    swissport_test(num_threads, parallel_version, file_path_1, file_path_2);
}


//    Stream stream_test;
//    for (int i=0; i<10;i++) {
//        cudaEvent_t start, stop;
//        float milliseconds = 0;
//        cudaEventCreate(&start);
//        cudaEventCreate(&stop);
//        cudaEventRecord(start, stream_test.cuda_stream());
//
//        warp<<<1, 32, 0, stream_test.cuda_stream()>>>();
//
//        cudaEventRecord(stop, stream_test.cuda_stream());
//        cudaEventSynchronize(stop);
//        cudaEventElapsedTime(&milliseconds, start, stop);
//        cudaEventDestroy(start);
//        cudaEventDestroy(stop);
//        printf("Warp Sync Time = %.3fms\n", milliseconds);
//    }
//
//    Stream stream_test_2;
//    for (int i=0; i<10;i++) {
//        cudaEvent_t start, stop;
//        float milliseconds = 0;
//        cudaEventCreate(&start);
//        cudaEventCreate(&stop);
//        cudaEventRecord(start, stream_test_2.cuda_stream());
//
//        block<<<1, 1024, 0, stream_test_2.cuda_stream()>>>();
//
//        cudaEventRecord(stop, stream_test_2.cuda_stream());
//        cudaEventSynchronize(stop);
//        cudaEventElapsedTime(&milliseconds, start, stop);
//        cudaEventDestroy(start);
//        cudaEventDestroy(stop);
//        printf("Block Sync Time = %.3fms\n", milliseconds);
//    }
//
//    Stream stream_test_3;
//    for (int i=0; i<10;i++) {
//        cudaEvent_t start, stop;
//        float milliseconds = 0;
//        cudaEventCreate(&start);
//        cudaEventCreate(&stop);
//        cudaEventRecord(start, stream_test_3.cuda_stream());
//
//        multi_block<<<2, 1024, 0, stream_test_3.cuda_stream()>>>();
//
//        cudaEventRecord(stop, stream_test_3.cuda_stream());
//        cudaEventSynchronize(stop);
//        cudaEventElapsedTime(&milliseconds, start, stop);
//        cudaEventDestroy(start);
//        cudaEventDestroy(stop);
//        printf("Multi-Block Sync Time = %.3fms\n", milliseconds);
//    }



//    test();

//    int blockSize = 32;
//    int numBlock = (d_matrix.size() + blockSize - 1) / blockSize;
//    compute<<<numBlock, blockSize, 0, 0>>>(p_d_matrix, d_matrix.size());


//    printf("Hello\n");
//    printDeviceVector(d_matrix);
//    mathKernel1<<<numBlock, blockSize, 0, 0>>>(p_d_matrix, d_matrix.size(), 32);
//    tiling<<<numBlock, blockSize, 0, 0>>>(p_d_matrix, d_matrix.size(), 1, m, n);
//    printDeviceVector(d_matrix);


//    dim3 block(32,32);
//    dim3 grid((m+block.x-1+1)/block.x,(n+block.y-1+1)/block.y);
//    printf("%d, %d, %d, %d\n", grid.x, grid.y, block.x, block.y);

//    compute2<<<grid, block>>>(p_d_matrix, m, n);
//    cudaDeviceSynchronize();
//    diagonal<<<grid, block>>>(p_d_matrix, m, n);

//    diagonal2<<<grid, block>>>(p_d_matrix, m+1, n+1, 2);
//    cudaDeviceSynchronize();
//    printDeviceVector2(d_matrix, n);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//    Stream stream1;
//    int n = 50;
//    int m = 50;
//    vector<int> matrix((n+1)*(m+1), 0);
//    thrust::device_vector<int> d_matrix(matrix);
//    int* p_d_matrix = thrust::raw_pointer_cast(d_matrix.data());
//
//    printDeviceVector2(d_matrix, n);
//
//    printf("\nhello\n\n");
//
//    int tile = 2;
//    dim3 block(16,16);
//    dim3 grid((m+block.x*tile)/(block.x*tile),(n+block.y*tile)/(block.y*tile));
//
//
////    single_table_parallel_test<<<grid, block>>>(p_d_matrix, m, n);
//
////    void *kernel_args[] = {(void*)&p_d_matrix, (void *)&m, (void *)&n};
////
////    cudaLaunchCooperativeKernel((void *)single_table_parallel_test, grid, block, kernel_args, 0, stream1.cuda_stream());
//
////    single_table_parallel_test_2<<<grid, block>>>(p_d_matrix, m, n);
//
////    single_table_parallel_test_3<<<grid, block>>>(p_d_matrix, m, n);
//
////    single_table_parallel_test_4<<<grid, block>>>(p_d_matrix, m, n);
//
//    int one_block = 32;
//    int one_grid = 2;
//
//    void *kernel_args[] = {(void*)&p_d_matrix, (void *)&m, (void *)&n};
//    cudaLaunchCooperativeKernel((void *)single_table_parallel_test_5, one_grid, one_block, kernel_args, 0, stream1.cuda_stream());
//
////    printf("hello\n");
//
//    CHECK_CUDA(cudaDeviceSynchronize());
//    cudaError_t error = cudaGetLastError();
//    if (error != cudaSuccess) {
//        printf("CUDA error: %s\n", cudaGetErrorString(error));
//        // Additional error handling or debugging steps
//    }
//
//    printDeviceVector2(d_matrix, n);
//
//    printf("\n%u, %u, %u, %u \n\n", block.x, block.y, grid.x, grid.y);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//    const int N = 256;
//    const int block_size = 128;
//    const int grid_size = (N + block_size - 1) / block_size;
//
//    int* data;
//    int step = 0;
//    cudaMalloc((void**)&data, N * sizeof(int));
//    CHECK_CUDA(cudaDeviceSynchronize());
//
//    myKernel<<<grid_size, block_size>>>(data);
//
//    void *kernel_args[] = {(void*)&data, (void *)&step};
//
//    cudaLaunchCooperativeKernel((void *) myKernel, grid_size, block_size, kernel_args);
//
//    CHECK_CUDA(cudaDeviceSynchronize());
//
//    cudaFree(data);


    return 0;
}

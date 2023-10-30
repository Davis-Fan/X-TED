#include "TED_C++.h"

//__device__ int min3_D(const int& a, const int& b, const int& c){
__device__ int min3_D(int a, int b, int c){
//    int min_val = a;
//    if (b < min_val) {
//        min_val = b;
//    }
//    if (c < min_val) {
//        min_val = c;
//    }
//    return min_val;
    return min(min(a,b),c);
}

__device__ void printQueue(Stream & stream, dev::Queue<int, uint32_t> d_queue) {
    for (int k = 0; k < d_queue.size(); k++) {
        int temp = d_queue[k];
        printf("%i ", temp);
    }
}

void printDeviceVector(thrust::device_vector<int> &d_a) {
    for (int k = 0; k < d_a.size(); k++) {
        int temp = d_a[k];
        printf("%i \n", temp);
    }
}

__global__ void printDeviceVector3(int* p_D_d, int size, int n) {
    int j = 0;
    printf("\n");
    for (int i=0; i<size; i++){
        printf("%u ", p_D_d[i]);
        j++;
        if (j == n){
            printf("\n");
            j=0;
        }
    }
}

__global__ void printArrayVector(ArrayView<int> d_a) {
    for (int k = TID_1D; k < d_a.size(); k+=TOTAL_THREADS_1D) {
        int temp = d_a[k];
        printf("This element is %i \n", temp);
    }
}

__global__ void printQueueVector(dev::Queue<int, uint32_t> d_queue) {
    for (int k = TID_1D; k < d_queue.size(); k+=TOTAL_THREADS_1D) {
        int temp = d_queue[k];
        printf("This element is %i \n", temp);
    }
    printf("\n");
}

__device__ void task_GPU_2(int* x_orl, int* x_kr, int* y_orl, int* y_kr, int* Delta, int* D, int* D_tree, int thread_in_number, int table_in_number, int L, int m, int n, int batch_size){
    int row = table_in_number / L;
    int column = table_in_number % L;
    int width = n+1;

    int i;
    int j;

    int i_0;
    int j_0;
    int i_max;
    int j_max;

    i_0 = x_kr[row];
    j_0 = y_kr[column];
    i_max = x_orl[i_0] + 1;
    j_max = y_orl[j_0] + 1;

    D[(i_max*width+j_max)*batch_size + thread_in_number]=0;

    for (i = i_max - 1; i > i_0 - 1; i--) {
        D[(i*width+j_max)*batch_size + thread_in_number] = 1 + D[((i+1)*width+j_max)*batch_size + thread_in_number];
    }

    for (j = j_max - 1; j > j_0 - 1; j--) {
        D[(i_max*width+j)*batch_size + thread_in_number] = 1 + D[(i_max*width+j+1)*batch_size + thread_in_number];
    }

    for (i = i_max - 1; i > i_0 - 1; i--) {
        for (j = j_max - 1; j > j_0 - 1; j--) {

            if ((x_orl[i] == x_orl[i_0]) & (y_orl[j] == y_orl[j_0])) {

                D[(i*width+j)*batch_size + thread_in_number] = min3_D(Delta[j+i*n]+D[((i+1)*width+j+1)*batch_size + thread_in_number],
                                                                      1+D[((i+1)*width+j)*batch_size + thread_in_number],
                                                                      1+D[(i*width+j+1)*batch_size + thread_in_number]);

                D_tree[j + i*n] = D[(i*width+j)*batch_size + thread_in_number];


            } else {
                D[(i*width+j)*batch_size + thread_in_number] = min3_D(D_tree[j + i*n]+D[((x_orl[i] + 1)*width+y_orl[j] + 1)*batch_size + thread_in_number],
                                                                      1+D[((i+1)*width+j)*batch_size + thread_in_number],
                                                                      1+D[(i*width+j+1)*batch_size + thread_in_number]);


            }
        }
    }
}

__device__ void single_unit_2(int* x_orl, int* y_orl, int i_0, int i_max, int j_0, int j_max, int i, int j, int* Delta, int* D, int* D_tree, int thread_in_number, int m, int n, int batch_size){
//    int num_table_offset = thread_in_number*(m+1)*(n+1);
    int width = n+1;
    int offset=width*batch_size;
    int index = (i*width+j)*batch_size + thread_in_number;
    if ((i == i_max) && (j == j_max)){
        D[index] = 0;
    }else if ((i <= i_max-1) && (i >= i_0) && (j == j_max)){
        D[index] = 1 + D[index + offset];
    }else if ((j <= j_max-1) && (j >= j_0) && (i == i_max)){
        D[index] = 1 + D[index + batch_size];
    }else if ((i <= i_max-1) && (i >= i_0) && (j <= j_max-1) && (j >= j_0)){
        if ((x_orl[i] == x_orl[i_0]) && (y_orl[j] == y_orl[j_0])) {

            D[index] = min3_D(Delta[j+i*n] + D[offset+batch_size+index],
                              1 + D[offset + index],
                              1 + D[index + batch_size]);
            D_tree[j + i*n] = D[index];

        }else{
            D[index] = min3_D(D_tree[j + i*n] + D[(x_orl[i]*width+y_orl[j])*batch_size + offset + batch_size + thread_in_number],
                              1 + D[offset + index],
                              1 + D[index + batch_size]);
        }
    }
}

__global__ void fetch_task(ArrayView<int> d_view, dev::Queue<int, uint32_t> d_queue, int current_depth){
    for(int i=TID_1D;i<d_view.size();i+=TOTAL_THREADS_1D) {
        if (d_view[i] == current_depth){
            d_queue.Append(i);
        }
    }
}

__global__ void fetch_size(ArrayView<int> d_size, dev::Queue<int, uint32_t> d_queue, int* x_orl, int* y_orl, int* x_kr, int* y_kr, int L){
    for (int i=TID_1D;i<d_queue.size();i+=TOTAL_THREADS_1D){
        int task = d_queue[i];
        int row = task / L;
        int column = task % L;
        int i_0 = x_kr[row];
        int j_0 = y_kr[column];
        d_size[i] =  (x_orl[i_0] + 1-i_0+1)*(y_orl[j_0] + 1-j_0+1);
//        d_size[i] =  (y_orl[j_0] + 1-j_0+1);
    }
}

__global__ void fetch_size_queue(dev::Queue<int, uint32_t> d_size, dev::Queue<int, uint32_t> d_queue, int* x_orl, int* y_orl, int* x_kr, int* y_kr, int L){
    for (int i=TID_1D;i<d_queue.size();i+=TOTAL_THREADS_1D){
        int task = d_queue[i];
        int row = task / L;
        int column = task % L;
        int i_0 = x_kr[row];
        int j_0 = y_kr[column];
        d_size[i] =  (x_orl[i_0] + 1-i_0+1)*(y_orl[j_0] + 1-j_0+1);
//        d_size[i] = y_orl[j_0] + 1-j_0+1;
    }
}

__global__ void filter(dev::Queue<int, uint32_t> d_queue1, dev::Queue<int, uint32_t> d_queue2, dev::Queue<int, uint32_t> large_queue, dev::Queue<int, uint32_t> block_size_queue, int threshold1, int threshold2){
    for (int i=TID_1D;i<d_queue1.size();i+=TOTAL_THREADS_1D){
        if (d_queue2[i] >= threshold1 && d_queue2[i] < threshold2){
            int task = d_queue1[i];
            large_queue.Append(task);
        }

        if(d_queue2[i] >= threshold2){
            block_size_queue.Append(d_queue1[i]);
        }
    }
}

__global__ void filter_new(dev::Queue<int, uint32_t> d_queue1, dev::Queue<int, uint32_t> d_queue2, dev::Queue<int, uint32_t> large_queue, dev::Queue<int, uint32_t> block_size_queue, dev::Queue<int, uint32_t> multi_block_queue, int threshold1, int threshold2, int threshold3){
    for (int i=TID_1D;i<d_queue1.size();i+=TOTAL_THREADS_1D){
        if (d_queue2[i] >= threshold1 && d_queue2[i] < threshold2){
            int task = d_queue1[i];
            large_queue.Append(task);
        }

        if(d_queue2[i] >= threshold2 && d_queue2[i] < threshold3){
            block_size_queue.Append(d_queue1[i]);
        }

        if(d_queue2[i] >= threshold3){
            multi_block_queue.Append(d_queue1[i]);
        }

    }
}



















__global__ void parallel_multi_table_1_flag(int flag, dev::Queue<int, uint32_t> d_queue, int* x_orl, int* x_kr, int* y_orl, int* y_kr, int* Delta, int* D, int* D_tree, int n, int m, int t, int k, int L){

    if (flag == 0) {
        for (int w = TID_1D; w < ((t + 1) * d_queue.size())/2; w += TOTAL_THREADS_1D) {

            int new_row = w / (t + 1);
            int new_column = w % (t + 1);
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
        }
    }else{
        for (int w = TID_1D; w < ((t + 1) * d_queue.size())/2; w += TOTAL_THREADS_1D) {
            w = w + ((t + 1) * d_queue.size())/2;
            int new_row = w / (t + 1);
            int new_column = w % (t + 1);
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
        }
    }

}

__global__ void parallel_multi_table_2_flag(int flag, dev::Queue<int, uint32_t> d_queue, int* x_orl, int* x_kr, int* y_orl, int* y_kr, int* Delta, int* D, int* D_tree, int n, int m, int t, int k, int L){

    if(flag == 0) {
        for (int w = TID_1D; w < (t + 1) * d_queue.size(); w += TOTAL_THREADS_1D) {

            int new_row = w / (t + 1);
            int new_column = w % (t + 1);
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

            single_unit(x_orl, y_orl, i_0, i_max, j_0, j_max, m - k + n + 1 - new_column, 0 + new_column, Delta, D,
                        D_tree,
                        new_row, m, n);
        }
    }else{
        for (int w = TID_1D; w < (t + 1) * d_queue.size(); w += TOTAL_THREADS_1D) {
            w = w + ((t + 1) * d_queue.size())/2;
            int new_row = w / (t + 1);
            int new_column = w % (t + 1);
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

            single_unit(x_orl, y_orl, i_0, i_max, j_0, j_max, m - k + n + 1 - new_column, 0 + new_column, Delta, D,
                        D_tree,
                        new_row, m, n);
        }
    }

}


void multi_table_flag(int flag, Stream& stream_mul, dev::Queue<int, uint32_t> d_queue, int size, int L, int blockSize, int n, int m, int *p_x_orl_d, int *p_x_kr_d, int *p_y_orl_d, int *p_y_kr_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){

    int numBlocks;

    int x = n + 1;
    int y = m + 1;
    int t;

    for (int k=1; k<=x+y-1; k++){
        if(k<=x){
            t = min(k-1,m);
            numBlocks = ((t+1)*size + blockSize) / blockSize;
            parallel_multi_table_1_flag<<<numBlocks, blockSize, 0, stream_mul.cuda_stream()>>>(flag, d_queue, p_x_orl_d, p_x_kr_d, p_y_orl_d, p_y_kr_d, p_Delta_d, p_D_d, p_D_tree_d, n, m, t, k ,L);
        }else{
            t = min(n, m-k+x);
            numBlocks = ((t+1)*size + blockSize) / blockSize;
            parallel_multi_table_2_flag<<<numBlocks, blockSize, 0, stream_mul.cuda_stream()>>>(flag,d_queue, p_x_orl_d, p_x_kr_d, p_y_orl_d, p_y_kr_d, p_Delta_d, p_D_d, p_D_tree_d, n, m, t, k ,L);
        }
    }

}
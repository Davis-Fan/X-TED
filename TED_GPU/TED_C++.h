#include <iostream>
#include <vector>
#include <time.h>
#include <string>
#include "fstream"
#include "thread"
#include "help.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "utils/array_view.h"
#include "utils/event.h"
#include "utils/launcher.h"
#include "utils/queue.h"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
using namespace std;

#ifndef TED_GPU_TED_C_H
#define TED_GPU_TED_C_H

extern long long Total_time;
extern float total_milliseconds;


int min3(int a, int b, int c);
vector<int> outermost_right_leaves(vector<vector<int>>& adj);
vector<int> key_roots(vector<int>& orl);
vector<vector<int>> standard_ted_2(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree);
vector<vector<int>> standard_ted_1 (vector<string>& x_node, vector<vector<int>>& x_adj, vector<string>& y_node, vector<vector<int>>& y_adj, int num_threads, int parallel_version);
int standard_ted (vector<string>& x_node, vector<vector<int>>& x_adj, vector<string>& y_node, vector<vector<int>>& y_adj, int num_threads, int parallel_version);
void print_vector(vector<int> x);


// GPU
void printDeviceVector(thrust::device_vector<int> &d_a);
void printDeviceVector2(thrust::device_vector<int> &d_a, int n);
__global__ void printDeviceVector3(int* p_D_d, int size, int n);


__device__ int min3_D(int a, int b, int c);
__global__ void fetch_task(ArrayView<int> d_view, dev::Queue<int, uint32_t> d_queue, int current_depth);
__global__ void fetch_size(ArrayView<int> d_size, dev::Queue<int, uint32_t> d_queue, int* x_orl, int* y_orl, int* x_kr, int* y_kr, int L);
__global__ void fetch_size_queue(dev::Queue<int, uint32_t> d_size, dev::Queue<int, uint32_t> d_queue, int* x_orl, int* y_orl, int* x_kr, int* y_kr, int L);
//__global__ void filter(dev::Queue<int, uint32_t> d_queue1, dev::Queue<int, uint32_t> d_queue2, dev::Queue<int, uint32_t> large_queue, int threshold);
__global__ void filter(dev::Queue<int, uint32_t> d_queue1, dev::Queue<int, uint32_t> d_queue2, dev::Queue<int, uint32_t> large_queue, dev::Queue<int, uint32_t> block_size_queue, int threshold1, int threshold2);
__global__ void filter_new(dev::Queue<int, uint32_t> d_queue1, dev::Queue<int, uint32_t> d_queue2, dev::Queue<int, uint32_t> large_queue, dev::Queue<int, uint32_t> block_size_queue, dev::Queue<int, uint32_t> multi_block_queue, int threshold1, int threshold2, int threshold3);

__device__ void __gpu_sync_range_update(int startBlockIdx, int endBlockIdx, int goalVal, volatile int *Arrayin, volatile int *Arrayout);
__device__ void __gpu_sync_range_fence(int startBlockIdx, int endBlockIdx, int goalVal, volatile int *Arrayin, volatile int *Arrayout);

__device__ void single_unit(int* x_orl, int* y_orl, int i_0, int i_max, int j_0, int j_max, int i, int j, int* Delta, int* D, int* D_tree, int thread_in_number, int m, int n);
__device__ void single_unit_2(int* x_orl, int* y_orl, int i_0, int i_max, int j_0, int j_max, int i, int j, int* Delta, int* D, int* D_tree, int thread_in_number, int m, int n, int batch_size);
__device__ void single_unit_10(int* x_orl, int* y_orl, int i_0, int i_max, int j_0, int j_max, int i, int j, int* Delta, int* D, int* D_tree, int thread_in_number, int m, int n);
__device__ void task_GPU(int* x_orl, int* x_kr, int* y_orl, int* y_kr, int* Delta, int* D, int* D_tree, int thread_in_number, int table_in_number, int L, int m, int n);

void simple_thread (Stream& stream_single, float& total_milliseconds, int limitation, dev::Queue<int, uint32_t> d_queue1, int L, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int*p_x_kr_d, int*p_y_kr_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d);
void simple_warp (Stream& stream_mul, float& total_milliseconds,int limitation, thrust::host_vector<int>& queue_h, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, int L, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d);
void simple_block (Stream& stream_mul, float& total_milliseconds, int limitation, thrust::host_vector<int>& queue_h, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, int L, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d);
void multi_block (Stream& stream_mul, float& total_milliseconds, int limitation, thrust::host_vector<int>& queue_h, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, int L, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d);
void last_table(Stream& stream_sing, float& total_milliseconds, int i, int thread_in_number, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, int L, int blockSize, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d);

__global__ void parallel_multi_table_1_flag(int flag, dev::Queue<int, uint32_t> d_queue, int* x_orl, int* x_kr, int* y_orl, int* y_kr, int* Delta, int* D, int* D_tree, int n, int m, int t, int k, int L);
__global__ void parallel_multi_table_2_flag(int flag, dev::Queue<int, uint32_t> d_queue, int* x_orl, int* x_kr, int* y_orl, int* y_kr, int* Delta, int* D, int* D_tree, int n, int m, int t, int k, int L);
void multi_table_flag(int flag, Stream& stream_mul, dev::Queue<int, uint32_t> d_queue, int size, int L, int blockSize, int n, int m, int *p_x_orl_d, int *p_x_kr_d, int *p_y_orl_d, int *p_y_kr_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d);
__global__ void initializeArrayToZero();

// parallel_version

vector<vector<int>> parallel_standard_ted_2_18(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threads, vector<vector<int>>& x_adj, vector<vector<int>>& y_adj);
vector<vector<int>> parallel_standard_ted_2_28(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threads, vector<vector<int>>& x_adj, vector<vector<int>>& y_adj);
vector<vector<int>> parallel_standard_ted_Zhang(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threads, vector<vector<int>>& x_adj, vector<vector<int>>& y_adj);
vector<vector<int>> parallel_standard_ted_test(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threads, vector<vector<int>>& x_adj, vector<vector<int>>& y_adj);


vector<vector<int>> parallel_standard_ted_test_threshold(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threads, vector<vector<int>>& x_adj, vector<vector<int>>& y_adj);
void single_table_new_9(Stream& stream_sing, int i, int thread_in_number, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, int L, int blockSize, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d);
__global__ void simple_parallel_new(int* x_orl, int* x_kr, int* y_orl, int* y_kr, int* Delta, int* D, int* D_tree, int n, int m, int L, dev::Queue<int, uint32_t> d_queue, int limitation);
void multi_table_warp_28 (Stream& stream_mul, float& total_milliseconds, thrust::host_vector<int>& queue_h, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, dev::Queue<int, uint32_t> d_queue, int size, int L, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d);
void multi_table_new_update_8 (Stream& stream_mul, float& total_milliseconds, thrust::host_vector<int>& queue_h, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, dev::Queue<int, uint32_t> d_queue, int size, int L, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d);




int ToInt(const string &str);
vector<int> To_Vector(string str);
vector<string> splitting(string str);
vector<vector<int>> pre_process(string str);
vector<string> node_process(string str);




void test_1(int num_nodes,int num_threads, int parallel_version);
void test_2(int num_threads, int parallel_version);
void test_3(int num_of_nodes, int num_threads, int parallel_version);
void sentiment_test(int num_threads, int parallel_version);
void bolzano_test(int num_threads, int parallel_version);
//void swissport_test(int num_threads, int parallel_version);
void swissport_test(int num_threads, int parallel_version, char* input_1, char* input_2);
void python_test(int num_threads, int parallel_version);

#endif //TED_GPU_TED_C_H

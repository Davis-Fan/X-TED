
#include <iostream>
#include <vector>
#include <time.h>
#include <string>
#include "fstream"
#include "thread"
#include "queue"
#include "unordered_set"
#include "set"
//#include </opt/homebrew/Cellar/libomp/15.0.7/include/omp.h>
using namespace std;

#ifndef TED_C___TED_C_H
#define TED_C___TED_C_H

extern long long Total_time;

int min3(int a, int b, int c);
vector<int> outermost_right_leaves(vector<vector<int>>& adj);
vector<int> key_roots(vector<int>& orl);
vector<vector<int>> standard_ted_2(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree);
vector<vector<int>> standard_ted_1 (vector<string>& x_node, vector<vector<int>>& x_adj, vector<string>& y_node, vector<vector<int>>& y_adj, int num_threads, int parallel_version);
int standard_ted (vector<string>& x_node, vector<vector<int>>& x_adj, vector<string>& y_node, vector<vector<int>>& y_adj, int num_threads, int parallel_version);
void print_vector(vector<int> x);

// parallel_version
vector<vector<int>> parallel_standard_ted_2(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree);
vector<vector<int>> parallel_standard_ted_2_2(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree);
vector<vector<int>> parallel_standard_ted_2_3(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree);
vector<vector<int>> parallel_standard_ted_2_4(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree);
vector<vector<int>> parallel_standard_ted_2_5(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree);
vector<vector<int>> parallel_standard_ted_2_6(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree);
vector<vector<int>> parallel_standard_ted_2_7(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree);
vector<vector<int>> parallel_standard_ted_2_8(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n);
vector<vector<int>> parallel_standard_ted_2_9(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n);
vector<vector<int>> parallel_standard_ted_2_10(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n);
vector<vector<int>> parallel_standard_ted_2_11(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threads);
vector<vector<int>> parallel_standard_ted_2_12(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threads);
vector<vector<int>> parallel_standard_ted_2_13(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threads);
vector<vector<int>> parallel_standard_ted_2_14(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threads);
vector<vector<int>> parallel_standard_ted_2_15(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threads);
vector<vector<int>> parallel_standard_ted_2_16(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threadsvector, vector<vector<int>>& x_adj, vector<vector<int>>& y_adj);
vector<vector<int>> parallel_standard_ted_2_17(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threadsvector, vector<vector<int>>& x_adj, vector<vector<int>>& y_adj);
vector<vector<int>> parallel_standard_ted_2_18(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threads, vector<vector<int>>& x_adj, vector<vector<int>>& y_adj);


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
void swissport_test(int num_threads, int parallel_version,  char* input_1, char* input_2);
void python_test(int num_threads, int parallel_version);
void dblp_test(int num_threads, int parallel_version);

#endif //TED_C___TED_C_H

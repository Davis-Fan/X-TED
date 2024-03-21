#include <iostream>
#include <vector>
#include <time.h>
#include <string>
#include "fstream"
#include "thread"
#include "queue"
#include "unordered_set"
#include "set"

using namespace std;

#ifndef TED_C___TED_C_H
#define TED_C___TED_C_H

int min3(int a, int b, int c);
vector<int> outermost_right_leaves(vector<vector<int>>& adj);
vector<int> key_roots(vector<int>& orl);
vector<vector<int>> sequential_cpu_ted(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree);
vector<vector<int>> standard_ted_1 (vector<string>& x_node, vector<vector<int>>& x_adj, vector<string>& y_node, vector<vector<int>>& y_adj, int num_threads, int parallel_version);
int standard_ted (vector<string>& x_node, vector<vector<int>>& x_adj, vector<string>& y_node, vector<vector<int>>& y_adj, int num_threads, int parallel_version);

// parallel_version
vector<vector<int>> parallel_cpu_ted(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threads, vector<vector<int>>& x_adj, vector<vector<int>>& y_adj);


int ToInt(const string &str);
vector<int> To_Vector(string str);
vector<string> splitting(string str);
vector<vector<int>> pre_process(string str);
vector<string> node_process(string str);


void test(int num_threads, int parallel_version,  char* input_1, char* input_2);

#endif //TED_C___TED_C_H

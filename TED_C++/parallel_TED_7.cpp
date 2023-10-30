#include "TED_C++.h"
//#include <fstream>

// A thread for each table (2 threads interleaved) & Internal: 100

void compute_7(int k, int l, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){
    int i;
    int j;

    int i_0;
    int j_0;
    int i_max;
    int j_max;

    i_0 = x_kr[k];
    j_0 = y_kr[l];
    i_max = x_orl[i_0] + 1;
    j_max = y_orl[j_0] + 1;
    D[i_max][j_max] = 0;


    for (i = i_max - 1; i > i_0 - 1; i--) {
        D[i][j_max] = 1 + D[i + 1][j_max];
    }

    for (j = j_max - 1; j > j_0 - 1; j--) {
        D[i_max][j] = 1 + D[i_max][j + 1];
    }

    for (i = i_max - 1; i > i_0 - 1; i--) {
        for (j = j_max - 1; j > j_0 - 1; j--) {

            if ((x_orl[i] == x_orl[i_0]) & (y_orl[j] == y_orl[j_0])) {

                D[i][j] = min3(Delta[i][j] + D[i + 1][j + 1], 1 + D[i + 1][j], 1 + D[i][j + 1]);
                D_tree[i][j] = D[i][j];

            } else {


                while (D_tree[i][j] == -1){
                    continue;
                }

                D[i][j] = min3(D_tree[i][j] + D[x_orl[i] + 1][y_orl[j] + 1], 1 + D[i + 1][j],
                               1 + D[i][j + 1]);

            }

        }
    }


    for (i=i_max; i>i_0-1; i--){
        for (j=j_max; j>j_0-1; j--){
            D[i][j] = -1;
        }
    }
}

void task_7(int n, int L, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){
    n--;
    int row = n / L;
    int column = n % L;
    compute_7(row,column,x_orl,x_kr,y_orl,y_kr,Delta, D, D_tree);
}

void worker_7_1(int K, int L, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){

    int i = 1;
    while(i<=250401){
        for (int j=0; j<100; j++){
            task_7(i+j, L, x_orl,x_kr,y_orl,y_kr,Delta, D, D_tree);
        }
        i = i + 200;
    }
}

void worker_7_2(int K, int L, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){

    int i = 101;
    while(i<=250301){
        for (int j=0; j<100; j++){
            task_7(i+j, L, x_orl,x_kr,y_orl,y_kr,Delta, D, D_tree);
        }
        i = i + 200;
    }
}



vector<vector<int>> parallel_standard_ted_2_7(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){

    int K = (int)x_kr.size();
    int L = (int)y_kr.size();

    int k;
    int l;


    vector<vector<int>> D_2 = D;

    thread t1(worker_7_1, K, L, ref(x_orl), ref(x_kr), ref(y_orl), ref(y_kr), ref(Delta), ref(D), ref(D_tree));
    thread t2(worker_7_2, K, L, ref(x_orl), ref(x_kr), ref(y_orl), ref(y_kr), ref(Delta), ref(D_2), ref(D_tree));
    t1.join();
    t2.join();


    vector<vector<int>> final_result = D_tree;
    return final_result;
}


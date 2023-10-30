#include "TED_C++.h"
//#include <fstream>

// A thread for each table (2 threads interleaved)

void compute_5(int k, int l, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){
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

void task(int n, int L, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){
    n--;
    int row = n / L;
    int column = n % L;
    compute_5(row,column,x_orl,x_kr,y_orl,y_kr,Delta, D, D_tree);
}

void worker_1(int K, int L, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){
    if((K*L)%2 == 1){
        int i = 1;
        while (i <= K*L){
            task(i, L, x_orl,x_kr,y_orl,y_kr,Delta, D, D_tree);
            i += 2;
        }
    }else{

        int i = 1;
        while (i <= (K*L-1)){
            task(i, L, x_orl,x_kr,y_orl,y_kr,Delta, D, D_tree);
            i += 2;
        }

    }
}

void worker_2(int K, int L, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){
    if((K*L)%2 == 1){
        int j = 2;
        while (j <= (K*L-1)){
            task(j, L, x_orl,x_kr,y_orl,y_kr,Delta, D, D_tree);
            j += 2;
        }
    }else{
        int j = 2;
        while (j <= (K*L)){
            task(j, L, x_orl,x_kr,y_orl,y_kr,Delta, D, D_tree);
            j += 2;
        }
//        cout << endl;
    }
//    cout << endl;
}


vector<vector<int>> parallel_standard_ted_2_5(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){

    int K = (int)x_kr.size();
    int L = (int)y_kr.size();

    int k;
    int l;

    vector<vector<int>> D_2 = D;

    thread t1(worker_1, K, L, ref(x_orl), ref(x_kr), ref(y_orl), ref(y_kr), ref(Delta), ref(D), ref(D_tree));
    thread t2(worker_2, K, L, ref(x_orl), ref(x_kr), ref(y_orl), ref(y_kr), ref(Delta), ref(D_2), ref(D_tree));
    t1.join();
    t2.join();

//    ofstream outfile("/Users/davis/CLionProjects/TED-C++/D_tree.txt");
//    for (int i=0;i<D_tree.size();i++){
//        for (int j=0; j<D_tree[0].size(); j++){
//            outfile << D_tree[i][j] << " ";
//        }
//        outfile << endl;
//    }
//    outfile.close();


    vector<vector<int>> final_result = D_tree;
    return final_result;
}

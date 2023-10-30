#include "TED_C++.h"

// Zhang's Parallel Version: Single Thread

void compute_8(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<vector<vector<int>>>>& D_forest_4D, vector<vector<int>>& D_tree, int i, int j, int i_0, int i_max, int j_0, int j_max, int k, int l){
    if ((i == i_max) && (j == j_max)){
        D_forest_4D[k][l][i][j] = 0;
    }else if ((i <= i_max-1) && (i >= i_0) && (j == j_max)){
        D_forest_4D[k][l][i][j_max] = 1 + D_forest_4D[k][l][i + 1][j_max];
    }else if ((j <= j_max-1) && (j >= j_0) && (i == i_max)){
        D_forest_4D[k][l][i_max][j] = 1 + D_forest_4D[k][l][i_max][j + 1];
    }else if ((i <= i_max-1) && (i >= i_0) && (j <= j_max-1) && (j >= j_0)){
        if ((x_orl[i] == x_orl[i_0]) && (y_orl[j] == y_orl[j_0])) {

            D_forest_4D[k][l][i][j] = min3(Delta[i][j] + D_forest_4D[k][l][i + 1][j + 1], 1 + D_forest_4D[k][l][i + 1][j], 1 + D_forest_4D[k][l][i][j + 1]);
            D_tree[i][j] = D_forest_4D[k][l][i][j];

        } else {

            while (D_tree[i][j] == -1){
                continue;
            }

            D_forest_4D[k][l][i][j] = min3(D_tree[i][j] + D_forest_4D[k][l][x_orl[i] + 1][y_orl[j] + 1], 1 + D_forest_4D[k][l][i + 1][j],
                                           1 + D_forest_4D[k][l][i][j + 1]);
        }
    }
//    else{
//        D_forest_4D[k][l][i][j] = -1;
//    }


}

void task_1(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<vector<vector<int>>>>& D_forest_4D, vector<vector<int>>& D_tree, int i, int j, int i_0, int i_max, int j_0, int j_max, int K, int L){
    int k;
    int l;
    for (k=0; k<(K/2); k++){
        for (l=0; l<L; l++){
            i_0 = x_kr[k];
            j_0 = y_kr[l];
            i_max = x_orl[i_0] + 1;
            j_max = y_orl[j_0] + 1;
            compute_8(x_orl, x_kr, y_orl, y_kr, Delta, D_forest_4D, D_tree, i, j, i_0, i_max, j_0, j_max, k, l);
        }
    }
}

void task_2(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<vector<vector<int>>>>& D_forest_4D, vector<vector<int>>& D_tree, int i, int j, int i_0, int i_max, int j_0, int j_max, int K, int L){
    int k;
    int l;
    for (k=K/2; k<K; k++){
        for (l=0; l<L; l++){
            i_0 = x_kr[k];
            j_0 = y_kr[l];
            i_max = x_orl[i_0] + 1;
            j_max = y_orl[j_0] + 1;
            compute_8(x_orl, x_kr, y_orl, y_kr, Delta, D_forest_4D, D_tree, i, j, i_0, i_max, j_0, j_max, k, l);
        }
    }
}

vector<vector<int>> parallel_standard_ted_2_8(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n){
    int K = (int)x_kr.size();
    int L = (int)y_kr.size();

    int k;
    int l;

    int i;
    int j;

    int i_0;
    int j_0;
    int i_max;
    int j_max;

//    int hit = 0;
//    int compute_times = 0;
    int table = 0;
    cout << "Total: " <<K*L << endl;

    vector<vector<int>> D_forest_2D (m+1, vector<int>(n+1,-1));
    vector<vector<vector<int>>> D_forest_3D(L,D_forest_2D);
    vector<vector<vector<vector<int>>>> D_forest_4D(K,D_forest_3D);
    cout << "4D Size is " << endl;
    cout << D_forest_4D.size() << endl;
    cout << D_forest_4D[0].size() << endl;
    cout << D_forest_4D[0][0].size() << endl;
    cout << D_forest_4D[0][0][0].size() << endl;
    cout << "m is " << m << ", n is " << n << endl;
    cout << "K*L = "<< K*L << endl;

    int inv = 0;
    while (n-inv >= 0){

        for (k=0; k<K; k++){
            for (l=0; l<L; l++){
                i_0 = x_kr[k];
                j_0 = y_kr[l];
                i_max = x_orl[i_0] + 1;
                j_max = y_orl[j_0] + 1;
                compute_8(x_orl, x_kr, y_orl, y_kr, Delta, D_forest_4D, D_tree, m, n-inv, i_0, i_max, j_0, j_max, k, l);
            }
        }


        int jec = 1;
        while (jec<=inv){

            for (k=0; k<K; k++){
                for (l=0; l<L; l++){
                    i_0 = x_kr[k];
                    j_0 = y_kr[l];
                    i_max = x_orl[i_0] + 1;
                    j_max = y_orl[j_0] + 1;
                    compute_8(x_orl, x_kr, y_orl, y_kr, Delta, D_forest_4D, D_tree, m-jec, n-inv+jec, i_0, i_max, j_0, j_max, k, l);
                }
            }

            jec++;
        }
        inv++;
    }


    int inv_2 = 1;
    while (m - inv_2 >= 0){
        for (k=0; k<K; k++){
            for (l=0; l<L; l++){
                i_0 = x_kr[k];
                j_0 = y_kr[l];
                i_max = x_orl[i_0] + 1;
                j_max = y_orl[j_0] + 1;
                compute_8(x_orl, x_kr, y_orl, y_kr, Delta, D_forest_4D, D_tree, m-inv_2, 0, i_0, i_max, j_0, j_max, k, l);
            }
        }

        int jec = 1;
        while (m - inv_2 - jec >= 0){
            for (k=0; k<K; k++){
                for (l=0; l<L; l++){
                    i_0 = x_kr[k];
                    j_0 = y_kr[l];
                    i_max = x_orl[i_0] + 1;
                    j_max = y_orl[j_0] + 1;
                    compute_8(x_orl, x_kr, y_orl, y_kr, Delta, D_forest_4D, D_tree, m-inv_2-jec, 0+jec, i_0, i_max, j_0, j_max, k, l);
                }
            }

            jec++;
        }
        inv_2++;
    }


    vector<vector<int>> final_result = D_tree;
    return final_result;
}



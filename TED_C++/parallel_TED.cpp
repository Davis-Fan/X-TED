#include "TED_C++.h"

//A thread for each row in a table

void initial_1(vector<vector<int>>& D, int& i_max, int& i_0, int& j_max){
    int i;
    for (i=i_max-1; i>i_0-1; i--){
        D[i][j_max] = 1 + D[i+1][j_max];
    }
}

void initial_2(vector<vector<int>>& D, int& j_max, int& j_0, int& i_max){
    int j;
    for (j=j_max-1; j>j_0-1; j--){
        D[i_max][j] = 1 + D[i_max][j+1];
    }
}

void compute(vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, vector<int>& x_orl, vector<int>& y_orl, int i, int j, int i_0, int j_0){


    if ((x_orl[i] == x_orl[i_0]) & (y_orl[j] == y_orl[j_0])){

        while ((D[i+1][j+1] == -1)
                || (D[i+1][j] == -1)
                || (D[i][j+1] == -1)){
            continue;
        }

        D[i][j] = min3(Delta[i][j] + D[i+1][j+1], 1 + D[i+1][j], 1 + D[i][j+1]);
        D_tree[i][j] = D[i][j];

    }else{
        while ((D[i+1][j] == -1)
                || (D[i][j+1] == -1)
                || (D[x_orl[i]+1][y_orl[j]+1] == -1)){
            continue;
        }
        D[i][j] = min3(D_tree[i][j] + D[x_orl[i]+1][y_orl[j]+1], 1 + D[i+1][j], 1 + D[i][j+1]);

    }

}

void task(vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, vector<int>& x_orl, vector<int>& y_orl, int i, int j, int i_0, int j_0, int j_max){

    for (j=j_max-1; j>j_0-1; j--){

//        while ((D[i+1][j+1] == -1)
//                || (D[i+1][j] == -1)
//                || (D[i][j+1] == -1)
//                || (D[x_orl[i]+1][y_orl[j]+1] == -1)){
//            continue;
//        }
        compute(Delta, D, D_tree, x_orl, y_orl, i, j, i_0, j_0);
    }
}


vector<vector<int>> parallel_standard_ted_2(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){
//    int m = (int)x_orl.size();
//    int n = (int)y_orl.size();

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

    for (k=0; k<K; k++){
        for (l=0; l<L; l++){

            i_0 = x_kr[k];
            j_0 = y_kr[l];
            i_max = x_orl[i_0] + 1;
            j_max = y_orl[j_0] + 1;
            D[i_max][j_max] = 0;


            thread t1(initial_1, ref(D), ref(i_max), ref(i_0), ref(j_max));
            thread t2(initial_2, ref(D), ref(j_max), ref(j_0), ref(i_max));

            t1.join();
            t2.join();


//            unsigned concurrent_count = thread::hardware_concurrency();
//            cout << "hardware_concurrency: " << concurrent_count << endl;
            vector<thread> threads;


            for (i=i_max-1; i>i_0-1; i--){

                threads.push_back(thread(task, ref(Delta), ref(D), ref(D_tree), ref(x_orl), ref(y_orl),i, j, i_0, j_0, j_max));

            }


            for (int f = 0; f < threads.size(); f++) {
//                cout<<threads[f].get_id()<<endl;
                threads[f].join();
            }

            // Clear D
//            for(auto &a: D){
//                fill(a.begin(),a.end(),0);
//            }
            for (i=i_max-1; i>i_0-1; i--){
                for (j=j_max-1; j>j_0-1; j--){
                    D[i][j] = -1;
                }
            }

        }

    }


    vector<vector<int>> final_result = D_tree;
    return final_result;
}

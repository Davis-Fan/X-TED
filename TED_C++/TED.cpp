#include "TED_C++.h"
//int hit = 0;
//int total = 0;
long long Total_time = 0;

int min3(int a, int b, int c){
    if (a<b){
        if (a<c){
            return a;
        }else{
            return c;
        }
    }else{
        if(b<c){
            return b;
        }else{
            return c;
        }
    }
}

vector<int> outermost_right_leaves(vector<vector<int>>& adj){
    int m = (int)adj.size();
    vector<int> orl(m,-1);
    int r;
    int i;

    for (i=0; i<m; i++){
        r = i;
        while(true){
            if((int)adj[r].size()==0){
                orl[i] = r;
                break;
            }
            if (orl[r] >= 0){
                orl[i] = orl[r];
                break;
            }
            r = adj[r][adj[r].size()-1];
        }
    }
    return orl;
}

vector<int> key_roots(vector<int>& orl){
    int m = (int)orl.size();
    vector<int> kr_view(m,-1);
    int K = 0;
    int i;
    int r;
    for (i=0; i<m; i++){
        r = orl[i];
        if(kr_view[r] < 0){
            kr_view[r] = i;
            K += 1;
        }
    }
    vector<int> key_roots_view(K,0);
    int k;
    int j;
    i = 0;
    for (k=0; k<K; k++){
        while(kr_view[i] < 0){
            i += 1;
        }
        j = k;
        while( (j>0) & (key_roots_view[j-1] < kr_view[i])){
            key_roots_view[j] = key_roots_view[j-1];
            j -= 1;
        }

        key_roots_view[j] = kr_view[i];
        i += 1;
    }
    return key_roots_view;
}


void compute_0(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D_forest, vector<vector<int>>& D_tree, int i, int j, int i_0, int i_max, int j_0, int j_max){
    if ((i == i_max) && (j == j_max)){
        D_forest[i][j] = 0;
    }else if ((i <= i_max-1) && (i >= i_0) && (j == j_max)){
        D_forest[i][j_max] = 1 + D_forest[i + 1][j_max];
    }else if ((j <= j_max-1) && (j >= j_0) && (i == i_max)){
        D_forest[i_max][j] = 1 + D_forest[i_max][j + 1];
    }else if ((i <= i_max-1) && (i >= i_0) && (j <= j_max-1) && (j >= j_0)){
        if ((x_orl[i] == x_orl[i_0]) && (y_orl[j] == y_orl[j_0])) {

            D_forest[i][j] = min3(Delta[i][j] + D_forest[i + 1][j + 1], 1 + D_forest[i + 1][j], 1 + D_forest[i][j + 1]);
            D_tree[i][j] = D_forest[i][j];

        } else {

            D_forest[i][j] = min3(D_tree[i][j] + D_forest[x_orl[i] + 1][y_orl[j] + 1], 1 + D_forest[i + 1][j],
                                           1 + D_forest[i][j + 1]);
        }
    }

}


vector<vector<int>> standard_ted_2(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){
//    int m = (int)x_orl.size();
//    int n = (int)y_orl.size();

    int K = (int)x_kr.size();
    int L = (int)y_kr.size();

//    ofstream output_file("/Users/davis/CLionProjects/TED_C++/x_orl.txt");
//    for (int dayi=0; dayi < x_orl.size(); dayi++){
//        output_file << x_orl[dayi] << endl;
//    }
//
//    ofstream output_file_2("/Users/davis/CLionProjects/TED_C++/x_kr.txt");
//    for (int dayi=0; dayi < x_kr.size(); dayi++){
//        output_file_2 << x_kr[dayi] << endl;
//    }
//    ofstream output_file_3("/Users/davis/CLionProjects/TED_C++/y_orl.txt");
//    for (int dayi=0; dayi < y_orl.size(); dayi++){
//        output_file_3 << y_orl[dayi] << endl;
//    }
//    ofstream output_file_4("/Users/davis/CLionProjects/TED_C++/y_kr.txt");
//    for (int dayi=0; dayi < y_kr.size(); dayi++){
//        output_file_4 << y_kr[dayi] << endl;
//    }



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
//    int table = 0;
//    cout << "Total: " <<K*L << endl;

    auto start_time = chrono::steady_clock::now();


    for (k = 0; k < K; k++) {
        for (l = 0; l < L; l++) {

            i_0 = x_kr[k];
            j_0 = y_kr[l];
            i_max = x_orl[i_0] + 1;
            j_max = y_orl[j_0] + 1;
            D[i_max][j_max] = 0;

            for (i = i_max - 1; i > i_0 - 1; i--) {
                D[i][j_max] = 1 + D[i + 1][j_max];
//                compute_0(x_orl,x_kr,y_orl,y_kr,Delta,D,D_tree,i,j_max,i_0,i_max,j_0,j_max);
            }

            for (j = j_max - 1; j > j_0 - 1; j--) {
                D[i_max][j] = 1 + D[i_max][j + 1];
//                compute_0(x_orl,x_kr,y_orl,y_kr,Delta,D,D_tree,i_max,j,i_0,i_max,j_0,j_max);
            }
//            cout << endl;
            for (i = i_max - 1; i > i_0 - 1; i--) {
                for (j = j_max - 1; j > j_0 - 1; j--) {

                    if ((x_orl[i] == x_orl[i_0]) & (y_orl[j] == y_orl[j_0])) {
                        D[i][j] = min3(Delta[i][j] + D[i + 1][j + 1], 1 + D[i + 1][j], 1 + D[i][j + 1]);
                        D_tree[i][j] = D[i][j];
                    } else {
                        D[i][j] = min3(D_tree[i][j] + D[x_orl[i] + 1][y_orl[j] + 1], 1 + D[i + 1][j],
                                       1 + D[i][j + 1]);

                    }
//                    compute_0(x_orl,x_kr,y_orl,y_kr,Delta,D,D_tree,i,j,i_0,i_max,j_0,j_max);


                }
            }

            if (k == K-1 && l == L-1){
                printf("Hello");
            }

//            }


            for (i = i_max; i > i_0 - 1; i--) {
                for (j = j_max; j > j_0 - 1; j--) {
                    D[i][j] = -1;
                }
            }

//            D[i_max][j_max] = -1;
//            if ((k == K-1)&&(l == L-1)){
//                cout << "Total computing is " << compute_times << endl;
//                cout << "The times of hit is "<< hit << endl;
//                cout << "Ratio is " << ((double)hit/compute_times) << endl;
//            }

        }

    }


//    ofstream outfile("/Users/davis/CLionProjects/TED-C++/D_tree.txt");
//    for (int da=0;da<D_tree.size();da++){
//        for (int yi=0; yi<D_tree[0].size(); yi++){
//            outfile << D_tree[da][yi] << " ";
//        }
//        outfile << endl;
//    }
//    outfile.close();

    auto end_time = chrono::steady_clock::now();
    auto ms = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
    Total_time = Total_time + ms;
    cout << "Basic Concurrent task finish, " << ms / 1000.0 << " ms consumed" << endl;
    cout << endl;

    vector<vector<int>> final_result = D_tree;
    return final_result;
}


vector<vector<int>> standard_ted_1 (vector<string>& x_node, vector<vector<int>>& x_adj, vector<string>& y_node, vector<vector<int>>& y_adj, int num_threads, int parallel_version){
    int m = (int)x_adj.size();
    int n = (int)y_adj.size();
    vector<vector<int>> Delta_view (m, vector<int>(n,0));

    int i;
    int j;

    for (i=0; i<m; i++){
        for (j=0; j<n; j++){
            if(x_node[i] != y_node[j]){
                Delta_view[i][j] =1;
            }
        }
    }
    x_node.clear();
    y_node.clear();

    vector<int> x_orl = outermost_right_leaves(x_adj);
    vector<int> x_kr = key_roots(x_orl);
    vector<int> y_orl = outermost_right_leaves(y_adj);
    vector<int> y_kr = key_roots(y_orl);

//    vector<vector<int>> D_forest (m+1, vector<int>(n+1,0));
    vector<vector<int>> D_forest (m+1, vector<int>(n+1,-1));
    vector<vector<int>> D_tree (m, vector<int>(n,-1));

//    vector<vector<int>> result = standard_ted_2(x_orl,x_kr,y_orl,y_kr,Delta_view,D_forest,D_tree);
    vector<vector<int>> result;

    vector<vector<int>> D_forest_2 (m+1, vector<int>(n+1,-1));
    vector<vector<int>> D_tree_2 (m, vector<int>(n,-1));
    switch (parallel_version) {
        case 0:{
            result=standard_ted_2(x_orl,x_kr,y_orl,y_kr,Delta_view,D_forest,D_tree);
            break;
        }

//    vector<vector<int>> result = parallel_standard_ted_2(x_orl,x_kr,y_orl,y_kr,Delta_view,D_forest,D_tree);
//    vector<vector<int>> result = parallel_standard_ted_2_2(x_orl,x_kr,y_orl,y_kr,Delta_view,D_forest, D_tree);
//    vector<vector<int>> result = parallel_standard_ted_2_3(x_orl,x_kr,y_orl,y_kr,Delta_view,D_forest, D_tree);
//    vector<vector<int>> result = parallel_standard_ted_2_4(x_orl,x_kr,y_orl,y_kr,Delta_view,D_forest, D_tree);
//    vector<vector<int>> result = parallel_standard_ted_2_5(x_orl,x_kr,y_orl,y_kr,Delta_view,D_forest, D_tree);
//    vector<vector<int>> result = parallel_standard_ted_2_6(x_orl,x_kr,y_orl,y_kr,Delta_view,D_forest, D_tree);
//    vector<vector<int>> result = parallel_standard_ted_2_7(x_orl,x_kr,y_orl,y_kr,Delta_view,D_forest, D_tree);
//    vector<vector<int>> result = parallel_standard_ted_2_8(x_orl,x_kr,y_orl,y_kr,Delta_view,D_forest, D_tree,m,n);
//    vector<vector<int>> result = parallel_standard_ted_2_9(x_orl,x_kr,y_orl,y_kr,Delta_view,D_forest, D_tree,m,n);
//    vector<vector<int>> result = parallel_standard_ted_2_10(x_orl,x_kr,y_orl,y_kr,Delta_view,D_forest, D_tree,m,n);

        case 11:{
            vector<vector<int>> result_11 = parallel_standard_ted_2_11(x_orl,x_kr,y_orl,y_kr,Delta_view,D_forest_2, D_tree_2,m,n, num_threads);
            result = result_11;
            break;
        }
        case 12:{
            vector<vector<int>> result_12 = parallel_standard_ted_2_12(x_orl,x_kr,y_orl,y_kr,Delta_view,D_forest_2, D_tree_2,m,n,num_threads);
            result = result_12;
            break;
        }
        case 13:{
            vector<vector<int>> result_13 = parallel_standard_ted_2_13(x_orl, x_kr, y_orl, y_kr, Delta_view, D_forest_2,D_tree_2, m, n, num_threads);
            result = result_13;
            break;
        }
        case 14:{
            vector<vector<int>> result_14 = parallel_standard_ted_2_14(x_orl, x_kr, y_orl, y_kr, Delta_view, D_forest_2,D_tree_2, m, n, num_threads);
            result = result_14;
            break;
        }
//        case 15:{
//            vector<vector<int>> result_15 = parallel_standard_ted_2_15(x_orl, x_kr, y_orl, y_kr, Delta_view, D_forest_2,D_tree_2, m, n, num_threads);
//            result = result_15;
//            break;
//        }
        case 16:{
            vector<vector<int>> result_16 = parallel_standard_ted_2_16(x_orl, x_kr, y_orl, y_kr, Delta_view, D_forest_2,D_tree_2, m, n, num_threads,x_adj,y_adj);
            result = result_16;
            break;
        }
        case 17:{
            vector<vector<int>> result_17 = parallel_standard_ted_2_17(x_orl, x_kr, y_orl, y_kr, Delta_view, D_forest_2,D_tree_2, m, n, num_threads,x_adj,y_adj);
            result = result_17;
            break;
        }
        case 18:{
            vector<vector<int>> result_18 = parallel_standard_ted_2_18(x_orl, x_kr, y_orl, y_kr, Delta_view, D_forest_2,D_tree_2, m, n, num_threads,x_adj,y_adj);
            result = result_18;
            break;
        }
    }
    return result;

}


int standard_ted (vector<string>& x_node, vector<vector<int>>& x_adj, vector<string>& y_node, vector<vector<int>>& y_adj, int num_threads, int parallel_version){
    int m = (int)x_adj.size();
    int n = (int)y_adj.size();
//    printf("%d, %d\n", m, n);
    if (m == 0){
        return n;
    }
    if (n == 0){
        return m;
    }
    vector<vector<int>> distance = standard_ted_1(x_node,x_adj,y_node,y_adj, num_threads, parallel_version);
    int k = distance[0][0];
//    printf("The distance is %d\n",k);
    return k;
}


void print_vector(vector<int> x){
    for (int i=0; i < x.size(); i++){
        printf("The Array[%d] is %d\n",i, x[i]);
    }
}

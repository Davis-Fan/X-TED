#include "TED_C++.h"


void compute_17(int n, int k, int l, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){
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

                D[i][j] = min3(D_tree[i][j] + D[x_orl[i] + 1][y_orl[j] + 1], 1 + D[i + 1][j],
                               1 + D[i][j + 1]);

            }
        }
    }

}

void task_17_1(int n, int L, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){
//    n--;
    int row = n / L;
    int column = n % L;
    compute_17(n, row,column,x_orl,x_kr,y_orl,y_kr,Delta, D, D_tree);
}

void task_17(vector<bool>& compute, vector<int>& depth, vector<int>& index, vector<int>& adjacent_list,  vector<int>& worklist_1, vector<int>& worklist_2, int i_begin, int interval, int final, int L, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree) {
    int i = i_begin;
    while (i < final) {
        int task = worklist_1[i];
        worklist_1[i] = -1;
        task_17_1(task,L,x_orl,x_kr,y_orl,y_kr,Delta, D, D_tree);
        i = i + interval;
    }
}

vector<vector<int>> parallel_standard_ted_2_17(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threads, vector<vector<int>>& x_adj, vector<vector<int>>& y_adj){

    int K = (int)x_kr.size();
    int L = (int)y_kr.size();

    int k;
    int l;
    int x_subtree_left;
    int x_subtree_right;
    int y_subtree_left;
    int y_subtree_right;

    int num_th = num_threads;
    vector<vector<vector<int>>> D_in_total;
    for (int i=0;i<num_th;i++){
        vector<vector<int>> D_new(m+1, vector<int>(n+1,-1));
        D_in_total.push_back(D_new);
    }

    vector<std::pair<int,int>> Table_index;
    for (int i=0; i<K; i++){
        for (int j=0; j<L; j++){
            Table_index.emplace_back(x_kr[i], y_kr[j]);
        }
    }


    vector<int> index(K*L+1, 0);
    vector<int> adjacent_list;
    vector<int> depth(K*L, 0);
    vector<int> depth2(K*L, 1);
    vector<bool> compute(K*L, 0);

    int number_x_nodes = x_adj.size();
    int number_y_nodes = y_adj.size();

    vector<int> x_keyroot_depth(number_x_nodes, -1);
    vector<int> y_keyroot_depth(number_y_nodes, -1);

    vector<int> x_keyroot_depth_2(K, -1);
    vector<int> y_keyroot_depth_2(L, -1);

    vector<bool> x_leaf_nodes(K*L, false);
    vector<bool> y_leaf_nodes(K*L, false);
//    for (int i=K-1;i>=0; i--){
//        int node = x_kr[i];
//        if(node == x_orl[node]){
//            x_leaf_nodes[node] = true;
////            x_keyroot_depth[i] = 0;
//        }
//    }
//
//
//    for (int i=L-1;i>=0; i--){
//        int node = y_kr[i];
//        if(node == y_orl[node]){
//            y_leaf_nodes[node] = true;
////            y_keyroot_depth[i] = 0;
//        }
//    }
//    for (int i=0; i<K*L;i++){
//        if (!x_leaf_nodes[Table_index[i].first] || !y_leaf_nodes[(Table_index[i].second)]) {
////            table_No.insert(i);
//        }else{
//            depth2[i] = 0;
//        }
//    }


    auto start_time_2 = chrono::steady_clock::now();
//    // Find leaf node
//    for (int i=K-1;i>=0; i--){
//        int node = x_kr[i];
//        if(node == x_orl[node]){
//            x_keyroot_depth[node] = 0;
//        }
//    }
//
//
//    for (int i=L-1;i>=0; i--){
//        int node = y_kr[i];
//        if(node == y_orl[node]){
//            y_keyroot_depth[node] = 0;
//        }
//    }
//
//    for (int i=K-1;i>=0;i--){
//        int x_node = x_kr[i];
//        for (auto k: x_adj[x_node]){
//            if (x_keyroot_depth[x_node] == -1 && x_keyroot_depth[k] == 0){
//                x_keyroot_depth[x_node] = 1;
//            }
//        }
//    }
//
//    for (int i=L-1;i>=0;i--){
//        int y_node = y_kr[i];
//        for (auto k: y_adj[y_node]){
//            if (y_keyroot_depth[y_node] == -1 && y_keyroot_depth[k] == 0){
//                y_keyroot_depth[y_node] = 1;
//            }
//        }
//    }
//
//    for (int i=K-1;i>=0;i--){
//        int x_node = x_kr[i];
//        for (auto k: x_adj[x_node]){
//            if (x_keyroot_depth[x_node] == -1 && x_keyroot_depth[k] == 1){
//                x_keyroot_depth[x_node] = 2;
//            }
//        }
//    }
//
//    for (int i=L-1;i>=0;i--){
//        int y_node = y_kr[i];
//        for (auto k: y_adj[y_node]){
//            if (y_keyroot_depth[y_node] == -1 && y_keyroot_depth[k] == 0){
//                y_keyroot_depth[y_node] = 1;
//            }
//        }
//    }

    for (int i=0;i<K;i++){
        int node = x_kr[i];
        if(node == x_orl[node]){
            x_keyroot_depth_2[i] = 0;
        }else{
            for (int j=0;j<i;j++){
                int node_2 = x_kr[j];
                if ((node <= node_2)&&(x_orl[node] >= node_2)){
                    x_keyroot_depth_2[i] = max(x_keyroot_depth_2[i],x_keyroot_depth_2[j]+1);
                }
            }
        }
    }

    for (int i=0;i<L;i++){
        int node = y_kr[i];
        if(node == y_orl[node]){
            y_keyroot_depth_2[i] = 0;
        }else{
            for (int j=0;j<i;j++){
                int node_2 = y_kr[j];
                if ((node <= node_2)&&(y_orl[node] >= node_2)){
                    y_keyroot_depth_2[i] = max(y_keyroot_depth_2[i],y_keyroot_depth_2[j]+1);
                }
            }
        }
    }






    cout << "x_keyroot_depth of 0 = " << x_keyroot_depth[0] << endl;
    cout << "y_keyroot_depth of 0 = " << y_keyroot_depth[0] << endl;

//    for (int i=K-1;i>=0;i--){
//        int x_node = x_kr[i];
//        if (x_keyroot_depth[x_node] != -1){
//            continue;
//        }else{
//            cout << "x - Keyroot " << x_node << " has no value" << endl;
//        }
//    }
//    for (int i=L-1;i>=0;i--){
//        int y_node = y_kr[i];
//        if (y_keyroot_depth[y_node] != -1){
//            continue;
//        }else{
//            cout << "y - Keyroot " << y_node << " has no value" << endl;
//        }
//    }

//    vector<bool> is_leaf_Table(K*L, false);
//    for (int i=0; i<K*L;i++){
//        if (x_leaf_nodes[Table_index[i].first] & y_leaf_nodes[Table_index[i].second]){
//            is_leaf_Table[i] = true;
//        }
//    }

//    for (int i=1; i<K*L; i++){
//        if(is_leaf_Table[i]){
//            continue;
//        } else{
//            for (int j=i-1; j>=0; j--){
//                if (is_leaf_Table[j]){
//                    continue;
//                }else {
//                    if (Table_index[j].first >= Table_index[i].first &
//                        Table_index[j].first <= x_orl[Table_index[i].first] &
//                        Table_index[j].second >= Table_index[i].second &
//                        Table_index[j].second <= y_orl[Table_index[i].second]) {
//                        depth2[i] = max(depth2[j] + 1, depth2[i]);
//                    }
////                    size_2++;
//                }
//            }
//        }
//    }


    for (int i=0;i<K*L;i++){
//        depth2[i] = x_keyroot_depth[Table_index[i].first] + y_keyroot_depth[Table_index[i].second];
//        depth[i] = x_keyroot_depth[x_kr[i/L]] + y_keyroot_depth[y_kr[i%L]];
        depth[i] = x_keyroot_depth_2[i/L] + y_keyroot_depth_2[i%L];
    }


//    for (int k=0;k<depth.size();k++){
//        if (depth[k] != depth2[k]){
//            cout << "Not Equal" << endl;
//        }
//    }

    int dayi = depth[561000];
    auto end_time_2 = chrono::steady_clock::now();
    auto ms_2 = chrono::duration_cast<chrono::microseconds>(end_time_2 - start_time_2).count();
    cout << "Preprocessing Method 4, the time is " << ms_2/1000.0 << " ms consumed" << endl;




    vector<int> worklist1(1000000,-1);
    vector<int> worklist2(1000000,-1);
    int worklist1_tail=0;
    int worklist2_tail=0;

    int current_depth = 0;
    for (int i=0; i<K*L; i++){
        if (depth[i] == current_depth){
            worklist1[worklist1_tail++] = i;
        }
    }


    cout << "Threads size = " << num_th << endl;
    auto start_time = chrono::steady_clock::now();
    while (worklist1[0] != -1 && worklist1[0] != K*L-1) {

        if (worklist1_tail >= num_th) {

            vector<thread> threads;

            for (int inter = 0; inter < num_th; inter++) {
                threads.push_back(thread(task_17, ref(compute), ref(depth), ref(index), ref(adjacent_list), ref(worklist1), ref(worklist2), inter, num_th, (int) worklist1_tail, L, ref(x_orl),ref(x_kr), ref(y_orl), ref(y_kr), ref(Delta), ref(D_in_total[inter]), ref(D_tree)));
            }

            for (auto &th: threads) {
                th.join();
            }

        } else {

            for (int i = 0; i < worklist1_tail; i++) {
                int task = worklist1[i];
                worklist1[i] = -1;
                task_17_1(task,L,x_orl,x_kr,y_orl,y_kr,Delta, D, D_tree);
            }
        }
        current_depth++;
        for (int i=0; i<K*L; i++){
            if (depth[i] == current_depth){
                worklist2[worklist2_tail++] = i;
            }
        }
        swap(worklist1, worklist2);
        worklist1_tail = worklist2_tail;
        worklist2_tail = 0;
    }
    task_17_1(K*L-1,L,x_orl,x_kr,y_orl,y_kr,Delta, D, D_tree);

    auto end_time = chrono::steady_clock::now();
    auto ms = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
    cout << "Parallel-17 Concurrent task finish, " << ms/1000.0 << " ms consumed" << endl;

    vector<vector<int>> final_result = D_tree;
    return final_result;

}








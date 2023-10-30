#include "TED_C++.h"

// Reference
vector<int> change(vector<int>& a){
    vector<int> result;
    result.push_back(10);
    a[1] = 1000;
    result.push_back(a[1]);
    return result;
}

// Pointer
vector<int> change2(const vector<int>* a){
    vector<int> result;
    result.push_back(10);
    result.push_back((*a)[1]);
    return result;
}

void traverse(int node, vector<vector<int>>& a_adj) {
    // print the current node
    std::cout << node << " ";

    // traverse the child nodes recursively
    for (auto child : a_adj[node]) {
        traverse(child,a_adj);
    }
}


int main(int argc, char* argv[]) {
//    bolzano_test();
//    sentiment_test();

    int num_nodes = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    int parallel_version = atoi(argv[3]);

    char* file_path_1 = argv[4];
    char* file_path_2 = argv[5];


//    test_1(num_nodes,num_threads,parallel_version);
//    test_2(num_threads,parallel_version);
//    test_3(num_nodes, num_threads,parallel_version);

//    bolzano_test(num_threads, parallel_version);
//    sentiment_test(num_threads, parallel_version);
    swissport_test(num_threads, parallel_version, file_path_1, file_path_2);
//    python_test(num_threads, parallel_version);
//    dblp_test(num_threads, parallel_version);


//    auto end_time = chrono::steady_clock::now();
//    auto ms = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
//    cout << "Final Concurrent task finish, " << ms << " ms consumed" << endl;
//    cout << endl;

//    vector<vector<int>> a_adj = {{1,4},{2,3},{},{},{}};
//    vector<vector<int>> b_adj = {{1,2},{},{3,4},{},{}};
//    traverse(0,a_adj);
//
//    vector<int> keyroot = {2,1,0};
//    set<int> a_leaf;
//    a_leaf.insert(2);
//    set<int> a_second;
//    for (int i=0;i<keyroot.size();i++){
//        for (auto k: a_adj[keyroot[i]]){
//            if (a_leaf.find(k) != a_leaf.end()){
//                a_second.insert(keyroot[i]);
//            }
//        }
//    }

//    vector<int> test1 = {1,2,3};
//    vector<int> test2 = {4,5,6};
//    vector<int> test3 = {7,8,9};
//    vector<int> test4 = {10,11,12};
//    vector<vector<int>> test = {test1, test2, test3, test4};
//
//    int i_max = 2;
//    int i_0 = 0;
//    int j_max = 1;
//    int j_0 = 0;
//
//    int x = j_max - j_0 + 1;
//    int y = i_max - i_0 + 1;
//    int t;
//
//    cout << endl;
//    for (int k=1; k<=x+y-1; k++){
//        if(k<=x){
//            t = min(k-1,i_max-i_0);
//            for(int w=0; w<=t; w++){
//                printf("%u ",test[i_max-w][j_max+1-k+w]);
//            }
//        }else{
//            t = min(j_max, i_max-i_0-k+x);
//            for(int w=0;w<=t; w++){
//                printf("%u ",test[i_max-k+x-w][0+w]);
//            }
//        }
//    }

//    int m1 = 8;
//    int n1 = 8;
////    vector<vector<int>> k = {{1,2,3,4},{5,6,7,8},{9,10,11,12}, {13,14,15,16}};
//    vector<vector<int>> test(8, vector<int>(8,0));
//
//    for (int step = 0; step < m1+n1-1; step++){
//        for (int j = max(0,step-n1+1); j<=min(step, m1-1); j++){
//            test[m1-1-j][n1-1-step+j]=step;
////            printf("i = %u, j = %u, %u \n",m1-1-j, n1-1-step+j, test[m1-1-j][n1-1-step+j]);
//            printf("i = %u j = %u ,  ", j, step-j);
//        }
//        printf("\n");
//    }
//
//    for (int i=0;i<8;i++){
//        for (int j=0; j<8; j++){
//            printf("%u ", test[i][j]);
//        }
//        printf("\n");
//    }
    return 0;
}

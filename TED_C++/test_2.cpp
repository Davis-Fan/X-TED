#include "TED_C++.h"
void test_2(int num_threads, int parallel_version){
//    vector<string> a_node = {"I", "am", "a","PhD","student"};
//    vector<string> a_node(31,"a");
    //    vector<string> a_node = {"a","b","c","d","e"};
//    vector<vector<int>> a_adj = {{1,2,3,17,18,19,20,21,22,23,24,25,26,27,28,29,30},{},{},{4,5,6,7,8,9,10,11,12,13,14,15,16}, {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}};
//    vector<string> b_node(31,"b");
//    vector<string> a_node = {"a", "b", "c","d","e"};
//    vector<vector<int>> a_adj = {{1,4},{2,3},{},{},{}};
//    vector<string> b_node = {"You", "should", "not", "be", "so", "mean"};
//    vector<string> b_node = {"a", "g", "b", "c", "f"};
//    vector<string> b_node = {"a", "g", "b","c","f"};
//    vector<vector<int>> b_adj = {{1,2},{},{3,4},{},{}};
//    vector<string> b_node = {"a", "a", "a","b","c"};
//    vector<vector<int>> b_adj = {{1,3},{2},{},{4},{}};
    //    vector<string> b_node = {"f", "g"};
//    vector<vector<int>> b_adj = {{1,2,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23},{},{3,4,5,6,7,8},{},{},{}, {},{},{}, {},{},{}, {},{},{}, {},{},{}, {},{},{}, {},{},{24,25,26,27,28,29,30},{},{}, {},{},{},{},{}};

    vector<string> a_node = {"a", "b", "c","d"};
    vector<vector<int>> a_adj = {{1},{2,3},{},{}};
    vector<string> b_node = {"a", "b", "f", "e"};
    vector<vector<int>> b_adj = {{1,3},{2},{},{}};

//    vector<string> a_node = {"a", "b", "c","d", "e"};
//    vector<vector<int>> a_adj = {{1},{2,3, 4},{},{}, {}};
//    vector<string> b_node = {"a", "b", "e", "c"};
//    vector<vector<int>> b_adj = {{1,3},{2},{},{}};

    int f = standard_ted(a_node, a_adj, b_node, b_adj,num_threads,parallel_version);
    cout << endl;
    printf("The final distance is %d\n",f);

}
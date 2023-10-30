#include "TED_C++.h"

void test_1(int num_nodes, int num_threads, int parallel_version){

//    int k = 5;
    int k = num_nodes;
    int i = 0;
    int j = 0;

    vector<vector<int>> A;
    vector<vector<int>> B;

    while (i<k){
        if ((1+i) <= (k/2)) {
            A.push_back({1 + i, k - i - 1});
//            printf("%d \n",i);
        } else{
            A.push_back({});
        }
        i++;
    }

//    printf("%d\n", A.size());

    while(j<k){
        if((j%2 == 0) && (j+2 < k)){
            B.push_back({j+1, j+2});
//            printf("%d \n",j);
        }else{
            B.push_back({});
        }
        j++;
    }

//    printf("%d\n", B.size());

    vector<string> A_node;
    vector<string> B_node;
    for (int f=0; f<k; f++){
        A_node.push_back("a");
        B_node.push_back("b");
    }
    int result = standard_ted(A_node,A,B_node,B, num_threads, parallel_version);
//    printf("The final distance is %d\n", result);
}



#include "TED_C++.h"

float total_milliseconds = 0;

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


    vector<vector<int>> D_forest (m+1, vector<int>(n+1,-1));
    vector<vector<int>> D_tree (m, vector<int>(n,-1));

    vector<vector<int>> result;

    vector<vector<int>> result_matrix = parallel_standard_ted_test(x_orl, x_kr, y_orl, y_kr, Delta_view, D_forest,D_tree, m, n, num_threads,x_adj,y_adj);

    result = result_matrix;

    return result;

}


int standard_ted (vector<string>& x_node, vector<vector<int>>& x_adj, vector<string>& y_node, vector<vector<int>>& y_adj, int num_threads, int parallel_version){
    int m = (int)x_adj.size();
    int n = (int)y_adj.size();
    if (m == 0){
        return n;
    }
    if (n == 0){
        return m;
    }
    vector<vector<int>> distance = standard_ted_1(x_node,x_adj,y_node,y_adj, num_threads, parallel_version);
    int k = distance[0][0];
    return k;
}



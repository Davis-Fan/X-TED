#include "TED_C++.h"


int main(int argc, char* argv[]) {

    int parallel_version = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    char* file_path_1 = argv[3];
    char* file_path_2 = argv[4];

    for (int i=0;i<20;i++) {
        test(num_threads, parallel_version, file_path_1, file_path_2);
    }
    return 0;
}

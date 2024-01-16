#include "TED_C++.h"


int main(int argc, char* argv[]) {

    char* file_path_1 = argv[1];
    char* file_path_2 = argv[2];

    for(int i=0; i<5;i++) {
        test(file_path_1, file_path_2);
        printf("\n");
    }
    return 0;
}

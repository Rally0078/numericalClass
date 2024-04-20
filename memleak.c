#include <stdlib.h>
#include <stdio.h>

int main(int argc, char* argv[]){
    int *x;
    FILE* file = fopen(argv[1], "r");
    if(file == NULL){
        printf("Error! File not found!");
        return -1;
    }
    char buffer[1024];
    fgets(buffer, 1024, file);
    printf("%s", buffer);

    return 0;
}
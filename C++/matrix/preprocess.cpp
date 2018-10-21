//
//  preprocess.cpp
//  matrix
//
//  Created by WangGuoan on 10/20/18.
//  Copyright © 2018 Guoan Wang. All rights reserved.
//

#include "preprocess.h"

/*
	read data from datafile.
 */
int readit(const char *filename, int *address_of_n, int *address_of_T, matrix *pmatrix)
{
    int readcode = 0, fscancode;
    FILE *datafile = NULL;
    char buffer[100];
    int n, T, i, j;
    
    datafile = fopen(filename, "r");
    if (!datafile) {
        printf("cannot open file %s\n", filename);
        readcode = 2;  return readcode;
    }
    
    printf("reading data file %s\n", filename);
    
    fscanf(datafile, "%s", buffer);
    fscancode = fscanf(datafile, "%s", buffer);
    if (fscancode == EOF) {
        printf("problem: premature file end at ...\n");
        readcode = 4; return readcode;
    }
    
    n = *address_of_n = atoi(buffer);
    
    fscanf(datafile, "%s", buffer);
    fscanf(datafile, "%s", buffer);
    T = *address_of_T = atoi(buffer);
    
    
    printf("n = %d\n", n);
    printf("T = %d\n", T);
    
    matrix price_matrix(n,T);
    
    fscanf(datafile, "%s", buffer); //"prices"
    
    
    for (i = 0; i < n; i++) {
        for (j = 0; j < T; j++) {
            fscanf(datafile, "%s", buffer);
            price_matrix[i*T + j] = atof(buffer);
        }
    }
    
    *pmatrix = price_matrix;
    
    fclose(datafile);
    
    
    return readcode;
}

void price_to_return(const matrix& price, matrix* preturn){
    int n = price.get_rows();
    int T = price.get_columns();
    matrix ret(n, T-1);
    for (int i = 0; i < n; ++i){
        for(int j = 0; j < T-1; ++j){
            ret(i,j) = (price(i,j+1) - price(i,j)) / price(i,j);
        }
    }
    *preturn = ret;
}

void train_test_split(const matrix& ret_matrix, matrix* px_train, matrix* py_train,
                      matrix* px_test, matrix* py_test, int offset){
    int n = ret_matrix.get_rows();
    int T = ret_matrix.get_columns();
    
    int T_train = ceil(1.0*T/2);
    matrix x_train(n, T_train-offset);
    matrix y_train(n, T_train-offset);
    matrix x_test(n, T-T_train-offset);
    matrix y_test(n, T-T_train-offset);
    
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < T_train-offset; ++j){
            x_train(i,j) = ret_matrix(i,j);
            y_train(i,j) = ret_matrix(i, j+offset);
        }
    }
    
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < T-T_train-offset; ++j){
            x_test(i,j) = ret_matrix(i,j+T_train);
            y_test(i,j) = ret_matrix(i, j+T_train+offset);
        }
    }
    *px_train = x_train;
    *py_train = y_train;
    *px_test = x_test;
    *py_test = y_test;

}


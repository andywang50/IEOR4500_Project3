//
//  main.cpp
//  matrix
//
//  Created by WangGuoan on 10/20/18.
//  Copyright © 2018 Guoan Wang. All rights reserved.
//
#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <cstdlib>
#include <time.h>
#include "matrix.h"
#include "preprocess.h"
#include "dnn.h"


int main(int argc, const char * argv[]) {
    
    // insert code here...
    srand (time(NULL));
    int retcode = 0;
    if (argc != 2) {
        printf("missing filename\n");
        retcode = 1;
        return retcode;
    }
    int n,T;
    matrix price_matrix;
    retcode = readit(argv[1], &n, &T, &price_matrix);
    //print_matrix(price_matrix);
    
    matrix ret_matrix;
    price_to_return(price_matrix, &ret_matrix);
    //print_matrix(ret_matrix);
    
    matrix x_train, y_train, x_test, y_test;
    train_test_split(ret_matrix, &x_train, &y_train, &x_test, &y_test);
    //print_matrix(x_train);
    //print_matrix(y_train);
    //print_matrix(x_test);
    //print_matrix(y_test);
    
    
    //printf("%f",loss(x_train, y_train));
    
    int num_layers = 3;
    int dim_hidden = 50;
    dnn clf = dnn(num_layers,dim_hidden);
    clf.fit(x_train, y_train);
    matrix y_train_hat = clf.predict(x_train);
    printf("In sample loss: %f\n", loss(y_train_hat, y_train));
    
    matrix y_test_hat = clf.predict(x_test);
    printf("Generalization loss: %f\n", loss(y_test_hat, y_test));

    
    return 0;
}


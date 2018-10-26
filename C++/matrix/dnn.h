//
//  dnn.hpp
//  matrix
//
//  Created by WangGuoan on 10/20/18.
//  Copyright © 2018 Guoan Wang. All rights reserved.
//

#ifndef dnn_hpp
#define dnn_hpp

#include <stdio.h>
#include <cfloat>
#include <vector>
#include <unordered_map>
#include "matrix.h"


matrix ReLu(const matrix&);
matrix grad_ReLu(const matrix&);
double loss(const matrix&, const matrix&);
matrix grad_loss(const matrix&, const matrix&);

class dnn{
private:
    std::vector<matrix> W_lst;
    std::vector<vec> b_lst;
    int num_layers;
    int dim_hidden;
    double lambdaval; // learning rate
    double tol;
    double beta1; //Adam
    double beta2; //Adam
    double eps; //Adam
public:
    dnn(int, int, double learning_rate=0.5 ,
        double tol=1e-4, double beta1=0.4, double beta2=0.8, double eps=1e-8);
    void fit(const matrix&, const matrix&, int num_iter=10000);
    matrix predict(const matrix&);
    
};
#endif /* dnn_hpp */

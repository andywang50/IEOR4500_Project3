//
//  preprocess.hpp
//  matrix
//
//  Created by WangGuoan on 10/20/18.
//  Copyright © 2018 Guoan Wang. All rights reserved.
//

#ifndef preprocess_hpp
#define preprocess_hpp
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <math.h>
#include "matrix.h"

int readit(const char *, int *, int *, matrix *);
void price_to_return(const matrix& price, matrix* preturn);
void train_test_split(const matrix&, matrix*, matrix*, matrix*, matrix*, int offset=10);

#endif /* preprocess_hpp */

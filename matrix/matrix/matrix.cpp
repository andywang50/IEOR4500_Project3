//
//  matrix.cpp
//  matrix
//
//  Created by WangGuoan on 10/20/18.
//  Copyright © 2018 Guoan Wang. All rights reserved.
//

#include "matrix.h"

/*
 Default Constructor
 */
matrix::matrix():rows(0), columns(0), entry(nullptr){}

/*
 Constructor with specified number of rows and number of columns;
 will initialize every entry to 0.0.
 */
matrix::matrix(int _row, int _col): rows(_row), columns(_col), entry(nullptr){
    try{
        entry = new double[rows*columns]();
    }
    catch(std::exception& e){
        std::cerr<<"Constructor failed to finish while pre-allocating space"<<std::endl;
        entry = nullptr;
    }
}

/*
 Copy constructor
 */
matrix::matrix(const matrix& copy):rows(copy.rows), columns(copy.columns), entry(nullptr){
    try { // to allocate new matrix of the same size
        entry = new double[rows*columns];
    }
    catch(std::exception& e) { // If error occurs
        std::cerr<<"matrix(copy) failed to allocate memory"<<std::endl;
        entry = nullptr;
    }
    if(entry) // equivalent to if(entry != nullptr)
        for(size_t i=0;i<rows*columns;++i)
            entry[i] = copy.entry[i];
}

/*
 Assignment operator
 */
matrix& matrix::operator=(matrix copy){
    copy.swap(*this);
    return *this;
}

/*
 Destructor
 */
matrix::~matrix(){
    if(entry) delete [] entry;
}

/*
 Swap
 */
void matrix::swap(matrix& other){
    std::swap(rows, other.rows);
    std::swap(columns, other.columns);
    std::swap(entry, other.entry);
}

/*
 [] operator
 */
double& matrix::operator[] (int index){
    return entry[index];
}

/*
 [] operator
 */
double matrix::operator[] (int index) const{
    return entry[index];
}

/*
 print the matrix
 */
void print_matrix(const matrix& mat) {
    int rows =mat.get_rows();
    int cols = mat.get_columns();
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%f, ", mat[i* + j]);
        }
        printf("\n");
    }
}

/*
 swap vec
 */
void vec::swap(vec& other){
    matrix::swap(other);
    std::swap(length, other.length);
}

/*
 default constructor of vec
 */
vec::vec():matrix(){}

/*
 constructor with specific length
 */
vec::vec(int _len):matrix(_len,1), length(_len){
    
}

/*
 copy constructor
 */
vec::vec(const vec& other):matrix(other){
    length = other.length;
}

/*
 assignemnt operator
 */
vec& vec::operator= (vec copy){
    copy.swap(*this);
    return *this;
}

/*
 convert a matrix to vec
 */

vec::vec(const matrix& other):matrix(other){
    if (other.get_rows() > 1) length = other.get_rows();
    else length = other.get_columns();
}

matrix add(const matrix& a, const matrix& b){
    
    if (!(a.shape_same(b))){
        throw "dimension mismatch in add: " + a.print_shape() + " " + b.print_shape() + "\n";
    }
    
    matrix result(a);
    for (int i = 0; i < a.get_rows() * a.get_columns(); ++i){
        result[i] += b[i];
    }
    
    return result;
}

matrix substract(const matrix &a, const matrix &b){
    return add(a, scalar_prod(-1.0, b));
}

matrix dot_prod(const matrix& a, const matrix& b){
    if (a.get_columns() != b.get_rows()){
        throw "dimension mismatch in dot_prod: " + a.print_shape() + " " + b.print_shape() + "\n";
    }
    
    int rows = a.get_rows();
    int columns = b.get_columns();
    
    int l = a.get_columns();
    
    matrix result(rows, columns);
    for (int i = 0; i < rows; ++ i){
        for (int j = 0; j < columns; ++ j){
            result[i*columns+j] = 0.0;
            for (int k = 0; k < l; ++k){
                result[i*columns+j] += a[i*l+k] * b[k*l+j];
            }
        }
    }
    return result;
}

matrix elementwise_prod(const matrix& a, const matrix& b){
    if (!(a.shape_same(b))){
        throw "dimension mismatch in elementwise_prod: " + a.print_shape() + " " + b.print_shape() + "\n";
    }
    
    matrix result(a);
    for (int i = 0; i < a.get_rows() * a.get_columns(); ++i){
        result[i] *= b[i];
    }
    
    return result;
}

matrix scalar_prod(const matrix& a, double b){
    return scalar_prod(b, a);
}

matrix scalar_prod(double b, const matrix& a){
    matrix result(a);
    for (int i = 0; i < a.get_rows() * a.get_columns(); ++i){
        result[i] *= b;
    }
    return result;
}

matrix scalar_division(const matrix&a, double b){
    return scalar_prod(a, 1.0/b);
}

matrix transpose(const matrix& a){
    int rows = a.get_columns();
    int columns = a.get_rows();
    
    matrix result(rows, columns);
    for (int i = 0; i < rows; ++i){
        for (int j = 0; j < columns; ++j){
            result[i*columns+j] = a[j*rows+i];
        }
    }
    return result;
}


matrix cross_prod(const vec& a, const vec& b){
    int rows = a.get_length();
    int columns = b.get_length();
    
    matrix result(rows, columns);
    for (int i = 0; i < rows; ++i){
        for (int j = 0; j < columns; ++j){
            result[i*columns+j] = a[i] * b[j];
        }
    }
    
    return result;
}

double dot_prod(const vec& a,const vec& b){
    if (a.get_length() != b.get_length()){
        throw "dimension mismatch in vec dot_prod: " + a.print_shape() + " " + b.print_shape() + "\n";
    }
    
    double result = 0.0;
    
    for (int i = 0; i < a.get_length(); ++i){
        result += a[i] * b[i];
    }
    
    return result;
    
}

vec dot_prod(const matrix& m, const vec& v){
    if (m.get_columns() != v.get_length()){
        throw "dimension mismatch in dot_prod: " + m.print_shape() + " " + v.print_shape() + "\n";
    }
    
    int length = m.get_rows();
    
    int l = v.get_length();
    
    vec result(length);
    for (int i = 0; i < length; ++i){
        result[i] = 0.0;
        for (int k = 0; k < l; ++k){
            result[i] += m[i*l+k] * v[k];
        }
        
    }
    return result;
}

matrix pow(const matrix& m, double n){
    matrix result(m);
    int num_entries = m.get_rows() * m.get_columns();
    for(int i = 0; i < num_entries; ++i){
        result[i] = pow(result[i], n);
    }
    return result;
}

double sum(const matrix& m){
    double result = 0.0;
    int num_entries = m.get_rows() * m.get_columns();
    for(int i = 0; i < num_entries; ++i){
        result += m[i];
    }
    return result;
}

double mean(const matrix& m){
    int num_entries = m.get_rows() * m.get_columns();
    return sum(m)/num_entries;
}

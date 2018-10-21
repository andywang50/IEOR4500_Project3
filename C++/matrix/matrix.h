//
//  matrix.hpp
//  matrix
//
//  Created by WangGuoan on 10/20/18.
//  Copyright © 2018 Guoan Wang. All rights reserved.
//

#ifndef matrix_hpp
#define matrix_hpp

#include <stdio.h>
#include <exception>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <string>

class matrix;
class vec;

void print_matrix(const matrix&);

matrix add(const matrix&, const matrix&);
matrix add(const matrix&, const vec&);
matrix add(const matrix&, double);
matrix substract(const matrix&, const matrix&);
matrix substract(const matrix&, const vec&);
matrix dot_prod(const matrix&, const matrix&);
matrix elementwise_prod(const matrix&, const matrix&);
matrix elementwise_division(const matrix&, const matrix&);
matrix scalar_prod(const matrix&, double);
matrix scalar_prod(double ,const matrix&);
matrix scalar_division(const matrix&a, double b);
matrix transpose(const matrix&);
matrix square(const matrix&);
matrix sqrt(const matrix&);
double sum(const matrix&);
double mean(const matrix&);
vec col_mean(const matrix&);

matrix cross_prod(const vec&, const vec&);
double dot_prod(const vec&,const vec&);
vec dot_prod(const matrix&, const vec&);

class matrix{
private:
    int columns;
    int rows;
    double* entry;
    
protected:
    void swap(matrix& other);

public:
    matrix();
    matrix(int _row, int _col, bool random=false);
    matrix(const matrix&);
    matrix& operator=(matrix copy);
    ~matrix();
    
    double& operator[] (int index);
    double operator[] (int index) const;
    
    double& operator() (int i, int j);
    double operator() (int i, int j) const;
    
    //friend void print_matrix(const matrix& mat);
    
    int get_rows() const{return rows;}
    int get_columns() const{return columns;}
    //double* get_entry() const{return entry;}
    
    bool shape_same(const matrix& other) const{
        return (rows == other.rows) && (columns == other.columns);
    }
    
    std::string print_shape() const{
        return "(" + std::to_string(rows) + "," + std::to_string(columns) + ")";
    }
};


class vec: public matrix{
private:
    int length;
protected:
    void swap(vec& other);
public:
    vec();
    vec(int _len, bool random=false);
    vec(const vec&);
    vec(const matrix&);
    vec& operator= (vec copy);
    
    int get_length() const{return length;}
    
    
};
#endif /* matrix_hpp */

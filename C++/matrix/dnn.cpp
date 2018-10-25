//
//  dnn.cpp
//  matrix
//
//  Created by WangGuoan on 10/20/18.
//  Copyright © 2018 Guoan Wang. All rights reserved.
//

#include "dnn.h"

matrix ReLu(const matrix& m){
    matrix result(m.get_rows(), m.get_columns());
    for(int i = 0; i < m.get_rows()* m.get_columns(); ++i){
        if (m[i] > 0){
            result[i] = m[i];
        }
        else{
            result[i] = 0.0;
        }
    }
    return result;
}

matrix grad_ReLu(const matrix& m){
    matrix result(m.get_rows(), m.get_columns());
    for(int i = 0; i < m.get_rows()* m.get_columns(); ++i){
        if (m[i] > 0){
            result[i] = 1.0;
        }
        else{
            result[i] = 0.0;
        }
    }
    return result;
}

double loss(const matrix& y_hat, const matrix& y_true){
    matrix y_diff = substract(y_true, y_hat);
    return mean(square(y_diff));
}

matrix grad_loss(const matrix& y_hat, const matrix& y_true){
    matrix y_diff = substract(y_true, y_hat);
    return scalar_prod(y_diff, -2.0/y_diff.get_columns() / y_diff.get_rows());
}

dnn::dnn(int num_layers, int dim_hidden, double learning_rate,
         double tol, double beta1, double beta2, double eps):num_layers(num_layers),
        dim_hidden(dim_hidden),lambdaval(learning_rate),
        tol(tol),beta1(beta1),beta2(beta2),eps(eps){}

void dnn::fit(const matrix& x, const matrix& y, int num_iter){
    int n = x.get_rows();
    int num_sample = x.get_columns();
    std::vector<matrix> a_lst(num_layers+1);
    std::vector<matrix> z_lst(num_layers+1);
    W_lst = std::vector<matrix>(num_layers);
    b_lst = std::vector<vec>(num_layers);
    
    matrix z_init(x);
    z_lst[0] = z_init;
    a_lst[0] = z_init;
    
    for (int i = 0; i < num_layers; ++i){
        if (i==0){
            matrix W(dim_hidden, n, true);
            W_lst[i] = W;
            vec b(dim_hidden, true);
            b_lst[i] = b;
            matrix a(dim_hidden, num_sample);
            matrix z(dim_hidden, num_sample);
            a_lst[i+1] = a;
			z_lst[i + 1] = z;
        }
        else if (i == num_layers - 1){
            matrix W(n, dim_hidden, true);
			W_lst[i] = W;
			vec b(n, true);
			b_lst[i] = b;
			matrix a(n, num_sample);
            matrix z(n, num_sample);
			a_lst[i + 1] = a;
			z_lst[i + 1] = z;
        }
        else{
            matrix W(dim_hidden, dim_hidden, true);
			W_lst[i] = W;
			vec b(dim_hidden, true);
			b_lst[i] = b;
			matrix a(dim_hidden, num_sample);
            matrix z(dim_hidden, num_sample);
			a_lst[i + 1] = a;
			z_lst[i + 1] = z;
        }
    }
    double previous_loss = DBL_MAX;
	matrix delta;
	std::unordered_map<int, matrix> w_mom_dict;
	std::unordered_map<int, matrix> w_reg_dict;
	std::unordered_map<int, vec> b_mom_dict;
	std::unordered_map<int, vec> b_reg_dict;

    for (int _it = 0; _it <= num_iter; ++_it){
        //evaluate each node (forward)
        for (int alpha = 1; alpha <= num_layers; ++alpha){
            matrix W = W_lst[alpha-1];
            vec b = b_lst[alpha-1];
            matrix z = add(dot_prod(W, a_lst[alpha-1]),b);
            z_lst[alpha] = z;
            a_lst[alpha] = ReLu(z);
        }
        delta = grad_loss(a_lst[num_layers], y);
        delta = elementwise_prod(delta, grad_ReLu(z_lst[num_layers]));
        // backward induction, use Adam

        for (int alpha = num_layers - 1; alpha >= 0; --alpha){
            matrix W = W_lst[alpha];
            vec b = b_lst[alpha];
            matrix a = a_lst[alpha];
            matrix dW = dot_prod(delta, transpose(a));
            vec db = col_mean(delta);
            
            if (w_mom_dict.find(alpha) == w_mom_dict.end()){
                w_mom_dict[alpha] = dW;
                b_mom_dict[alpha] = db;
                w_reg_dict[alpha] = square(dW);
                b_reg_dict[alpha] = square(db);
            }
            else{
                w_mom_dict[alpha] = add(scalar_prod(dW,1-beta1),
                                        scalar_prod(w_mom_dict[alpha], beta1));
                b_mom_dict[alpha] = add(scalar_prod(db, 1-beta1),
                                        scalar_prod(b_mom_dict[alpha], beta1));
                w_reg_dict[alpha] = add(scalar_prod(square(dW), 1-beta2),
                                        scalar_prod(w_reg_dict[alpha], beta2));
                b_reg_dict[alpha] = add(scalar_prod(square(db), 1-beta2),
                                        scalar_prod(b_reg_dict[alpha], beta2));
				//printf("!");

            }
            
            matrix w_mom_tilde = scalar_division(w_mom_dict[alpha], 1-pow(beta1, _it+1));
            vec b_mom_tilde = scalar_division(b_mom_dict[alpha], 1-pow(beta1, _it+1));
            matrix w_reg_tilde = scalar_division(w_reg_dict[alpha], 1-pow(beta2, _it+1));
            vec b_reg_tilde = scalar_division(b_reg_dict[alpha], 1-pow(beta2, _it+1));
            
            dW = scalar_prod(lambdaval, elementwise_division(w_mom_tilde,
                                                             add(sqrt(w_reg_tilde),eps)));
            db = scalar_prod(lambdaval, elementwise_division(b_mom_tilde,
                                                             add(sqrt(b_reg_tilde),eps)));
            
            W = substract(W, dW);

            b = substract(b, db);
            W_lst[alpha] = W;
            b_lst[alpha] = b;
            //std::cout<<transpose(W).print_shape()<<"\n";
            //std::cout<<W.print_shape()<<"\n";

            delta = elementwise_prod(dot_prod(transpose(W), delta), grad_ReLu(z_lst[alpha]));
        }
        double current_loss = loss(a_lst[num_layers],y);
        if (fabs(previous_loss - current_loss) < tol){
            //printf("%f, %f, %f", current_loss, previous_loss, fabs(previous_loss - current_loss));
            break;
        }
        previous_loss = current_loss;
        if (_it % 1 == 0){
            printf("iteration: %d, %f\n", _it, current_loss);
        }
    }
    
}
matrix dnn::predict(const matrix& x){
    matrix v(x);
    for (int alpha = 1; alpha <= num_layers; ++ alpha){
        matrix W = W_lst[alpha-1];
        vec b = b_lst[alpha-1];
        v = dot_prod(W, v);
        v = add(v, b);
        v = ReLu(v);
    }
    return v;
}


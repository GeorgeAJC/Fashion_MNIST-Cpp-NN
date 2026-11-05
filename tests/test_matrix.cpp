#include <cmath>
#include <iostream>
#include "matrix.hpp"

static int test_fails = 0; // counter for test fails

static bool near(double a, double b){
    return std::fabs(a-b) < 1e-12; //compare floating point nums
}

//basic transpose function for tests
static Matrix transpose(const Matrix& M){
    Matrix T(M.cols, M.rows, 0.0);
    for(int i = 0; i < M.rows; ++i){ 
        for(int j = 0; j < M.cols; ++j){ 
            T(j, i) = M(i, j);
        }
    }
    return T;
}
//check if two matricies are identical 
static bool same(const Matrix& A, const Matrix& B){
    if(A.rows != B.rows || A.cols != B.cols){
        return false;
    }
    for(std::size_t i = 0; i < A.data.size(); ++i){
        if(!near(A.data[i], B.data[i])){
            return false;
        }
    }
    return true;
}

int main(){
    Matrix A(2, 3, 0.0), B(3, 2, 0.0); //create test matricies
    const double a[] = {1,2,3,4,5,6}; //define values to populate matrices
    const double b[] = {7,8,9,10,11,12};
    for(int i = 0; i < 2; ++i){ // fill matrices with values from arrays
        for(int j = 0; j < 3; ++j){
            A(i, j) = a[i * 3 + j];
        }
    }
    for(int i = 0; i < 3; ++i){
        for(int j = 0; j < 2; ++j){
            B(i, j) = b[i * 2 + j];
        }
    }

    //check calculated values are correct for matmulAB
    Matrix C = matmulAB(A,B);
    if(!near(C(0,0),58)){ //checks top left value is calculated to be 58
        std::cerr << "Test Failed at line" << 51 << ": near(C(0,0),58)" << std::endl; //print result of tests
        ++test_fails;
    }
    if(!near(C(0,1),64)){ //checks top right value is calculated to be 64
        std::cerr << "Test Failed at line" << 55 << ": near(C(0,1),64)" << std::endl;
        ++test_fails;
    }
    if(!near(C(1,0),139)){ //checks bottom left value is calculated to be 139
        std::cerr << "Test Failed at line" << 59 << ": near(C(1,0),139)" << std::endl;
        ++test_fails;
    }
    if(!near(C(1,1),154)){ //checks bottom right value is calculated to be 154
        std::cerr << "Test Failed at line" << 63 << ": near(C(1,1),154)" << std::endl;
        ++test_fails;
    }
    //check transposed multiplication functions work for A and B transposition
    if(!same(matmulAtB(B, B), matmulAB(transpose(B), B))){ //checks transposed A multiplication matrix gives same result as test transpose method
        std::cerr << "Test Failed at line" << 68 << ": same(matmulAtB(B, B), matmulAB(transpose(B), B))" << std::endl;
        ++test_fails;
    }
    if(!same(matmulABt(A, A), matmulAB(A, transpose(A)))){ //checks transposed B multiplication matrix gives same result as test transpose method
        std::cerr << "Test Failed at line" << 72 << ": same(matmulABt(A, A), matmulAB(A, transpose(A)))" << std::endl;
        ++test_fails;
    }

    if (test_fails == 0){
        std::cout << "Tests passed"; //return success message if no tests failed 
        return 0;
    }
    std::cout << "Number of Tests Failed:" << test_fails; //return number of failed tests
    return 1;
}
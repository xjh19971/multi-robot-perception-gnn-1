//
//  Conduction.cpp
//  GNN
//
//  Created by Yue Zhou on 12/10/20.
//  Copyright © 2020 Yue Zhou. All rights reserved.
//

#include "Conduction.hpp"


//transform matrix from Matrix<3,3> R_BC to double R[3][3]
void transmatrix1(Matrix<3, 3> R_BC) {
    // input: Matrix<3, 3> R_BC, output: double R[3][3]
    double R[3][3];
    
    R[0][0] = R_BC(0, 0);
    R[0][1] = R_BC(0, 1);
    R[0][2] = R_BC(0, 2);
    
    R[1][0] = R_BC(1, 0);
    R[1][1] = R_BC(1, 1);
    R[1][2] = R_BC(1, 2);
    
    R[2][0] = R_BC(2, 0);
    R[2][1] = R_BC(2, 1);
    R[2][2] = R_BC(2, 2);
    
    //return R[3][3];
}

//transform from double R[3][3] to Matrix<3,3> R_BC
void transmatrix2( double R[3][3] ){
    //typedef Matrix<double, 3, 3> Matrix3f;
    //input: double R[3][3],output: Matrix<3, 3> R_BC
    
    Matrix<3, 3> R_BC;
    
    R_BC(0, 0) = R[0][0] ;
    R_BC(0, 1) = R[0][1] ;
    R_BC(0, 2) = R[0][2] ;
    
    //row 2
    R_BC(1, 0) = R[1][0] ;
    R_BC(1, 1) = R[1][1] ;
    R_BC(1, 2) = R[1][2] ;
    
    //row3
    R_BC(2, 0) = R[2][0] ;
    R_BC(2, 1) = R[2][1] ;
    R_BC(2, 2) = R[2][2] ;
    
    //return R_BC;
}

//对接
Vector<3> B_r_BC ((Scalar)relpose_T_[i][0],(Scalar)relpose_T_[i][1],(Scalar)relpose_T_[i][2]);

vector<double> camera_1.relpose_T;
camera_1.relpose_T.push_back((Scalar)relpose_T_[0][0]);
camera_1.relpose_T.push_back((Scalar)relpose_T_[0][1]);
camera_1.relpose_T.push_back((Scalar)relpose_T_[0][2]);

vector<double> camera_2.relpose_T; camera_2.relpose_T.push_back((Scalar)relpose_T_[1][0]);
camera_2.relpose_T.push_back((Scalar)relpose_T_[1][1]);
camera_2.relpose_T.push_back((Scalar)relpose_T_[1][2]);

vector<double> camera_2.relpose_T; camera_2.relpose_T.push_back((Scalar)relpose_T_[2][0]);
camera_2.relpose_T.push_back((Scalar)relpose_T_[2][1]);
camera_2.relpose_T.push_back((Scalar)relpose_T_[2][2]);

Matrix<3, 3> R_BC_1 = Quaternion((Scalar)relpose_R_[0][0], (Scalar)relpose_R_[0][1], (Scalar)relpose_R_[0][2], (Scalar)relpose_R_[0][3]).toRotationMatrix();

Matrix<3, 3> R_BC_2 = Quaternion((Scalar)relpose_R_[1][0], (Scalar)relpose_R_[1][1], (Scalar)relpose_R_[1][2], (Scalar)relpose_R_[1][3]).toRotationMatrix()

Matrix<3, 3> R_BC_3 = Quaternion((Scalar)relpose_R_[2][0], (Scalar)relpose_R_[2][1], (Scalar)relpose_R_[2][2], (Scalar)relpose_R_[2][3]).toRotationMatrix()

double camera_1.relpose_R = transmatrix1(Matrix<3, 3> R_BC_1);
double camera_2.relpose_R = transmatrix1(Matrix<3, 3> R_BC_2);
double camera_3.relpose_R = transmatrix1(Matrix<3, 3> R_BC_3);
//以上对接完毕

//
//  quaternion.cpp
//  
//
//  Created by Yue Zhou on 11/25/20.
//

#include "quaternion.hpp"
#include <iostream>
#include <string>
#include <iostream>
#include <Eigen/Dense>


using namespace std;
using namespace Eigen;

int scene_id  = 0;
float main_loop_freq= 50.0;
bool unity_render = true;

int fov = 90;
double width = 720;
double height = 480;
int num_camera = 3;
const char *camera_names={"camera_1","camera_2","camera_3"};


struct camera_1{
    vector<double> relpose_T;
    relpose_T.push_back(0.0);
    relpose_T.push_back(0.0);
    relpose_T.push_back(0.3);
    
    vector<double> relpose_R;
    relpose_R.push_back(1.0);
    relpose_R.push_back(0.0);
    relpose_R.push_back(0.0);
    relpose_R.push_back(0.0);
};

struct camera_2{
    vector<double> relpose_T;
    relpose_T.push_back(-3.0);
    relpose_T.push_back(0.0);
    relpose_T.push_back(0.1);
    
    vector<double> relpose_R;
    relpose_R.push_back(0.924);
    relpose_R.push_back(0.0);
    relpose_R.push_back(0.0);
    relpose_R.push_back(0.383);
};

struct camera_3{
    vector<double> relpose_T;
    relpose_T.push_back(3.0);
    relpose_T.push_back(0.0);
    relpose_T.push_back(0.1);
    
    vector<double> relpose_R;
    relpose_R.push_back(0.924);
    relpose_R.push_back(0.0);
    relpose_R.push_back(0.0);
    relpose_R.push_back(-0.383);
};

//normalization equation1
vector<double> normalize_array(vector<double> arr){
    vector<double> tmp(arr), ret(arr.size());
    
    sort(tmp.begin(), tmp.end());
    
    for (int i =0; i < arr.size(); i++){
        vector<double>::iterator iter = find(tmp.begin(), tmp.end(), arr[i]);
        ret[i] = std::distance(tmp.begin(),iter);
    }
    return ret;
}

//normalization equation 2
double normalization (vector<double> arr){
    double length = sqrt(arr[0]*arr[0]+arr[1]*arr[1]+arr[2]*arr[2]);
    
    for (int i = 0, i < 3, i++){
        arr[i] = arr[i] /length;
    }
    return arr;
}

//define cross product
void crossProduct (int v_A[], int v_B[], int c_P[]){
    c_P[0] = v_A[1] * v_B[2] - v_A[2] * v_B[1];
    c_P[1] = -(v_A[0] * v_B[2] - v_A[2] * v_B[0]);
    c_P[2] = v_A[0] * v_B[1] - v_A[1] * v_B[0];
}

void dotProduct (vector<double> vec1, vector<double> vec2){
    double product = 0.0;
    
    for (int i = 0; i < vec1.size(); i++)
        
        product = product + vec1[i] * vec2[i];
    return product;
}
//https://www.programiz.com/cpp-programming/examples/matrix-multiplication

double matrix_dotProduct (m[3][3],n[3][3]){
    double mult[3][3];
    for (i = 0; i < 3 ; i++){
        for(j = 0; j < 3; j++){
            for(k =0; k < 3; k++){
                mult[i][j] += m[i][k] * n[k][j];
            }
        }
    }
    return mult;
}

//get o1
void o1_position(c1, o1, depth1){
    vector<double> o1;
    vector<double> c1o1;
    
    c1o1.push_back(o1[0]-c1[0]);
    c1o1.push_back(o1[1]-c1[1]);
    c1o1.push_back(o1[2]-c1[2]);
    
    normalization(c1o1);
    
    o1.push_back(c1[0] + c1o1[0]*depth1);
    o1.push_back(c1[1] + c1o1[1]*depth1);
    o1.push_back(c1[2] + c1o1[2]*depth1);
    
    //return o1;
}

double find_o2(vector<double> relpose_R, vector<double> relpose_T, o1){
    
    double R[3][3] = {
        {1- 2*relpose_R[1]*relpose_R[1] - 2*relpose_R[2]*relpose_R[2], 2*relpose_R[0]*relpose_R[2] - 2*relpose_R[2]*relpose_R[3], 2*relpose_R[0]*relpose_R[2] + 2*relpose_R[1]*relpose_R[3]},
        {2*relpose_R[0]*relpose_R[1] + 2*relpose_R[2]*relpose_R[3], 1-2*relpose_R[0]*relpose_R[0]-2*relpose_R[2]*relpose_R[2], 2*relpose_R[1]*relpose_R[2] - 2*relpose_R[0]*relpose_R[3]},
        {2*relpose_R[0]*relpose_R[2] - 2*relpose_R[1]*relpose_R[3], 2*relpose_R[1]*relpose_R[2] + 2*relpose_R[0]*relpose_R[3],
            1- 2*relpose_R[0]*relpose_R[0] -2*relpose_R[1]*relpose_R[1]}
    };
    
    vector<double> p;
    p.push_back(o1[0] - relpose_T[0]);
    p.push_back(o1[1] - relpose_T[1]);
    p.push_back(o1[2] - relpose_T[2]);
    
    vector<double> second_vector;
    second_vector.push_back(R[0][0]*p[0] + R[0][1]*p[1] + R[0][2]*p[2]);
    second_vector.push_back(R[1][0]*p[0] + R[1][1]*p[1] + R[1][2]*p[2]);
    second_vector.push_back(R[2][0]*p[0] + R[2][1]*p[1] + R[2][2]*p[2]);
    
    //o2 normalization
    for (int i =0, i< 3, i++){
        second_vector[i] = second_vector[i] /
    }
    
    vector<double> o2;
    o2.push_back(relpose_T[0] + second_vector[0] * depth2);
    o2.push_back(relpose_T[1] + second_vector[1] * depth2);
    o2.push_back(relpose_T[1] + second_vector[1] * depth2);
    
    return o2;
}

void rotation_matrix_from_vectors(o1, o2, relpose_T){
    vector<double> vec1;
    vec1.push_back(o1[0] - relpose_T[0]);
    vec1.push_back(o1[1] - relpose_T[1]);
    vec1.push_back(o1[2] - relpose_T[2]);
    
    vector<double> vec2;
    vec2.push_back(o2[0] - relpose_T[0]);
    vec2.push_back(o2[1] - relpose_T[1]);
    vec2.push_back(o2[2] - relpose_T[2]);
    
    double a;
    double b;
    a = normalization(vec1);
    b = normalization(vec2);
    
    v = crossProduct(a, b);
    c = dotProduct(a, b);
    s = normalization(v);
    
    double vx[3][3] = {
        {0, -v[2], v[1]},
        {v[2], 0, -v[0]},
        {-v[1], v[0], 0}
    };
    
    //first step to compute rotation matrix
    double part1[3][3] = {
        {1, -v[2], v[1]},
        {v[2], 1, -v[0]},
        {-v[1], v[0], 1}
    };
    
    //use part2 to compute part3
    double part2[3][3] = matrix_dotProduct(vx,vx);
    
    double part3 = (1 - c) / (s * s)
    
    double R[3][3];
    
    for(int i = 0; i < 3; i++){
        for(int j =0; j < 3, j++){
            R[i][j] = part1[i][j] + part3 * part2[i][j];
        }
    }
    double rotation_matrix[2][2] = R;
}

//transform matrix from Matrix<3,3> R_BC to double R[3][3]
void transmatrix1(Matrix<3, 3> R_BC){
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
void transmatrix2( double R[3][3], Matrix<3, 3> R_BC){
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
    
void find_depth(cv::Mat image){
    int rows = image.rows;
    int cols = image.cols;
    
    double center[5][5];
    //height
    int center_row = ceil(rows/2);
    int center_col = ceil(cols/2);
}
//o1 is the object position
//c1 is the first camera

// call o1_position to get o1
//use o1 and the quaternion of c1 to find o2
//use o1 and o2 find rotation_matrix for each camera


//对接
Vector<3> B_r_BC ((Scalar)relpose_T_[i][0],(Scalar)relpose_T_[i][1],(Scalar)relpose_T_[i][2]);

vector<double> camera_1.relpose_T; camera_1.relpose_T.push_back((Scalar)relpose_T_[0][0]);
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


void main (o1, camera_1, depth1){
    
    o1_position(camera_1.relpose_T, o1, depth1);
    //o1 position is returned
    

    
    
    find_o2(camera_1.relpose_R, camera_1.relpose_T, o1);
    //o2,object2, location is returned
    
    rotation_matrix_from_vectors(o1, o2, camera_1.relpose_T);
    double rotation_matrix_c1[2][2] =rotation_matrix;
    //save the result of rotation matrix of c1
    
    rotation_matrix_from_vectors(o1, o2, camera_2.relpose_T);
    double rotation_matrix_c2[2][2] =rotation_matrix;
    //save the result of rotation matrix of c2
    
    rotation_matrix_from_vectors(o1, o2, camera_3.relpose_T);
    double rotation_matrix_c3[2][2] =rotation_matrix;
}

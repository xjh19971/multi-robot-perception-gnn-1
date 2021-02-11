//
//  camera_pose.cpp
//  GNN
//
//  Created by Yue Zhou on 2/1/21.
//  Copyright © 2021 Yue Zhou. All rights reserved.
//

//https://stackoverflow.com/questions/8610994/direction-vector-from-quaternion

//https://answers.unity.com/questions/525952/how-i-can-converting-a-quaternion-to-a-direction-v.html

#include "camera_pose.hpp"
#include <iostream>
#include <string>
#include <iostream>
#include <vector>
#include <Magick++.h>
#include <opencv2/opencv.hpp>
//#include <Eigen/Dense>

/*-----------------------------
rgb_camera_ = std::make_shared<RGBCamera>();
Vector<3> B_r_BC(0.0, 0.0, 0.3);
Matrix<3, 3> R_BC = Quaternion(1.0, 0.0, 0.0, 0.0).toRotationMatrix();
std::cout << R_BC << std::endl;
rgb_camera_->setFOV(90);
rgb_camera_->setWidth(720);
rgb_camera_->setHeight(480);
rgb_camera_->setRelPose(B_r_BC, R_BC);
quad_ptr_->addRGBCamera(rgb_camera_);
-------------------------------*/

using namespace std;
using namespace Magick;
//using namespace Eigen;

int scene_id  = 0;
float main_loop_freq= 50.0;
bool unity_render = true;

int fov = 90;
double width = 720;
double height = 480;
int num_camera = 3;
//const char *camera_names={"camera_1","camera_2","camera_3"};


//global variable o1, only used in initialization
vector<double> o1;
o1 = initilization(cv:mat image, vector<double> c1);

//global varaibles
vector<double> o1Direction;

//camera
struct camera{
    vector<double> relpose_T_;
    vector<double> relpose_R_;
};

//function: normalizing a vector(equation1)
vector<double> normalize_array(vector<double> arr){
    vector<double> tmp(arr), ret(arr.size());
    
    sort(tmp.begin(), tmp.end());
    
    for (int i =0; i < arr.size(); i++){
        vector<double>::iterator iter = find(tmp.begin(), tmp.end(), arr[i]);
        ret[i] = std::distance(tmp.begin(),iter);
    }
    return ret;
}

//function: normalizing a vector(equation 2)
vector<double> normalization (vector<double> arr){
    double length = sqrt(arr[0]*arr[0]+arr[1]*arr[1]+arr[2]*arr[2]);
    
    for (int i = 0; i < 3; i++){
        arr[i] = arr[i] /length;
    }
    return arr;
}

double find_norm (vector<double> arr){
    double length = sqrt(arr[0]*arr[0]+arr[1]*arr[1]+arr[2]*arr[2]);
    return length;
}

//define cross product
vector<double> crossProduct (vector<double> v_A, vector<double> v_B){
    double c_P[3];
    vector<double> vec3;
    c_P[0] = v_A[1] * v_B[2] - v_A[2] * v_B[1];
    c_P[1] = -(v_A[0] * v_B[2] - v_A[2] * v_B[0]);
    c_P[2] = v_A[0] * v_B[1] - v_A[1] * v_B[0];
    
    for (int i =0; i<3; i++){
        vec3.push_back(c_P[i]);
    }
    return vec3;
}


//function: dot product of two vectors
double dotProduct (vector<double> vec1, vector<double> vec2){
    double product = 0.0;
    
    for (int i = 0; i < vec1.size(); i++){
        
        product += vec1[i] * vec2[i];
    }
    return product;
}

//https://www.programiz.com/cpp-programming/examples/matrix-multiplication
//https://www.programiz.com/cpp-programming/multidimensional-arrays

//function: dotProduct of matrices
double matrix_dotProduct (double m[3][3], double n[3][3]){
    double mult[3][3]= {{0,0,0},{0,0,0},{0,0,0} };
    for (int i = 0; i < 3 ; i++){
        for(int j = 0; j < 3; j++){
             mult[i][j] = 0;
            for(int k =0; k < 3; k++){
                mult[i][j] += m[i][k] * n[k][j];
            }
        }
    }
    return mult[2][2];
}

//function: from quaternion to vector
//https://forum.unity.com/threads/convert-quaternion-to-vector3.106789/
//https://docs.unity3d.com/ScriptReference/Quaternion-eulerAngles.html

double rotation_matrix_from_vectors(vector<double> o1, vector<double> o2, vector<double> relpose_T){
    vector<double> vec1;
    vec1.push_back(o1[0] - relpose_T[0]);
    vec1.push_back(o1[1] - relpose_T[1]);
    vec1.push_back(o1[2] - relpose_T[2]);
    
    vector<double> vec2;
    vec2.push_back(o2[0] - relpose_T[0]);
    vec2.push_back(o2[1] - relpose_T[1]);
    vec2.push_back(o2[2] - relpose_T[2]);
    
    vector<double> a;
    vector<double> b;
    a = normalization(vec1);
    b = normalization(vec2);
    
    vector<double> v;
    double cos;
    double sin;
    v = crossProduct(a, b);
    cos = dotProduct(a, b);
    sin = find_norm(v);
    
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
    //double part2[3][3];
    double part2[3][3] ={{0,0,0},{0,0,0},{0,0,0}};
    part2[2][2] = matrix_dotProduct(vx,vx);
    
    double part3 = (1 - cos) / (sin * sin);
    
/*    double part4[3][3];
    
    for(int i = 0; i < 3; i++){
            for(int j =0; j < 3; j++){
                part4[i][j] = part3 * part2[i][j];
            }
        }
    */
    
    double R[3][3];

    for (int i = 0; i < 3; i++) {
        for(int j =0; j < 3; j++){
            R[i][j] = part1[i][j] + part3 * part2[i][j];
          }
      }
    //double rotation_matrix[2][2] = R;
    return R[2][2];
}


//Local Functions

//functions: convert initialized quaternion of c1 to vector
using UnityEngine;

public static Quaternion Euler(float yaw, float pitch, float roll) {
        yaw*=Mathf.Deg2Rad;
        pitch*=Mathf.Deg2Rad;
        roll*=Mathf.Deg2Rad;

        double yawOver2 = yaw * 0.5f;
        float cosYawOver2 = (float)System.Math.Cos(yawOver2);
        float sinYawOver2 = (float)System.Math.Sin(yawOver2);
        double pitchOver2 = pitch * 0.5f;
        float cosPitchOver2 = (float)System.Math.Cos(pitchOver2);
        float sinPitchOver2 = (float)System.Math.Sin(pitchOver2);
        double rollOver2 = roll * 0.5f;
        float cosRollOver2 = (float)System.Math.Cos(rollOver2);
        float sinRollOver2 = (float)System.Math.Sin(rollOver2);
        Quaternion result;
        result.w = cosYawOver2 * cosPitchOver2 * cosRollOver2 + sinYawOver2 * sinPitchOver2 * sinRollOver2;
        result.x = sinYawOver2 * cosPitchOver2 * cosRollOver2 + cosYawOver2 * sinPitchOver2 * sinRollOver2;
        result.y = cosYawOver2 * sinPitchOver2 * cosRollOver2 - sinYawOver2 * cosPitchOver2 * sinRollOver2;
        result.z = cosYawOver2 * cosPitchOver2 * sinRollOver2 - sinYawOver2 * sinPitchOver2 * cosRollOver2;

        return result;
}

public static Quaternion ToQ (Vector3 v)
{
    return ToQ (v.y, v.x, v.z);
}

public static Quaternion ToQ (float yaw, float pitch, float roll)
{
    yaw *= Mathf.Deg2Rad;
    pitch *= Mathf.Deg2Rad;
    roll *= Mathf.Deg2Rad;
    float rollOver2 = roll * 0.5f;
    float sinRollOver2 = (float)Math.Sin ((double)rollOver2);
    float cosRollOver2 = (float)Math.Cos ((double)rollOver2);
    float pitchOver2 = pitch * 0.5f;
    float sinPitchOver2 = (float)Math.Sin ((double)pitchOver2);
    float cosPitchOver2 = (float)Math.Cos ((double)pitchOver2);
    float yawOver2 = yaw * 0.5f;
    float sinYawOver2 = (float)Math.Sin ((double)yawOver2);
    float cosYawOver2 = (float)Math.Cos ((double)yawOver2);
    Quaternion result;
    result.w = cosYawOver2 * cosPitchOver2 * cosRollOver2 + sinYawOver2 * sinPitchOver2 * sinRollOver2;
    result.x = cosYawOver2 * sinPitchOver2 * cosRollOver2 + sinYawOver2 * cosPitchOver2 * sinRollOver2;
    result.y = sinYawOver2 * cosPitchOver2 * cosRollOver2 - cosYawOver2 * sinPitchOver2 * sinRollOver2;
    result.z = cosYawOver2 * cosPitchOver2 * sinRollOver2 - sinYawOver2 * sinPitchOver2 * cosRollOver2;

    return result;
}

public static Vector3_FromQ2 (Quaternion q1)
{
    float sqw = q1.w * q1.w;
    float sqx = q1.x * q1.x;
    float sqy = q1.y * q1.y;
    float sqz = q1.z * q1.z;
    float unit = sqx + sqy + sqz + sqw; // if normalised is one, otherwise is correction factor
    float test = q1.x * q1.w - q1.y * q1.z;
    Vector3 v;

    if (test>0.4995f*unit) { // singularity at north pole
        v.y = 2f * Mathf.Atan2 (q1.y, q1.x);
        v.x = Mathf.PI / 2;
        v.z = 0;
        return NormalizeAngles (v * Mathf.Rad2Deg);
    }
    if (test<-0.4995f*unit) { // singularity at south pole
        v.y = -2f * Mathf.Atan2 (q1.y, q1.x);
        v.x = -Mathf.PI / 2;
        v.z = 0;
        return NormalizeAngles (v * Mathf.Rad2Deg);
    }
    Quaternion q = new Quaternion (q1.w, q1.z, q1.x, q1.y);
    v.y = (float)Math.Atan2 (2f * q.x * q.w + 2f * q.y * q.z, 1 - 2f * (q.z * q.z + q.w * q.w));     // Yaw
    v.x = (float)Math.Asin (2f * (q.x * q.z - q.w * q.y));                             // Pitch
    v.z = (float)Math.Atan2 (2f * q.x * q.y + 2f * q.z * q.w, 1 - 2f * (q.y * q.y + q.z * q.z));      // Roll
    return NormalizeAngles (v * Mathf.Rad2Deg);
}

static Vector3 NormalizeAngles (Vector3 angles)
{
    angles.x = NormalizeAngle (angles.x);
    angles.y = NormalizeAngle (angles.y);
    angles.z = NormalizeAngle (angles.z);
    return angles;
}

static float NormalizeAngle (float angle)
{
    while (angle>360)
        angle -= 360;
    while (angle<0)
        angle += 360;
    return angle;
}

//function:get o1(expected to be given: c1 position and o1Direction converted from the initialized quaternion)
vector<double> o1_position(vector<double> c1, vector<double> o1Direction, double depth1){
    vector<double> o1;
    //vector<double> c1o1;
    
    //c1o1.push_back(o1[0]-c1[0]);
    //c1o1.push_back(o1[1]-c1[1]);
    //c1o1.push_back(o1[2]-c1[2]);
    
    normalization(o1Direction);
    
    o1.push_back(c1[0] + o1Direction[0]*depth1);
    o1.push_back(c1[1] + o1Direction[1]*depth1);
    o1.push_back(c1[2] + o1Direction[2]*depth1);
    
    return o1;
}

//function: find o2
vector<double> find_o2(vector<double> relpose_R, vector<double> relpose_T, vector<double> o1, double depth){
    //the input depth here should be the updated depth, which is used to calculate o2
    
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
    
    normalization(second_vector);
    
    //o2 normalization
    //for (int i =0, int i< 3, i++){
    //    second_vector[i] = second_vector[i] /
    //}
    
    vector<double> o2;
    o2.push_back(relpose_T[0] + second_vector[0] * depth);
    o2.push_back(relpose_T[1] + second_vector[1] * depth);
    o2.push_back(relpose_T[2] + second_vector[2] * depth);
    
    return o2;
}

//https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/897677#897677






//https://stackoverflow.com/questions/38352891/converting-depth-image-into-open-cv-mat-format
//https://stackoverflow.com/questions/13840013/opencv-how-to-visualize-a-depth-image

double get_depth(cv::Mat image){
    int rows = image.rows;
    int cols = image.cols;
    double bgrPixel;
    double sum = 0.0;
    int center_step = 5;
    int center = rows/2;
    int center_size = center_step * center_step;
    double depth =0.0;
    
    vector<double> conter;
    //height
    int center_row = ceil(rows/2);
    int center_col = ceil(cols/2);
    
    cv::Mat depthImage;
    depthImage = cv::imread(cv::Mat image, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR ); // Read the file
    depthImage.convertTo(depthImage, CV_32F); // convert the image data to float type
    
    for(int i = center -2; i < center +2; i++){
        for(int j = center -2; j < center +2; j++){
            sum += depthImage.at<double>(i,j);
            }
        }
    double depth = sum / center_size;
    return depth;
}
//o1 is the object position
//c1 is the first camera

// call o1_position to get o1
//use o1 and the quaternion of c1 to find o2
//use o1 and o2 find rotation_matrix for each camera






void take_screen_shot(cv::mat image,vector<double> o1, camera_1.relpose_T, camera_1.relpose_R, camera_2.relpose_T, camera_3.relpose_T ){
    get_depth(cv::mat image);
    
    update_o1(); //现在的o2是o1，为下一张截图和o2做准备
    
    
    
    
    
}

/* void main (cv:mat image){
    
    //at the very beginning, you take a screenshot with c1, so the object is located in the center of the image from c1.
    //so the info you have is c1_pose, depth.
    // you can calculate o1 by c1_pose and depth.
    //then you take 2nd image, you get o2 and c1_roation by then. In addition, you get c2_roation, and c3_rotaion.
    
    //In conclusion, the beginning set up is 2 screen shots with c1, and the object should be in the middle of c1.
    
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
    
    
} */

// seal the functions below
vector<double> initilization(cv::Mat image, vector<double> c1){
    vector<double> o1;
    vector<double> o1Direction = {0.0, -1.0 , 0.0};
    double depth;
    
    depth = getDepthMap(image);
    
    vector<double> o1 = o1_position(vector<double> c1, vector<double> o1Direction, double depth);
    
    return o1;
}


//seal the functions above

void main (cv:mat image, vector<double> c1){
    
    //at the very beginning, you take a screenshot with c1, so the object is located in the center of the image from c1.
    //so the info you have is c1_pose, depth.
    // you can calculate o1 by c1_pose and depth.
    //then you take 2nd image, you get o2 and c1_roation by then. In addition, you get c2_roation, and c3_rotaion.
    
    //In conclusion, the beginning set up is 2 screen shots with c1, and the object should be in the middle of c1.
    
    vector<double> initial_v;
    initial_v.push_back(0.0);
    initial_v.push_back(1.0);
    initial_v.push_back(0.0);
    
    vector<double> o1;
    o1 = initilization(cv:mat image, vector<double> initial_v);
    //o1 position is returned
    // o1 is only called once during entire duration(at the first screenshot)
    
    //the functions below are called whenever you take a screen shot, except the first screenshot.
    //you take the 2nd screenshot here
    vector<double> depth2;
    vector<double> o2;
    depth2 = get_depth(cv::mat image2);
    o2 = find_o2(vector<double> relpose_R, vector<double> relpose_T, vector<double> o1, double depth2);
    find_o2(camera_1.relpose_R, camera_1.relpose_T, o1);
    //o2,which is the object2, location is returned
    
    
    double relpose_R1[3][3];
    double relpose_R2[3][3];
    double relpose_R2[3][3];

    double rotation_matrix_c1[2][2] = rotation_matrix_from_vectors (vector<double> o1, vector<double> o2, vector<double> camera_1.relpose_T);
        //save the result of rotation matrix of c1

    double rotation_matrix_c1[2][2] = rotation_matrix_from_vectors (vector<double> o1, vector<double> o2, vector<double> camera_2.relpose_T);
    //save the result of rotation matrix of c2

    double rotation_matrix_c1[2][2] = rotation_matrix_from_vectors (vector<double> o1, vector<double> o2, vector<double> camera_3.relpose_T);
    
    
    //save the result of rotation matrix of c3

    //now, give the relpose_R to c2 and c3, and capture screenshots from c2 and c3
    //update new o1 as current o2, preparing for the next image capture
    double o1 = o2;
    
}

int main(int argc, const char * argv[]){
    InitializeMagick("");
}



//need (Scalar)relpose_T_[0-i][0-2]
//need (Scalar)relpose_R_[0-i][0-3]



  // add multiple_cameras
  for (size_t i = 0; i < num_camera_; ++i){
    rgb_cameras_.push_back(std::make_shared<RGBCamera>());
    Vector<3> B_r_BC ((Scalar)relpose_T_[i][0],(Scalar)relpose_T_[i][1],(Scalar)relpose_T_[i][2]);
    Matrix<3, 3> R_BC = Quaternion((Scalar)relpose_R_[i][0], (Scalar)relpose_R_[i][1], (Scalar)relpose_R_[i][2], (Scalar)relpose_R_[i][3]).toRotationMatrix();
    std::cout << R_BC << std::endl;
      
    //o1 position is returned
    // o1 is only called once during entire duration(at the first screenshot)
      
    //the functions below are called whenever you take a screen shot, except the first screenshot.
    //you take the 2nd screenshot here
    vector<double> depth2;
    vector<double> o2;
    vector<double> camera1_T_;
    camera1_T_.push_back(relpose_T_[0][0]);
    camera1_T_.push_back(relpose_T_[0][1]);
    camera1_T_.push_back(relpose_T_[0][2]);
    
    vector<double> camera1_R_;
       camera1_R_.push_back(relpose_R_[0][0]);
       camera1_R_.push_back(relpose_R_[0][1]);
       camera1_R_.push_back(relpose_R_[0][2]);
       camera1_R_.push_back(relpose_R_[0][3]);
      
      
    depth2 = get_depth(cv::mat image2);
    o2 = find_o2(vector<double> camera1_T_, vector<double> camera1_R_, vector<double> o1, double depth2);
    find_o2(camera_1.relpose_R, camera_1.relpose_T, o1);
      
    //using o1 and o2 to calculate relposes for c2 to ci
    rgb_cameras_[i]->setFOV(fov_);
    rgb_cameras_[i]->setWidth(width_);
    rgb_cameras_[i]->setHeight(height_);
    rgb_cameras_[i]->setRelPose(B_r_BC, R_BC);
    rgb_cameras_[i]->setPostProcesscing(std::vector<bool>{true, false, false});
    quad_ptr_->addRGBCamera(rgb_cameras_[i]);
    pub_RGBImages_.push_back(it_.advertise("flight_pilot/"+camera_names_[i]+"/RGBImage", 1));
    pub_DepthMaps_.push_back(it_.advertise("flight_pilot/"+camera_names_[i]+"/DepthMap", 1));
}

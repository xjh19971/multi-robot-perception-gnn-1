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
//#include <Magick++.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

using  std::cout;
using std::cin;
//using namespace Magick;
using namespace Eigen;


//global variable o1, only used in initialization
vector<double> o1;
o1 = initilization(cv:mat image, vector<double> c1);

//global varaibles
vector<double> o1Direction;

class OP
{
    public:

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
}

        //function: from quaternion to vector
        //https://forum.unity.com/threads/convert-quaternion-to-vector3.106789/
        //https://docs.unity3d.com/ScriptReference/Quaternion-eulerAngles.html
class Pose:: Op
{
    public:

        vector<double> initilization(cv::Mat image, vector<double> c1){
        vector<double> o1;
        vector<double> o1Direction = {0.0, -1.0 , 0.0};
        double depth;
    
        depth = getDepthMap(image);
    
        vector<double> o1 = o1_position(vector<double> c1, vector<double> o1Direction, double depth);
    
        return o1;
        }
        
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

}


// seal the functions below

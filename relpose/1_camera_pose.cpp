
#include <iostream>
#include "flightros/flight_pilot.hpp"
#include "flightros/camera_pose.hpp"
#include <string>
#include <iostream>
#include <vector>
//#include <Magick++.h>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

/*
normalize_array
normalization
matrix_dotproduct

rotation_matrix_from_vectors

o1_position
find_o2
get_depth
initilization
*/


using std::cout;
using std::cin;
//using namespace Magick;
using namespace Eigen;

int scene_id  = 0;
float main_loop_freq= 50.0;
bool unity_render = true;

int fov = 90;
double width = 720;
double height = 480;
int num_camera = 3;
//const char *camera_names={"camera_1","camera_2","camera_3"};

// Define the scalar type used.
using Scalar = float;  // numpy float32

// Define frame id for unity
using FrameID = uint64_t;

// Define frame id for unity
using SceneID = size_t;



//function: normalizing a vector(equation1)
/*
Vector<3> normalize_array(Vector<3> arr){
    Vector<3> tmp(arr), ret(arr.size());
    
    sort(tmp.begin(), tmp.end());
    
    for (int i =0; i < arr.size(); i++){
        Vector<3>:: iterator iter = find(tmp.begin(), tmp.end(), arr[i]);
        ret[i] = std::distance(tmp.begin(),iter);
    }
    return ret;
}
*/

class Op
{
    public:
        Vector<3> arr;
        Matrix<3,3> m,n;

     //function: normalizing a vector(equation 2)
        Vector<3> normalization (Vector<3> arr){
            double length;
            double length = sqrt(arr(1)*arr(1)+arr(2)*arr(2)+arr(3)*arr(3));
    
             for (int i = 1; i < 4; i++){
                arr(i) = arr(i) /length;
                }
            return arr;
        }

        double find_norm (Vector<3> arr){
             double length = sqrt(arr(1)*arr(1)+arr(2)*arr(2)+arr(3)*arr(3));
             return length;
        }

        //define cross product
        //function: dot product of two vectors
        //https://www.programiz.com/cpp-programming/examples/matrix-multiplication
        //https://www.programiz.com/cpp-programming/multidimensional-arrays

        //function: dotProduct of matrices
        Matrix<3,3> matrix_dotProduct (Matrix<3,3> m, Matrix<3,3> n){
            Matrix<3,3> mult;
            for (int i = 1; i < 4 ; i++){
              for(int j = 1; j < 4; j++){
                    mult(i,j) = 0;
                 for(int k =1; k < 4; k++){
                        mult(i,j) += m(i,k) * n(k,j);
                   }
                }
            }
         return mult;
        }

        //function: from quaternion to vector
        //https://forum.unity.com/threads/convert-quaternion-to-vector3.106789/
        //https://docs.unity3d.com/ScriptReference/Quaternion-eulerAngles.html

        Matrix<3,3> rotation_matrix_from_vectors(Vector<3> o1, Vector<3> o2, Vector<3> B_r_BC){
            Vector<3> vec1;
            vec1 = ((Scalar)(o1(0) - B_r_BC(0)),(Scalar)(o1(1) - B_r_BC(1)),(Scalar)(o1(2) - B_r_BC(2)));
    
            Vector<3> vec2;
            vec2 = ((Scalar)(o2(0) - B_r_BC(0)),(Scalar)(o2(1) - B_r_BC(1)),(Scalar)(o2(2) - B_r_BC(2)));

    
            Vector<3> a;
            Vector<3> b;
            a = normalization(vec1);
            b = normalization(vec2);
    
            Vector<3> v;
            (Scalar) cos;
            (Scalar) sin;
            v = a.cross(b);
            cos = a.dot(b);
            sin = find_norm(v);
    
            Matrix<3,3> vx = (
                (0, -v(3), v(2)),
                (v(3), 0, -v(1)),
                (-v(2), v(1), 0)
            );
    
            //first step to compute rotation matrix
            Matrix<3,3> part1 = (
                (1, -v(3), v(2)),
                (v(3), 1, -v(1)),
                (-v(2), v(1), 1)
            );
    
            //use part2 to compute part3
            //double part2[3][3];
            //Matrix<3,3> part2 =((0,0,0),(0,0,0),(0,0,0));
            Matrix<3,3> part2 = matrix_dotProduct(vx,vx);
    
            (Scalar) part3 = (1 - cos) / (sin * sin);
    
        /*    double part4[3][3];
    
            for(int i = 0; i < 3; i++){
                    for(int j =0; j < 3; j++){
                        part4[i][j] = part3 * part2[i][j];
                    }
                }
            */
    
            Matrix<3,3> R;

            for (int i = 1; i < 4; i++) {
                for(int j =1; j < 4; j++){
                    R(i,j) = part1(i,j) + part3 * part2(i,j);
                }
            }
            //double rotation_matrix[2][2] = R;
            return Matrix<3,3> R;
        }

}

class Pose: public Op
{
    public:
        //global variable o1, only used in initialization
        Vector<3> o1;
        // o1 = initilization(cv:mat image, vector<3> c1);

        //global varaibles
        Vector<3> o1Direction;
        //Local Functions
        double depth;
        double depth1;


        //initialize o1
        Vector<3> initilization(cv::Mat image, Vector<3> c1){
            Vector<3> o1;
            Vector<3> o1Direction = (0.0, -1.0 , 0.0);
            Matrix<3,3> R_BC;
            
    
            depth = getDepthMap(image);
    
            Vector<3> o1 = o1_position(Vector<3> c1, Vector<3> o1Direction, double depth);
    
            return o1;
        }

        //function:get o1(expected to be given: c1 position and o1Direction converted from the initialized quaternion)
        Vector<3> o1_position(Vector<3> c1, Vector<3> o1Direction, (Scalar) depth1){
            Vector<3> o1;
            normalization(o1Direction);
    
            o1= ((Scalar)(c1(1) + o1Direction(1)*depth1),(Scalar)(c1(2) + o1Direction(2)*depth1),(Scalar)(c1(3) + o1Direction(3)*depth1));
    
            return o1;
        }

        //https://stackoverflow.com/questions/38352891/converting-depth-image-into-open-cv-mat-format
        //https://stackoverflow.com/questions/13840013/opencv-how-to-visualize-a-depth-image

        double get_depth(cv::Mat image){
            int rows = image.rows;
            int cols = image.cols;
            (Scalar) bgrPixel;
            (Scalar) sum = 0.0;
            int center_step = 5;
            int center = rows/2;
            int center_size = center_step * center_step;
            (Scalar) depth =0.0;
    
            Vector<3> center;
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
            (Scalar) depth = sum / center_size;
            return depth;
        }
        //o1 is the object position
        //c1 is the first camera

        // call o1_position to get o1
        //use o1 and the quaternion of c1 to find o2
        //use o1 and o2 find rotation_matrix for each camera



        //functions: convert initialized quaternion of c1 to vector



        //function: find o2
        Vector<3> find_o2(Matrix<3,3> R_BC, Vector<3> B_r_BC, Vector<3> o1, (Scalar) depth){
            Vector<3> o2;
            //inputs are, rotation matrix, current camera location, o1 location, and depth
            //the input depth here should be the updated depth, which is used to calculate o2
    
            /*
            Matrix<3,3> R = (
                (1- 2*relpose_R[1]*relpose_R[1] - 2*relpose_R[2]*relpose_R[2], 2*relpose_R[0]*relpose_R[2] - 2*relpose_R[2]*relpose_R[3], 2*relpose_R[0]*relpose_R[2] + 2*relpose_R[1]*relpose_R[3]),
                (2*relpose_R[0]*relpose_R[1] + 2*relpose_R[2]*relpose_R[3], 1-2*relpose_R[0]*relpose_R[0]-2*relpose_R[2]*relpose_R[2], 2*relpose_R[1]*relpose_R[2] - 2*relpose_R[0]*relpose_R[3]),
                (2*relpose_R[0]*relpose_R[2] - 2*relpose_R[1]*relpose_R[3], 2*relpose_R[1]*relpose_R[2] + 2*relpose_R[0]*relpose_R[3],
                    1- 2*relpose_R[0]*relpose_R[0] -2*relpose_R[1]*relpose_R[1])
            );
            */
    
            Vector<3> p;
            p= ((Scalar)(o1(1) - B_r_BC(1)),(Scalar)(o1(2) - B_r_BC(2)),(Scalar)(o1(3) - B_r_BC(3)));
    
            Vector<3> second_vector;
            second_vector = ((Scalar)(R_BC(1,1)*p(1) + R_BC(1,2)*p(2) + R_BC(1,3)*p(3)),(Scalar)(R_BC(2,1)*p(1) + R_BC(2,2)*p(2) + R_BC(2,3)*p(3)),(Scalar)(R_BC(3,1)*p(1) + R_BC(3,2)*p(2) + R_BC(3,3)*p(3)));
    
            normalization(second_vector);
    
            //o2 normalization
            //for (int i =0, int i< 3, i++){
            //    second_vector[i] = second_vector[i] /
            //}
    

            o2 = ((Scalar)(B_r_BC(1) + second_vector(1) * depth),(Scalar)(B_r_BC(2) + second_vector(2) * depth),(Scalar)(B_r_BC(3) + second_vector(3) * depth));
    
            return o2;
        }

        //https://)math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/897677#897677
}





int main(){
    return 0;
}




/*
void take_screen_shot(cv::mat image,vector<double> o1, camera_1.relpose_T, camera_1.relpose_R, camera_2.relpose_T, camera_3.relpose_T ){
    get_depth(cv::mat image);
    
    update_o1(); //现在的o2是o1，为下一张截图和o2做准备
}

 void main (cv:mat image){
    
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


//seal the functions above


/*

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

*/




//need (Scalar)relpose_T_[0-i][0-2]
//need (Scalar)relpose_R_[0-i][0-3]



 

/*
class relpose 
{
    public: 
        void setPoint(vector<3> vec)
        {

        }

        vector<3> find_o2()
        {

        }

        double get_depth()
        {

        }

        double rotation_matrix_from_vectors()
        {

        }
}
*/ 

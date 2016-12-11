/****
使用方法：
1. FaceRecognize f = new FaceRecognize(<path/to/haar>, <path/to/csv>);
2.
  for(;;) {
    vector<FaceInfo> face_infos = f.predict_face(<video_Mat>);
    for(i in face_infos.length){
      info = face_infos.pop;
      putText(<video_Mat>, info.predict_name, Point(info.pos_x, info.pos_y), FONT_HERSHEY_PLAIN, 3.0, CV_RGB(0,255,0), 2.0);
    }
    imshow("face_recognizer", <video_Mat>);
    char key = (char) waitKey(20);
    if(key == 27) break;
  }

****/


#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace cv;
using namespace cv::face;
using namespace std;

struct FaceInfo{
  string predict_name;
  int pos_x;
  int pos_y;
};

class FaceRecognize{
  public:
    FaceRecognize(string, string);
    static void read_csv(const string&, vector<Mat>&, vector<int>&, char);
  private:
    string fn_haar;
    string fn_csv;
    vector<Mat> images;
    vector<int> labels;
    int im_width;
    int im_height;
    Ptr<FaceRecognizer> model;
    CascadeClassifier haar_cascade;
};

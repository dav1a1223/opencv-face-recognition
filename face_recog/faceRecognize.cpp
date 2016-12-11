#include "faceRecognize.h"
using namespace cv;
using namespace cv::face;
using namespace std;

FaceRecognize::FaceRecognize(string haar_path, string csv_path){
  string fn_haar = haar_path;
  string fn_csv = csv_path;

  try {
      read_csv(fn_csv, images, labels);
  } catch (cv::Exception& e) {
      cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
      exit(1);
  }
  int im_width = images[0].cols;
  int im_height = images[0].rows;

  Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
  model->train(images, labels);

  CascadeClassifier haar_cascade;
  haar_cascade.load(fn_haar);
}

static void FaceRecognize::read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

vector<FaceInfo> FaceRecognize::predict_face(Mat video_frame){
  Mat original = video_frame.clone();
  Mat gray;
  cvtColor(original, gray, CV_BGR2GRAY);
  vector< Rect_<int> > faces;
  haar_cascade.detectMultiScale(gray, faces);
  vector<FaceInfo> infos;

  for(int i = 0; i < faces.size(); i++) {
      Rect face_i = faces[i];
      Mat face = gray(face_i);
      Mat face_resized;
      cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);

      int prediction = model->predict(face_resized);
      rectangle(original, face_i, CV_RGB(0, 255,0), 1);
      string name;
      if(prediction == 1){
          name = "Pei-Shuan Wu";
      }
      else if(prediction == 2){
          name = "Kuan-Ting Lai";
      }
      else if(prediction == 3){
          name = "Yu-Hao Liu";
      }
      else if(prediction == 4){
          name = "Yih-Kuen Tsay";
      }

      FaceInfo info;
      info.predict_name = format("%s", name.c_str());
      info.pos_x = std::max(face_i.tl().x - 10, 0);
      info.pos_y = std::max(face_i.tl().y - 10, 0);

      infos.push_back(info);
  }
  return infos;
}

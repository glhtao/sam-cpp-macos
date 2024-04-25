#include <opencv2/opencv.hpp>
#define STRIP_FLAG_HELP 1
#include <gflags/gflags.h>
#include <thread>
#include "sam.h"

DEFINE_string(encoder, "mobile_sam/mobile_sam_preprocess.onnx", "Path to the encoder model");
DEFINE_string(decoder, "mobile_sam/mobile_sam.onnx", "Path to the decoder model");
DEFINE_string(image, "images/macos.jpg", "Path to the image");
DEFINE_string(device, "cpu", "cpu or cuda:0(1,2,3...)");
DEFINE_bool(h, false, "Show help");

int main(int argc, char** argv) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if(FLAGS_h){
    std::cout<<"Example: ./build/sam_cpp_test -encoder=\"mobile_sam/mobile_sam_preprocess.onnx\" "
               "-decoder=\"mobile_sam/mobile_sam.onnx\" "
               "-image=\"david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg\" -device=\"cpu\""<< std::endl;
    return 0;
  }
  Sam sam;
  if(FLAGS_encoder.find("sam_hq") != std::string::npos){
    sam.changeMode(HQSAM);
  }else if(FLAGS_encoder.find("efficientsam") != std::string::npos){
    sam.changeMode(EfficientSAM);
  }else if(FLAGS_encoder.find("edge_sam") != std::string::npos){
    sam.changeMode(EdgeSAM);
  }


  std::cout<<"loadModel started"<<std::endl;
  bool successLoadModel = sam.loadModel(FLAGS_encoder, FLAGS_decoder, std::thread::hardware_concurrency(), FLAGS_device);
  if(!successLoadModel){
    std::cout<<"loadModel error"<<std::endl;
    return 1;
  }
  std::cout<<"preprocessImage started"<<std::endl;
  cv::Mat image = cv::imread(FLAGS_image, cv::IMREAD_COLOR);
  if (image.empty()) {
      std::cout << "Image loading failed" << std::endl;
      return -1;
  }
  cv::Size imageSize = cv::Size(image.cols, image.rows);
  std::cout << "Iimage size  " << imageSize << std::endl;
  cv::Size inputSize = sam.getInputSize();
  if (inputSize.empty()) {
      std::cout << "Sam initialization failed" << std::endl;
      return -1;
  }
  std::cout << "Resize image to " << inputSize << std::endl;
  cv::resize(image, image, inputSize);

  std::cout << "preprocessImage image..." << std::endl;
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  bool successPreprocessImage = sam.preprocessImage(image);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "sec = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 <<std::endl;
  if(!successPreprocessImage){
    std::cout<<"preprocessImage error"<<std::endl;
    return 1;
  }

  std::cout << "Now click on the image (press q/esc to quit; press c to clear selection; press a "
      "to run automatic segmentation)\n"
      << "Ctrl+Left click to select foreground, Ctrl+Right click to select background, "
      << "Middle click and drag to select a region\n";

  std::list<cv::Point3i> clickedPoints;
  cv::Point3i newClickedPoint(-1, 0, 0);
  std::list<cv::Rect> rects;
  cv::Rect roi;
  cv::Mat outImage = image.clone();

  auto g_windowName = "Segment Anything CPP Demo";
  cv::namedWindow(g_windowName, 0);
  cv::setMouseCallback(
      g_windowName,
      [](int event, int x, int y, int flags, void* userdata) {
          int code = -1;
          if (event == cv::EVENT_LBUTTONDOWN) {
              code = 2;
          }
          else if (event == cv::EVENT_RBUTTONDOWN) {
              code = 0;
          }
          else if (event == cv::EVENT_MBUTTONDOWN ||
              ((flags & cv::EVENT_FLAG_MBUTTON) && event == cv::EVENT_MOUSEMOVE)) {
              code = 4;
          }
          else if (event == cv::EVENT_MBUTTONUP) {
              code = 5;
          }

          if (code >= 0) {
              if (code <= 2 && (flags & cv::EVENT_FLAG_CTRLKEY) == cv::EVENT_FLAG_CTRLKEY) {
                  // If ctrl is pressed, then append it to the list later
                  code += 1;
              }
              *(cv::Point3i*)userdata = { x, y, code };
          }
      },
      &newClickedPoint);

#define SHOW_TIME                                                     \
  std::cout << "Time elapsed: "                                       \
            << std::chrono::duration_cast<std::chrono::milliseconds>( \
                   std::chrono::system_clock::now() - timeNow)        \
                   .count()                                           \
            << " ms" << std::endl;

  bool bRunning = true;
  while (bRunning) {
      const auto timeNow = std::chrono::system_clock::now();

      if (newClickedPoint.x > 0) {
          std::list<cv::Point> points, nagativePoints;
          if (newClickedPoint.z == 5) {
              roi = {};
          }
          else if (newClickedPoint.z == 4) {
              if (roi.empty()) {
                  roi = cv::Rect(newClickedPoint.x, newClickedPoint.y, 1, 1);
              }
              else {
                  auto tl = roi.tl(), np = cv::Point(newClickedPoint.x, newClickedPoint.y);
                  // construct a rectangle from two points
                  roi = cv::Rect(cv::Point(std::min(tl.x, np.x), std::min(tl.y, np.y)),
                      cv::Point(std::max(tl.x, np.x), std::max(tl.y, np.y)));
                  std::cout << "Box: " << roi << std::endl;
              }
          }
          else {
              if (newClickedPoint.z % 2 == 0) {
                  clickedPoints = { newClickedPoint };
              }
              else {
                  clickedPoints.push_back(newClickedPoint);
              }
          }

          for (auto& p : clickedPoints) {
              if (p.z >= 2) {
                  points.push_back({ p.x, p.y });
              }
              else {
                  nagativePoints.push_back({ p.x, p.y });
              }
          }

          newClickedPoint.x = -1;
          if (points.empty() && nagativePoints.empty() && roi.empty()) {
              continue;
          }

          //cv::Mat mask = sam.getMask(points, nagativePoints, roi);
          int previousMaskIdx = -1;
          bool isNextGetMask = true;
          rects.clear();
          rects.push_back(roi);
          cv::Mat mask = sam.getMask(points, nagativePoints, rects, previousMaskIdx, isNextGetMask);
        //  cv::imwrite("mask-box.png", mask);
        //  cv::resize(mask, mask, imageSize, 0, 0, cv::INTER_NEAREST);
        //  cv::imwrite("mask-box2.png", mask);

          SHOW_TIME

              // apply mask to image
              outImage = cv::Mat::zeros(image.size(), CV_8UC3);
          for (int i = 0; i < image.rows; i++) {
              for (int j = 0; j < image.cols; j++) {
                  auto bFront = mask.at<uchar>(i, j) > 0;
                  float factor = bFront ? 1.0 : 0.2;
                  outImage.at<cv::Vec3b>(i, j) = image.at<cv::Vec3b>(i, j) * factor;
              }
          }

          for (auto& p : points) {
              cv::circle(outImage, p, 2, { 0, 255, 255 }, -1);
          }
          for (auto& p : nagativePoints) {
              cv::circle(outImage, p, 2, { 255, 0, 0 }, -1);
          }
      }
      else if (newClickedPoint.x == -2) {
 
      }

      if (!roi.empty()) {
          cv::rectangle(outImage, roi, { 255, 255, 255 }, 2);
      }

      cv::imshow(g_windowName, outImage);
      int key = cv::waitKeyEx(100);
      switch (key) {
      case 27:
      case 'Q':
      case 'q': {
          bRunning = false;
      } break;
      case 'C':
      case 'c': {
          clickedPoints.clear();
          newClickedPoint.x = -1;
          roi = {};
          outImage = image.clone();
      } break;
      case 'A':
      case 'a': {
          clickedPoints.clear();
          newClickedPoint.x = -2;
          outImage = image.clone();
      }
      }
  }

  cv::destroyWindow(g_windowName);

  return 0;

  /*
  std::cout<<"getMask started"<<std::endl;
  std::list<cv::Point> points, nagativePoints;
  std::list<cv::Rect> rects;
  // box
  int previousMaskIdx = -1;
  bool isNextGetMask = true;
  cv::Rect rect = cv::Rect(1215 * inputSize.width / imageSize.width,
                           125 * inputSize.height / imageSize.height,
                           508 * inputSize.width / imageSize.width,
                           436 * inputSize.height / imageSize.height);
  rects.push_back(rect);
  cv::Mat mask = sam.getMask(points, nagativePoints, rects, previousMaskIdx, isNextGetMask);
  previousMaskIdx++;
  cv::resize(mask, mask, imageSize, 0, 0, cv::INTER_NEAREST);
  cv::imwrite("mask-box.png", mask);


  // positive point
  isNextGetMask = false;
  cv::Point point = cv::Point(1255 * inputSize.width / imageSize.width,
                              360 * inputSize.height / imageSize.height);
  points.push_back(point);
  mask = sam.getMask(points, nagativePoints, rects, previousMaskIdx, isNextGetMask);
  previousMaskIdx++;
  cv::resize(mask, mask, imageSize, 0, 0, cv::INTER_NEAREST);
  cv::imwrite("mask-positive_point.png", mask);
  return 0;*/
}

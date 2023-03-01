//
// Created by qinhua on 2/25/23.
//
#define USE_FFMPEG  1
#define USE_OPENCV  1
#define USE_BMCV    1
#ifndef OPENCV_YOLOV5_YOLOV5_H
#define OPENCV_YOLOV5_YOLOV5_H

#include <string>
#include <vector>
#include <array>
#include <numeric>
#include <algorithm>
#include <fstream>
#include "opencv2/opencv.hpp"
#include <map>
#include <inireader.hpp>
#include <sstream>
#include <iostream>

#include <cvwrapper.h>
#include <engine.h>
#include <tensor.h>
#include <queue>

using namespace std;
struct crop_param {
    int crop_x0;
    int crop_y0;
    int crop_h;
    int crop_w;
};

struct output_struct {
    cv::Mat output_mat;
    vector<int> output_shapes;
    cv::Mat input_mat;
};

struct sail_tensor_with_mat {
    sail::Tensor input_tensor;
    cv::Mat input_mat;
};

struct sail_image_with_mat {
    sail::BMImage input_image;
    cv::Mat input_mat;
};

class yolo {
public:
    yolo();

    ~yolo();

    void set_finish();

    void push_video(vector<string> video_names, vector<int> dev_id_);


    void init_yolo_preprocess(bool letter_box_flag, int resize_w, int resize_h, sail::PaddingAtrr &padding_in,
                              crop_param &crop_param_);

    void init_yolo_preprocess(bool letter_box_flag, int resize_w, int resize_h);

    void start_yolo_preprocess_threads(int num, vector<int> device_list);

    void start_yolo_process_threads(int num, string bmodel_path, vector<int> device_list);

    void start_yolo_post_process_threads(int num, float modelScoreThreshold, float modelNMSThreshold);

    void write_to_txt(vector<cv::Rect> &boxes, vector<int> &nms_result, cv::Mat &ori_mat);

    void end_yolo_thread();


private:
    void decode_video(string video_name, int tpu_id);

    void yolo_preproess_thread(int dev_id);

    void yolo_process_thread(int dev_id, string bmodel_path);

    void yolo_postprocess_thread(float modelScoreThreshold, float modelNMSThreshold);

    void NMSBoxes(std::vector<cv::Rect> &boxes, std::vector<float> &confidences,
                  std::vector<int> &nms_resultm, float modelScoreThreshold, float modelNMSThreshold);

    void
    draw(cv::Mat &image_ori, std::vector<int> &nms_result, std::vector<int> &class_ids, std::vector<float> &confidences,
         std::vector<cv::Rect> &boxes, vector<string> &classes);

    //只定义共享变量
    struct Detection {
        int class_id{0};
        std::string className{};
        float confidence{0.0};
        cv::Scalar color{};
        cv::Rect box{};
    };
    bool finish_flag_ = false;
    int mode = 2;
    bool letter_box_flag = false;
    sail::PaddingAtrr yolo_padding_in;
    crop_param yolo_crop_param;
    vector<thread> preprocess_thread_pool;
    vector<thread> decode_thread_pool;
    vector<thread> process_thread_pool;
    vector<thread> postprocess_thread_pool;
    int resize_w;
    int resize_h;
    std::queue<sail::BMImage> before_preprocess_queue;
    std::queue<sail_image_with_mat> after_preprocess_queue;
    std::queue<output_struct> after_process_queue;
    mutex before_preprocess_queue_mutex;
    mutex after_preprocess_queue_mutex;
    mutex after_process_queue_mutex;
    condition_variable before_preprocess_queue_cv;
    condition_variable after_preprocess_queue_cv;
    condition_variable after_process_queue_cv;


};


#endif //OPENCV_YOLOV5_YOLOV5_H

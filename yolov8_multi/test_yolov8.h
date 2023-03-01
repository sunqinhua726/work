#define USE_OPENCV 1
#define USE_FFMPEG 1
#define USE_BMCV 1

#include <sail/cvwrapper.h>
#include <sail/decoder_multi.h>
#include <sail/engine.h>
#include <sail/tensor.h>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <stdio.h>
#include <mutex>
#include <thread>

using namespace std;
using namespace sail;


class MultiProcessor {
public:
    MultiProcessor(int tpu_id, std::vector<std::string> video_list, std::string bmodel_name,
                   sail_resize_type resize_type, int queue_in_size, int queue_out_size);

    ~MultiProcessor();

    std::tuple<std::map<std::string, sail::Tensor *>, std::vector<cv::Mat>, std::vector<int>, std::vector<int>, std::vector<std::vector<int>>>
    GetBatchData();

    int get_input_width();

    int get_input_height();

    vector<int> get_output_shape();

private:
    std::vector<std::string> video_list_;
    int tpu_id_;
    sail::MultiDecoder *multi_decoder_ = NULL;
    EngineImagePreProcess *engine_image_pre_process_ = NULL;
    std::map<int, std::string> video_list_map;

    std::string bmodel_name_;

    std::tuple<std::pair<float, float>, std::pair<float, float>, std::pair<float, float>> alpha_beta;

    std::mutex mutex_exit;
    bool flag_exit_;

    std::mutex mutex_thread_ended;
    bool thread_ended;
private:
    void InitEngineImagePreProcess(int tpu_id, std::string bmodel_name, sail_resize_type resize_type, int queue_in_size,
                                   int queue_out_size);

    void decoder_and_preprocess();

    bool get_exit_flag();

    void set_exit_flag(bool);

    void start_decoder_thread();
};

struct ObjRect {
    unsigned int class_id;
    float score;
    float left;
    float top;
    float right;
    float bottom;
    float width;
    float height;
};


class Yolov8_PostForward {
public:
    Yolov8_PostForward(float modelScoreThreshold_, float modelNMSThreshold_, int MAX_OUTPUT_QUEUE_,
                       std::vector<std::string> classes_, vector<int> output_shapes ,bool padding_flag_);

    ~Yolov8_PostForward();

    void set_yolov8_post_threads(int num);

    void start_yolov8_post_threads();

    void end_yolov8_post_threads();

    void push_data_to_output_queue(vector<float *> &output_tensor, vector<cv::Mat> &output_mat, vector<int> &channel,
                                   vector<int> &index);

    void set_mutidecode_is_finish();

//    void set_output_shape(vector<int> output_shape);

    bool mutidecode_is_finish = false;
private:
    pair<float, float> calculate_resize_scale(vector<float> &image_size, vector<float> &target_size);

    pair<float, float>
    calculate_resize_offset(vector<float> &image_size, vector<float> &target_size, float &resize_scale);

    void
    draw(cv::Mat &image_ori, std::vector<int> &nms_result, std::vector<int> &class_ids, std::vector<float> &confidences,
         std::vector<cv::Rect> &boxes);

    void yolov8_post_process();

    void NMSBoxes(std::vector<cv::Rect> &boxes, std::vector<float> &confidences, std::vector<int> &nms_result);

    struct Detection {
        int class_id{0};
        std::string className{};
        float confidence{0.0};
        cv::Scalar color{};
        cv::Rect box{};
    };
    struct Data {
        float *output_tensor_point;
        cv::Mat output_mat;
        int channel;
        int index;
    };
    queue<Data> Data_queue;
    float modelScoreThreshold{0.45};
    float modelNMSThreshold{0.50};
    condition_variable output_queue_cv;
    mutex output_queue_mutex;
    vector<int> output_shapes;
    pair<int, int> input_h_w;
    int MAX_OUTPUT_QUEUE = 100;

//    int flag_output_queue_is_not_full = 1;
    std::vector<std::thread> pool_post;
    bool padding_flag;
    int num_post_threads = 0;
    std::vector<std::string> classes = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                                        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                                        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                                        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                        "skis",
                                        "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                                        "skateboard",
                                        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
                                        "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                                        "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
                                        "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                                        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                                        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
};
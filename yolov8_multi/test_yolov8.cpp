#include "test_yolov8.h"
#include <sys/time.h>
#include <unistd.h>
#include <numeric>
#include <random>
#include "cstring"

using namespace cv;

double get_time_us() {
#ifdef _WIN32
    // 从1601年1月1日0:0:0:000到1970年1月1日0:0:0:000的时间(单位100ns)
#define EPOCHFILETIME   (116444736000000000UL)
        FILETIME ft;
        LARGE_INTEGER li;
        double tt = 0;
        GetSystemTimeAsFileTime(&ft);
        li.LowPart = ft.dwLowDateTime;
        li.HighPart = ft.dwHighDateTime;
        // 从1970年1月1日0:0:0:000到现在的微秒数(UTC时间)
        tt = (li.QuadPart - EPOCHFILETIME) /10;
        return tt;
#else
    timeval tv;
    gettimeofday(&tv, 0);
    return (double) tv.tv_sec * 1000000 + (double) tv.tv_usec;
#endif // _WIN32
    return 0;
}

float overlap_FM(float x1, float w1, float x2, float w2) {
    float l1 = x1;
    float l2 = x2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1;
    float r2 = x2 + w2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection_FM(ObjRect a, ObjRect b) {
    float w = overlap_FM(a.left, a.width, b.left, b.width);
    float h = overlap_FM(a.top, a.height, b.top, b.height);
    if (w < 0 || h < 0) return 0;
    float area = w * h;
    return area;
}

float box_union_FM(ObjRect a, ObjRect b) {
    float i = box_intersection_FM(a, b);
    float u = a.width * a.height + b.width * b.height - i;
    return u;
}

//float box_iou_FM(ObjRect a, ObjRect b) {
//    return box_intersection_FM(a, b) / box_union_FM(a, b);
//}

MultiProcessor::MultiProcessor(int tpu_id, std::vector<std::string> video_list, std::string bmodel_name,
                               sail_resize_type resize_type, int queue_in_size, int queue_out_size) {
    tpu_id_ = tpu_id;
    for (int i = 0; i < video_list.size(); ++i) {
        video_list_.push_back(video_list[i]);
    }
    multi_decoder_ = new MultiDecoder(10, tpu_id_);
    multi_decoder_->set_local_flag(true);
    for (int i = 0; i < video_list.size(); ++i) {
        // int channel_index = multi_decoder_->add_channel(video_list[i],1);
        int channel_index = multi_decoder_->add_channel(video_list[i]);
        video_list_map.insert(std::pair<int, std::string>(channel_index, video_list[i]));
    }
    flag_exit_ = false;
    thread_ended = true;
    alpha_beta = std::make_tuple(std::pair<float, float>(0.003921, 0), std::pair<float, float>(0.003921, 0),
                                 std::pair<float, float>(0.003921, 0));
    InitEngineImagePreProcess(tpu_id, bmodel_name, resize_type, queue_in_size, queue_out_size);
    start_decoder_thread();
}

MultiProcessor::~MultiProcessor() {
    SPDLOG_INFO("Start set_exit_flag.....");
    set_exit_flag(true);
    SPDLOG_INFO("End set_exit_flag, and waiting for thread to finish....");

    while (true) {
        {
            std::lock_guard<std::mutex> lock_mutex(mutex_thread_ended);
            if (thread_ended) {
                break;
            }
        }
        SPDLOG_INFO("Thread Not finished, sleep 500ms!");
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        continue;
    }

    SPDLOG_INFO("Thread Finished!");
    SPDLOG_INFO("Start delete multi_decoder_");
    delete multi_decoder_;
    SPDLOG_INFO("Start delete engine_image_pre_process_");
    delete engine_image_pre_process_;
}

void MultiProcessor::InitEngineImagePreProcess(int tpu_id, std::string bmodel_name, sail_resize_type resize_type,
                                               int queue_in_size, int queue_out_size) {
    engine_image_pre_process_ = new EngineImagePreProcess(bmodel_name, tpu_id, true);
    engine_image_pre_process_->InitImagePreProcess(resize_type, true, queue_in_size, queue_out_size);
    engine_image_pre_process_->SetPaddingAtrr();
    engine_image_pre_process_->SetConvertAtrr(alpha_beta);
}

void MultiProcessor::decoder_and_preprocess() {
    int image_idx = 0;
    sail::Handle handle(tpu_id_);
    sail::Bmcv bmcv(handle);
    while (true) {
        thread_ended = false;
        if (get_exit_flag()) {
            break;
        }
        auto iter = video_list_map.begin();
        while (iter != video_list_map.end()) {
            int channel_index = iter->first;
            sail::BMImage image_temp;
            int ret = multi_decoder_->read(channel_index, image_temp);
            if (ret == 0) {
                ret = engine_image_pre_process_->PushImage(channel_index, image_idx, image_temp);
                if (ret == 0)
                    image_idx++;
            }
            iter++;
        }
    }
    std::lock_guard<std::mutex> lock_mutex(mutex_thread_ended);
    thread_ended = true;
    SPDLOG_INFO("decoder_and_preprocess thread finished.....");
}

bool MultiProcessor::get_exit_flag() {
    std::lock_guard<std::mutex> lock_mutex(mutex_exit);
    return flag_exit_;
}

void MultiProcessor::set_exit_flag(bool flag) {
    std::lock_guard<std::mutex> lock_mutex(mutex_exit);
    flag_exit_ = flag;
}

void MultiProcessor::start_decoder_thread() {
    std::thread thread_decoder = std::thread(&MultiProcessor::decoder_and_preprocess, this);
    thread_decoder.detach();
}

int MultiProcessor::get_input_width() {
    return engine_image_pre_process_->get_input_width();
}

int MultiProcessor::get_input_height() {
    return engine_image_pre_process_->get_input_height();
}

vector<int> MultiProcessor::get_output_shape() {
    string output_name = engine_image_pre_process_->get_output_names()[0];
    return engine_image_pre_process_->get_output_shape(output_name);
}

std::tuple<std::map<std::string, sail::Tensor *>, std::vector<cv::Mat>, std::vector<int>, std::vector<int>, std::vector<std::vector<int>>>
MultiProcessor::GetBatchData() {
    return std::move(engine_image_pre_process_->GetBatchData_CV());
}


Yolov8_PostForward::Yolov8_PostForward(float modelScoreThreshold_, float modelNMSThreshold_, int MAX_OUTPUT_QUEUE_,
                                       std::vector<std::string> classes_, vector<int> output_shapes_,
                                       bool padding_flag_)
        : modelNMSThreshold(modelNMSThreshold_),
          modelScoreThreshold(modelScoreThreshold_),
          MAX_OUTPUT_QUEUE(MAX_OUTPUT_QUEUE_), output_shapes(output_shapes_), padding_flag(padding_flag_) {
    classes = move(classes_);

}

Yolov8_PostForward::~Yolov8_PostForward() {

    cout << "destory Yolov8_PostForward" << endl;

}


void
Yolov8_PostForward::NMSBoxes(std::vector<cv::Rect> &boxes, std::vector<float> &confidences,
                             std::vector<int> &nms_result) {
    CV_Assert(boxes.size() == confidences.size());
    nms_result.clear();
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    auto cmp = [&](int a, int b) { return confidences[a] > confidences[b]; };
    std::sort(order.begin(), order.end(), cmp);

    std::vector<float> areas(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        areas[i] = (boxes[i].width + 1) * (boxes[i].height + 1);
    }

    while (!order.empty()) {
        int idx = order.front();
        if (confidences[idx] < modelScoreThreshold) {
            break;
        }
        nms_result.push_back(idx);
        order.erase(order.begin());

        for (auto it = order.begin(); it != order.end();) {
            int overlap_idx = *it;
            float x1 = std::max(boxes[idx].x, boxes[overlap_idx].x);
            float y1 = std::max(boxes[idx].y, boxes[overlap_idx].y);
            float x2 = std::min(boxes[idx].x + boxes[idx].width, boxes[overlap_idx].x + boxes[overlap_idx].width);
            float y2 = std::min(boxes[idx].y + boxes[idx].height, boxes[overlap_idx].y + boxes[overlap_idx].height);
            float w = std::max(0.0f, x2 - x1 + 1);
            float h = std::max(0.0f, y2 - y1 + 1);
            float overlap = w * h / (areas[idx] + areas[overlap_idx] - w * h);
            if (overlap > modelNMSThreshold) {
                it = order.erase(it);
            } else {
                ++it;
            }
        }
    }

}


void Yolov8_PostForward::draw(cv::Mat &image_ori, std::vector<int> &nms_result, std::vector<int> &class_ids,
                              std::vector<float> &confidences,
                              std::vector<cv::Rect> &boxes) {
    std::vector<Detection> detections{};
    for (unsigned long i = 0; i < nms_result.size(); ++i) {
        int idx = nms_result[i];

        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255);
        result.color = cv::Scalar(dis(gen),
                                  dis(gen),
                                  dis(gen));
        result.className = classes[result.class_id];
        result.box = boxes[idx];


        detections.push_back(result);
    }
    auto output = detections;
    int detections_size = output.size();
    std::cout << "Number of detections:" << detections_size << std::endl;
    for (int i = 0; i < detections_size; ++i) {
        Detection detection = output[i];
        cv::Rect box = detection.box;
        cv::Scalar color = detection.color;
        if (padding_flag == false) {
            box.x = box.x / 640.0 * image_ori.cols;
            box.y = box.y / 640.0 * image_ori.rows;
            box.width = box.width * (float(image_ori.cols) / 640.0);
            box.height = box.height * (float(image_ori.rows) / float(640));
        } else {
            auto ratio_letter = min(640.0 / image_ori.cols, 640.0 / image_ori.rows);
            box.height = box.height / ratio_letter;
            box.width = box.width / ratio_letter;
            box.x = box.x / ratio_letter;
            box.y = box.y / ratio_letter;
        }
        cv::rectangle(image_ori, box, color, 2);
        std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

        cv::rectangle(image_ori, textBox, color, cv::FILLED);
        cv::putText(image_ori, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1,
                    cv::Scalar(0, 0, 0), 2, 0);
    }
    class_ids.clear();
    confidences.clear();
    boxes.clear();
    nms_result.clear();
}

void Yolov8_PostForward::yolov8_post_process() {
    std::vector<int> nms_result;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    while (true) {
        unique_lock<mutex> lock(output_queue_mutex);
        output_queue_cv.wait(lock, [this] { return (!Data_queue.empty()) || mutidecode_is_finish; });
        if (mutidecode_is_finish && Data_queue.empty())
            break;
        auto output_tensor = Data_queue.front().output_tensor_point;
        auto output_mat = move(Data_queue.front().output_mat);
        auto output_index = move(Data_queue.front().index);
        auto output_channel = move(Data_queue.front().channel);
        Data_queue.pop();
        lock.unlock();
        output_queue_cv.notify_all();
        if (output_shapes[1] < output_shapes[2]) {
            cv::Mat mat_(output_shapes[1], output_shapes[2], CV_32FC1, output_tensor);
            cv::Mat mat;
            cv::transpose(mat_, mat);
            delete output_tensor;
            float *data = (float *) mat.data;
            for (int i = 0; i < output_shapes[2]; ++i) {
                float *classes_scores = data + 4;
                cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
                cv::Point class_id;
                double maxClassScore;
                minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
                if (maxClassScore > modelScoreThreshold) {
                    confidences.push_back(maxClassScore);
                    class_ids.push_back(class_id.x);
                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];
                    int left = int(x - 0.5 * w);
                    int top = int(y - 0.5 * h);
                    int width = int(w);
                    int height = int(h);
                    boxes.push_back(move(cv::Rect(left, top, width, height)));
                }
                data += output_shapes[1];
            }
        } else {
            cv::Mat mat_(output_shapes[1], output_shapes[2], CV_32FC1, output_tensor);
            float *data = (float *) mat_.data;

            for (int i = 0; i < output_shapes[1]; ++i) {
                {
                    float confidence = data[4];

                    if (confidence >= 0.25) {
                        float *classes_scores = data + 5;

                        cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
                        cv::Point class_id;
                        double max_class_score;

                        minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

                        if (max_class_score > modelScoreThreshold) {
                            confidences.push_back(confidence);
                            class_ids.push_back(class_id.x);

                            float x = data[0];
                            float y = data[1];
                            float w = data[2];
                            float h = data[3];

                            int left = int((x - 0.5 * w));
                            int top = int((y - 0.5 * h));

                            int width = int(w);
                            int height = int(h);

                            boxes.push_back(cv::Rect(left, top, width, height));
                        }
                    }
                }

                data += output_shapes[2];
            }

        }

        double time_start = get_time_us();
        NMSBoxes(boxes, confidences, nms_result);
        double time_end = get_time_us();
        cout << "### NMS time  " + to_string(time_end - time_start) + "  us" + "  channel " + to_string(output_channel)
                + " index " + to_string(output_index) << endl;
        cout << "### NMS BOXS number : " + to_string(boxes.size()) << endl;
        pair<float, float> ori_image_size = make_pair(output_mat.rows, output_mat.cols);
        pair<float, float> resized_image_size = make_pair(640, 640);

        draw(output_mat, nms_result, class_ids, confidences, boxes);
        cv::imwrite("channel_" + to_string(output_channel) + "_index_" + to_string(output_index) + ".jpg", output_mat);
    }


}

void Yolov8_PostForward::set_yolov8_post_threads(int num) {
    num_post_threads = num;
}

void Yolov8_PostForward::start_yolov8_post_threads() {
    for (int i = 0; i < num_post_threads; ++i) {
        pool_post.emplace_back(&Yolov8_PostForward::yolov8_post_process, this);
    }
}

void Yolov8_PostForward::end_yolov8_post_threads() {
    for (int i = 0; i < num_post_threads; i++) {
        pool_post[i].join();
    }
}

void Yolov8_PostForward::set_mutidecode_is_finish() {
    mutidecode_is_finish = true;
}

std::vector<std::string> classes = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                                    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                                    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                                    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
                                    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                                    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
                                    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                                    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
                                    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                                    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                                    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};


int flag_output_queue_is_not_full = 1;


void Yolov8_PostForward::push_data_to_output_queue(vector<float *> &output_tensor, vector<cv::Mat> &output_mat,
                                                   vector<int> &channel, vector<int> &index) {
    unique_lock<mutex> lock(output_queue_mutex);
    output_queue_cv.wait(lock, [&] { return Data_queue.size() < MAX_OUTPUT_QUEUE; });

    for (int i = 0; i < output_shapes[0]; ++i) {
        Data data{output_tensor[i], move(output_mat[i]), channel[i], index[i]};
        Data_queue.push(data);
    }
    output_queue_cv.notify_all();

}

int main() {
    int tpu_id = 0;
    int max_count = 100;
    std::string bmodel_name = "/home/qinhua/muli/yolov8n_fp16_b4_640_84X/compilation.bmodel";
//    std::string bmodel_name = "/home/qinhua/models/yolov5/yolov5s_batch1_fp32_84X/compilation.bmodel";
    sail_resize_type resize_type = BM_PADDING_VPP_NEAREST;


    bool padding_flag = true;
    if (resize_type == BM_RESIZE_VPP_NEAREST || resize_type == BM_RESIZE_TPU_NEAREST ||
        resize_type == BM_RESIZE_TPU_NEAREST || resize_type == BM_RESIZE_TPU_BICUBIC) {
        padding_flag = false;
    }
    std::vector<std::string> video_list;
    int queue_in_size = 10;
    int queue_out_size = 5;

    video_list.push_back("/home/qinhua/videos/001.mp4");
    video_list.push_back("/home/qinhua/videos/002.mp4");
    video_list.push_back("/home/qinhua/videos/003.mp4");
    video_list.push_back("/home/qinhua/videos/004.mp4");

    sail::Handle handle(tpu_id);
    sail::Bmcv bmcv(handle);
    MultiProcessor process(tpu_id, video_list, bmodel_name, resize_type, queue_in_size, queue_out_size);
    std::vector<int> strides;
    vector<int> output_shapes = process.get_output_shape();
    Yolov8_PostForward yolov8_post(0.45, 0.5, 50, classes, output_shapes, padding_flag);
    yolov8_post.set_yolov8_post_threads(1);
    yolov8_post.start_yolov8_post_threads();
    strides.push_back(8);
    strides.push_back(16);
    strides.push_back(32);

    std::cout << "::::::main thread: " << getpid() << "," << gettid() << endl;
    std::tuple<std::map<std::string, sail::Tensor *>, std::vector<cv::Mat>, std::vector<int>, std::vector<int>, std::vector<std::vector<int>>> result;
    for (int i = 0; i < max_count; ++i) {

        double time_start = get_time_us();
        if (flag_output_queue_is_not_full) {
            result = process.GetBatchData();
        }
        double time_end = get_time_us();

        auto result_map = std::move(std::get<0>(result));
        auto iter = result_map.begin();
        if (iter == result_map.end()) {
            printf("Out put tensor map is empty!\n");
            return 1;
        }

        sail::Tensor *output_tensor = iter->second;
        std::vector<cv::Mat> imgs = std::move(std::get<1>(result));
        std::vector<int> channel = std::move(std::get<2>(result));
        std::vector<int> index = std::move(std::get<3>(result));
        std::vector<std::vector<int>> padding_atrr = std::move(std::get<4>(result));
        output_tensor->sync_d2s();
        float *output_tensor_data = (float *) output_tensor->sys_data();
        vector<float *> point_vector;
        for (int j = 0; j < output_shapes[0]; ++j) {
            point_vector.emplace_back(new float[output_shapes[1] * output_shapes[2]]);
            std::memcpy(point_vector[j], output_tensor_data + j * output_shapes[1] * output_shapes[2],
                        output_shapes[0] * output_shapes[1] * output_shapes[2]);
        }
        delete output_tensor;
        yolov8_post.push_data_to_output_queue(point_vector, imgs, channel, index);
        printf("### Get data time: %.0f us, [%d]\n", time_end - time_start, i);
    }
    yolov8_post.set_mutidecode_is_finish();
    yolov8_post.end_yolov8_post_threads();
    return 0;
}

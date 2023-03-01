//
// Created by qinhua on 2/25/23.
//

#include <random>
#include "yolo.h"
#include "iostream"
#include "opencv2/opencv.hpp"

using namespace std;

static int num_decode_queue_size = 50;
static int after_preprocess_queue_size = 50;
static int after_process_queue_size = 50;

yolo::yolo() {

    cout << "CREATE YOLO " << endl;
}

yolo::~yolo() {
    cout << "DESTORY YOLO " << endl;
}

void yolo::set_finish() {
    finish_flag_ = true;
}

void yolo::push_video(vector<string> video_names, vector<int> dev_id_) {
    for (int i = 0; i < video_names.size(); i++) {
        decode_thread_pool.emplace_back(&yolo::decode_video, this, video_names[i], dev_id_[i]);
    }
    cout << " video list contain " + to_string(video_names.size()) << endl;
    cout << "start " + to_string(video_names.size()) + "decode threads" << endl;
}

void yolo::decode_video(string video_name, int dev_id) {
    sail::Decoder capture(video_name, true, dev_id);
//    cv::Mat temp_mat = cv::imread(video_name, cv::IMREAD_COLOR, 0);

    sail::Handle handle(dev_id);
    sail::Bmcv bmcv(handle);
    while (true) {
        sail::BMImage temp_bmimage;
        cv::Mat temp_mat;
        capture.read(handle, temp_bmimage);
        bmcv.bm_image_to_mat(temp_bmimage, temp_mat);
        unique_lock<mutex> lock(before_preprocess_queue_mutex);
        before_preprocess_queue_cv.wait(lock, [&] {
            return before_preprocess_queue.size() < num_decode_queue_size || finish_flag_;
        });
        before_preprocess_queue.push(move(temp_bmimage));
        before_preprocess_queue_cv.notify_all();
    }

}

void yolo::init_yolo_preprocess(bool letter_box_flag_, int resize_w_, int resize_h_, sail::PaddingAtrr &padding_in_,
                                crop_param &crop_param__) {
    letter_box_flag = letter_box_flag_;
    resize_w = resize_w_;
    resize_h = resize_h_;
    yolo_padding_in = move(padding_in_);
    yolo_crop_param = move(crop_param__);

}

void yolo::init_yolo_preprocess(bool letter_box_flag_, int resize_w_, int resize_h_) {
    letter_box_flag = letter_box_flag_;
    resize_w = resize_w_;
    resize_h = resize_h_;
}

void yolo::start_yolo_preprocess_threads(int num, vector<int> device_list) {
    for (int i = 0; i < num; ++i) {
        preprocess_thread_pool.emplace_back(&yolo::yolo_preproess_thread, this, device_list[i]);
    }
}

void yolo::end_yolo_thread() {
    for (int i = 0; i < preprocess_thread_pool.size(); i++) {
        preprocess_thread_pool[i].join();
    }
    for (int i = 0; i < process_thread_pool.size(); ++i) {
        process_thread_pool[i].join();
    }
    for (int i = 0; i < postprocess_thread_pool.size(); ++i) {
        postprocess_thread_pool[i].join();
    }
}

void yolo::yolo_preproess_thread(int dev_id) {
    while (true) {
        sail::Handle handle(dev_id);
        sail::Bmcv bmcv(handle);
        unique_lock<mutex> lock(before_preprocess_queue_mutex);
        before_preprocess_queue_cv.wait(lock, [&] { return !before_preprocess_queue.empty() || finish_flag_; });
        if (finish_flag_ == true) {
            break;
        }
        sail::BMImage temp_image(move(before_preprocess_queue.front()));
        cv::Mat ori_mat = bmcv.bm_image_to_mat(temp_image);

        before_preprocess_queue.pop();
        lock.unlock();
        before_preprocess_queue_cv.notify_all();
        if (letter_box_flag == true) {
            yolo_padding_in.dst_crop_stx = 0;
            yolo_padding_in.dst_crop_sty = 0;
            auto a = temp_image.width();
            auto b = temp_image.height();
            auto radio = std::max(temp_image.width() / 640.0, temp_image.height() / 640.0);
            yolo_padding_in.dst_crop_w =
                    temp_image.width() / radio;
            yolo_padding_in.dst_crop_h =
                    temp_image.height() / radio;
            yolo_padding_in.padding_b = 114;
            yolo_padding_in.padding_r = 114;
            yolo_padding_in.padding_g = 114;
            yolo_crop_param.crop_y0 = 0;
            yolo_crop_param.crop_x0 = 0;
            yolo_crop_param.crop_h = temp_image.height();
            yolo_crop_param.crop_w = temp_image.width();
        } else {
            yolo_padding_in.dst_crop_stx = 0;
            yolo_padding_in.dst_crop_sty = 0;
            yolo_padding_in.dst_crop_w = 640;
            yolo_padding_in.dst_crop_h = 640;
            yolo_padding_in.padding_b = 114;
            yolo_padding_in.padding_r = 114;
            yolo_padding_in.padding_g = 114;
            yolo_crop_param.crop_y0 = 0;
            yolo_crop_param.crop_x0 = 0;
            yolo_crop_param.crop_h = temp_image.height();
            yolo_crop_param.crop_w = temp_image.width();
        }
        sail::BMImage rgb_bmimg(handle, temp_image.height(), temp_image.width(), FORMAT_RGB_PLANAR,
                                DATA_TYPE_EXT_1N_BYTE);
        bmcv.convert_format(temp_image, rgb_bmimg);

        sail::BMImage resize_padding_bmimag(handle, 640, 640, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE);

        bmcv.vpp_crop_and_resize_padding(rgb_bmimg, resize_padding_bmimag, yolo_crop_param.crop_x0,
                                         yolo_crop_param.crop_y0,
                                         yolo_crop_param.crop_w,
                                         yolo_crop_param.crop_h,
                                         resize_w, resize_h, yolo_padding_in);
        bmcv.imwrite("./letter.jpg", resize_padding_bmimag);
        sail::BMImage convert_bmimg(handle, 640, 640, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_FLOAT32);
        bmcv.convert_to(resize_padding_bmimag, convert_bmimg,
                        make_tuple(make_pair(0.00392, 0), make_pair(0.00392, 0),
                                   make_pair(0.00392, 0)));


        sail::Tensor input_tensor(handle, {1, 3, 640, 640}, BM_FLOAT32, true, true);
        bmcv.bm_image_to_tensor(convert_bmimg, input_tensor);

        unique_lock<mutex> lock1(after_preprocess_queue_mutex);
        after_preprocess_queue_cv.wait(lock1, [&] {
            return finish_flag_ || after_preprocess_queue.size() < after_preprocess_queue_size;
        });
        sail_image_with_mat temp;
        temp.input_mat = move(ori_mat);
        temp.input_image = move(convert_bmimg);
        after_preprocess_queue.push(move(temp));
        after_preprocess_queue_cv.notify_all();
    }
}

void yolo::start_yolo_process_threads(int num, string bmodel_path, vector<int> device_list) {
    for (int i = 0; i < num; ++i) {
        preprocess_thread_pool.emplace_back(&yolo::yolo_process_thread, this, device_list[i], bmodel_path);
    }
}

void yolo::yolo_process_thread(int dev_id, string bmodel_path) {
    sail::Engine engine(dev_id);
    engine.load(bmodel_path);
    auto graph_name = engine.get_graph_names().front();
    auto input_name = engine.get_input_names(graph_name).front();
    auto output_name = engine.get_output_names(graph_name).front();
    std::vector<int> input_shape = {1, 3, 640, 640};
    std::map<std::string, std::vector<int>> input_shapes;
    input_shapes[input_name] = input_shape;
    auto output_shape = engine.get_output_shape(graph_name, output_name);
    auto input_dtype = engine.get_input_dtype(graph_name, input_name);
    auto output_dtype = engine.get_output_dtype(graph_name, output_name);

    // set io_mode SYSO:Both input and output tensors are in system memory.
    engine.set_io_mode(graph_name, sail::SYSO);

    // get handle to create input and output tensors
    sail::Handle handle = engine.get_handle();
    sail::Bmcv bmcv(handle);
    while (true) {
        unique_lock<mutex> lock(after_preprocess_queue_mutex);
        after_preprocess_queue_cv.wait(lock, [&] { return finish_flag_ || !after_preprocess_queue.empty(); });
        if (finish_flag_ == true)
            break;
        sail::BMImage input_image(move(after_preprocess_queue.front().input_image));
        cv::Mat ori_mat(move(after_preprocess_queue.front().input_mat));

        after_preprocess_queue.pop();
        lock.unlock();
        after_preprocess_queue_cv.notify_all();
        sail::Tensor output_tensor(handle, output_shape, output_dtype, true, true);
        sail::Tensor input_tensor(handle, input_shape, input_dtype, true, true);
        std::map<std::string, sail::Tensor *> input_tensors = {{input_name, &input_tensor}};
        std::map<std::string, sail::Tensor *> output_tensors = {{output_name, &output_tensor}};
        bmcv.bm_image_to_tensor(input_image, input_tensor);
        engine.process(graph_name, input_tensors, input_shapes, output_tensors);

        cv::Mat mat_(output_shape[1], output_shape[2], CV_32FC1, reinterpret_cast<float *>(output_tensor.sys_data()));
        auto mat = mat_.clone();

        vector<int> real_output_shape = engine.get_output_shape(graph_name, output_name);

        unique_lock<mutex> lock1(after_process_queue_mutex);
        after_process_queue_cv.wait(lock1, [&] {
            return finish_flag_ || after_process_queue.size() < after_process_queue_size;
        });
        after_process_queue.push({mat, real_output_shape, ori_mat});
        after_process_queue_cv.notify_all();
    }
}


void yolo::start_yolo_post_process_threads(int num, float modelScoreThreshold, float modelNMSThreshold) {
    for (int i = 0; i < num; ++i) {
        postprocess_thread_pool.emplace_back(&yolo::yolo_postprocess_thread, this, modelScoreThreshold,
                                             modelNMSThreshold);

    }
}


void yolo::yolo_postprocess_thread(float modelScoreThreshold, float modelNMSThreshold) {

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

    while (true) {
        std::vector<int> nms_result;
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        unique_lock<mutex> lock(after_process_queue_mutex);
        after_process_queue_cv.wait(lock, [&] { return finish_flag_ || !after_process_queue.empty(); });
        auto output_mat = after_process_queue.front().output_mat;
        vector<int> output_shapes = move(after_process_queue.front().output_shapes);
        cv::Mat ori_mat = move(after_process_queue.front().input_mat);
        after_process_queue.pop();
//        cv::Mat mat_(output_shapes[1], output_shapes[2], CV_32FC1, output_fp32);
        lock.unlock();
        after_process_queue_cv.notify_all();
        if (output_mat.cols > output_mat.rows) {
//            cv::Mat mat;
            cv::transpose(output_mat, output_mat);
            float *data = (float *) output_mat.data;
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
        } else if (output_mat.cols < output_mat.rows) {
//            cv::Mat mat = output_mat.clone();

            float *data = (float *) output_mat.data;

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

        NMSBoxes(boxes, confidences, nms_result, modelScoreThreshold, modelNMSThreshold);


        cout << "### NMS BOXS number : " + to_string(boxes.size()) << endl;
//        write_to_txt(boxes, nms_result, ori_mat);
        draw(ori_mat, nms_result, class_ids, confidences, boxes, classes);

    }


}

void yolo::write_to_txt(vector<cv::Rect> &boxes, vector<int> &nms_result, cv::Mat &ori_mat) {
    for (int i: nms_result) {
        if (letter_box_flag == false) {
            boxes[i].x = boxes[i].x / 640.0 * ori_mat.cols;
            boxes[i].y = boxes[i].y / 640.0 * ori_mat.rows;
            boxes[i].width = boxes[i].width * (float(ori_mat.cols) / 640.0);
            boxes[i].height = boxes[i].height * (float(ori_mat.rows) / float(640));
        } else {
            auto ratio_letter = min(640.0 / ori_mat.cols, 640.0 / ori_mat.rows);
            boxes[i].height = boxes[i].height / ratio_letter;
            boxes[i].width = boxes[i].width / ratio_letter;
            boxes[i].x = boxes[i].x / ratio_letter;
            boxes[i].y = boxes[i].y / ratio_letter;
        }

        string filename = "./data.txt";
        ifstream infile(filename);
        if (infile.good()) {
            // 文件存在，打开文件并定位到最后一行
            ofstream outfile(filename, ios_base::app);
            outfile << endl;
            outfile.close();
        } else {
            // 文件不存在，创建文件并打开
            ofstream outfile(filename);
            outfile.close();
        }
        ofstream outfile(filename, ios_base::app);
        outfile << "x : " + boxes[i].x << ", y :" << boxes[i].y << ", width :" << boxes[i].width << ", height : "
                << boxes[i].height;
        outfile << " ";
        outfile << endl;
        outfile.close();
    }

}

void yolo::NMSBoxes(std::vector<cv::Rect> &boxes, std::vector<float> &confidences, std::vector<int> &nms_result,
                    float modelScoreThreshold, float modelNMSThreshold) {
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


void yolo::draw(cv::Mat &image_ori, std::vector<int> &nms_result, std::vector<int> &class_ids,
                std::vector<float> &confidences,
                std::vector<cv::Rect> &boxes, vector<string> &classes) {
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
        if (letter_box_flag == false) {
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
    auto ran = std::rand();
    cv::imwrite("./" + to_string(ran) + ".jpg", image_ori);
}
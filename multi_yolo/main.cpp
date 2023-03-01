//
// Created by qinhua on 2/25/23.
//
#include "yolo.h"

int main() {
//    std::string bmodel_name = "/home/qinhua/models/yolov5/yolov5s_batch1_fp32_84X/compilation.bmodel";
    std::string bmodel_name = "/home/qinhua/models/yolov8/compilation/compilation.bmodel";
    std::vector <std::string> video_list;
    video_list.push_back("/home/qinhua/models/videos/001.mp4");
    video_list.push_back("/home/qinhua/models/videos/002.mp4");
    video_list.push_back("/home/qinhua/models/videos/003.mp4");
    video_list.push_back("/home/qinhua/models/videos/004.mp4");


    yolo net;
    net.push_video(video_list, {0, 0, 0, 0});
    net.init_yolo_preprocess(true, 640, 640);
    net.start_yolo_preprocess_threads(4, {0, 0, 0, 0, 0, 0, 0, 0, 0});
    net.start_yolo_process_threads(4, bmodel_name, {0, 0, 0, 0, 0, 0});
    net.start_yolo_post_process_threads(4, 0.45, 0.5);
    net.end_yolo_thread();
    net.set_finish();
}

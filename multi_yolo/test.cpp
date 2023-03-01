////
//// Created by qinhua on 2/25/23.
////
//#define USE_FFMPEG  1
//#define USE_OPENCV  1
//#define USE_BMCV    1
//#include "sail/cvwrapper.h"
//#include <engine.h>
//#include <tensor.h>
//#include <inireader.hpp>
//int main(){
//    sail::Handle handle(0);
//    sail::Bmcv bmcv(handle);
//    sail::Decoder decoder("/home/qinhua/libtorch-yolov5/images/bus.jpg", true, 0);
//    sail::BMImage img0(handle, 1080, 810, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE);
//    decoder.read(handle,img0);
//    cout << to_string(img0.width()) << to_string(img0.height()) <<endl;
//    bmcv.imwrite("./test.bmp",img0);
//
//
//}
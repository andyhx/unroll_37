#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"


namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
//   //hujun:
//  bool has_processed=true;
//  int input_image_h=bottom[0]->height();
//  int input_image_w=bottom[0]->width();
//  int input_image_channels=bottom[0]->channels();
//  const Dtype* image_data=bottom[0]->cpu_data()+2*bottom[0]->count(1,4);
//  cv::Mat input_image(input_image_h,input_image_w,CV_8UC1);
//  int image_data_index=0;
//  for (int h = 0; h < input_image_h; ++h) {
//      uchar* ptr = input_image.ptr<uchar>(h);
//      int img_index = 0;
//      for (int w = 0; w < input_image_w; ++w) {
//          for (int c = 0; c < input_image_channels; ++c) {
//
//              image_data_index = (c * input_image_h + h) * input_image_w + w;
//
//              // int image_data_index = (c * height + h) * width + w;
//              Dtype pixel = image_data[image_data_index];
//              if(has_processed){
//                  pixel=pixel*255;
//                  if(pixel<0){
//                      //                  std::cout << "unrolling_layer :pixel < 0" << std::endl;
//                      pixel=0;
//                  }
//                  if(pixel>255){
//                      //                std::cout << "unrolling_layer:pixel > 255" << std::endl;
//                      pixel=255;
//                  }
//              }
//              ptr[img_index++]=static_cast<uchar>(pixel);
//          }
//      }
//  }
//  cv::imshow("loss_img1",input_image);
//  cv::waitKey(0);
//  image_data=bottom[1]->cpu_data()+2*bottom[1]->count(1,4);
//  cv::Mat isomap_image(input_image_h,input_image_w,CV_8UC1);
//  image_data_index=0;
//  for (int h = 0; h < input_image_h; ++h) {
//      uchar* ptr = isomap_image.ptr<uchar>(h);
//      int img_index = 0;
//      for (int w = 0; w < input_image_w; ++w) {
//          for (int c = 0; c < input_image_channels; ++c) {
//
//              image_data_index = (c * input_image_h + h) * input_image_w + w;
//
//              // int image_data_index = (c * height + h) * width + w;
//              Dtype pixel = image_data[image_data_index];
//              if(has_processed){
//                  pixel=pixel*255;
//                  if(pixel<0){
//                      //                  std::cout << "unrolling_layer :pixel < 0" << std::endl;
//                      pixel=0;
//                  }
//                  if(pixel>255){
//                      //                std::cout << "unrolling_layer:pixel > 255" << std::endl;
//                      pixel=255;
//                  }
//              }
//              ptr[img_index++]=static_cast<uchar>(pixel);
//          }
//      }
//  }
//  cv::imshow("isomap_img1",isomap_image);
//  cv::waitKey(0);
//  //hujun:
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      //DLOG(INFO) << "top[0]->cpu_diff()[0]" << top[0]->cpu_diff()[0];
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);

}  // namespace caffe

#ifndef CAFFE_POOLING_LAYER_HPP_
#define CAFFE_POOLING_LAYER_HPP_


#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <vector>
#include <iostream>
#include <fstream>

#include <algorithm>
#include <cfloat>
#include <vector>

#include <future>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "caffe/util/math_functions.hpp"

#include "cuda_runtime.h"
#include "cublas_v2.h"

namespace caffe {

/**
 * @brief Pools the input image by taking the max, average, etc. within regions.
 *
 */
template <typename Dtype>
class Weight_pmLossLayer : public  LossLayer<Dtype> {
 public:
  explicit Weight_pmLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Weight_pmLoss"; }
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

/*  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }
*/
  void load_txt(cv::Mat& input,int height,int width,std::string file_loc,int type);
  void compute_alpha_beta_triangular_index();
  inline int ExactNumBottomBlobs() const { return 4; }
  std::string mean_shape_location;
  std::string shape_basis_location;
  std::string mean_exp_location;
  std::string exp_basis_location;
  std::string para_std_location;
  std::string para_mean_location;
  //triangle vertice index
  std::string tvi_location;
  std::string texcoord_location;
  //@ test_type==0 denotes we use our test method.
  int test_type;

  cv::Mat mean_shape,mean_exp,shape_basis,exp_basis,para_std,para_mean;
  cv::Mat duv_dtheta,s_duv_dtheta;
  //pid:199 point pexp:29 point
  //cv::Mat pid,pexp;
  //cv::Mat input_image;
  int isomap_h,isomap_w;
  int input_image_h,input_image_w,input_image_channels,input_image_num;
  int vertices_num;
  //to write image by the order of count_num;
  int count_num=0;
  // to reduce the number of vertices from 50000+ to 40000+
  std::vector<int> shape_index;
  //the show whether the input of this layer is gray or RGB image
  bool debug=true;
  Blob<Dtype> weight;
  int pm_len=0;
  float *d_a,*d_b,*d_c;
  cv::Mat merged_mat;
};
}  // namespace caffe

#endif  // CAFFE_POOLING_LAYER_HPP_

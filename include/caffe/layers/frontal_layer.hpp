#ifndef CAFFE_POOLING_LAYER_HPP_
#define CAFFE_POOLING_LAYER_HPP_


#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


#include "eos/render/utils.hpp"
#include "eos/render/texture_extraction.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <vector>
#include <iostream>
#include <fstream>

#include <algorithm>
#include <cfloat>
#include <vector>

#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "caffe/util/math_functions.hpp"

namespace caffe {

/**
 * @brief Pools the input image by taking the max, average, etc. within regions.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class FrontalLayer : public Layer<Dtype> {
 public:
  explicit FrontalLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  bool LoadPca();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Frontal"; }
/*
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  // MAX POOL layers can output an extra top blob for the mask;
  // others can only output the pooled inputs.
  virtual inline int MaxTopBlobs() const {
    return (this->layer_param_.pooling_param().pool() ==
            PoolingParameter_PoolMethod_MAX) ? 2 : 1;
  }
*/
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  void load_txt(cv::Mat& input,int height,int width,std::string file_loc,int type);
  cv::Mat frontal(eos::render::Mesh mesh,cv::Mat affine_matrix,cv::Mat image,int num);
  void frontal_backward_cpu(Dtype dV, const Dtype* U, const Dtype px,
		const Dtype py, Dtype& dpx, Dtype& dpy);
  cv::Mat interpolate_black_line(cv::Mat isomap);
  cv::Mat WeakProjection(cv::Mat,cv::Mat);
  void weight_pca_pm();
  void compute_alpha_beta_triangular_index();
  void compute_duv_dtheta();


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
  //cv::Mat alpha,beta,triangular_index_0,triangular_index_1,triangular_index_2,backward_affine_matrix;
  cv::Mat duv_dtheta,s_duv_dtheta;

  cv::Mat mean_lm2; 

  //pid:199 point pexp:29 point
  //cv::Mat pid,pexp;
  //cv::Mat input_image;
  int frontal_h,frontal_w;
  int input_image_h,input_image_w,input_image_channels,input_image_num;
  int vertices_num;
  int count_num=0;
  // to reduce the number of vertices from 50000+ to 40000+
  cv::Mat part_index;
  eos::render::Mesh mesh;
  //the show whether the input of this layer is gray or RGB image
  bool is_gray;
  bool has_processed;
  bool debug=true;
  Dtype n_times;
  vector<Dtype> norm_exp_vec;

  std::vector<int> im_index={17098, 17313, 16973, 16740, 30219, 31333, 32323, 32905, 33110, 33310, 33869, 34870, 36011, 28126, 27843, 27513, 27691, 28771, 28992, 29069, 29111, 29152, 29326, 29363, 29405, 29480, 29708, 8162, 8178, 8188, 8193, 6516, 7244, 8205, 9164, 9884, 2216, 3887, 4921, 5829, 4802, 3641, 10456 ,11354, 12384, 14067, 12654, 11493, 5523, 6026, 7496, 8216, 8936, 10396, 10796, 9556, 8837, 8237, 7637, 6916, 5910, 7385, 8224, 9065, 10538,8830,8230,7630};
//  std::vector<int> im_index{    21874,  22150,	21654,	21037,	43237,	44919,	46167,	47136,	47915,	48696,	49668,	50925,	52614,	33679,	33006,	32470,	32710,	38696,	39393,	39783,	39988,	40155,	40894,	41060,	41268,	41662,	42368,	8162,	8178,	8188,	8193,	6516,	7244,	8205,	9164,	9884,	2216,	3887,	4921,	5829,	4802,	3641,	10456,	11354,	12384,	14067,	12654,	11493,	5523,	6026,	7496,	8216,	8936,	10396,	10796,	9556,	8837,	8237,	7637,	6916,	5910,	7385,	8224,	9065,	10538,	8830,	8230,	7630};
  Blob<Dtype> input_grid; //// corresponding coordinate on input image after projection for each output pixel.
  Blob<Dtype> visible_grid;//// visiblility about above points
  Blob<Dtype> alpha,beta,triangular_index_0,triangular_index_1,triangular_index_2;
  //Blob<Dtype> duv_dtheta_blob,s_duv_dtheta_blob;
  float *d_a,*d_b,*d_c;
  cv::Mat merged_shape_exp;
  cv::Mat mean_sum_shape;
};

}  // namespace caffe

#endif  // CAFFE_POOLING_LAYER_HPP_


/*
 * this code takes as input the vertice,tvi,texcoords of 3d model and the tvi of the image,image.However actually,we can only get vertice,tvi,texcoords,affine_matrix of 3d model and image;
 */
/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: examples/fit-model.cpp
 *
 * Copyright 2016 Patrik Huber
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "caffe/layers/weight_pm_layer.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
using cv::Mat;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using std::vector;
using std::string;

/**
 * Reads an ibug .pts landmark file and returns an ordered vector with
 * the 68 2D landmark coordinates.
 *
 * @param[in] filename Path to a .pts file.
 * @return An ordered vector with the 68 ibug landmarks.
 */


/**
 * This app demonstrates estimation of the camera and fitting of the shape
 * model of a 3D Morphable Model from an ibug LFPW image with its landmarks.
 * In addition to fit-model-simple, this example uses blendshapes, contour-
 * fitting, and can iterate the fitting.
 *
 * 68 ibug landmarks are loaded from the .pts file and converted
 * to vertex indices using the LandmarkMapper.
 */
namespace caffe {

    using std::min;
    using std::max;
    template <typename Dtype>
        void Weight_pmLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top) {

	    //LOG(INFO)<< "what the fuck";
	    //this is important
	      LossLayer<Dtype>::LayerSetUp(bottom, top);
	    //this is important

            Weight_pmParameter weight_pm_param = this->layer_param_.weight_pm_param();

            mean_shape_location=weight_pm_param.mean_shape();
            shape_basis_location=weight_pm_param.shape_basis();
            mean_exp_location=weight_pm_param.mean_exp();
            exp_basis_location=weight_pm_param.exp_basis();

            vertices_num=weight_pm_param.vertices_num();
            tvi_location=weight_pm_param.tvi();
            std::ifstream read_tvi(tvi_location);


            if(weight_pm_param.has_para_std()){
                para_std_location=weight_pm_param.para_std();
                para_std=cv::Mat(1,242,CV_32FC1);
                std::ifstream read_para_std(para_std_location);
                float std_ite;
                int i=0;
                while(read_para_std>>std_ite){
                    para_std.at<Dtype>(0,i)=std_ite;
                    i++;
                }
            }
            if(weight_pm_param.has_para_mean()){
                para_mean_location=weight_pm_param.para_mean();
                para_mean=cv::Mat(1,242,CV_32FC1);
                std::ifstream read_para_mean(para_mean_location);
                float mean_ite;
                int i=0;
                while(read_para_mean>>mean_ite){
                    para_mean.at<Dtype>(0,i)=mean_ite;
                    i++;
                }
            }


            mean_shape=cv::Mat(vertices_num,1,CV_32FC1);
            load_txt(mean_shape,vertices_num,1,mean_shape_location,0);

            shape_basis=cv::Mat(vertices_num,199,CV_32FC1);
            load_txt(shape_basis,vertices_num,199,shape_basis_location,1);

            mean_exp=cv::Mat(vertices_num,1,CV_32FC1);
            load_txt(mean_exp,vertices_num,1,mean_exp_location,0);

            exp_basis=cv::Mat(vertices_num,29,CV_32FC1);
            load_txt(exp_basis,vertices_num,29,exp_basis_location,0);


	    //CV_ASSERT(A.cols == B.cols&&A.type() == B.type());
	    int totalCols = shape_basis.cols + exp_basis.cols;

	    merged_mat=cv::Mat(shape_basis.rows, totalCols, shape_basis.type());
	    Mat submat = merged_mat.colRange(0, shape_basis.cols);
	    shape_basis.copyTo(submat);
	    submat = merged_mat.colRange(shape_basis.cols, totalCols);
	    exp_basis.copyTo(submat);
		//std::cout << merged_mat.rows*merged_mat.cols << std::endl;
		//getchar();
			cudaMalloc((void**)&d_a,merged_mat.rows*merged_mat.cols*sizeof(float));
			cudaMalloc((void**)&d_b,228*sizeof(float));
			cudaMalloc((void**)&d_c,merged_mat.rows*1*sizeof(float));
			cublasSetVector(merged_mat.rows*merged_mat.cols,sizeof(float),merged_mat.ptr(),1,d_a,1);
/*
			status=cublasSetVector(mergedMat.rows*mergedMat.cols,sizeof(Dtype),mergedMat.ptr(),1,d_a,1);
			if(status!=CUBLAS_STATUS_SUCCESS){
				std::cout << "d_a " << std::endl;	
				getchar();
			}
*/

        }

    template <typename Dtype>
        void Weight_pmLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top) {
            // the channels of isomap is the same as input image.
            //std::cout << "check bottom[0]->channels==199" << bottom[0]->channels() << std::endl;
            //getchar(); 
	    //this is important
	    LossLayer<Dtype>::Reshape(bottom, top);
	    //this is important

            pm_len=bottom[0]->width();
            //std::cout << "pm_len_oh_no" << pm_len << std::endl;
            top[0]->Reshape(1, 1, 1,1);
            weight.Reshape(bottom[0]->num(),1,1,pm_len);
        }

 
    template <typename Dtype>
        void Weight_pmLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top) {
            //this is important
            bool debug=false;
	    //std::cout << "what hahahaahasdfasdfah";
            if(debug) std::cout << "beginning hahahah" << std::endl;
            Dtype* weight_data=weight.mutable_cpu_data();
            caffe_set(weight.count(),(Dtype).0,weight_data);
            const Dtype* gt_theta_data=bottom[0]->cpu_data();
            const Dtype* pid_data = bottom[1]->cpu_data();
            const Dtype* pexp_data=bottom[2]->cpu_data();
            const Dtype* affine_matrix_data=bottom[3]->cpu_data();
            Dtype* top_data=top[0]->mutable_cpu_data();


            for(int i_num=0;i_num<bottom[0]->num();i_num++){

		//clock_t start=clock();
                std::vector<std::future<void>> results;
                Dtype sum_w=0;

                        //cv::Mat gt_pid=cv::Mat::zeros(199,1,CV_32FC1);
			cv::Mat gt_228_theta=cv::Mat::zeros(228,1,CV_32FC1);
                        for(int i_p=0;i_p<199;i_p++){
                            gt_228_theta.at<Dtype>(i_p,0)=gt_theta_data[i_p+i_num*236];
                            //std::cout << "pid.at<Dtype>(i_p,0):" << pid.at<Dtype>(i_p,0) <<  std::endl;
                        } 

                        //cv::Mat gt_pexp=cv::Mat::zeros(29,1,CV_32FC1);

                        for(int i_exp=0;i_exp<29;i_exp++){
                            gt_228_theta.at<Dtype>(i_exp+199,0)=gt_theta_data[i_exp+i_num*236+199];
                        }
                        cv::Mat gt_affine_matrix=cv::Mat(2,4,CV_32FC1);

                        int affine_matrix_index=0;
                        //std::cout << "affine_matrix: ";
                        for(int i=0;i<2;i++){
                            for(int j=0;j<4;j++){
                                gt_affine_matrix.at<Dtype>(i,j)=gt_theta_data[affine_matrix_index+i_num*236+228];
                                ++affine_matrix_index;
                            }  
                        }


                        for(int i=0;i<199;i++){
                            gt_228_theta.at<Dtype>(i,0)=gt_228_theta.at<Dtype>(i,0)*para_std.at<Dtype>(0,i+14)+para_mean.at<Dtype>(0,i+14);
                        }
                        for(int i=0;i<29;i++){
                            gt_228_theta.at<Dtype>(i+199,0)=gt_228_theta.at<Dtype>(i+199,0)*para_std.at<Dtype>(0,i+213)+para_mean.at<Dtype>(0,i+213);
                        }

                        gt_affine_matrix.at<Dtype>(0,0)=gt_affine_matrix.at<Dtype>(0,0)*para_std.at<Dtype>(0,0)+para_mean.at<Dtype>(0,0);
                        gt_affine_matrix.at<Dtype>(0,1)=gt_affine_matrix.at<Dtype>(0,1)*para_std.at<Dtype>(0,1)+para_mean.at<Dtype>(0,1);
                        gt_affine_matrix.at<Dtype>(0,2)=gt_affine_matrix.at<Dtype>(0,2)*para_std.at<Dtype>(0,2)+para_mean.at<Dtype>(0,2);
                        gt_affine_matrix.at<Dtype>(0,3)=gt_affine_matrix.at<Dtype>(0,3)*para_std.at<Dtype>(0,3)+para_mean.at<Dtype>(0,3);
                        gt_affine_matrix.at<Dtype>(1,0)=gt_affine_matrix.at<Dtype>(1,0)*para_std.at<Dtype>(0,4)+para_mean.at<Dtype>(0,4);
                        gt_affine_matrix.at<Dtype>(1,1)=gt_affine_matrix.at<Dtype>(1,1)*para_std.at<Dtype>(0,5)+para_mean.at<Dtype>(0,5);
                        gt_affine_matrix.at<Dtype>(1,2)=gt_affine_matrix.at<Dtype>(1,2)*para_std.at<Dtype>(0,6)+para_mean.at<Dtype>(0,6);
                        gt_affine_matrix.at<Dtype>(1,3)=gt_affine_matrix.at<Dtype>(1,3)*para_std.at<Dtype>(0,7)+para_mean.at<Dtype>(0,7);
			//gpu acce
			cublasHandle_t gt_handle;
			cublasCreate(&gt_handle);

			cublasSetVector(gt_228_theta.rows*gt_228_theta.cols,sizeof(float),gt_228_theta.ptr(),1,d_b,1);

			cudaThreadSynchronize();
			float a=1,b=0;


			cublasSgemm (
			gt_handle, 
			CUBLAS_OP_T,	
			CUBLAS_OP_T,	
			merged_mat.rows, 
			gt_228_theta.cols,  
			merged_mat.cols,    
		 	&a,   
			d_a,   
			merged_mat.cols,    
			d_b,    
			gt_228_theta.cols,  
			&b,    
			d_c,    
			merged_mat.rows   
			);

			cudaThreadSynchronize();

			Dtype* h_c=(Dtype*)malloc(merged_mat.rows*merged_mat.cols*sizeof(float));
			cublasGetVector(merged_mat.rows*gt_228_theta.cols,sizeof(float),d_c,1,h_c,1);


			cv::Mat gt_mat_shape_exp(merged_mat.rows,gt_228_theta.cols,CV_32FC1);
			memcpy(gt_mat_shape_exp.ptr(),h_c,merged_mat.rows*sizeof(float));

			free(h_c);
			cublasDestroy (gt_handle);
			//end gpu 


			cv::Mat gt_vertex3d;
                        gt_vertex3d=mean_shape.clone();
                        gt_vertex3d+=mean_exp;
                        gt_vertex3d+=gt_mat_shape_exp;


                        cv::Mat gt_reshape_vertex=gt_vertex3d.reshape(0,vertices_num/3);
                        cv::Mat gt_trans_matr;
                        cv::transpose(gt_reshape_vertex,gt_trans_matr);
                        cv::Mat gt_vertex_matrix;
                        copyMakeBorder( gt_trans_matr, gt_vertex_matrix, 0, 1, 0, 0, cv::BORDER_CONSTANT,(Dtype)1.0  );
                        cv::Mat gt_U=gt_affine_matrix.clone()*gt_vertex_matrix;
                        //IplImage temp1=gt_U;




                for(int i_w=0;i_w<236;i_w++){


                        //std::cout << "i_num:" << i_num << "  i_w:" << i_w << std::endl;
                        cv::Mat es_theta(228,1,CV_32FC1);
                        //cv::Mat pid=cv::Mat::zeros(199,1,CV_32FC1);
                        for(int i_p=0;i_p<199;i_p++){
                            es_theta.at<Dtype>(i_p,0)=gt_theta_data[i_p+i_num*236];
                            if(i_p==i_w) es_theta.at<Dtype>(i_p,0)=pid_data[i_p+i_num*199]; 

                        } 

                        //cv::Mat pexp=cv::Mat::zeros(29,1,CV_32FC1);



                        for(int i_exp=0;i_exp<29;i_exp++){
                            es_theta.at<Dtype>(i_exp+199,0)=gt_theta_data[i_exp+i_num*236+199];
                            if((i_exp+199)==i_w) es_theta.at<Dtype>(i_exp+199,0)=pexp_data[i_exp+i_num*29]; 

                        }
                        cv::Mat affine_matrix=cv::Mat(2,4,CV_32FC1);

                        int affine_matrix_index=0;
                        //std::cout << "affine_matrix: ";
                        for(int i=0;i<2;i++){
                            for(int j=0;j<4;j++){
                                affine_matrix.at<Dtype>(i,j)=gt_theta_data[affine_matrix_index+i_num*236+228];
                                if(i_w==(228+i*4+j)) affine_matrix.at<Dtype>(i,j)=affine_matrix_data[affine_matrix_index+i_num*8];
                                ++affine_matrix_index;
                            }  
                        }


                        for(int i=0;i<199;i++){
                            es_theta.at<Dtype>(i,0)=es_theta.at<Dtype>(i,0)*para_std.at<Dtype>(0,i+14)+para_mean.at<Dtype>(0,i+14);
                        }
                        for(int i=0;i<29;i++){
                            es_theta.at<Dtype>(i+199,0)=es_theta.at<Dtype>(i+199,0)*para_std.at<Dtype>(0,i+213)+para_mean.at<Dtype>(0,i+213);
                        }

                        affine_matrix.at<Dtype>(0,0)=affine_matrix.at<Dtype>(0,0)*para_std.at<Dtype>(0,0)+para_mean.at<Dtype>(0,0);
                        affine_matrix.at<Dtype>(0,1)=affine_matrix.at<Dtype>(0,1)*para_std.at<Dtype>(0,1)+para_mean.at<Dtype>(0,1);
                        affine_matrix.at<Dtype>(0,2)=affine_matrix.at<Dtype>(0,2)*para_std.at<Dtype>(0,2)+para_mean.at<Dtype>(0,2);
                        affine_matrix.at<Dtype>(0,3)=affine_matrix.at<Dtype>(0,3)*para_std.at<Dtype>(0,3)+para_mean.at<Dtype>(0,3);
                        affine_matrix.at<Dtype>(1,0)=affine_matrix.at<Dtype>(1,0)*para_std.at<Dtype>(0,4)+para_mean.at<Dtype>(0,4);
                        affine_matrix.at<Dtype>(1,1)=affine_matrix.at<Dtype>(1,1)*para_std.at<Dtype>(0,5)+para_mean.at<Dtype>(0,5);
                        affine_matrix.at<Dtype>(1,2)=affine_matrix.at<Dtype>(1,2)*para_std.at<Dtype>(0,6)+para_mean.at<Dtype>(0,6);
                        affine_matrix.at<Dtype>(1,3)=affine_matrix.at<Dtype>(1,3)*para_std.at<Dtype>(0,7)+para_mean.at<Dtype>(0,7);
/*
			std::cout << affine_matrix.at<Dtype>(0,0)-gt_affine_matrix.at<Dtype>(0,0) << std::endl;
			std::cout << affine_matrix.at<Dtype>(0,1)-gt_affine_matrix.at<Dtype>(0,1) << std::endl;
			std::cout << affine_matrix.at<Dtype>(0,2)-gt_affine_matrix.at<Dtype>(0,2) << std::endl;
			std::cout << affine_matrix.at<Dtype>(0,3)-gt_affine_matrix.at<Dtype>(0,3) << std::endl;
			std::cout << affine_matrix.at<Dtype>(1,0)-gt_affine_matrix.at<Dtype>(1,0) << std::endl;
			std::cout << affine_matrix.at<Dtype>(1,1)-gt_affine_matrix.at<Dtype>(1,1) << std::endl;
			std::cout << affine_matrix.at<Dtype>(1,2)-gt_affine_matrix.at<Dtype>(1,2) << std::endl;
			std::cout << affine_matrix.at<Dtype>(1,3)-gt_affine_matrix.at<Dtype>(1,3) << std::endl;

			getchar();
*/
			//for gpu
			cublasStatus_t status;
			cublasHandle_t handle;
			status=cublasCreate(&handle);


			status=cublasSetVector(es_theta.rows*es_theta.cols,sizeof(float),es_theta.ptr(),1,d_b,1);
			if(status!=CUBLAS_STATUS_SUCCESS){
				std::cout << "d_b " << std::endl;	
				getchar();
			}
			cudaThreadSynchronize();
			float a=1,b=0;


			status=cublasSgemm (
			handle, 
			CUBLAS_OP_T,	
			CUBLAS_OP_T,	
			merged_mat.rows, 
			es_theta.cols,  
			merged_mat.cols,    
		 	&a,   
			d_a,   
			merged_mat.cols,    
			d_b,    
			es_theta.cols,  
			&b,    
			d_c,    
			merged_mat.rows   
			);
			if(status!=CUBLAS_STATUS_SUCCESS){
				std::cout << "cublasSgemm " << std::endl;	
				getchar();
			}
			
			cudaThreadSynchronize();

			h_c=(Dtype*)malloc(merged_mat.rows*es_theta.cols*sizeof(float));
			cublasGetVector(merged_mat.rows*es_theta.cols,sizeof(float),d_c,1,h_c,1);


			cv::Mat mat_shape_exp(merged_mat.rows,es_theta.cols,CV_32FC1);
			memcpy(mat_shape_exp.ptr(),h_c,merged_mat.rows*sizeof(float));


			free(h_c);
			cublasDestroy(handle);


                        cv::Mat vertex3d;
                        vertex3d=mean_shape.clone();
                        vertex3d+=mean_exp; 
                        vertex3d+=mat_shape_exp;
			//for(int i=0;i<50000;i++){if((vertex3d.at<Dtype>(i,0)-gt_vertex3d.at<Dtype>(i,0)>10000)) std::cout << gt_vertex3d.at<Dtype>(i,0) << std::endl;}
			//for(int i=34300*3;i<34350*3;i++){std::cout << vertex3d.at<Dtype>(i,0) << std::endl;}
			//getchar();
              

/*
			//test gpu
			cv::Mat true_vertex3d;
	                true_vertex3d=mean_shape.clone();
	                true_vertex3d+=mean_exp;
	                true_vertex3d+=merged_mat*es_theta;
			int ver_count=0;
			for(int i=0;i<10000;i++){
				if((true_vertex3d.at<Dtype>(i,0)-vertex3d.at<Dtype>(i,0))>1){
		 			std::cout << "i:"<< i<<" true-cublas:" << true_vertex3d.at<Dtype>(i,0)-vertex3d.at<Dtype>(i,0) << " " << std::endl;
					ver_count++;
				}
			}
			std::cout << ver_count <<  std::endl;
			getchar();			
			//end:test gpu
*/       
                        cv::Mat reshape_vertex=vertex3d.reshape(0,vertices_num/3);
                        cv::Mat trans_matr;
                        cv::transpose(reshape_vertex,trans_matr);


                        cv::Mat vertex_matrix;
                        copyMakeBorder( trans_matr, vertex_matrix, 0, 1, 0, 0, cv::BORDER_CONSTANT,(Dtype)1.0  );

                        cv::Mat U=affine_matrix.clone()*vertex_matrix;
/*
			std::cout << "U" << std::endl;
			for(int i_urow=0;i_urow<50050;i_urow++){
				if((U.at<Dtype>(1,i_urow)-gt_U.at<Dtype>(1,i_urow))>100)
				  std::cout << "i_urow:" << i_urow << "U.at<Dtype>(1,i_urow)" <<  U.at<Dtype>(1,i_urow) << " " << gt_U.at<Dtype>(1,i_urow) << "           " << std::endl;
			}
			std::cout << "end U " << std::endl;
   			getchar();
*/
                        IplImage temp=U;
			IplImage temp1=gt_U;
/*
			std::ofstream write_u("/home/brl/u.txt");
			for(int j_u=0;j_u<U.cols;j_u++){for(int i_u=0;i_u<U.rows;i_u++) write_u << U.at<Dtype>(i_u,j_u) << " ";}

			std::ofstream write_gt_u("/home/brl/gt_u.txt");
			for(int j_u=0;j_u<U.cols;j_u++){for(int i_u=0;i_u<U.rows;i_u++) write_gt_u << gt_U.at<Dtype>(i_u,j_u) << " ";}
			write_u.close();
			write_gt_u.close();
			getchar();
*/
                        weight_data[i_num*236+i_w]=cvNorm((CvArr*)&temp,(CvArr*)&temp1,CV_L2);
			//std::cout << "weight_data[i_num*236+i_w]" << weight_data[i_num*236+i_w] << std::endl;


                        if(debug) {std::cout << weight_data[i_num*236+i_w] << std::endl;}
                        sum_w+=weight_data[i_num*236+i_w];
                } 
  
                for(int i_su=0;i_su<236;i_su++){
		    //std::cout << "sum_w:" << sum_w << std::endl;
		    //getchar();
                    weight_data[i_num*236+i_su]=weight_data[i_num*236+i_su]/sum_w;
                    if(debug) std::cout << "weight_data:"<< weight_data[i_num*236+i_su]<<  " sum_w:" <<sum_w << std::endl;
                }
                Dtype loss_sum=0;
                for(int i_top=0;i_top<199;i_top++){
                    loss_sum+=weight_data[i_num*236+i_top]*(gt_theta_data[i_num*236+i_top]-pid_data[i_num*199+i_top])*
                        (gt_theta_data[i_num*236+i_top]-pid_data[i_num*199+i_top]);
                }
                for(int i_top=0;i_top<29;i_top++){
                    loss_sum+=weight_data[i_num*236+i_top+199]*(gt_theta_data[i_num*236+i_top+199]-pexp_data[i_num*28+i_top])*
                        (gt_theta_data[i_num*236+i_top+199]-pexp_data[i_num*28+i_top]);
                }
                for(int i_top=0;i_top<8;i_top++){
                    loss_sum=weight_data[i_num*236+i_top+228]*(gt_theta_data[i_num*236+i_top+228]-affine_matrix_data[i_num*8+i_top])*(gt_theta_data[i_num*236+i_top+228]-affine_matrix_data[i_num*8+i_top]);
                }
                if(debug) std::cout << "loss_num:"  << loss_sum << std::endl;
                top_data[0]+=loss_sum;
	        //clock_t finish=clock();
	        //std::cout << "time per iter" << (Dtype)(finish-start)/CLOCKS_PER_SEC << std::endl;
            }
	    top_data[0]=top_data[0]/bottom[0]->num();
	    //top[0]->mutable_cpu_diff()[0]=top_data[0];

	    //std::cout << "loss in weight pm" << top_data[0] << std::endl;
        }

    template<typename Dtype>
        void Weight_pmLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
            debug=false;
            string prefix = "\t\t Backward_cpu: \t";
		
            if(debug) std::cout<<prefix<<"Starting!"<<std::endl;

            const Dtype* weight_data=weight.cpu_data();
            const Dtype* top_diff_data = top[0]->cpu_diff();

            Dtype* d_shape=bottom[1]->mutable_cpu_diff();
            Dtype* d_exp=bottom[2]->mutable_cpu_diff();
            Dtype* d_m=bottom[3]->mutable_cpu_diff();


            const Dtype* gt_theta_data=bottom[0]->cpu_data();
            const Dtype* pid_data = bottom[1]->cpu_data();
            const Dtype* pexp_data=bottom[2]->cpu_data();
            const Dtype* affine_matrix_data=bottom[3]->cpu_data();

            //caffe_set(bottom[0]->count(), (Dtype)0, dU);
            caffe_set(bottom[1]->count(), (Dtype)0, d_shape);
            caffe_set(bottom[2]->count(), (Dtype)0, d_exp);
            caffe_set(bottom[3]->count(), (Dtype)0, d_m);
            // for each image in batch
            for(int i_num = 0; i_num  < bottom[0]->num(); ++i_num ) {
		//std::cout << "top_diff_data[0]" << top_diff_data[0] << std::endl;
                for(int i_diff=0;i_diff<199;i_diff++){
                    d_shape[i_num*199+i_diff]=-2*top_diff_data[0]*weight_data[i_num*236+i_diff]*(gt_theta_data[i_num*236+i_diff]-pid_data[i_num*199+i_diff])/bottom[0]->num();
		    if(debug) std::cout << (gt_theta_data[i_num*236+i_diff]-pid_data[i_num*199+i_diff]) << " ";
                }
                for(int i_diff=0;i_diff<29;i_diff++){
                    d_exp[i_num*29+i_diff]=-2*top_diff_data[0]*weight_data[i_num*236+199+i_diff]*(gt_theta_data[i_num*236+i_diff+199]-pexp_data[i_num*29+i_diff])/bottom[0]->num();
		    //std::cout << (gt_theta_data[i_num*236+i_diff+199]-pexp_data[i_num*29+i_diff]) << " ";
                }
                for(int i_diff=0;i_diff<8;i_diff++){
                    d_m[i_num*8+i_diff]=-2*top_diff_data[0]*weight_data[i_num*236+228+i_diff]*(gt_theta_data[i_num*236+i_diff+228]-affine_matrix_data[i_num*8+i_diff])/bottom[0]->num();
                }
            }

        }

#ifdef CPU_ONLY
    STUB_GPU(Weight_pmLossLayer);
#endif

    INSTANTIATE_CLASS(Weight_pmLossLayer);
    REGISTER_LAYER_CLASS(Weight_pmLoss); 


    // namespace caffe
    template<typename Dtype>
        void Weight_pmLossLayer<Dtype>::load_txt(cv::Mat& input,int height,int width,std::string file_loc,int type)
        {
            //@hujun type==1 denote we load this txt by pthread

            if(type==1){
                std::vector<std::future<void>> results;
                //while(1==1){} //@hujun just for multi-thread testing.
                for (int i=1;i<=width;i++) {

                    // Note: If there's a performance problem, there's no need to capture the whole mesh - we could capture only the three required vertices with their texcoords.
                    auto read_txt = [i, &input, height, file_loc]() {
                        std::string final_filename;
                        if (i<10){
                            final_filename=file_loc+"00"+std::to_string(i)+".txt";
                        }
                        else if (i>=10 && i<100){
                            final_filename=file_loc+"0"+std::to_string(i)+".txt";
                        }
                        else{
                            final_filename=file_loc+std::to_string(i)+".txt";
                        }

                        std::ifstream fts(final_filename);
                        Dtype num=0;
                        int j=0;
                        while(fts>>num){
                            //std::cout <<"type==1  j: " << j <<"i: " << i <<  std::endl;
                            input.at<Dtype>(j,i-1)=num;
                            //if(j==0) {std::cout << "i: " << i << " num: " << num << std::endl;std::cout << input.at<Dtype>(j,i-1) << std::endl; }
                            ++j;
                        }
                        if(fts.is_open()) fts.close();
                        CHECK((j<=height));

                    }; // end lambda auto extract_triangle();
                    results.emplace_back(std::async(read_txt));
                } 
                // Collect all the launched tasks:
                for (auto&& r : results) {
                    r.get();
                }    
                return;

            }
            else{
                int i=0,j=0;
                Dtype num;
                std::ifstream read_txt(file_loc); 
                while(read_txt>>num){
                    //std::cout <<"type==0  j: " << j <<"  i: " << i   << std::endl;
                    input.at<Dtype>(j,i)=num;
                    ++j;
                    if(j==height){j=0;++i;}
                }
                if(read_txt.is_open()) read_txt.close();
                CHECK((i<=width));
            }
        }
}//namespace caffe

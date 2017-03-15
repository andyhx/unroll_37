/*
 * 
 * this code takes as input the vertice,tvi,texcoords of 3d model and the tvi of the image,image.However actually,we can only get vertice,tvi,texcoords,affine_matrix of 3d model and image;
 */

#include "caffe/layers/frontal_layer.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace eos;
using cv::Mat;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using std::vector;
using std::string;

namespace caffe {

    using std::min;
    using std::max;
    template <typename Dtype>
        void FrontalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top) {

            FrontalParameter frontal_param = this->layer_param_.frontal_param();

            frontal_h=frontal_param.frontal_h();
            frontal_w=frontal_param.frontal_w();
            mean_shape_location=frontal_param.mean_shape();
            shape_basis_location=frontal_param.shape_basis();
            mean_exp_location=frontal_param.mean_exp();
            exp_basis_location=frontal_param.exp_basis();
            test_type=frontal_param.test_type();
            vertices_num=frontal_param.vertices_num();
            CHECK((frontal_param.has_is_gray())) << "you must tell us whether the input image is gray or not";
            is_gray=frontal_param.is_gray();  

            std::string part_index_location=frontal_param.part_index();
            std::string shape_basis_index_location=frontal_param.shape_basis_index();
            std::string mean_lm2_location=frontal_param.mean_lm2();

            has_processed=frontal_param.has_processed();


            tvi_location=frontal_param.tvi();
            std::ifstream read_tvi(tvi_location);
            float float_num,float_num1,float_num2;

            while(read_tvi>>float_num){
                read_tvi>>float_num1;read_tvi>>float_num2;
                mesh.tvi.push_back(std::array<int, 3>({static_cast<int>(float_num),static_cast<int>(float_num1),static_cast<int>(float_num2)}));
            }
            if(read_tvi.is_open()) read_tvi.close();

            //maybe use it in the future
            if(frontal_param.has_para_std()){
                para_std_location=frontal_param.para_std();
                para_std=cv::Mat(1,242,CV_32FC1);
                std::ifstream read_para_std(para_std_location);
                float std_ite;
                int i=0;
                while(read_para_std>>std_ite){
                    para_std.at<Dtype>(0,i)=std_ite;
                    i++;
                }
            }
            if(frontal_param.has_para_mean()){
                para_mean_location=frontal_param.para_mean();
                para_mean=cv::Mat(1,242,CV_32FC1);
                std::ifstream read_para_mean(para_mean_location);
                float mean_ite;
                int i=0;
                while(read_para_mean>>mean_ite){
                    para_mean.at<Dtype>(0,i)=mean_ite;
                    i++;
                }
            }
            //end:maybe use it in the future

            cv::Mat ori_mean_shape=cv::Mat(vertices_num,1,CV_32FC1);
            load_txt(ori_mean_shape,vertices_num,1,mean_shape_location,0);

            cv::Mat ori_shape_basis=cv::Mat(vertices_num,199,CV_32FC1);
            load_txt(ori_shape_basis,vertices_num,199,shape_basis_location,1);

            cv::Mat ori_mean_exp=cv::Mat(vertices_num,1,CV_32FC1);
            load_txt(ori_mean_exp,vertices_num,1,mean_exp_location,0);

            cv::Mat ori_exp_basis=cv::Mat(vertices_num,29,CV_32FC1);
            load_txt(ori_exp_basis,vertices_num,29,exp_basis_location,0);

            cv::Mat shape_basis_index=cv::Mat(109071,1,CV_32FC1);
            load_txt(shape_basis_index,109071,1,shape_basis_index_location,0);

            part_index=cv::Mat(36357,1,CV_32FC1);
            load_txt(part_index,36357,1,part_index_location,0);

            mean_lm2=cv::Mat(2,68,CV_32FC1);
            load_txt(mean_lm2,2,68,mean_lm2_location,0);

            for(int i_m=0;i_m<68;i_m++){
                mean_lm2.at<Dtype>(1,i_m)=128+1-mean_lm2.at<Dtype>(1,i_m);
            }
            //  shape basis and exp basis reduce points 53215--->36357
            mean_shape=cv::Mat(36357*3,1,CV_32FC1);
            mean_exp=cv::Mat(36357*3,1,CV_32FC1);
            shape_basis=cv::Mat(36357*3,199,CV_32FC1);
            exp_basis=cv::Mat(36357*3,29,CV_32FC1);
            for(int i=0; i < shape_basis_index.rows; i++){
                mean_shape.at<Dtype>(i, 0) = ori_mean_shape.at<Dtype>(shape_basis_index.at<Dtype>(i, 0)-1, 0);
                mean_exp.at<Dtype>(i, 0) = ori_mean_exp.at<Dtype>(shape_basis_index.at<Dtype>(i, 0)-1, 0);
                for (int k=0; k<199; k++){
                    shape_basis.at<Dtype>(i, k) = ori_shape_basis.at<Dtype>(shape_basis_index.at<Dtype>(i, 0)-1, k);
                } 
                for (int k=0; k<29; k++){
                    exp_basis.at<Dtype>(i, k) = ori_exp_basis.at<Dtype>(shape_basis_index.at<Dtype>(i, 0)-1, k);
                } 
            }

            mean_sum_shape=mean_shape.clone()+mean_exp.clone();

            // add 199 and 29 to 228 cols
            int totalCols=shape_basis.cols+exp_basis.cols;
            merged_shape_exp=cv::Mat(shape_basis.rows, totalCols, shape_basis.type());
            Mat submat = merged_shape_exp.colRange(0, shape_basis.cols);
            shape_basis.copyTo(submat);
            submat = merged_shape_exp.colRange(shape_basis.cols, totalCols);
            exp_basis.copyTo(submat);
            //std::cout << "merged_shape_exp.rows" << merged_shape_exp.rows << std::endl;
            //std::cout << "merged_shape_exp.cols" << merged_shape_exp.cols << std::endl;
            //getchar();
            cudaMalloc((void**)&d_a,merged_shape_exp.rows*merged_shape_exp.cols*sizeof(float));
            cudaMalloc((void**)&d_b,228*sizeof(float));
            cudaMalloc((void**)&d_c,merged_shape_exp.rows*1*sizeof(float));
            cublasSetVector(merged_shape_exp.rows*228,sizeof(float),merged_shape_exp.ptr(),1,d_a,1);
        }

/*
    template<typename Dtype>
        void FrontalLayer<Dtype>::compute_alpha_beta_triangular_index(){
            alpha=cv::Mat(frontal_h,frontal_w,CV_32FC1);
            beta=cv::Mat(frontal_h,frontal_w,CV_32FC1);
            triangular_index_0=cv::Mat::zeros(frontal_h,frontal_w,CV_32FC1);
            triangular_index_1=cv::Mat::zeros(frontal_h,frontal_w,CV_32FC1);
            triangular_index_2=cv::Mat::zeros(frontal_h,frontal_w,CV_32FC1);
            for(int i=0;i<mesh.tvi.size();i++){
                cv::Point2f dst_tri[3];
                dst_tri[0] = cv::Point2f((frontal_w)*mesh.texcoords[mesh.tvi[i][0]-1][0], (frontal_h)*mesh.texcoords[mesh.tvi[i][0]-1][1] );
                dst_tri[1] = cv::Point2f((frontal_w)*mesh.texcoords[mesh.tvi[i][1]-1][0], (frontal_h)*mesh.texcoords[mesh.tvi[i][1]-1][1] );
                dst_tri[2] = cv::Point2f((frontal_w)*mesh.texcoords[mesh.tvi[i][2]-1][0], (frontal_h)*mesh.texcoords[mesh.tvi[i][2]-1][1] );
                for (int x = min(dst_tri[0].x, min(dst_tri[1].x, dst_tri[2].x)); x < max(dst_tri[0].x, max(dst_tri[1].x, dst_tri[2].x)); ++x) {
                    for (int y = min(dst_tri[0].y, min(dst_tri[1].y, dst_tri[2].y)); y < max(dst_tri[0].y, max(dst_tri[1].y, dst_tri[2].y)); ++y) {
                        if (eos::render::detail::is_point_in_triangle   (cv::Point2f(x, y), dst_tri[0], dst_tri[1], dst_tri[2])) {
                            Dtype fenzi=(x-dst_tri[0].x)*(dst_tri[2].y-dst_tri[0].y)-(y-dst_tri[0].y)*(dst_tri[2].x-dst_tri[0].x);
                            Dtype fenmu=(dst_tri[1].x-dst_tri[0].x)*(dst_tri[2].y-dst_tri[0].y)-(dst_tri[1].y-dst_tri[0].y)*(dst_tri[2].x-dst_tri[0].x);
                            alpha.at<Dtype>(x,y)= fenzi/fenmu;

                            Dtype beta_fenzi=(y-dst_tri[0].y)*(dst_tri[1].x-dst_tri[0].x)-(x-dst_tri[0].x)*(dst_tri[1].y-dst_tri[0].y);
                            //  Dtype beta_fenmu=(dst_tri[1].x-dst_tri[0].x)*(dst_tri[2].y-dst_tri[0].y)-(dst_tri[1].y-dst_tri[0].y)*(dst_tri[2].x-dst_tri[0].x);
                            beta.at<Dtype>(x,y)=beta_fenzi/fenmu;
                            triangular_index_0.at<Dtype>(x,y)=static_cast<Dtype>(mesh.tvi[i][0]-1);
                            triangular_index_1.at<Dtype>(x,y)=static_cast<Dtype>(mesh.tvi[i][1]-1);
                            triangular_index_2.at<Dtype>(x,y)=static_cast<Dtype>(mesh.tvi[i][2]-1);
                        }
                    }
                }
            }

               std::string basic_dir="/home/scw4750/github/frontal/zero/test_backward";
               std::ofstream write_alpha(basic_dir+"/alpha.txt");
               std::ofstream write_beta(basic_dir+"/beta.txt");
               std::ofstream write_triangular_0(basic_dir+"/triangular_0.txt");
               std::ofstream write_triangular_1(basic_dir+"/triangular_1.txt");
               std::ofstream write_triangular_2(basic_dir+"/triangular_2.txt");
               for(int i_h=0;i_h<frontal_h;i_h++){
               for(int i_w=0;i_w<frontal_w;i_w++){
               write_alpha<< alpha.at<Dtype>(i_h,i_w) << " ";
               write_beta<< beta.at<Dtype>(i_h,i_w)<< " ";
               write_triangular_0<< triangular_index_0.at<Dtype>(i_h,i_w)<< " ";
               write_triangular_1<< triangular_index_1.at<Dtype>(i_h,i_w)<< " ";
               write_triangular_2<< triangular_index_2.at<Dtype>(i_h,i_w)<< " ";
               }
               }

        }
*/
    template <typename Dtype>
        void FrontalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top) {
            CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
                << "corresponding to (num, channels, height, width)";
            input_image_num=bottom[0]->num();
            input_image_channels = bottom[0]->channels();
            input_image_h= bottom[0]->height();
            input_image_w = bottom[0]->width();


            // the channels of frontal is the same as input image.
            top[0]->Reshape(bottom[0]->num(), input_image_channels, frontal_h,frontal_w);
            //assume the input image must be gray image for convience.
            input_grid.Reshape(bottom[0]->num(),2,frontal_h,frontal_w);
            visible_grid.Reshape(bottom[0]->num(),1,frontal_h,frontal_w);
            alpha.Reshape(bottom[0]->num(),1,frontal_h,frontal_w);
            beta.Reshape(bottom[0]->num(),1,frontal_h,frontal_w);
            triangular_index_0.Reshape(bottom[0]->num(),1,frontal_h,frontal_w);
            triangular_index_1.Reshape(bottom[0]->num(),1,frontal_h,frontal_w);
            triangular_index_2.Reshape(bottom[0]->num(),1,frontal_h,frontal_w);

        }


    template <typename Dtype>
        void FrontalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top) {
            //this is important
            //std::cout << "forward_cpu beging hahahahahaha" << std::endl;
            Dtype* input_grid_data = input_grid.mutable_cpu_data();
            caffe_set(input_grid.count(), (Dtype)0, input_grid_data);
            Dtype* visible_grid_data=visible_grid.mutable_cpu_data();
            caffe_set(visible_grid.count(),(Dtype)0,visible_grid_data);

	    Dtype* alpha_data=alpha.mutable_cpu_data();
	    caffe_set(alpha.count(),(Dtype)0,alpha_data);
	    Dtype* beta_data=beta.mutable_cpu_data();
	    caffe_set(beta.count(),(Dtype)0,beta_data);
	    Dtype* triangular_index_0_data=triangular_index_0.mutable_cpu_data();
	    caffe_set(triangular_index_0.count(),(Dtype)0,triangular_index_0_data);

	    Dtype* triangular_index_1_data=triangular_index_1.mutable_cpu_data();
	    caffe_set(triangular_index_1.count(),(Dtype)0,triangular_index_1_data);

	    Dtype* triangular_index_2_data=triangular_index_2.mutable_cpu_data();
	    caffe_set(triangular_index_2.count(),(Dtype)0,triangular_index_2_data);


/*
            Dtype* alpha_data=alpha.mutable_cpu_data();
            caffe_set(alpha.count(),(Dtype)0,alpha_data);
            Dtype* beta_data=beta.mutable_cpu_data();
            caffe_set(beta.count(),(Dtype)0,beta_data);
	    Dtype* tri_data=triangular_index_0.mutable_cpu_data();
	    caffe_set(triangular_index_0.count(),(Dtype)0,tri_data);
	    tri_data=triangular_index_1.mutable_cpu_data();
	    caffe_set(triangular_index_1.count(),(Dtype)0,tri_data);
	    tri_data=triangular_index_2.mutable_cpu_data();
	    caffe_set(triangular_index_2.count(),(Dtype)0,tri_data);
*/
            const Dtype* pid_data = bottom[1]->cpu_data();
            const Dtype* pexp_data=bottom[2]->cpu_data();
            const Dtype* affine_matrix_data=bottom[3]->cpu_data();
            //this is important
            for(int i_num=0;i_num<input_image_num;i_num++){

                //clock_t forward_start, forward_finish;
                //forward_start=clock();


                //this is important  
                mesh.vertices.clear();
                mesh.texcoords.clear();
                //end:this is important




                //cv::Mat pid=cv::Mat::zeros(199,1,CV_32FC1);
                cv::Mat es_theta=cv::Mat::zeros(228,1,CV_32FC1);
                for(int i=0;i<199;i++){
                    es_theta.at<Dtype>(i,0)=pid_data[i+i_num*199]; 
                    //std::cout << es_theta.at<Dtype>(i.0) << std::endl;
                } 



                //cv::Mat pexp=cv::Mat::zeros(29,1,CV_32FC1);
                for(int i=0;i<29;i++){
                    es_theta.at<Dtype>(i+199,0)=pexp_data[i+i_num*29]; 
                }

                cv::Mat affine_matrix=cv::Mat(3,4,CV_32FC1);
                int affine_matrix_index=0;
                //std::cout << "affine_matrix: ";
                for(int i=0;i<3;i++){
                    for(int j=0;j<4;j++){
                        if(i<2){
                            affine_matrix.at<Dtype>(i,j)=affine_matrix_data[affine_matrix_index+i_num*8];
                            //std::cout << affine_matrix_data[affine_matrix_index] <<" ";
                            ++affine_matrix_index;
                        }
                        else{
                            affine_matrix.at<Dtype>(i,j)=0;
                        }	
                    }  
                }
                affine_matrix.at<Dtype>(2,3)=1;


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
                float a=1;
                float b=0;

                status=cublasSgemm(
                        handle,
                        CUBLAS_OP_T,
                        CUBLAS_OP_T,
                        merged_shape_exp.rows,
                        es_theta.cols,
                        merged_shape_exp.cols,
                        &a,
                        d_a,
                        merged_shape_exp.cols,    
                        d_b,
                        es_theta.cols,//es_theta.cols,
                        &b,
                        d_c,
                        merged_shape_exp.rows// es_theta.cols,
                        );
                if(status!=CUBLAS_STATUS_SUCCESS){
                    std::cout << "sgemm " << std::endl;	
                    getchar();
                }
                cudaThreadSynchronize();

                /*
                   float* h_a=(float*)malloc(sizeof(float)*100);
                   cublasGetVector(100,sizeof(float),d_a,100,h_a,1);
                   for(int i=0;i<100;i++){
                   std::cout << h_a[i] << " ";
                   }
                   getchar();
                   */


                float* h_c=(float*)malloc(merged_shape_exp.rows*es_theta.cols*sizeof(float));
                cublasGetVector(merged_shape_exp.rows*es_theta.cols,sizeof(float),d_c,1,h_c,1);


                cv::Mat mat_shape_exp(merged_shape_exp.rows,es_theta.cols,CV_32FC1);
                memcpy(mat_shape_exp.ptr(),h_c,merged_shape_exp.rows*sizeof(float));

                free(h_c);
                cublasDestroy(handle);

                cv::Mat vertex3d;
                vertex3d=mean_sum_shape.clone();
                vertex3d+=mat_shape_exp;
                //end for gpu


                Dtype first_vertex,second_vertex,third_vertex;
                int i=0;
                //std::ofstream write_vertex("/home/brl/vertex.txt");
                while(i<vertex3d.rows){
                    first_vertex=vertex3d.at<Dtype>(i,0);
                    //write_vertex << first_vertex << " ";
                    //std::cout << first_vertex << std::endl;
                    //getchar();
                    ++i;
                    second_vertex=vertex3d.at<Dtype>(i,0);
                    //write_vertex << second_vertex << " ";
                    ++i;
                    third_vertex=vertex3d.at<Dtype>(i,0);
                    //write_vertex << third_vertex << " ";
                    ++i;
                    //if( (i/3==21874)) std::cout << first_vertex << " " << second_vertex << " " << third_vertex << std::endl; 
                    mesh.vertices.push_back(glm::tvec4<float>(first_vertex,second_vertex, third_vertex,1));
                }
                //need a project to get affine matrix to fit our goal 




                // compute new camera projection parameters for fit the EOS code
                cv::Mat src_lm3=cv::Mat::zeros(68,3,CV_32FC1);
                cv::Mat dst_lm2=cv::Mat::zeros(68,2,CV_32FC1);

                Vec3f res;
                //cv::Mat temp_affine_matrix = eos::render::detail::calculate_affine_z_direction(affine_matrix).clone();
                cv::Mat temp_affine_matrix=affine_matrix.clone();
                for(int vertices_index=0;vertices_index<68;vertices_index++){
                    /*
                       std::cout << "mesh.vertices[im_index[vertices_index]-1][0]" << mesh.vertices[im_index[vertices_index]-1][0] << std::endl;
                       std::cout << "mesh.vertices[im_index[vertices_index]-1][1]" << mesh.vertices[im_index[vertices_index]-1][1] << std::endl;
                       std::cout << "mesh.vertices[im_index[vertices_index]-1][2]" << mesh.vertices[im_index[vertices_index]-1][2] << std::endl;
                       getchar();
                       */
                    src_lm3.at<Dtype>(vertices_index,0)=mesh.vertices[im_index[vertices_index]-1][0];
                    src_lm3.at<Dtype>(vertices_index,1)=mesh.vertices[im_index[vertices_index]-1][1];
                    src_lm3.at<Dtype>(vertices_index,2)=mesh.vertices[im_index[vertices_index]-1][2];
                    cv::Vec4f temp(mesh.vertices[im_index[vertices_index]-1][0],mesh.vertices[im_index[vertices_index]-1][1],mesh.vertices[im_index[vertices_index]-1][2],1);
                    res = Mat(temp_affine_matrix.clone() * Mat(temp));
                    dst_lm2.at<Dtype>(vertices_index,0)=res[0];
                    dst_lm2.at<Dtype>(vertices_index,1)=input_image_h-res[1]+1;
                }
                /*
                   for(int i=0;i<68;i++){
                   std::cout << dst_lm2.at<Dtype>(i,0) << " " << dst_lm2.at<Dtype>(i,1) << " ";
                   }

                   getchar();
                   */   
                cv::Mat new_affine_matrix=WeakProjection(src_lm3,dst_lm2);
                cv::Mat final_affine_matrix= cv::Mat::zeros(3,4,CV_32FC1);
                for(int i=0;i<2;++i){
                    for(int j=0;j<4;++j){
                        final_affine_matrix.at<Dtype>(i,j)=new_affine_matrix.at<Dtype>(i,j);
                    }
                }
                final_affine_matrix.at<Dtype>(2,3)=1;
                //end need


                // new projected shape in new frontal image
                cv::Mat frontal_affine_matrix=WeakProjection(src_lm3, mean_lm2.t());
/*
                std::cout << frontal_affine_matrix.row(0) << std::endl;
                std::cout << frontal_affine_matrix.row(1) << std::endl;
                
                   std::cout << src_lm3.col(0) << std::endl;
                   std::cout << src_lm3.col(1) << std::endl;
                   std::cout << src_lm3.col(2) << std::endl;
                   std::cout << mean_lm2.row(0) << std::endl;
                   std::cout << mean_lm2.row(1) << std::endl;
                   */
                cv::Mat homo_shape = vertex3d.reshape(0, 36357);
//                std::cout << homo_shape.rows<< "  "<< homo_shape.cols<< std::endl;

                cv::Mat temp_ones = cv::Mat::ones(homo_shape.rows, 1, CV_32FC1);
                cv::hconcat(homo_shape, temp_ones, homo_shape);
/*
                std::cout<< homo_shape.row(0)<<std::endl;
                std::cout<< homo_shape.row(1)<<std::endl;
                std::cout<< homo_shape.row(36356)<<std::endl;
*/
                cv::Mat projected_shape = frontal_affine_matrix * homo_shape.t();

                /*
                   texcoord_location=frontal_param.texcoord();
                   std::ifstream read_texcoord(texcoord_location);
                   float num,num1;
                   mesh.texcoords.clear();//as the same reason above
                   while(read_texcoord>>num){
                   read_texcoord>>num1;
                   mesh.texcoords.push_back(glm::tvec2<float>(num,num1));
                   }
                   if(read_texcoord.is_open()) read_texcoord.close();
                   */

                std::ofstream write_shape("/home/scw4750/github/shape.txt");
                for(int i_c=0;i_c<36357;i_c++){
                    Dtype coord_x=projected_shape.at<Dtype>(0,i_c);
                    Dtype coord_y=projected_shape.at<Dtype>(1,i_c);
                    //std::cout << coord_x << " " << coord_y << std::endl;
                    //getchar();
                    write_shape<< coord_x << " " << coord_y << " ";
                    mesh.texcoords.push_back(glm::tvec2<float>(coord_x,coord_y));
                }
/*
                cv::Mat test_project=cv::Mat::zeros(128,128,CV_32FC1);
                for(int i_pr=0;i_pr<36357;i_pr++){
                    cv::circle(test_project,cv::Point(projected_shape.at<Dtype>(0,i_pr),projected_shape.at<Dtype>(1,i_pr)),1, cv::Scalar(255,255,255),CV_FILLED, 8,0);
                    //cv::circle(test_project,cv::Point(mean_lm2.at<Dtype>(0,i_pr),mean_lm2.at<Dtype>(1,i_pr)),1, cv::Scalar(255,255,255),CV_FILLED, 8,0);
                }
                cv::imshow("test",test_project);
                cv::waitKey(0);
                write_shape.close();
*/
                if(!is_gray){
                    /*
                    //do frontal and write the result to top. 
                    for(int i=0;i<input_image_num;i++){
                    const Dtype* image_data=bottom[0]->cpu_data();
                    input_image=cv::Mat(input_image_h,input_image_w,CV_8UC3);
                    int image_data_index=0;
                    for (int h = 0; h < input_image_h; ++h) {
                    uchar* ptr = input_image.ptr<uchar>(h);
                    int img_index = 0;
                    for (int w = 0; w < input_image_w; ++w) {
                    for (int c = 0; c < input_image_channels; ++c) {

                    image_data_index = (c * input_image_h + h) * input_image_w + w;

                    // int image_data_index = (c * height + h) * width + w;
                    Dtype pixel = image_data[image_data_index];
                    if(has_processed){
                    pixel=pixel*128+127.5;
                    if(pixel<0){
                    std::cout << "frontal_layer :pixel < 0" << std::endl;
                    pixel=0;
                    }
                    if(pixel>255){
                    std::cout << "frontal_layer:pixel > 0" << std::endl;
                    pixel=255;
                    }
                    }
                    ptr[img_index++]=static_cast<uchar>(pixel);
                    }

                    }
                    }
                    //cv::imshow("tet",input_image);
                    //cv::waitKey(0);
                    cv::Mat frontal=frontal(mesh,final_affine_matrix,input_image,i);

                    //cv::Mat transformed_frontal=cv::Mat::zeros(frontal_h,frontal_w,CV_8UC3);
                    //cv::cvtColor(frontal,transformed_frontal,CV_RGBA2RGB);

                    Dtype* transformed_data = top[0]->mutable_cpu_data()+i*top[0]->count(1,4);
                    int top_index;
                    for (int h = 0; h < frontal_h; ++h) {
                    const uchar* ptr = frontal.ptr<uchar>(h);
                    int img_index = 0;
                    for (int w = 0; w < frontal_w; ++w) {

                    for (int c = 0; c < input_image_channels; ++c) {

                    top_index = (c * frontal_h + h) * frontal_w + w;

                    // int top_index = (c * height + h) * width + w;
                    Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
                    transformed_data[top_index] = pixel/255.0 ;
                    //the reason why I do this because the channel of frontal is 4
                    if(c==2) ++img_index;
                    }

                    }
                    }

                    }
                    */
                }
                else{
                    //do frontal and write the result to top. 
                    const Dtype* image_data=bottom[0]->cpu_data()+i_num*bottom[0]->count(1,4);
                    cv::Mat input_image(input_image_h,input_image_w,CV_8UC1);
                    int image_data_index=0;
                    for (int h = 0; h < input_image_h; ++h) {
                        uchar* ptr = input_image.ptr<uchar>(h);
                        int img_index = 0;
                        for (int w = 0; w < input_image_w; ++w) {
                            for (int c = 0; c < input_image_channels; ++c) {

                                image_data_index = (c * input_image_h + h) * input_image_w + w;

                                // int image_data_index = (c * height + h) * width + w;
                                Dtype pixel = image_data[image_data_index];
                                if(has_processed){
                                    pixel=pixel*128+127.5;
                                    if(pixel<0){
                                        //                  std::cout << "frontal_layer :pixel < 0" << std::endl;
                                        pixel=0;
                                    }
                                    if(pixel>255){
                                        //                std::cout << "frontal_layer:pixel > 255" << std::endl;
                                        pixel=255;
                                    }
                                }
                                ptr[img_index++]=static_cast<uchar>(pixel);
                            }
                        }
                    }
                    //cv::imshow("img",input_image);
                    //cv::waitKey(0);
                    cv::Mat temp(input_image_h,input_image_w,CV_8UC3); cv::cvtColor(input_image,temp,CV_GRAY2RGB); cv::Mat to_image=std::move(temp);
                    //cv::imshow("input_image",to_image);
                    //cv::waitKey(0);
                    //clock_t frontal_start=clock();
                    cv::Mat frontal_img=frontal(mesh,final_affine_matrix,to_image,i_num);
                    /*
                       count_num++;
                       if(count_num%50==0){ 
                       cv::imshow("frontal",frontal);
                       cv::waitKey(0);
                       }
                       */
                    //cv::Mat transformed_frontal=cv::Mat::zeros(frontal_h,frontal_w,CV_8UC3);
                    //cv::cvtColor(frontal,transformed_frontal,CV_RGBA2RGB);
                    Dtype* transformed_data = top[0]->mutable_cpu_data()+i_num*top[0]->count(1,4);
                    int top_index;
                    for (int h = 0; h < frontal_h; ++h) {
                        const uchar* ptr = frontal_img.ptr<uchar>(h);
                        int img_index = 0;
                        for (int w = 0; w < frontal_w; ++w) {

                            for (int c = 0; c < input_image_channels; ++c) {

                                top_index = (c * frontal_h + h) * frontal_w + w;

                                // int top_index = (c * height + h) * width + w;
                                Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
                                transformed_data[top_index] = pixel/255.0 ;
                                //the reason why I do this because the channel of frontal is 4
                                img_index=img_index+3;
                            }

                        }
                    }


                }

                //forward_finish=clock();
                //std::cout << "forward_cpu time/iteration:" << double(forward_finish-forward_start)/ CLOCKS_PER_SEC << std::endl;
            }  
        }

    template<typename Dtype>
        void FrontalLayer<Dtype>::frontal_backward_cpu(Dtype dV,const Dtype* U,const Dtype px,const Dtype py,Dtype& dpx,Dtype& dpy){

            //string prefix = "\t\tSpatial Transformer Layer:: transform_backward_cpu: \t";
            //if(debug) std::cout<<prefix<<"Starting!"<<std::endl;
            // position (x,y)
            Dtype x = px;
            Dtype y = py;
            //if(debug) std::cout<<prefix<<"(x, y) = ("<<x<<", "<<y<<")"<<std::endl;

            for(int m = floor(x); m <= ceil(x); ++m){
                for(int n = floor(y); n <= ceil(y); ++n) {
                    //if(debug) std::cout<<prefix<<"(m, n) = ("<<m<<", "<<n<<")"<<std::endl;
                    //if(debug) std::cout << "m * input_image_w + n:" << m * input_image_w + n << std::endl;
                    if(m >= 0 && m < input_image_h && n >= 0 && n < input_image_w) {
                        //dpx += caffe_sign<Dtype>(m - x) * (1 - abs(y - n)) * U[m * input_image_w + n] * dV ;
                        //dpy += caffe_sign<Dtype>(n - y) * (1 - abs(x - m)) * U[m * input_image_w + n] * dV ;
                        //if((m * input_image_w + n)>=10000) std::cout <<"exceed 1w and m:n:input_image_w" << m << " " << n << " " << input_image_w << std::endl; 
                        //std::cout << "U[m * input_image_w + n]" << U[m * input_image_w + n] << std::endl;
                        dpx += caffe_sign<Dtype>(m - x) * (1 - abs(y - n)) * U[m * input_image_w + n] * dV ;
                        dpy += caffe_sign<Dtype>(n - y) * (1 - abs(x - m)) * U[m * input_image_w + n] * dV ;
                    }
                }
            }
            //if(debug) std::cout<<prefix<<"Finished."<<std::endl;
        }

    template<typename Dtype>
        void FrontalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

            if(propagate_down[1]==true){
                debug=true;
                /*
                   int write_temp_variable=false;

                   std::string basic_dir="/home/scw4750/github/frontal/zero/test_backward";
                   std::ofstream write_pid(basic_dir+"/pid.txt");
                   std::ofstream write_pexp(basic_dir+"/pexp.txt");
                   std::ofstream write_pm(basic_dir+"/pm.txt");
                   std::ofstream write_duv_dtheta(basic_dir+"/duv_dtheta.txt");
                   std::ofstream write_dv(basic_dir+"/dv.txt");
                   */
                string prefix = "\t\tSpatial Transformer Layer:: Backward_cpu: \t";

                if(debug) std::cout<<prefix<<"Starting!"<<std::endl;

                const Dtype* dV = top[0]->cpu_diff();
                Dtype* image_data= bottom[0]->mutable_cpu_data();
                caffe_scal(bottom[0]->count(),(Dtype)128.0,image_data);
                caffe_add_scalar(bottom[0]->count(),(Dtype)127.5,image_data);
                caffe_scal(bottom[0]->count(),(Dtype)0.003921,image_data);

                //Dtype* dU = bottom[0]->mutable_cpu_diff();
                Dtype* d_shape=bottom[1]->mutable_cpu_diff();
                Dtype* d_exp=bottom[2]->mutable_cpu_diff();
                Dtype* d_m=bottom[3]->mutable_cpu_diff();

                //caffe_set(bottom[0]->count(), (Dtype)0, dU);
                caffe_set(bottom[1]->count(), (Dtype)0, d_shape);
                caffe_set(bottom[2]->count(), (Dtype)0, d_exp);
                caffe_set(bottom[3]->count(), (Dtype)0, d_m);
                // for each image in batch

                for(int i_num = 0; i_num  < input_image_num; ++i_num ) {
                    //clock_t backward_start=clock();
                    const Dtype* pid_data = bottom[1]->cpu_data();
                    cv::Mat pid=cv::Mat::zeros(199,1,CV_32FC1);
                    //int pid_index=0;
                    for(int i=0;i<199;i++){
                        pid.at<Dtype>(i,0)=pid_data[i+i_num*199]; 
                        //if(write_temp_variable) write_pid << pid.at<Dtype>(i,0) << " ";
                    } 
                    const Dtype* pexp_data=bottom[2]->cpu_data();
                    cv::Mat pexp=cv::Mat::zeros(29,1,CV_32FC1);

                    for(int i=0;i<29;i++){
                        pexp.at<Dtype>(i,0)=pexp_data[i+i_num*29]; 
                        //if(write_temp_variable) write_pexp << pexp.at<Dtype>(i,0) << " ";

                    }
                    const Dtype* affine_matrix_data=bottom[3]->cpu_data();
                    cv::Mat affine_matrix=cv::Mat(2,4,CV_32FC1);
                    int affine_matrix_index=0;
                    for(int i=0;i<2;i++){
                        for(int j=0;j<4;j++){
                            affine_matrix.at<Dtype>(i,j)=affine_matrix_data[affine_matrix_index+i_num*8];
                            //if(write_temp_variable) write_pm <<affine_matrix_data[affine_matrix_index+i_num*8]  << " ";


                            ++affine_matrix_index;
                        }  
                    }

                    //   cv::Mat vertex3d;

                    for(int i=0;i<199;i++){
                        pid.at<Dtype>(i,0)=pid.at<Dtype>(i,0)*para_std.at<Dtype>(0,i+14)+para_mean.at<Dtype>(0,i+14);
                    }
                    for(int i=0;i<29;i++){
                        pexp.at<Dtype>(i,0)=pexp.at<Dtype>(i,0)*para_std.at<Dtype>(0,i+213)+para_mean.at<Dtype>(0,i+213);
                    }

                    affine_matrix.at<Dtype>(0,0)=affine_matrix.at<Dtype>(0,0)*para_std.at<Dtype>(0,0)+para_mean.at<Dtype>(0,0);
                    affine_matrix.at<Dtype>(0,1)=affine_matrix.at<Dtype>(0,1)*para_std.at<Dtype>(0,1)+para_mean.at<Dtype>(0,1);
                    affine_matrix.at<Dtype>(0,2)=affine_matrix.at<Dtype>(0,2)*para_std.at<Dtype>(0,2)+para_mean.at<Dtype>(0,2);
                    affine_matrix.at<Dtype>(0,3)=affine_matrix.at<Dtype>(0,3)*para_std.at<Dtype>(0,3)+para_mean.at<Dtype>(0,3);
                    affine_matrix.at<Dtype>(1,0)=affine_matrix.at<Dtype>(1,0)*para_std.at<Dtype>(0,4)+para_mean.at<Dtype>(0,4);
                    affine_matrix.at<Dtype>(1,1)=affine_matrix.at<Dtype>(1,1)*para_std.at<Dtype>(0,5)+para_mean.at<Dtype>(0,5);
                    affine_matrix.at<Dtype>(1,2)=affine_matrix.at<Dtype>(1,2)*para_std.at<Dtype>(0,6)+para_mean.at<Dtype>(0,6);
                    affine_matrix.at<Dtype>(1,3)=affine_matrix.at<Dtype>(1,3)*para_std.at<Dtype>(0,7)+para_mean.at<Dtype>(0,7);
                    //std::cout << "affine_matrix"<<std::endl;

                    cv::Mat s_duv_dtheta=cv::Mat(part_index.rows,236*2,CV_32FC1);
                    for(int iv=0;iv<part_index.rows;iv++){
                        cv::Mat dvertex_u = cv::Mat(1,236,CV_32FC1);
                        cv::Mat dvertex_v = cv::Mat(1,236,CV_32FC1);
                        Dtype sum_1 = 0;
                        Dtype sum_2 = 0;
                        Dtype sum_3 = 0;
                        for(int ia=0;ia<199;ia++){
                            Dtype a1 = shape_basis.at<Dtype>(iv*3,ia);
                            Dtype a2 = shape_basis.at<Dtype>(iv*3+1,ia);
                            Dtype a3 = shape_basis.at<Dtype>(iv*3+2,ia);
                            s_duv_dtheta.at<Dtype>(iv,ia)=a1*affine_matrix.at<Dtype>(0,0) + a2*affine_matrix.at<Dtype>(0,1)+a3*affine_matrix.at<Dtype>(0,2);
                            s_duv_dtheta.at<Dtype>(iv,ia+236)= -(a1*affine_matrix.at<Dtype>(1,0) + a2*affine_matrix.at<Dtype>(1,1)+a3*affine_matrix.at<Dtype>(1,2));
                            sum_1 = sum_1 + shape_basis.at<Dtype>(iv*3,ia) * pid.at<Dtype>(ia, 0);
                            sum_2 = sum_2 + shape_basis.at<Dtype>(iv*3+1,ia) * pid.at<Dtype>(ia, 0);
                            sum_3 = sum_3 + shape_basis.at<Dtype>(iv*3+2,ia) * pid.at<Dtype>(ia, 0);

                        }
                        for(int ia=0;ia<29;ia++){
                            Dtype a1 = exp_basis.at<Dtype>(iv*3,ia);
                            Dtype a2 = exp_basis.at<Dtype>(iv*3+1,ia);
                            Dtype a3 = exp_basis.at<Dtype>(iv*3+2,ia);
                            s_duv_dtheta.at<Dtype>(iv,199 + ia) = a1*affine_matrix.at<Dtype>(0,0) + a2*affine_matrix.at<Dtype>(0,1)+a3*affine_matrix.at<Dtype>(0,2);
                            s_duv_dtheta.at<Dtype>(iv,435+ ia) = -(a1*affine_matrix.at<Dtype>(1,0) + a2*affine_matrix.at<Dtype>(1,1)+a3*affine_matrix.at<Dtype>(1,2));
                            sum_1 = sum_1 + exp_basis.at<Dtype>(iv*3,ia) * pexp.at<Dtype>(ia, 0);
                            sum_2 = sum_2 + exp_basis.at<Dtype>(iv*3+1,ia) * pexp.at<Dtype>(ia, 0);
                            sum_3 = sum_3 + exp_basis.at<Dtype>(iv*3+2,ia) * pexp.at<Dtype>(ia, 0);
                        }
                        for(int ia=0;ia<8;ia++){
                            Dtype a1 = mean_sum_shape.at<Dtype>(iv*3,0)  + sum_1;
                            Dtype a2 = mean_sum_shape.at<Dtype>(iv*3+1,0)  + sum_2;
                            Dtype a3 = mean_sum_shape.at<Dtype>(iv*3+2,0)  +sum_3;

                            s_duv_dtheta.at<Dtype>(iv, 228) = a1; 
                            s_duv_dtheta.at<Dtype>(iv, 229) = a2; 
                            s_duv_dtheta.at<Dtype>(iv, 230) = a3; 
                            s_duv_dtheta.at<Dtype>(iv, 231) = 1;  
                            s_duv_dtheta.at<Dtype>(iv, 232) = 0;  
                            s_duv_dtheta.at<Dtype>(iv, 233) = 0;  
                            s_duv_dtheta.at<Dtype>(iv, 234) = 0;  
                            s_duv_dtheta.at<Dtype>(iv, 235) = 0;  
                            s_duv_dtheta.at<Dtype>(iv, 228+236) = 0;
                            s_duv_dtheta.at<Dtype>(iv, 229+236) = 0;
                            s_duv_dtheta.at<Dtype>(iv, 230+236) = 0;
                            s_duv_dtheta.at<Dtype>(iv, 231+236) = 0;
                            s_duv_dtheta.at<Dtype>(iv, 232+236) = -a1;
                            s_duv_dtheta.at<Dtype>(iv, 233+236) = -a2;
                            s_duv_dtheta.at<Dtype>(iv, 234+236) = -a3;
                            s_duv_dtheta.at<Dtype>(iv, 235+236) = -1;
			    //std::cout << s_duv_dtheta.at<Dtype>(iv, 228) << std::endl;
			    //	getchar();
                        }

                    }

                    Dtype px, py, dpx, dpy, delta_dpx, delta_dpy;

                    int count=0;
                   // cv::Mat all_dpx=cv::Mat::zeros(frontal_h,frontal_w,CV_32FC1);
                    //cv::Mat all_dpy=cv::Mat::zeros(frontal_h,frontal_w,CV_32FC1);
                    //int zero_count=0;
                    for(int i_h = 0; i_h < frontal_h; ++i_h){
                        for(int i_w = 0; i_w < frontal_w; ++i_w) {
                            //std::cout << visible_grid.data_at(i_num,0,i_h,i_w) << std::endl;
                            /*
                               if(write_temp_variable){
                               write_dv<<dV[top[0]->offset(i_num, 0, i_h, i_w)] << " ";
                               }
                               */
                            if(visible_grid.data_at(i_num,0,i_h,i_w)!=0){
                                //if(true){
				
                                count+=1;
                                //std::cout << "dv" << dV[top[0]->offset(i_num, 0, i_h, i_w)] << " ";
                                for(int j = 0; j<1; ++j) {
                                    //if(debug) std::cout << "i_bum:" << i_num << "  i_h:" << i_h << "   i_w:" << i_w  << "  j:" << j << std::endl;  
                                    dpx=0;
                                    dpy=0;
                                    px=input_grid.data_at(i_num,0,i_h,i_w);
                                    py=input_grid.data_at(i_num,1,i_h,i_w);

                                    delta_dpx = delta_dpy = (Dtype)0;
                                    //if(debug) std::cout << "bottom[0]->count :" << bottom[0]->count() << "bottom[0]->offset:" << bottom[0]->offset(i_num, j, 0, 0) << std::endl;
                                    frontal_backward_cpu(dV[top[0]->offset(i_num, j, i_h, i_w)], image_data + bottom[0]->offset(i_num, j, 0, 0),px, py, delta_dpx, delta_dpy);
                                    //if(abs(dV[top[0]->offset(i_num, j, i_h, i_w)])<0.0000001) zero_count++;
                                    dpx = delta_dpx;
                                    dpy = delta_dpy;
                                    //std::cout << "alpha" << alpha.data_at(i_num,0,i_h,i_w) << " beta" << beta.data_at(i_num,0,i_h,i_w);
                                    //all_dpx.at<Dtype>(i_h,i_w)=dpx;
                                    //all_dpy.at<Dtype>(i_h,i_w)=dpy;
                                    Dtype d_w_x_d_ui=(1-alpha.data_at(i_num,0,i_h,i_w)-beta.data_at(i_num,0,i_h,i_w));
                                    //Dtype d_w_y_d_ui=0;
                                    Dtype d_w_x_d_uj=alpha.data_at(i_num,0,i_h,i_w);
                                    //Dtype d_w_y_d_uj=0;
                                    Dtype d_w_x_d_uk=beta.data_at(i_num,0,i_h,i_w);
                                    //Dtype d_w_y_d_uk=0;
                                    //Dtype d_w_x_d_vi=0;
                                    Dtype d_w_y_d_vi=(1-alpha.data_at(i_num,0,i_h,i_w)-beta.data_at(i_num,0,i_h,i_w));
                                    //Dtype d_w_x_d_vj=0;
                                    Dtype d_w_y_d_vj=alpha.data_at(i_num,0,i_h,i_w);
                                    //Dtype  d_w_x_d_vk=0;
                                    Dtype d_w_y_d_vk=beta.data_at(i_num,0,i_h,i_w);
                                    int i_tri_index=static_cast<int>(triangular_index_0.data_at(i_num,0,i_h,i_w)-1);
                                    int j_tri_index=static_cast<int>(triangular_index_1.data_at(i_num,0,i_h,i_w)-1);
                                    int k_tri_index=static_cast<int>(triangular_index_2.data_at(i_num,0,i_h,i_w)-1);
                                    for(int i_shape=0;i_shape<199;i_shape++){
                                        Dtype d_w_x_dpid=
                                            d_w_x_d_ui*s_duv_dtheta.at<Dtype>(i_tri_index,i_shape)+
                                            d_w_x_d_uj*s_duv_dtheta.at<Dtype>(j_tri_index,i_shape)+
                                            d_w_x_d_uk*s_duv_dtheta.at<Dtype>(k_tri_index,i_shape);
                                        Dtype d_w_y_dpid=
                                            d_w_y_d_vi*s_duv_dtheta.at<Dtype>(i_tri_index,i_shape+236)+
                                            d_w_y_d_vj*s_duv_dtheta.at<Dtype>(j_tri_index,i_shape+236)+
                                            d_w_y_d_vk*s_duv_dtheta.at<Dtype>(k_tri_index,i_shape+236);
                                        d_shape[i_num*199+i_shape]+=(dpx*d_w_x_dpid+dpy*d_w_y_dpid);
                                    }

                                    
                                    for(int i_exp=0;i_exp<29;i_exp++){
                                        Dtype d_w_x_dpexp=
                                            d_w_x_d_ui*s_duv_dtheta.at<Dtype>(i_tri_index,i_exp+199)+
                                            d_w_x_d_uj*s_duv_dtheta.at<Dtype>(j_tri_index,i_exp+199)+
                                            d_w_x_d_uk*s_duv_dtheta.at<Dtype>(k_tri_index,i_exp+199);
                                        Dtype d_w_y_dpexp=
                                            d_w_y_d_vi*s_duv_dtheta.at<Dtype>(i_tri_index,i_exp+236+199)+
                                            d_w_y_d_vj*s_duv_dtheta.at<Dtype>(j_tri_index,i_exp+236+199)+
                                            d_w_y_d_vk*s_duv_dtheta.at<Dtype>(k_tri_index,i_exp+236+199);
                                        d_exp[i_num*29+i_exp]+=(dpx*d_w_x_dpexp+dpy*d_w_y_dpexp);
                                    }

                                    for(int i_m=0;i_m<8;i_m++){
                                        Dtype d_w_x_dm=
                                            d_w_x_d_ui*s_duv_dtheta.at<Dtype>(i_tri_index,i_m+228)+
                                            d_w_x_d_uj*s_duv_dtheta.at<Dtype>(j_tri_index,i_m+228)+
                                            d_w_x_d_uk*s_duv_dtheta.at<Dtype>(k_tri_index,i_m+228);
                                        Dtype d_w_y_dm=
                                            d_w_y_d_vi*s_duv_dtheta.at<Dtype>(i_tri_index,i_m+236+228)+
                                            d_w_y_d_vj*s_duv_dtheta.at<Dtype>(j_tri_index,i_m+236+228)+
                                            d_w_y_d_vk*s_duv_dtheta.at<Dtype>(k_tri_index,i_m+236+228);
                                        d_m[i_num*8+i_m]+=(dpx*d_w_x_dm+dpy*d_w_y_dm);
                                    }

                                }
                            }
                            }
                        }

                        //clock_t backward_finish=clock();
                        //std::cout << "backward time/iter" << double(backward_finish-backward_start)/ CLOCKS_PER_SEC << std::endl;
                        /*
                           std::ofstream write_visible(basic_dir+"/visible.txt");
                           std::ofstream write_dpx(basic_dir+"/dpx.txt");
                           std::ofstream write_dpy(basic_dir+"/dpy.txt");
                           for(int i_dv_h=0;i_dv_h<frontal_h;i_dv_h++){
                           for(int i_dv_w=0;i_dv_w<frontal_w;i_dv_w++){
                           write_visible<< visible_grid.data_at(i_num,0,i_dv_h,i_dv_w) << " ";
                           write_dpx << all_dpx.at<Dtype>(i_dv_h,i_dv_w) << " ";
                           write_dpy << all_dpy.at<Dtype>(i_dv_h,i_dv_w) << " ";
                           }
                           }

                           write_visible.close();
                           write_dpx.close();
                           write_dpy.close();
                           */
                        if(debug){ std::cout << "visible points:" << count << std::endl;}//std::cout << "zero_count" << zero_count << std::endl;}
                    for(int i_shape=0;i_shape<199;i_shape++){
                        d_shape[i_num*199+i_shape]=d_shape[i_num*199+i_shape]*para_std.at<Dtype>(0,i_shape+14);
                        //d_shape[i_num*199+i_shape]=d_shape[i_num*199+i_shape];
                    }	
                    for(int i_exp=0;i_exp<29;i_exp++){
                        d_exp[i_num*29+i_exp]=d_exp[i_num*29+i_exp]*para_std.at<Dtype>(0,i_exp+213);
                        //d_exp[i_num*29+i_exp]=d_exp[i_num*29+i_exp];

                    }
                    for(int i_m=0;i_m<8;i_m++){
                        d_m[i_num*8+i_m]=d_m[i_num*8+i_m]*para_std.at<Dtype>(0,i_m);
                        //d_m[i_num*8+i_m]=d_m[i_num*8+i_m];
                    }	
                }
                if(debug){
                    for(int i = 0; i < input_image_num; ++i) {
                        std::cout << std::endl;
                        std::cout << std::endl;
			//for(int i_diff=0;i_diff<128*128;i_diff){
	                //        std::cout << " top_diff: " << dV[i*128*128+i_diff] << " ";
			//}
                        std::cout << std::endl;
                        std::cout << std::endl;
                        std::cout << " shape" << std::endl;
                        for(int i_shape=0;i_shape<199;i_shape++){

                            std::cout << d_shape[i*199+i_shape]<<",";
                        }	
                        std::cout << std::endl << " exp:" << std::endl;
                        for(int i_exp=0;i_exp<29;i_exp++){

                            std::cout << d_exp[i*29+i_exp] << ",";

                        }
                        std::cout << std::endl << " m:" << std::endl;
                        for(int i_m=0;i_m<8;i_m++){

                            std::cout << d_m[i*8+i_m] << ",";
                        }	
                    }
                }
                /*
                   if(write_temp_variable){
                   write_pid.close();
                   write_pexp.close();
                   write_pm.close();
                   write_duv_dtheta.close();
                   write_dv.close();
                   }
                   std::cout << "finish" << std::endl;

                   getchar();
                   */
            }

        }

#ifdef CPU_ONLY
    STUB_GPU(FrontalLayer);
#endif

    INSTANTIATE_CLASS(FrontalLayer);
    REGISTER_LAYER_CLASS(Frontal); 


    // namespace caffe
    template<typename Dtype>
        void FrontalLayer<Dtype>::load_txt(cv::Mat& input,int height,int width,std::string file_loc,int type)
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

    template<typename Dtype>
        cv::Mat FrontalLayer<Dtype>::frontal(eos::render::Mesh mesh,cv::Mat old_affine_camera_matrix,cv::Mat image,int num){

            using cv::Mat;
            using cv::Vec2f;
            using cv::Vec3f;
            using cv::Vec4f;
            using cv::Vec3b;
            using std::min;
            using std::max;
            using std::floor;
            using std::ceil;

            cv::Mat affine_camera_matrix = eos::render::detail::calculate_affine_z_direction(old_affine_camera_matrix).clone();
            //cv::Mat lined_image=image.clone();


            /*
            //std::string output_dir="/home/scw4750/github/unrolling/zero/A_net/A_net_by_frontal/data/train_test_data/lined_image";
            count_num++;
            cv::Mat lined_image=image.clone();
            for (const auto& triangle_indices : mesh.tvi){
                cv::Point2f src_tri[3];
                Vec4f vec(mesh.vertices[triangle_indices[0]][0], mesh.vertices[triangle_indices[0]][1], mesh.vertices[triangle_indices[0]][2], 1.0f);
                Vec4f res = Mat(affine_camera_matrix * Mat(vec));
                src_tri[0] = Vec2f(res[0], res[1]);

                vec = Vec4f(mesh.vertices[triangle_indices[1]][0], mesh.vertices[triangle_indices[1]][1], mesh.vertices[triangle_indices[1]][2], 1.0f);
                res = Mat(affine_camera_matrix * Mat(vec));
                src_tri[1] = Vec2f(res[0], res[1]);

                vec = Vec4f(mesh.vertices[triangle_indices[2]][0], mesh.vertices[triangle_indices[2]][1], mesh.vertices[triangle_indices[2]][2], 1.0f);
                res = Mat(affine_camera_matrix * Mat(vec));
                src_tri[2] = Vec2f(res[0], res[1]);
                glm::vec2 p1=glm::vec2(src_tri[0].x,src_tri[0].y);
                glm::vec2 p2=glm::vec2(src_tri[1].x,src_tri[1].y);
                glm::vec2 p3=glm::vec2(src_tri[2].x,src_tri[2].y);

                //if (eos::render::detail::are_vertices_ccw_in_screen_space(glm::vec2(p1), glm::vec2(p2), glm::vec2(p3)))
                {
                    //std::cout << "begin" << std::endl;
                    //std::cout << p1.x << " " << p1.y << std::endl;
                    //std::cout << p2.x << " " << p2.y << std::endl;
                    //std::cout << p3.x << " " << p3.y << std::endl;
                    cv::line(lined_image, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y), cv::Scalar(0, 255, 0, 255));
                    cv::line(lined_image, cv::Point(p2.x, p2.y), cv::Point(p3.x, p3.y), cv::Scalar(0, 255, 0, 255));
                    cv::line(lined_image, cv::Point(p3.x, p3.y), cv::Point(p1.x, p1.y), cv::Scalar(0, 255, 0, 255));
                }
            }
            cv::imshow("lined_image",lined_image);
            cv::waitKey(0);
            //cv::imwrite(output_dir+"/"+std::to_string(count_num)+"lined_image.png", lined_image);

            */

            cv::Mat depthbuffer;
            //std::cout << "depthbuffer" << std::endl;
            std::tie(std::ignore, depthbuffer) = render::render_affine(mesh, affine_camera_matrix, image.cols, image.rows);
            //std::cout << "asdfasdfasdfasdfdepthbuffer" << std::endl;
            cv::Mat frontal_img = cv::Mat::zeros(frontal_h, frontal_w, CV_8UC4);

            // #Todo: We should handle gray images, but output a 4-channel frontal nevertheless I think.

            std::vector<std::future<void>> results;
            for (const auto& triangle_indices : mesh.tvi) {

                // Note: If there's a performance problem, there's no need to capture the whole mesh - we could capture only the three required vertices with their texcoords.
                auto extract_triangle = [&mesh, &affine_camera_matrix, &triangle_indices, &depthbuffer, &frontal_img, &image,num,this]() {

                    // Find out if the current triangle is visible:
                    // We do a second rendering-pass here. We use the depth-buffer of the final image, and then, here,
                    // check if each pixel in a triangle is visible. If the whole triangle is visible, we use it to extract
                    // the texture.
                    // Possible improvement: - If only part of the triangle is visible, split it

                    // This could be optimized in 2 ways though:
                    // - Use render(), or as in render(...), transfer the vertices once, not in a loop over all triangles (vertices are getting transformed multiple times)
                    // - We transform them later (below) a second time. Only do it once.

                    cv::Vec4f v0_as_Vec4f(mesh.vertices[triangle_indices[0]-1].x, mesh.vertices[triangle_indices[0]-1].y, mesh.vertices[triangle_indices[0]-1].z, mesh.vertices[triangle_indices[0]-1].w);
                    cv::Vec4f v1_as_Vec4f(mesh.vertices[triangle_indices[1]-1].x, mesh.vertices[triangle_indices[1]-1].y, mesh.vertices[triangle_indices[1]-1].z, mesh.vertices[triangle_indices[1]-1].w);
                    cv::Vec4f v2_as_Vec4f(mesh.vertices[triangle_indices[2]-1].x, mesh.vertices[triangle_indices[2]-1].y, mesh.vertices[triangle_indices[2]-1].z, mesh.vertices[triangle_indices[2]-1].w);                                     

                    // Project the triangle vertices to screen coordinates, and use the depthbuffer to check whether the triangle is visible:
                    const Vec4f v0 = Mat(affine_camera_matrix * Mat(v0_as_Vec4f));
                    const Vec4f v1 = Mat(affine_camera_matrix * Mat(v1_as_Vec4f));
                    const Vec4f v2 = Mat(affine_camera_matrix * Mat(v2_as_Vec4f));
                    if (eos::render::detail::is_triangle_visible(glm::tvec4<float>(v0[0], v0[1], v0[2], v0[3]), glm::tvec4<float>(v1[0], v1[1], v1[2], v1[3]), glm::tvec4<float>(v2[0], v2[1], v2[2], v2[3]), depthbuffer))
                    {
                        //continue;
                        //std::cout<<"is_triangle_visible and i: " << std::endl;
                        return;
                    }
                    //std::cout<<"not is_triangle_visible and i: " << std::endl;
                    float alpha_value;

                    // no visibility angle computation - if the triangle/pixel is visible, set the alpha chan to 255 (fully visible pixel).
                    alpha_value = 255.0f;


                    // Todo: Documentation
                    cv::Point2f src_tri[3];
                    cv::Point2f dst_tri[3];

                    Vec4f vec(mesh.vertices[triangle_indices[0]-1][0], mesh.vertices[triangle_indices[0]-1][1], mesh.vertices[triangle_indices[0]-1][2], 1.0f);
                    Vec4f res = Mat(affine_camera_matrix * Mat(vec));
                    src_tri[0] = Vec2f(res[0], res[1]);

                    vec = Vec4f(mesh.vertices[triangle_indices[1]-1][0], mesh.vertices[triangle_indices[1]-1][1], mesh.vertices[triangle_indices[1]-1][2], 1.0f);
                    res = Mat(affine_camera_matrix * Mat(vec));
                    src_tri[1] = Vec2f(res[0], res[1]);

                    vec = Vec4f(mesh.vertices[triangle_indices[2]-1][0], mesh.vertices[triangle_indices[2]-1][1], mesh.vertices[triangle_indices[2]-1][2], 1.0f);
                    res = Mat(affine_camera_matrix * Mat(vec));
                    src_tri[2] = Vec2f(res[0], res[1]);

                    dst_tri[0] = cv::Point2f(mesh.texcoords[triangle_indices[0]-1][0], mesh.texcoords[triangle_indices[0]-1][1] );
                    dst_tri[1] = cv::Point2f(mesh.texcoords[triangle_indices[1]-1][0], mesh.texcoords[triangle_indices[1]-1][1] );
                    dst_tri[2] = cv::Point2f(mesh.texcoords[triangle_indices[2]-1][0], mesh.texcoords[triangle_indices[2]-1][1] );
                    // We now have the source triangles in the image and the source triangle in the frontal
                    // We use the inverse/ backward mapping approach, so we want to find the corresponding texel (texture-pixel) for each pixel in the frontal

                    // Get the inverse Affine Transform from original image: from dst (pixel in frontal) to src (in image)
                    Mat warp_mat_org_inv = cv::getAffineTransform(dst_tri, src_tri);
                    warp_mat_org_inv.convertTo(warp_mat_org_inv, CV_32FC1);

                    // We now loop over all pixels in the triangle and select, depending on the mapping type, the corresponding texel(s) in the source image
                    for (int x = min(dst_tri[0].x, min(dst_tri[1].x, dst_tri[2].x)); x < max(dst_tri[0].x, max(dst_tri[1].x, dst_tri[2].x)); ++x) {
                        for (int y = min(dst_tri[0].y, min(dst_tri[1].y, dst_tri[2].y)); y < max(dst_tri[0].y, max(dst_tri[1].y, dst_tri[2].y)); ++y) {
                            if (eos::render::detail::is_point_in_triangle   (cv::Point2f(x, y), dst_tri[0], dst_tri[1], dst_tri[2])&& x>=0 && x<frontal_w &&y>=0 && y<frontal_h) {
                                Vec3f homogenous_dst_coord(x, y, 1.0f);
                                Vec2f src_texel = Mat(warp_mat_org_inv * Mat(homogenous_dst_coord));







                                // calculate euclidean distances to next 4 texels
                                using std::sqrt;
                                using std::pow;
                                float distance_upper_left = sqrt(pow(src_texel[0] - floor(src_texel[0]), 2) + pow(src_texel[1] - floor(src_texel[1]), 2));
                                float distance_upper_right = sqrt(pow(src_texel[0] - floor(src_texel[0]), 2) + pow(src_texel[1] - ceil(src_texel[1]), 2));
                                float distance_lower_left = sqrt(pow(src_texel[0] - ceil(src_texel[0]), 2) + pow(src_texel[1] - floor(src_texel[1]), 2));
                                float distance_lower_right = sqrt(pow(src_texel[0] - ceil(src_texel[0]), 2) + pow(src_texel[1] - ceil(src_texel[1]), 2));

                                // normalise distances that the sum of all distances is 1
                                float sum_distances = distance_lower_left + distance_lower_right + distance_upper_left + distance_upper_right;
                                distance_lower_left /= sum_distances;
                                distance_lower_right /= sum_distances;
                                distance_upper_left /= sum_distances;
                                distance_upper_right /= sum_distances;
                                //for backward
                                if ((cvRound(src_texel[1]) < image.rows) && (cvRound(src_texel[0]) < image.cols) && cvRound(src_texel[0]) > 0 && cvRound(src_texel[1]) > 0){
                                    visible_grid.mutable_cpu_data()[visible_grid.offset(num,0,x,y)]=(Dtype)1.0;
                                    //std::cout << "in frontal: " << visible_grid.data_at(num,0,x,y) << std::endl;
                                    input_grid.mutable_cpu_data()[input_grid.offset(num,0,x,y)]=src_texel[1];
                                    input_grid.mutable_cpu_data()[input_grid.offset(num,1,x,y)]=src_texel[0];
                                    Dtype fenzi=(x-dst_tri[0].x)*(dst_tri[2].y-dst_tri[0].y)-(y-dst_tri[0].y)*(dst_tri[2].x-dst_tri[0].x);
                                    Dtype fenmu=(dst_tri[1].x-dst_tri[0].x)*(dst_tri[2].y-dst_tri[0].y)-(dst_tri[1].y-dst_tri[0].y)*(dst_tri[2].x-dst_tri[0].x);
				    alpha.mutable_cpu_data()[alpha.offset(num,0,x,y)]=fenzi/fenmu;

                                    Dtype beta_fenzi=(y-dst_tri[0].y)*(dst_tri[1].x-dst_tri[0].x)-(x-dst_tri[0].x)*(dst_tri[1].y-dst_tri[0].y);
                                    beta.mutable_cpu_data()[beta.offset(num,0,x,y)]=beta_fenzi/fenmu;

				    triangular_index_0.mutable_cpu_data()[triangular_index_0.offset(num,0,x,y)]=static_cast<Dtype>(triangle_indices[0]);
                                    triangular_index_1.mutable_cpu_data()[triangular_index_1.offset(num,0,x,y)]=static_cast<Dtype>(triangle_indices[1]);
				    triangular_index_2.mutable_cpu_data()[triangular_index_2.offset(num,0,x,y)]=static_cast<Dtype>(triangle_indices[2]);



                                    // set color depending on distance from next 4 texels
                                    for (int color = 0; color < 3; ++color) {
                                        float color_upper_left = image.at<Vec3b>(floor(src_texel[1]), floor(src_texel[0]))[color] * distance_upper_left;
                                        float color_upper_right = image.at<Vec3b>(floor(src_texel[1]), ceil(src_texel[0]))[color] * distance_upper_right;
                                        float color_lower_left = image.at<Vec3b>(ceil(src_texel[1]), floor(src_texel[0]))[color] * distance_lower_left;
                                        float color_lower_right = image.at<Vec3b>(ceil(src_texel[1]), ceil(src_texel[0]))[color] * distance_lower_right;

                                        frontal_img.at<cv::Vec4b>(y, x)[color] = color_upper_left + color_upper_right + color_lower_left + color_lower_right;
                                        frontal_img.at<cv::Vec4b>(y, x)[3] = static_cast<uchar>(alpha_value);
                                    }
                                }

                            }
                        }
                    }
                }; // end lambda auto extract_triangle();
                results.emplace_back(std::async(extract_triangle));
            } // end for all mesh.tvi
            // Collect all the launched tasks:
            for (auto&& r : results) {
                r.get();
            }
            //cv::imshow("frontal_in_c++",frontal);
            //cv::waitKey(0);
            //frontal=interpolate_black_line(frontal);
            //cv::imwrite(output_dir+"/"+std::to_string(count_num)+"frontal.png", frontal_img);
            //std::cout << "success at frontal function" << std::endl; 
            return frontal_img;
        }
    template<typename Dtype>
        inline cv::Mat FrontalLayer<Dtype>::interpolate_black_line(cv::Mat frontal)
        {
            assert(frontal.type() == CV_8UC4);
            // Replace the vertical black line ("missing data"):
            int col = frontal.cols / 2;
            for (int row = 0; row < frontal.rows; ++row)
            {
                if (frontal.at<cv::Vec4b>(row, col) == cv::Vec4b(0, 0, 0, 0))
                {
                    frontal.at<cv::Vec4b>(row, col) = frontal.at<cv::Vec4b>(row, col - 1) * 0.5f + frontal.at<cv::Vec4b>(row, col + 1) * 0.5f;
                }
            }

            // Replace the horizontal line around the mouth that occurs in the
            // frontals of resolution 512x512 and higher:
            if (frontal.rows == 512) // num cols is 512 as well
            {
                int r = 362;
                for (int c = 206; c <= 306; ++c)
                {
                    if (frontal.at<cv::Vec4b>(r, c) == cv::Vec4b(0, 0, 0, 0))
                    {
                        frontal.at<cv::Vec4b>(r, c) = frontal.at<cv::Vec4b>(r - 1, c) * 0.5f + frontal.at<cv::Vec4b>(r + 1, c) * 0.5f;
                    }
                }
            }
            if (frontal.rows == 1024) // num cols is 1024 as well
            {
                int r = 724;
                for (int c = 437; c <= 587; ++c)
                {
                    if (frontal.at<cv::Vec4b>(r, c) == cv::Vec4b(0, 0, 0, 0))
                    {
                        frontal.at<cv::Vec4b>(r, c) = frontal.at<cv::Vec4b>(r - 1, c) * 0.5f + frontal.at<cv::Vec4b>(r + 1, c) * 0.5f;
                    }
                }
                r = 725;
                for (int c = 411; c <= 613; ++c)
                {
                    if (frontal.at<cv::Vec4b>(r, c) == cv::Vec4b(0, 0, 0, 0))
                    {
                        frontal.at<cv::Vec4b>(r, c) = frontal.at<cv::Vec4b>(r - 1, c) * 0.5f + frontal.at<cv::Vec4b>(r + 1, c) * 0.5f;
                    }
                }
            }
            // Higher resolutions are probably affected as well but not used so far in practice.

            return frontal;
        }
    template<typename Dtype>
        cv::Mat FrontalLayer<Dtype>::WeakProjection(cv::Mat src_lm3, cv::Mat dst_lm2){
            cv::Mat T = cv::Mat::zeros(2, 4, CV_32FC1);
            cv::Mat A = cv::Mat::zeros(dst_lm2.rows * 2, 8, CV_32FC1);
            cv::Mat b = cv::Mat::zeros(dst_lm2.rows * 2, 1, CV_32FC1);

            for (int i = 0; i < dst_lm2.rows; i++)
            {
                A.at<Dtype>((i * 2), 0) = src_lm3.at<Dtype>(i, 0);
                A.at<Dtype>((i * 2), 1) = src_lm3.at<Dtype>(i, 1);
                A.at<Dtype>((i * 2), 2) = src_lm3.at<Dtype>(i, 2);
                A.at<Dtype>((i * 2), 3) = 1;
                A.at<Dtype>((i * 2), 4) = 0;
                A.at<Dtype>((i * 2), 5) = 0;
                A.at<Dtype>((i * 2), 6) = 0;
                A.at<Dtype>((i * 2), 7) = 0;

                A.at<Dtype>((i * 2 + 1), 0) = 0;
                A.at<Dtype>((i * 2 + 1), 1) = 0;
                A.at<Dtype>((i * 2 + 1), 2) = 0;
                A.at<Dtype>((i * 2 + 1), 3) = 0;
                A.at<Dtype>((i * 2 + 1), 4) = src_lm3.at<Dtype>(i, 0);
                A.at<Dtype>((i * 2 + 1), 5) = src_lm3.at<Dtype>(i, 1);
                A.at<Dtype>((i * 2 + 1), 6) = src_lm3.at<Dtype>(i, 2);
                A.at<Dtype>((i * 2 + 1), 7) = 1;

                b.at<Dtype>((i * 2), 0) = dst_lm2.at<Dtype>(i, 0);
                b.at<Dtype>((i * 2 + 1), 0) = dst_lm2.at<Dtype>(i, 1);
            }

            cv::Mat Aa = cv::Mat::zeros(8, dst_lm2.rows * 2, CV_32FC1);
            cv::Mat X = cv::Mat::zeros(8, 1, CV_32FC1);;
            invert(A, Aa, cv::DECOMP_SVD);
            X = Aa*b;

            T.at<Dtype>(0, 0) = X.at<Dtype>(0, 0);
            T.at<Dtype>(0, 1) = X.at<Dtype>(1, 0);
            T.at<Dtype>(0, 2) = X.at<Dtype>(2, 0);
            T.at<Dtype>(0, 3) = X.at<Dtype>(3, 0);
            T.at<Dtype>(1, 0) = X.at<Dtype>(4, 0);
            T.at<Dtype>(1, 1) = X.at<Dtype>(5, 0);
            T.at<Dtype>(1, 2) = X.at<Dtype>(6, 0);
            T.at<Dtype>(1, 3) = X.at<Dtype>(7, 0);
            /*
               std::cout << T.at<Dtype>(0,0) << std::endl;
               std::cout << T.at<Dtype>(0,1) << std::endl;
               std::cout << T.at<Dtype>(0,2) << std::endl;
               std::cout << T.at<Dtype>(0,3) << std::endl;
               std::cout << T.at<Dtype>(1,0) << std::endl;
               std::cout << T.at<Dtype>(1,1) << std::endl;
               std::cout << T.at<Dtype>(1,0) << std::endl;
               std::cout << T.at<Dtype>(1,0) << std::endl;
               */
            return T;
        }
}//namespace caffe

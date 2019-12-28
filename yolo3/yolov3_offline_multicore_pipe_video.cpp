/*!
*  Copyright (c) 2018 by Contributors
* \file yolov3_offline_multicore_pipe.cpp
* \brief
*
* \author
*/
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <algorithm>
#include <condition_variable> // NOLINT
#include <iomanip>
#include <iosfwd>
#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <thread> // NOLINT
#include <utility>
#include <vector>
#include <cmath>
#include <atomic>

#include "../include/detection_out.hpp"
#include "../include/runtime.hpp"
#include "../include/blocking_queue.hpp"
#include "../include/func_runner.hpp"

using std::map;
using std::max;
using std::min;
using std::queue;
using std::thread;
using std::stringstream;
using std::vector;

using namespace std;
using namespace cv;

std::condition_variable condition;
std::mutex condition_m;
bool ready_start = false;
bool use_rtctx = false;

int height = 416;
int width = 416;
int result_num = 0;

#ifdef USE_OPENCV

static constexpr int kSplicePerThreadUnion2 = 8;

void get_point_position(const vector<float> pos,
                        cv::Point* p1, cv::Point* p2, int h, int w) {
  int left = (pos[3] - pos[5] / 2) * w;
  int right = (pos[3] + pos[5] / 2) * w;
  int top = (pos[4] - pos[6] / 2) * h;
  int bottom = (pos[4] + pos[6] / 2) * h;
  if (left < 0) left = 0;
  if (top < 0) top = 0;
  if (right > w) right = w;
  if (bottom > h) bottom = h;
  p1->x = left;
  p1->y = top;
  p2->x = right;
  p2->y = bottom;
  return;
}

typedef struct {
  float x_min;
  float y_min;
  float x_max;
  float y_max;
} detection_bbox;


const char *obj_sparse_tag[20]={"aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"};


std::ostream& operator<<(std::ostream& os, const mof::MShape& mshape) {
  os << " nchw = [(" << mshape.n << ", " << mshape.c << ", "
     << mshape.h << ", " << mshape.w << ")]";
  return os;
}
std::ostream& operator<<(std::ostream& os, const detection_bbox& tmp) {
  os << " x,x,y,y,score,id = [(" << tmp.x_min << ", " << tmp.y_min << ", "
     << tmp.x_max << ", " << tmp.y_max << ")]";
  return os;
}

DEFINE_string(offlinemodel, "",
    "The prototxt file used to find net configuration");
DEFINE_string(mean_file, "",
    "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "123.675,116.28,103.53",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel (default channel order is RGB). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(stdt_value, "58.395,57.120,50.625", "stdt value which default channel order is RGB");
DEFINE_string(use_mean, "on",
    "if it does not need to subtract mean, then use_mean must be 'off'.");
DEFINE_int32(core_num, 32,
    "number of cores running in on device");
DEFINE_bool(duplicate_channel, false,
    "duplicate command and wight");
DEFINE_int32(data_parallelism, 1,
    "recommendation is [1 2 4 8 16 32]");
DEFINE_int32(model_parallelism, 1,
    "recommendation is [1 2 4 8], number of cores that used to process one batch.");
DEFINE_int32(threads, 1,
    "recommendation is [1 2 4]");
DEFINE_int32(device_id, 0, "device id for mlu devices");
DEFINE_int32(int8, 0,
    "0 or 1, int8 mode");
DEFINE_int32(iter_num, 1, "iterate number for offline model list");
DEFINE_double(scale, 1.0,
    "Optional; sometimes the input needs to be scaled, MobileNet for example");
DEFINE_string(img_dir, "",
    "path to images");
DEFINE_string(images, "",
    "file name of images");
DEFINE_double(confidence_threshold, 0.50,
    "Only store detections with score higher than the threshold.");
DEFINE_string(output_mode, "picture",
    "picture(default) or text");
DEFINE_int32(data_provider_num, 1,
    "The number of dataproviders in each thread");
DEFINE_int32(post_processor_num, 1,
    "The number of postprocessors in each thread");
DEFINE_int32(max_images_num,
    2048,
    "The number of images that current application handle");
DEFINE_bool(pre_read, true, "read all image before pipeline");

class PostProcessor;

class Inferencer {
 public:
  Inferencer(
    const int& thread_id,
    const int& data_parallelism,
    const cnrtModel_t& model);
  int n() {return in_n_;}
  int c() {return in_c_;}
  int h() {return in_h_;}
  int w() {return in_w_;}
  int out_n(unsigned int i) {return out_n_[i];}
  int out_c(unsigned int i) {return out_c_[i];}
  int out_h(unsigned int i) {return out_h_[i];}
  int out_w(unsigned int i) {return out_w_[i];}
  unsigned int output_chw(unsigned int i) {return out_c_[i] * out_h_[i] * out_w_[i];}
  void pushValidInputData(void** data);
  void pushFreeInputData(void** data);
  void** popValidInputData();
  void** popFreeInputData();
  void pushValidOutputData(void** data);
  void pushFreeOutputData(void** data);  void** popValidOutputData();
  void** popFreeOutputData();
  void pushValidInputNames(vector<string> imgs);
  vector<string> popValidInputNames();
  void notify();
  void run();
  void run_with_rtctx();
  ~Inferencer();

  cnrtDataDescArray_t inputDescS() {return inputDescS_;}
  cnrtDataDescArray_t outputDescS() {return outputDescS_;}
  mof::BlockingQueue<void**> validInputFifo_;
  mof::BlockingQueue<void**> freeInputFifo_;
  mof::BlockingQueue<void**> validOutputFifo_;
  mof::BlockingQueue<void**> freeOutputFifo_;
  mof::BlockingQueue<vector<string> > imagesFifo_;

  cnrtDataDescArray_t inputDescS_, outputDescS_;
  cnrtQueue_t queue_;
  int inputNum, outputNum;
  cnrtModel_t model_;
  cnrtFunction_t function;
  unsigned int in_n_, in_c_, in_h_, in_w_;
  unsigned int out_n_[54], out_c_[54], out_h_[54], out_w_[54];
  int out_count_[54];
  cnrtDim3_t dim_;
  int running_;
  int thread_id_;
  int data_parallelism_;
  cnrtFunctionType_t func_type_;
  vector<PostProcessor*> post_processor_;
  double invoke_time;
  cnrtInvokeFuncParam_t invoke_param_;
  std::mutex data_mtx;
  std::mutex post_mtx;
  int post_processor_num_;
};

class PostProcessor {
 public:
  PostProcessor();
  void run();
  void RectangleAndDrawResult(float* ids_data,
                              float* bbox_data,
                              float* score_data,
                              float conf_thresh,
                              string intorigin_img,
                              int idx);
  void RectangleAndPrintResult(float* ids_data,
                               float* bbox_data,
                               float* score_data,
                               float conf_thresh,
                               string intorigin_img,
                               int idx);

  Inferencer* inferencer_;
  int thread_id_;
  unsigned int total_;
};

template<class datatype>
class DataProvider {
 public:
  DataProvider(
               const string& mean_file,
               const string& mean_value,
               const queue<string>& images,
               float scale = 1.0);
  void SetMean(const string&, const string&);
  void SetStdt(const string&);
  void preRead();
  void PreData(int batch_size);
  void run();
  void WrapInputLayer(std::vector<std::vector<cv::Mat> >* input_imgs);
  void Preprocess(const std::vector<cv::Mat>& imgs);
  cv::Mat mean_;
  cv::Mat stdt_;
  void** cpu_data_;
  int in_n_, in_c_, in_h_, in_w_;
  float scale_;
  queue<string> images_;
  Inferencer* inferencer_;
  cv::Size input_geometry_;
  int thread_id_;
  bool need_mean_;
  // vector<vector<cv::Mat> > v_images;
  vector<vector<string> > v_names;
  float* offset_data;
  float* arange_data;
  float* anchor_data0;
  float* anchor_data1;
  float* anchor_data2;
};
vector<vector<cv::Mat> > v_images;
template<class datatype>
DataProvider<datatype>::DataProvider(
               const string& mean_file,
               const string& mean_value,
               const queue<string>& images,
               float scale) {
  images_ = images;
  thread_id_ = 0;
  scale_ = scale;

  need_mean_ = true;
  if (FLAGS_use_mean == "off") {   // use_mean = on
    need_mean_ = false;
  }
}

template<class datatype>
void DataProvider<datatype>::PreData(int batch_size) {
  cout << "arange——data——offset——anchor——";
  arange_data = reinterpret_cast<float*>(cpu_data_[1]);
  offset_data = reinterpret_cast<float*>(cpu_data_[2]);
  anchor_data0 = reinterpret_cast<float*>(cpu_data_[3]);
  anchor_data1 = reinterpret_cast<float*>(cpu_data_[6]);
  anchor_data2 = reinterpret_cast<float*>(cpu_data_[9]);

  // pre offset data
  for (int i = 0; i < 128; i++) {
    for (int j = 0; j < 128; j++) {
      offset_data[i * 2 + j * 128 * 2] = i;
      offset_data[i * 2 * 128 + j * 2 + 1] = i;
    }
  }

  // pre arange data
  for (int i = 0; i < 20; i++) {
    arange_data[i] = i;
  }

  // pre anchor data
  anchor_data0[0] = 116.0; anchor_data0[1] = 90.0; anchor_data0[2] = 156.0;
  anchor_data0[3] = 198.0; anchor_data0[4]= 373.0; anchor_data0[5] = 326.0;

  anchor_data1[0] = 30.0; anchor_data1[1] = 61.0; anchor_data1[2] = 62.0;
  anchor_data1[3] = 45.0; anchor_data1[4] = 59.0; anchor_data1[5] = 119.0;

  anchor_data2[0] = 10.0; anchor_data2[1] = 13.0; anchor_data2[2] = 16.0;
  anchor_data2[3] = 30.0; anchor_data2[4] = 33.0; anchor_data2[5] = 23.0;

  // tile data
  for (int i = 0; i < batch_size; i++) {
    memcpy(offset_data + i*128*128*2, offset_data, 128*128*2*sizeof(float));
    memcpy(arange_data + i*20, arange_data, 20*sizeof(float));
    memcpy(anchor_data0 + i*6, anchor_data0, 6*sizeof(float));
    memcpy(anchor_data1 + i*6, anchor_data1, 6*sizeof(float));
    memcpy(anchor_data2 + i*6, anchor_data2, 6*sizeof(float));
  }

  memcpy(reinterpret_cast<float*>(cpu_data_[4]), reinterpret_cast<float*>(cpu_data_[1]),
        sizeof(float) * batch_size * 20);
  memcpy(reinterpret_cast<float*>(cpu_data_[7]), reinterpret_cast<float*>(cpu_data_[1]),
        sizeof(float) * batch_size * 20);

  memcpy(reinterpret_cast<float*>(cpu_data_[5]), reinterpret_cast<float*>(cpu_data_[2]),
        sizeof(float)* batch_size * 128 * 128 * 2);
  memcpy(reinterpret_cast<float*>(cpu_data_[8]), reinterpret_cast<float*>(cpu_data_[2]),
        sizeof(float)* batch_size * 128 * 128 * 2);
}

template<class datatype>
void DataProvider<datatype>::preRead() { // 读图片
  in_n_ = inferencer_->n();
  cout << "读图片int in_nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn-----------"  << in_n_ << endl;
  std::string img_dir = FLAGS_img_dir;
  while (images_.size()) {
    vector<cv::Mat> imgs;
    vector<string> img_names;
    int left_num = images_.size();
    cout << left_num << "------------------------left_num--------------" << endl;
    for (int i = 0; i < in_n_; i++) {
      if (i < left_num) {
        string file = images_.front();
        images_.pop();
        cv::Mat img;
        // use cv::imread to read image and the default channel order is BGR
        img = cv::imread(img_dir + file);
        imgs.push_back(img);
        img_names.push_back(file);
      } else {
        cv::Mat img;
        // use cv::imread to read image and the default channel order is BGR
        img = cv::imread(img_dir + img_names[0]);
        imgs.push_back(img);
        img_names.push_back("null");
      }
    }
    v_images.push_back(imgs);
    v_names.push_back(img_names);
  }
}

template<class datatype>
void DataProvider<datatype>::run() {
  // LOG(INFO) << "################# DataProvider::run()!";
  cnrtDev_t dev;
  CNRT_CHECK(cnrtGetDeviceHandle(&dev, FLAGS_device_id));
  CNRT_CHECK(cnrtSetCurrentDevice(dev));

  if (!use_rtctx && FLAGS_data_parallelism * FLAGS_model_parallelism <= 8) {
    cout << "cnrtSetCurrentChannel" << endl;
    cnrtSetCurrentChannel((cnrtChannelType_t)(thread_id_ % CHANNEL_NUM));
  }

  cpu_data_ = reinterpret_cast<void**>(malloc(sizeof(void*) * 10));   // 10 input
  in_n_ = inferencer_->n();
  in_c_ = inferencer_->c();
  in_h_ = inferencer_->h();
  in_w_ = inferencer_->w();
  cout << in_n_<< "-----"<< in_c_<< "-----"<< in_h_<< "-----"<< in_w_<< "-----"<<endl;
  cpu_data_[0] = reinterpret_cast<void*>
                 (malloc(in_n_ * in_c_ * in_h_ * in_w_ * sizeof(float)));
  for (int i = 0; i < 3; i++) {
    // arange data
    cpu_data_[1+i*3] = reinterpret_cast<void*>
        (malloc(in_n_ * 1 * 20 * 1 * sizeof(float)));
    // offset data
    cpu_data_[2+i*3] = reinterpret_cast<void*>
        (malloc(in_n_ * 128 * 128 * 2 * sizeof(float)));
    // anchor data
    cpu_data_[3+i*3] = reinterpret_cast<void*>
        (malloc(in_n_ * 1 * 3 * 2 * sizeof(float)));
  }
  // arange/offset/anchor input
  PreData(in_n_);
  input_geometry_ = cv::Size(in_w_, in_h_);
  SetMean(FLAGS_mean_file, FLAGS_mean_value);
  SetStdt(FLAGS_stdt_value);
  std::string img_dir = FLAGS_img_dir;

  std::unique_lock<std::mutex> lk(condition_m);
  LOG(INFO) << "Waiting ...";
  condition.wait(lk, [](){return ready_start;});
  lk.unlock();

  if (FLAGS_pre_read) {
    cout << v_images.size() <<"vimages-------------------------- vector<vector<cv::Mat> >" << endl;
    for (int i = 0; i < v_images.size(); i++) {
      
      mof::Timer prepareInput;
      vector<cv::Mat> imgs = v_images[i];
      cout << imgs.size() <<"放图片的imgs---------------------------"<<endl;
      vector<string> img_names = v_names[i];
      Preprocess(imgs);
      prepareInput.duration("prepare input data ...");

      void** mlu_data = inferencer_->popFreeInputData();
      mof::Timer copyin;
      if (use_rtctx) {
        CNRT_CHECK(cnrtMemcpyByDescArray(
            mlu_data, cpu_data_, inferencer_->inputDescS(),
            inferencer_->inputNum, CNRT_MEM_TRANS_DIR_HOST2DEV));
      } else {
        CNRT_CHECK(cnrtMemcpyBatchByDescArray(
            mlu_data, cpu_data_, inferencer_->inputDescS(),
            inferencer_->inputNum, FLAGS_data_parallelism, CNRT_MEM_TRANS_DIR_HOST2DEV));
      }
      copyin.duration("copyin time ...");
      std::unique_lock<std::mutex> lock(inferencer_->data_mtx);
      inferencer_->pushValidInputData(mlu_data);
      inferencer_->pushValidInputNames(img_names);
      lock.unlock();
    }
  }
   else {
    while (images_.size()) {
      mof::Timer prepareInput;
      vector<cv::Mat> imgs;
      vector<string> img_names;
      int left_num = images_.size();
      for (int i = 0; i < in_n_; i++) {
        if (i < left_num) {
          string file = images_.front();
          images_.pop();
          cv::Mat img;
          // use cv::imread to read image and the default channel order is BGR
          img = cv::imread(img_dir + file);
          imgs.push_back(img);
          img_names.push_back(file);
        } else {
          cv::Mat img;
          // use cv::imread to read image and the default channel order is BGR
          img = cv::imread(img_dir + img_names[0]);
          imgs.push_back(img);
          img_names.push_back("null");
        }
      }
      Preprocess(imgs);
      prepareInput.duration("prepare input data ...");

      void** mlu_data = inferencer_->popFreeInputData();
      mof::Timer copyin;
      if (use_rtctx) {
        CNRT_CHECK(cnrtMemcpyByDescArray(
            mlu_data, cpu_data_, inferencer_->inputDescS(),
            inferencer_->inputNum, CNRT_MEM_TRANS_DIR_HOST2DEV));
      } else {
        CNRT_CHECK(cnrtMemcpyBatchByDescArray(
            mlu_data, cpu_data_, inferencer_->inputDescS(),
            inferencer_->inputNum, FLAGS_data_parallelism, CNRT_MEM_TRANS_DIR_HOST2DEV));
      }
      copyin.duration("copyin time ...");
      std::unique_lock<std::mutex> lock(inferencer_->data_mtx);
      inferencer_->pushValidInputData(mlu_data);
      inferencer_->pushValidInputNames(img_names);
      lock.unlock();
    }
  }
  inferencer_->notify();
  inferencer_->pushValidInputData(nullptr);
  for (int i = 0; i < 10; i++) {
    free(cpu_data_[i]);
  }
  free(cpu_data_);
}

template<class datatype>
void DataProvider<datatype>::Preprocess(const std::vector<cv::Mat>& imgs) {
  cout << "preprocess预处理---------------------------" << endl;
  /* Convert the input image to the input image format of the network. */
  datatype* input_data = reinterpret_cast<datatype*>(cpu_data_[0]);
  for (int i = 0; i < imgs.size(); ++i) {
    cv::Mat sample;
    if (imgs[i].channels() == 3 && in_c_ == 1)
      cv::cvtColor(imgs[i], sample, cv::COLOR_BGR2GRAY);
    else if (imgs[i].channels() == 4 && in_c_ == 1)
      cv::cvtColor(imgs[i], sample, cv::COLOR_BGRA2GRAY);
    else if (imgs[i].channels() == 4 && in_c_ == 3)
      cv::cvtColor(imgs[i], sample, cv::COLOR_BGRA2BGR);
    else if (imgs[i].channels() == 1 && in_c_ == 3)
      cv::cvtColor(imgs[i], sample, cv::COLOR_GRAY2BGR);
    else
      sample = imgs[i];
    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
      cv::resize(sample, sample_resized, input_geometry_);
    else
      sample_resized = sample;
    cv::Mat sample_normalized;
    if (!FLAGS_int8) {
      cv::Mat sample_float;
      cv::Mat sample_normalized_temp;
      if (in_c_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
      else
        sample_resized.convertTo(sample_float, CV_32FC1);
      if (need_mean_ == true) {
        cv::subtract(sample_float, mean_, sample_normalized_temp);
        cv::divide(sample_normalized_temp, stdt_, sample_normalized);
      } else {
        sample_normalized = sample_float;
      }

      if (scale_ != 1.0) {
        sample_normalized *= scale_;
      }
    } else {
      cv::Mat sample_float;
      if (in_c_ == 3)
        sample_resized.convertTo(sample_float, CV_8UC3);
      else
        sample_resized.convertTo(sample_float, CV_8UC1);
      sample_normalized = sample_float;
    }
    // hwc to chw && BGR to RGB
    std::vector<cv::Mat> SingleChannelImgs;
    SingleChannelImgs.resize(in_c_);
    cv::split(sample_normalized, SingleChannelImgs);
    for (int c = 0; c < in_c_; c++) {
      for (int h = 0; h < in_h_; h++) {
        const datatype* p = reinterpret_cast<const datatype*>((SingleChannelImgs[in_c_ - c - 1].ptr(h)));
        memcpy(input_data, p, sizeof(datatype) * in_w_);
        input_data += in_w_;
      }
    }
  }
}

template<class datatype>
void DataProvider<datatype>::SetMean(const string& mean_file,
                           const string& mean_value) {
  if (need_mean_ == false)
    return;
  cout << "setmean-----------------------------"<< endl;
  if (!mean_file.empty()) {
    CHECK(mean_value.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    cv::Scalar channel_mean;
    unsigned int mean_size = in_c_ * in_h_ * in_w_;
    float* mean_data = new float[mean_size];
    std::ifstream fin(mean_file.c_str(), std::ios::in);
    for (unsigned int i = 0; i < mean_size; i++) {
      fin >> mean_data[i];
    }
    fin.close();

    /* The format of the mean file is planar 32-bit float RGB or grayscale.
     * We will turn the channel order to BGR since images with channel order
     * BGR during its preprocess.
     * */
    std::vector<cv::Mat> channels(in_c_);
    for (int i = 0; i < in_c_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(in_h_, in_w_, CV_32FC1, mean_data);
      // Fill channel value in reverse order, so they are in order BGR.
      channels[in_c_ - i - 1] = channel;
      mean_data += in_h_ * in_w_;
    }
    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);
    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
    delete [] mean_data;
  }

  if (!mean_value.empty()) {
    CHECK(mean_file.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    /* The channel order of the mean_value is RGB or grayscale.
     * We will turn the channel order to BGR since images with channel order
     * BGR during its preprocess.
     * */
    stringstream ss(mean_value);
    vector<float> values(in_c_);
    string item;
    int index_for_BGR = in_c_ - 1;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      // Fill mean value in reverse order, so they are in order BGR.
      values[index_for_BGR--] = value;
    }
    CHECK(values.size() == 1 || values.size() == in_c_) <<
      "Specify either 1 mean_value or as many as channels: " << in_c_;
    std::vector<cv::Mat> channels;
    for (int i = 0; i < in_c_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}

template <class datatype>
void DataProvider<datatype>::SetStdt(const string& stdt_value) {
  cout << "setstdt------------------------------" << endl;
  if (!stdt_value.empty()) {
    /* The channel order of the stdt_value is RGB or grayscale.
     * We will turn the channel order to BGR since images with channel order
     * BGR during its preprocess.
     * */
    stringstream ss(stdt_value);
    vector<float> values(in_c_);
    string item;
    int index_for_BGR = in_c_ - 1;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      // Fill stdt value in reverse order, so they are in order BGR.
      values[index_for_BGR--] = value;
    }
    CHECK(values.size() == 1 || values.size() == in_c_)
        << "Specify either 1 mean_value or as many as channels: " << in_c_;
    std::vector<cv::Mat> channels;
    for (int i = 0; i < in_c_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height,
                      input_geometry_.width,
                      CV_32FC1,
                      cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, stdt_);
  } else {
    /* default stdt = '1,1,1' */
    std::vector<cv::Mat> channels;
    for (int i = 0; i < in_c_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height,
                      input_geometry_.width,
                      CV_32FC1,
                      cv::Scalar(1.0));
      channels.push_back(channel);
    }
    cv::merge(channels, stdt_);
  }
}

Inferencer::Inferencer(
    const int& thread_id,
    const int& data_parallelism,
    const cnrtModel_t& model) {

  invoke_time = 0.;
  running_ = FLAGS_data_provider_num;
  post_processor_num_ = FLAGS_post_processor_num;
  thread_id_ = thread_id;
  data_parallelism_ = data_parallelism;
  model_ = model;
  unsigned int dev_num;
  cnrtGetDeviceCount(&dev_num);
  if (dev_num == 0) {
    LOG(ERROR) << "no device found";
    exit(-1);
  }
  cnrtDev_t dev;
  CNRT_CHECK(cnrtGetDeviceHandle(&dev, FLAGS_device_id));
  CNRT_CHECK(cnrtSetCurrentDevice(dev));

  if (!use_rtctx) {
    switch (FLAGS_data_parallelism) {
      case 1:
        func_type_ = CNRT_FUNC_TYPE_BLOCK;
        break;
      case 2:
        func_type_ = CNRT_FUNC_TYPE_BLOCK1;
        break;
      case 4:
        func_type_ = CNRT_FUNC_TYPE_UNION1;
        break;
      case 8:
        func_type_ = CNRT_FUNC_TYPE_UNION2;
        break;
      case 16:
        func_type_ = CNRT_FUNC_TYPE_UNION4;
        break;
      case 32:
        func_type_ = CNRT_FUNC_TYPE_UNION8;
        break;
      default:
        LOG(ERROR) << "not support data_parallelism: " << FLAGS_data_parallelism;
        exit(-1);
    }

    // func_type_ = CNRT_FUNC_TYPE_BLOCK;
    if (FLAGS_data_parallelism * FLAGS_model_parallelism <= 8) {
      cnrtSetCurrentChannel((cnrtChannelType_t)(thread_id_ % CHANNEL_NUM));
      // func_type_ = CNRT_FUNC_TYPE_UNION2;
    }
  }

  // 2. get function
  string name = "fusion_0";
  cnrtCreateFunction(&function);
  cnrtExtractFunction(&function, model_, name.c_str());
  // 3. get function's I/O DataDesc
  cnrtGetInputDataDesc(&inputDescS_, &inputNum , function);
  CHECK_EQ(inputNum, 10);
  cnrtGetOutputDataDesc(&outputDescS_, &outputNum, function);
  CHECK_EQ(outputNum, 54);
  // 4. allocate I/O data space on CPU memory and prepare Input data
  int in_count;

  // LOG(INFO) << "input blob num is " << inputNum;
  for (int i = 0; i < inputNum; i++) {
    unsigned int in_n, in_c, in_h, in_w;
    cnrtDataDesc_t inputDesc = inputDescS_[i];
    cnrtGetHostDataCount(inputDesc, &in_count);
    if (FLAGS_int8) {
      cnrtSetHostDataLayout(inputDesc, CNRT_UINT8, CNRT_NCHW);
    } else {
      cnrtSetHostDataLayout(inputDesc, CNRT_FLOAT32, CNRT_NCHW);
    }
    cnrtGetDataShape(inputDesc, &in_n, &in_c, &in_h, &in_w);
    if (!use_rtctx) {
      in_count *= FLAGS_data_parallelism;
      in_n *= FLAGS_data_parallelism;
    } else {
      LOG(INFO) << "in_n is "<< in_n << " in_count is "<< in_count;
    }
    // LOG(INFO) << "################# i:" << i;
    // LOG(INFO) << "shape " << in_n;
    // LOG(INFO) << "shape " << in_c;
    // LOG(INFO) << "shape " << in_h;
    // LOG(INFO) << "shape " << in_w;
    if (i == 0) {
      in_n_ = in_n;
      in_c_ = in_c;
      in_w_ = in_w;
      in_h_ = in_h;
    } else {
      cnrtGetHostDataCount(inputDesc, &in_count);
    }
  }

  for (int i = 0; i < outputNum; i++) {
    cnrtDataDesc_t outputDesc = outputDescS_[i];
    cnrtSetHostDataLayout(outputDesc, CNRT_FLOAT32, CNRT_NCHW);
    cnrtGetHostDataCount(outputDesc, &out_count_[i]);
    cnrtGetDataShape(outputDesc, &out_n_[i], &out_c_[i], &out_h_[i], &out_w_[i]);
    if (!use_rtctx) {
      out_count_[i] *= FLAGS_data_parallelism;
      // out_n_[i] *= FLAGS_data_parallelism;
    }
    // LOG(INFO) << "##### out shape: " << i;
    // LOG(INFO) << "output shape " << out_n_[i];
    // LOG(INFO) << "output shape " << out_c_[i];
    // LOG(INFO) << "output shape " << out_h_[i];
    // LOG(INFO) << "output shape " << out_w_[i];
  }

  // 5. allocate I/O data space on MLU memory and copy Input data
  void** inputMluPtrS;
  void** outputMluPtrS;
  int validDataSets = FLAGS_data_provider_num > FLAGS_post_processor_num ?
    FLAGS_data_provider_num : FLAGS_post_processor_num;
  for (int i = 0; i < 2 * validDataSets; i++) {
    if (!use_rtctx) {
      cnrtMallocBatchByDescArray(
          &inputMluPtrS ,
          inputDescS_,
          inputNum,
          FLAGS_data_parallelism);
      cnrtMallocBatchByDescArray(
          &outputMluPtrS,
          outputDescS_,
          outputNum,
          FLAGS_data_parallelism);
    } else {
      cnrtMallocByDescArray(
        &inputMluPtrS ,
        inputDescS_,
        inputNum);
      cnrtMallocByDescArray(
        &outputMluPtrS,
        outputDescS_,
        outputNum);
    }

    freeInputFifo_.push(inputMluPtrS);
    freeOutputFifo_.push(outputMluPtrS);
  }

  dim_ = {1, 1, 1};
  if (!use_rtctx) {
    invoke_param_.data_parallelism = &FLAGS_data_parallelism;
    invoke_param_.end = CNRT_PARAM_END;
  }
}

void** Inferencer::popFreeInputData() {
  return freeInputFifo_.pop();
}

void** Inferencer::popValidInputData() {
  return validInputFifo_.pop();
}

void Inferencer::pushFreeInputData(void** data) {
  freeInputFifo_.push(data);
}

void Inferencer::pushValidInputData(void** data) {
  validInputFifo_.push(data);
}

void** Inferencer::popFreeOutputData() {
  return freeOutputFifo_.pop();
}

void** Inferencer::popValidOutputData() {
  return validOutputFifo_.pop();
}

void Inferencer::pushFreeOutputData(void** data) {
  freeOutputFifo_.push(data);
}

void Inferencer::pushValidOutputData(void** data) {
  validOutputFifo_.push(data);
}

void Inferencer::pushValidInputNames(vector<string> images) {
  imagesFifo_.push(images);
}

vector<string> Inferencer::popValidInputNames() {
  return imagesFifo_.pop();
}

void Inferencer::notify() {
  running_--;
}

void Inferencer::run() {
  cnrtDev_t dev;
  CNRT_CHECK(cnrtGetDeviceHandle(&dev, FLAGS_device_id));
  CNRT_CHECK(cnrtSetCurrentDevice(dev));

  // func_type_ = CNRT_FUNC_TYPE_BLOCK;
  if (FLAGS_data_parallelism * FLAGS_model_parallelism <= 8) {
    cnrtSetCurrentChannel((cnrtChannelType_t)(thread_id_ % CHANNEL_NUM));
    // func_type_ = CNRT_FUNC_TYPE_UNION2;
  }

  cnrtFunction_t function_;
  cnrtCreateFunction(&function_);
  cnrtCopyFunction(&function_, function);
  bool muta = false;
  cnrtInitFuncParam_t init_param;
  init_param.muta = &muta;
  init_param.data_parallelism = &FLAGS_data_parallelism;
  init_param.end = CNRT_PARAM_END;

  // 6. initialize function memory
  cnrtInitFunctionMemory_V2(function_, &init_param);

  cnrtCreateQueue(&queue_);
  // initliaz function memory
  cnrtInitFunctionMemory_V2(function_, &init_param);
  // create start_notifier and end_notifier
  cnrtNotifier_t notifier_start, notifier_end;
  cnrtCreateNotifier(&notifier_start);
  cnrtCreateNotifier(&notifier_end);
  float notifier_time_use;
  // void **param = (void **)malloc(sizeof(void *) * (inputNum + outputNum));
  void **param = reinterpret_cast<void **>(malloc(sizeof(void *) * (inputNum + outputNum)));
  while (running_ || validInputFifo_.size()) {
    void** mlu_input_data = validInputFifo_.pop();
    if (mlu_input_data) {
      void** mlu_output_data = freeOutputFifo_.pop();
      // LOG(INFO) << "Invoke function ...";
      for (int i = 0; i < inputNum; i++) {
        param[i] = mlu_input_data[i];
      }
      for (int i = 0; i < outputNum; i++) {
        param[inputNum + i] = mlu_output_data[i];
      }
      cnrtPlaceNotifier(notifier_start, queue_);
      CNRT_CHECK(cnrtInvokeFunction_V2(function_, dim_, param,
                         func_type_, queue_, reinterpret_cast<void*>(&invoke_param_)));
      cnrtPlaceNotifier(notifier_end, queue_);
      CNRT_CHECK(cnrtSyncQueue(queue_))
      cnrtNotifierDuration(notifier_start, notifier_end, &notifier_time_use);
      invoke_time += notifier_time_use;
      LOG(INFO) << " execution time: " << notifier_time_use << " us";

      pushValidOutputData(mlu_output_data);
      pushFreeInputData(mlu_input_data);
    }
  }

  /* push exiting notify into ValidOutputData,
   * the number of exiting notify is equal the number of post_processors
   */
  for (int i = 0; i < post_processor_num_; i++) {
    pushValidOutputData(nullptr);
  }
  free(param);
  cnrtDestroyNotifier(&notifier_start);
  cnrtDestroyNotifier(&notifier_end);
  cnrtDestroyFunction(function_);
}

void Inferencer::run_with_rtctx() {
  LOG(INFO) << "use run_with_rtctx FLAGS_duplicate_channel is " << FLAGS_duplicate_channel;
  cnrtDev_t dev;
  CNRT_CHECK(cnrtGetDeviceHandle(&dev, 0));
  CNRT_CHECK(cnrtSetCurrentDevice(dev));

  // not need copy function,use function directly
  cnrtRet_t ret;
  cnrtRuntimeContext_t ctx;

  // 1. create runtime context with function
  ret = cnrtCreateRuntimeContext(&ctx, function, NULL);
  if (ret != CNRT_RET_SUCCESS) {
    LOG(FATAL) << "create runtime context failed!";
    return;
  }

  // 2. set channel and device id
  cnrtChannelType_t channel = FLAGS_duplicate_channel ? CNRT_CHANNEL_TYPE_DUPLICATE : CNRT_CHANNEL_TYPE_NONE;
  cnrtSetRuntimeContextChannel(ctx, channel);
  cnrtSetRuntimeContextDeviceId(ctx, FLAGS_device_id);
  ret = cnrtInitRuntimeContext(ctx, NULL);
  if (ret != CNRT_RET_SUCCESS) {
    LOG(FATAL) << "Init runtime context failed! ";
    return;
  }

  // 3. create queue with ctx
  cnrtQueue_t queue;
  cnrtNotifier_t notifier_start;
  cnrtNotifier_t notifier_end;
  cnrtRuntimeContextCreateQueue(ctx, &queue);
  cnrtRuntimeContextCreateNotifier(ctx, &notifier_start);
  cnrtRuntimeContextCreateNotifier(ctx, &notifier_end);

  // 4. runtime context run in queue
  float notifier_time_use = 0.0f;
  void **param = reinterpret_cast<void **>(malloc(sizeof(void *) * (inputNum + outputNum)));
  while (running_ || validInputFifo_.size()) {
    void** mlu_input_data = validInputFifo_.pop();
    if (mlu_input_data) {
      void** mlu_output_data = freeOutputFifo_.pop();
      for (int i = 0; i < inputNum; i++) {
        param[i] = mlu_input_data[i];
      }
      for (int i= 0; i < outputNum; i++) {
        param[inputNum + i] = mlu_output_data[i];
      }
      cnrtPlaceNotifier(notifier_start, queue);
      CNRT_CHECK(cnrtInvokeRuntimeContext(ctx, param, queue, NULL));
      cnrtPlaceNotifier(notifier_end, queue);
      CNRT_CHECK(cnrtSyncQueue(queue));
      cnrtNotifierDuration(notifier_start, notifier_end, &notifier_time_use);
      invoke_time += notifier_time_use;
      LOG(INFO) << " execution time: " << notifier_time_use << " us";

      pushValidOutputData(mlu_output_data);
      pushFreeInputData(mlu_input_data);
    }
  }

  // 5. notfiy to post and free runtime context resource
  /* push exiting notify into ValidOutputData,
   * the number of exiting notify is equal the number of post_processors
   */
  for (int i = 0; i < post_processor_num_; i++) {
    pushValidOutputData(nullptr);
  }
  free(param);
  cnrtDestroyQueue(queue);
  cnrtDestroyRuntimeContext(ctx);
  cnrtDestroyNotifier(&notifier_start);
  cnrtDestroyNotifier(&notifier_end);
}

Inferencer::~Inferencer() {
  while (freeInputFifo_.size()) {
    cnrtFreeArray(freeInputFifo_.pop(), inputNum);
  }
  while (freeOutputFifo_.size()) {
    cnrtFreeArray(freeOutputFifo_.pop(), outputNum);
  }
  if (!use_rtctx) {
    cnrtDestroyQueue(queue_);
  }
  cnrtDestroyFunction(function);
}

PostProcessor::PostProcessor() {
  thread_id_ = 0;
  total_ = 0;
}

void PostProcessor::RectangleAndDrawResult(float* ids_data,
                                           float* bbox_data,
                                           float* score_data,
                                           float conf_thresh,
                                           string intorigin_img,
                                           int idx) {
  cv::Mat *result_img;
  std::string img_dir = FLAGS_img_dir + intorigin_img;
  cv::Mat img = cv::imread(img_dir, -1);
  result_img = &img;
  // string id2name[20] = {
  //         "aeroplane", "bicycle", "bird", "boat",
  //         "bottle", "bus", "car", "cat", "chair",
  //         "cow", "diningtable", "dog", "horse",
  //         "motorbike", "person", "pottedplant",
  //         "sheep", "sofa", "train", "tvmonitor"};
    string id2name[2] = {"hat", "person"};

  detection_bbox *boxs = reinterpret_cast<detection_bbox*>(bbox_data);
  int index;

  for (int i = 0; i < 100; i++) {
    if (ids_data[i] < 0 || score_data[i] < conf_thresh) {
      continue;
    }
    index = static_cast<int>(ids_data[i]);
    cv::Point p1, p2;
    p1.x = boxs[i].x_min * img.cols / (width * 1.0);
    p1.y = boxs[i].y_min * img.rows / (height * 1.0);
    p2.x = boxs[i].x_max * img.cols / (width * 1.0);
    p2.y = boxs[i].y_max * img.rows / (height * 1.0);
    cv::rectangle(*result_img, p1, p2, cv::Scalar(0, 255, 0), 8, 8, 0);
    std::stringstream s0;
    s0 << score_data[i];
    string s00 = s0.str();

    cv::putText(*result_img, id2name[index],
        cv::Point(p1.x, (p1.y + p2.y)/2 - 10), 2, 0.5,
        cv::Scalar(255, 0, 0), 0, 8, 0);
    cv::putText(*result_img, s00.c_str(),
        cv::Point(p1.x, (p1.y + p2.y)/2 + 10), 2, 0.5,
        cv::Scalar(255, 0, 0), 0, 8, 0);
  }

  string img_name;
  std::stringstream ss;
  ss << "./yolov3/detect_" << intorigin_img;
  ss >> img_name;
  cv::imwrite(img_name, *result_img);

  total_++;
}

void PostProcessor::RectangleAndPrintResult(float* ids_data,
                                            float* bbox_data,
                                            float* score_data,
                                            float conf_thresh,
                                            string intorigin_img,
                                            int idx) {
  detection_bbox *boxs = reinterpret_cast<detection_bbox*>(bbox_data);
  int index;
  // string id2name[20] = {
  //         "aeroplane", "bicycle", "bird", "boat",
  //         "bottle", "bus", "car", "cat", "chair",
  //         "cow", "diningtable", "dog", "horse",
  //         "motorbike", "person", "pottedplant",
  //         "sheep", "sofa", "train", "tvmonitor"};
  string id2name[2] = {"hat", "person"};

  for (int i = 0; i < 100; i++) {
    boxs[i].x_max /= (width * 1.0);
    boxs[i].x_min /= (width * 1.0);
    boxs[i].y_max /= (height * 1.0);
    boxs[i].y_min /= (height * 1.0);
  }

  std::ofstream fout("yolov3/" + intorigin_img + ".txt");
  if (!fout) {
    std::cout << "yolov3/" + intorigin_img + ".txt" << std::endl;
    exit(0);
  }

  for (int i = 0; i < 100; i++) {
    if (ids_data[i] < 0 || score_data[i] < conf_thresh) {
      continue;
    }
    index = static_cast<int>(ids_data[i]);
    fout << id2name[index] << " " << score_data[i] << " "
         << boxs[i].x_min << " " << boxs[i].y_min << " "
         << boxs[i].x_max << " " << boxs[i].y_max << "\n";
  }

  fout.close();
  total_++;
}

void PostProcessor::run() {
  cnrtDev_t dev;
  CNRT_CHECK(cnrtGetDeviceHandle(&dev, FLAGS_device_id));
  CNRT_CHECK(cnrtSetCurrentDevice(dev));

  if (!use_rtctx && FLAGS_data_parallelism * FLAGS_model_parallelism <= 8) {
    cnrtSetCurrentChannel((cnrtChannelType_t)(thread_id_ % CHANNEL_NUM));
  }

  void** outputCpuPtrS;
  outputCpuPtrS = reinterpret_cast<void**>(malloc (sizeof(void*) * inferencer_->outputNum));
  for (int i = 0; i < inferencer_->outputNum; i++) {
    outputCpuPtrS[i] = reinterpret_cast<void*>
      (malloc(sizeof(float) * inferencer_->out_count_[i]));
  }

  while (1 /* run until get a exiting notify */) {
    // use mutex to ensure result is match for origin_img
    std::unique_lock<std::mutex> lock(inferencer_->post_mtx);
    void** mlu_output_data = inferencer_->validOutputFifo_.pop();
    if (mlu_output_data) {
      // get img names matching the results
      vector<string> origin_img = inferencer_->popValidInputNames();
      // get result && img names unlock
      lock.unlock();

      LOG(INFO) << "memcpy to host ...";
      if (!use_rtctx) {
        CNRT_CHECK(cnrtMemcpyBatchByDescArray(
            outputCpuPtrS, mlu_output_data, inferencer_->outputDescS_,
            inferencer_->outputNum, FLAGS_data_parallelism, CNRT_MEM_TRANS_DIR_DEV2HOST));
      } else {
        CNRT_CHECK(cnrtMemcpyByDescArray(
            outputCpuPtrS, mlu_output_data, inferencer_->outputDescS_,
            inferencer_->outputNum, CNRT_MEM_TRANS_DIR_DEV2HOST));
      }

      inferencer_->pushFreeOutputData(mlu_output_data);

      float* ids_result = reinterpret_cast<float*>(outputCpuPtrS[51]);
      float* score_result = reinterpret_cast<float*>(outputCpuPtrS[52]);
      float* bbox_result = reinterpret_cast<float*>(outputCpuPtrS[53]);

      unsigned int output_chw[3];
      for (unsigned int i = 51; i < 54; i++) {
        output_chw[i-51] = inferencer_->output_chw(i);
      }

      // LOG(INFO) << "output_nchw[0] = " << output_chw[0];
      // LOG(INFO) << "output_nchw[1] = " << output_chw[1];
      // LOG(INFO) << "output_nchw[2] = " << output_chw[2];
      // std::cout << "output_chw = " << output_chw << std::endl;
      unsigned int addr_offset = 0;

      auto ids_n_ = inferencer_->out_n(51);
      auto ids_c_ = inferencer_->out_c(51);
      auto ids_h_ = inferencer_->out_h(51);
      auto ids_w_ = inferencer_->out_w(51);
      // auto ids_size = ids_n_ * ids_c_ * ids_h_ * ids_w_;

      auto score_n_ = inferencer_->out_n(52);
      auto score_c_ = inferencer_->out_c(52);
      auto score_h_ = inferencer_->out_h(52);
      auto score_w_ = inferencer_->out_w(52);
      // auto score_size = score_n_ * score_c_ * score_h_ * score_w_;

      auto bbox_n_ = inferencer_->out_n(53);
      auto bbox_c_ = inferencer_->out_c(53);
      auto bbox_h_ = inferencer_->out_h(53);
      auto bbox_w_ = inferencer_->out_w(53);

      float conf_thresh = 0.0;

      float *ids_data = ids_result;
      float *bbox_data = bbox_result;
      float *score_data = score_result;

      for (unsigned int i = 0; i < inferencer_->n(); i++) {
        if (origin_img[i] != "null") {
          int idx = i;
          if (FLAGS_output_mode == "picture") {
            conf_thresh = 0.5;
            RectangleAndDrawResult(ids_data, bbox_data, score_data, conf_thresh,
                                  origin_img[i], idx);
          } else {
            conf_thresh = 0.0;
            RectangleAndPrintResult(ids_data, bbox_data, score_data, conf_thresh,
                                  origin_img[i], idx);
          }
        }

        ids_data += output_chw[0];
        score_data += output_chw[1];
        bbox_data += output_chw[2];
      }
    } else {
      // get exiting notify and unlock the lock
      lock.unlock();
      break;
    }
  }

  for (int i = 0; i < inferencer_->outputNum; i++) {
    free(outputCpuPtrS[i]);
  }
  free(outputCpuPtrS);
}

template<class datatype>
class Pipeline {
 public:
  Pipeline(const string& mean_file,
            const string& mean_value,
            const int& thread_id,
            const int& data_parallelism,
            vector<queue<string> > images,
            const cnrtModel_t& model,
            float scale = 1.0);
  ~Pipeline();

  vector<DataProvider<datatype>*> data_provider_;
  Inferencer* inferencer_;
  vector<PostProcessor*> post_processor_;
  void run();
};

template<class datatype>
Pipeline<datatype>::Pipeline(const string& mean_file,
                   const string& mean_value,
                   const int& thread_id,
                   const int& data_parallelism,
                   vector<queue<string> > images,
                   const cnrtModel_t& model,
                   float scale) : data_provider_(FLAGS_data_provider_num),
                   post_processor_(FLAGS_post_processor_num) {
  inferencer_ = new Inferencer(thread_id,
      data_parallelism,
      model);
  for (int i = 0; i < FLAGS_data_provider_num; i++) {
    data_provider_[i] = new DataProvider<datatype>(
        mean_file,
        mean_value,
        images[thread_id * FLAGS_data_provider_num + i],
        scale);
    cout << "images[[[[[[[[[[[]]]]]]]]]]]]]]]]]]=++++++++++++++++++++" << typeid(images[thread_id * FLAGS_data_provider_num + i]).name() << endl;
    data_provider_[i]->inferencer_ = inferencer_;
    data_provider_[i]->thread_id_ = thread_id;
    // if (FLAGS_pre_read) {
    //   data_provider_[i]->preRead(); //读图片
    // }
  }
  for (int i = 0; i < FLAGS_post_processor_num; i++) {
    post_processor_[i] = new PostProcessor();
    post_processor_[i]->inferencer_ = inferencer_;
    post_processor_[i]->thread_id_ = thread_id;
  }
  inferencer_->post_processor_ = post_processor_;
  inferencer_->thread_id_ = thread_id;
}

template<class datatype>
Pipeline<datatype>::~Pipeline() {
  for (int i = 0; i < FLAGS_data_provider_num; i++) {
    if (data_provider_[i] != nullptr) {
      delete data_provider_[i];
      data_provider_[i] = nullptr;
    }
  }
  if (inferencer_ != nullptr) {
    delete inferencer_;
    inferencer_ = nullptr;
  }
  for (int i = 0; i < FLAGS_post_processor_num; i++) {
    if (post_processor_[i] != nullptr) {
      delete post_processor_[i];
      post_processor_[i] = nullptr;
    }
  }
}

template<class datatype>
void Pipeline<datatype>::run() {
  vector<thread*> threads(FLAGS_data_provider_num + FLAGS_post_processor_num + 1, nullptr);
  for (int i = 0; i < FLAGS_data_provider_num; i++) {
    cout << "预处理线程------------------DataProvider" << ""<< endl;
    threads[i] = new thread(&DataProvider<datatype>::run, data_provider_[i]);
  }
  if (!use_rtctx) {
    cout << "推理线程------------------inference" << ""<< endl;
    threads[FLAGS_data_provider_num] = new thread(&Inferencer::run, inferencer_);
  } else {
    LOG(INFO) << "use runtime context";
    threads[FLAGS_data_provider_num] = new thread(&Inferencer::run_with_rtctx, inferencer_);
  }
  for (int i = 0; i < FLAGS_post_processor_num; i++) {
    cout << "后处理线程------------------PostProcessor" << ""<< endl;
    threads[FLAGS_data_provider_num + 1 + i] = new thread(&PostProcessor::run, post_processor_[i]);
  }
  for (auto th : threads)
    th->join();
  for (auto &th : threads) {
    if (th != nullptr) {
      delete th;
      th = nullptr;
    }
  }
}

void check_args(void) {
  if (FLAGS_use_mean != "on" && FLAGS_use_mean != "off") {
    LOG(ERROR) << "use_mean should be set on or off";
    exit(-1);
  }
  if (!use_rtctx) {
    if (FLAGS_data_parallelism < 1 || FLAGS_data_parallelism > 32) {
      LOG(ERROR) << "data_parallelism should be LE 32, recommendation is [1 2 4 8 16 32]";
      exit(-1);
    }
    if (FLAGS_model_parallelism < 1 || FLAGS_model_parallelism > 32) {
      LOG(ERROR) << "model_parallelism should be LE 32, recommendation is [1 2 4 8]";
      exit(-1);
    }
    if (FLAGS_data_parallelism * FLAGS_model_parallelism > 32) {
      LOG(ERROR) << "model_parallelism * data_parallelism should be LE 32";
      exit(-1);
    }
  }

  if (FLAGS_int8 != 0 && FLAGS_int8 != 1) {
    LOG(ERROR) << "int8 should be set 0 or 1";
    exit(-1);
  }
  if (FLAGS_output_mode != "picture" && FLAGS_output_mode != "text" && FLAGS_output_mode != "screenonly") {
    LOG(ERROR) << "output_mode should be set picture or text or screenonly";
    exit(-1);
  }
  return;
}

template<class datatype>
void Process(vector<queue<string>> img_list) {
  vector<thread*> pipelines;
  vector<Pipeline<datatype>*> pipeline_instances;
  /* load model */
  cnrtModel_t model;
  LOG(INFO) << "load file: " << FLAGS_offlinemodel.c_str();
  CNRT_CHECK(cnrtLoadModel(&model, FLAGS_offlinemodel.c_str()));
  int count = 0;
  for (int i = 0; i < FLAGS_threads; i++) {
    if (img_list[i * FLAGS_data_provider_num].size()) {
      Pipeline<datatype>* pipeline = new Pipeline<datatype>(FLAGS_mean_file,
          FLAGS_mean_value,
          i,
          FLAGS_data_parallelism,
          img_list,
          model,
          FLAGS_scale);
      pipeline_instances.push_back(pipeline);
      pipelines.push_back(new thread(&Pipeline<datatype>::run, pipeline));
      count++;
    }
  }
  cout << count<<"count" << endl;
  cout << img_list.size() << " img_list.size()--------------------------" << endl; 
  cout << pipeline_instances.size() << "pipeline_instances.size()--------------------------" << endl; 
  cout << pipelines.size() << "pipelines.size()--------------------------" << endl; 

  double time_use;
  struct timeval tpend, tpstart;
  gettimeofday(&tpstart, NULL);
  {
    std::lock_guard<std::mutex> lk(condition_m);
    ready_start = true;
    LOG(INFO) << "Notify to start ...";
  }
  condition.notify_all();
  for (int i = 0; i < FLAGS_threads; i++) {
    pipelines[i]->join();
  }
  cnrtUnloadModel(model);
  gettimeofday(&tpend, NULL);

  int64_t total = 0;
  double avg_invoke_time = 0.;
  for (int i = 0; i < FLAGS_threads; i++) {
    for (int j = 0; j < FLAGS_post_processor_num; j++) {
      total += pipeline_instances[i]->post_processor_[j]->total_;
    }
    avg_invoke_time += pipeline_instances[i]->inferencer_->invoke_time;
  }

  std::cout << "---------------------" << std::endl;
  std::cout << "Detecting: " << total << std::endl;
  // avg_invoke_time = avg_invoke_time / (double)FLAGS_threads;
  unsigned int real_parallelism = FLAGS_threads;
  if (!use_rtctx && FLAGS_data_parallelism * FLAGS_model_parallelism * FLAGS_threads > 32)
    real_parallelism = 32 / (FLAGS_data_parallelism * FLAGS_model_parallelism);
  avg_invoke_time = avg_invoke_time / static_cast<double>(real_parallelism);
  if (!use_rtctx) {
    std::cout << "data_parallelism: " << FLAGS_data_parallelism << " "
      << "and there are " << FLAGS_threads << " threads."<< std::endl;
  }

  for (int i = 0; i < FLAGS_threads; i++) {
    std::cout << "Thread" << i << " hardware time sum = "
              << pipeline_instances[i]->inferencer_->invoke_time << " us" << std::endl;
  }
  std::cout << "for only inference, " << total / (avg_invoke_time / 1000000)
    << " fps" << std::endl;

  time_use = 1000000 * (tpend.tv_sec - tpstart.tv_sec)
    + tpend.tv_usec - tpstart.tv_usec;
  std::cout << "for end2end, " << total / (time_use / 1000000)
    << " fps" << std::endl;

  if (FLAGS_output_mode == "text") {
    char result_char[32] = {0};
    char *result_str = result_char;
    snprintf(result_str, sizeof(result_char), "./run.sh yolov3 %d", result_num);
    std::string cmd = std::string(result_char);
    std::system(cmd.c_str());
  }
  for (int i = 0; i < FLAGS_threads; i++) {
    if (pipelines[i] != nullptr) {
      delete pipelines[i];
      pipelines[i] = nullptr;
    }
    if (pipeline_instances[i] != nullptr) {
      delete pipeline_instances[i];
      pipeline_instances[i] = nullptr;
    }
  }
  return;
}

int main(int argc, char* argv[]) {
  const char* rtctx_env = getenv("MXNET_EXEC_FUSE_MLU_OPS_USE_RTCTX");
  if (rtctx_env && !strcmp(rtctx_env, "true")) {
    use_rtctx = true;
  }

  LOG(INFO) << "use rtctx : "<< use_rtctx;

  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::ifstream files_tmp(FLAGS_images.c_str(), std::ios::in);
  cout << FLAGS_images.c_str() << "++++++++++++++++++++++++++++++++flags.images" << endl;
  int image_num = 0;
  vector<string> files;
  std::string line_tmp;
  int queue_num = FLAGS_threads * FLAGS_data_provider_num;
  vector<queue<string>> img_list(queue_num);
  if (files_tmp.fail()) {
    LOG(ERROR) << "open " << FLAGS_images  << " file fail!";
    return 1;
  } else {
    while (getline(files_tmp, line_tmp)) {
      img_list[image_num%queue_num].push(line_tmp);
      image_num++;
      if (image_num >= FLAGS_max_images_num) {
        break;
      }
    }
  }
  for (int i = 0;i<img_list.size();i++){
    cout << img_list[i].size() << "img_list_queue----------=============" << endl;
  }
  cout << img_list.size() << "--------------------------imglist" << endl; 
  files_tmp.close();
  result_num = image_num;
  check_args();

  cnrtInit(0);

  VideoCapture capture;
  Mat frame;
  frame= capture.open("/home/Cambricon-Test/nanbei.mp4");
  if(!capture.isOpened())
  {
      printf("can not open ...\n");
      return -1;
  }
  vector<cv::Mat> imgs;
  while (capture.read(frame))
  {
    if (FLAGS_int8) {

    imgs.push_back(frame);
    v_images.push_back(imgs);
    Process<uint8_t>(img_list);
    } else {
    Process<float>(img_list);
    }
    imgs.clear();
    v_images.clear();
  }

  // for (int i = 0; i < FLAGS_iter_num; ++i) {
  //   if (FLAGS_int8) {
  //     Process<uint8_t>(img_list);
  //   } else {
  //     Process<float>(img_list);
  //   }
  // }
  cnrtDestroy();
}
#endif

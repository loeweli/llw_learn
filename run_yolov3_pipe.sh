#!/bin/bash
# name: run_yolov3_pipe.sh
current_dir=$(dirname "${0}")
pushd ${current_dir}

if [ -z $MXNET_MODELS_DIR ]; then
    echo "MXNET_MODELS_DIR is not exit!"
    exit -1
fi

if [ -z $MXNET_DATA_DIR ]; then
    echo "MXNET_DATA_DIR is not exit!"
    exit -1
fi

if [[ $# = 7 ]];
then
    datatype="$1"
    mode="$2"
    batch="$3"
    data_parallelism="$4"
    model_parallelism="$5"
    threads="$6"
    output_mode="$7"
    data_preprocess_strategy="CNML_CPU_MLU_BALANCE"
elif [[ $# = 8 ]];
then
    datatype="$1"
    mode="$2"
    batch="$3"
    data_parallelism="$4"
    model_parallelism="$5"
    threads="$6"
    output_mode="$7"
    data_preprocess_strategy="$8"

else
    echo ""
    echo "Usage: ./run_yolov3_pipe.sh datatype mode batch data_parallelism model_parallelism threads output_mode data_preprocess_strategy"
    echo "datatype: float16 or int8"
    echo "mode: dense or sparse"
    echo "batch: 1/2/4/etc"
    echo "data_parallelism:recommendation is [1/2/4/8/16/32]"
    echo "model_parallelism:recommendation is [1/2/4/8]. Note: (dp * mp) must be in [1 32]"
    echo "threads:recommendation is [1/2/4]"
    echo "output_mode: picture or text. picture:plot picture, text:output to text"
    echo "[option] data_preprocess_strategy:[CNML_CPU_MLU_BALANCE/CNML_CPU_PRIORITY/CNML_MLU_PRIORITY]"
    exit -1
fi

if [ -d "yolov3" ]; then
    /bin/rm -rf yolov3
fi
mkdir yolov3

if [[ $mode == "dense" ]];
then
    mode="dense"
elif [[ $mode == "sparse" ]];
then
    mode="sparse"
    echo "yolov3 only support float16 dense mode so far."
    exit -1
fi

if [[ $datatype == "int8" ]];
then
    is_int8="int8"
    int8="1"
    path_postfix="int8"
    echo "yolov3 only support float16 dense mode so far."
    exit -1

elif [[ $datatype == "float16" ]];
then
    is_int8="float16"
    int8="0"
    path_postfix="float16"
fi

if [[ $output_mode == "picture" ]];
then
    output_mode="picture"
elif [[ $output_mode == "text" ]];
then
    output_mode="text"
fi

function init {
    exe_path="/build/examples/yolov3_offline_multicore_pipe"
    model_path="${MXNET_MODELS_DIR}/offline/"
    images="detection_img_list"
    #images="img_list_dete"
    use_mean="on"
    img_dir="${MXNET_DATA_DIR}/VOC2007_original/JPEGImages/"
    scale="1"
    pipe_flag="pipe"
    height="416"
    width="416"
    offlinemodel=${model_path}yolov3_darknet53_${datatype}_${mode}-symbol_batch${batch}_mp${model_parallelism}_${data_preprocess_strategy}_fusion_1.cambricon
    int8=0
    data_provider_num=1
    post_processor_num=1
    max_images_num=10
    pre_read=true
}

init

function run {
    cmd=".${exe_path} -offlinemodel ${offlinemodel} -data_parallelism $data_parallelism -model_parallelism
         ${model_parallelism} -threads $threads -img_dir $img_dir -use_mean $use_mean -output_mode ${output_mode}
          -scale ${scale} -images $images -int8 $int8 -data_provider_num ${data_provider_num}
          -post_processor_num ${post_processor_num} -max_images_num ${max_images_num} -pre_read=${pre_read}"
    echo $cmd
    $cmd 2>&1 | tee ${offlinemodel}${pipe_flag}log
}

run
echo "$offlinemodel run over"
popd


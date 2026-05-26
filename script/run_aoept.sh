#!/bin/bash

arch=$1
dataset=$2
missing_type=$3
missing_rate=$4
n_clusters=${5:-256}
python core/main.py --config-name "${arch}_${dataset}_infer" data.missing_type="${missing_type}" data.missing_rate="${missing_rate}" para.missing_type="${missing_type}"
python core/model/AOEPT/preprocess/cluster_tokens.py --input "cache/collect_token/${arch}-${dataset}-${missing_type}-${missing_rate}-mean.pt" --output "cache/collect_token_cluster_${n_clusters}/${arch}-${dataset}-${missing_type}-${missing_rate}-mean.pt" --n_clusters "${n_clusters}"
python core/main.py --config-name "AOEPT_${arch}_${dataset}_${missing_type}" data.missing_type="${missing_type}" data.missing_rate="${missing_rate}" para.missing_type="${missing_type}" para.init_from_token="cache/collect_token_cluster_${n_clusters}/${arch}-${dataset}-${missing_type}-${missing_rate}-mean.pt"

##########################################################################################################
# Author: Peter Karacsonyi                                                                               #
# Last updated: 2025 apr 7                                                                              #
# Runner script for HuggingFace Embedding Inference Server                                               #
# Main github project site: https://github.com/huggingface/text-embeddings-inference                     #
# API endpoints: https://huggingface.github.io/text-embeddings-inference/                                #
# Tags: https://github.com/huggingface/text-embeddings-inference/pkgs/container/text-embeddings-inference#
##########################################################################################################

# WARNING: sometimes it takes 5-6 minutes for the model to start, for example:
# 2024-12-22T22:00:29.017555Z  INFO text_embeddings_backend_candle: backends/candle/src/lib.rs:373: Starting FlashQwen2 model on Cuda(CudaDevice(DeviceId(1)))
# 2024-12-22T22:05:36.201611Z  INFO text_embeddings_router: router/src/lib.rs:248: Warming up model

# or 20+ mins, after warming up
# 2025-04-03T13:59:39.238702Z  INFO text_embeddings_router: router/src/lib.rs:252: Warming up model
# 2025-04-03T14:19:01.323126Z  INFO text_embeddings_router::http::server: router/src/http/server.rs:1804: Starting HTTP server: 0.0.0.0:3000

# Dynamically set the location to the directory of this script
Set-Location -Path $PSScriptRoot

$model = "NovaSearch/stella_en_1.5B_v5"
$volume = "${PSScriptRoot}\models\" # Use the script's location for models directory
$prompt_name = "s2p_query"
# either we use the default_prompt of the name
# $default_prompt = "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "
$docker_container_name = "huggingface-embeddings-inference"
$containerport = 3000 # the huggingface server inside listens on this port
$localport = 3001
# best image reference for ADA RTX https://github.com/huggingface/text-embeddings-inference?tab=readme-ov-file#docker-images
$image = "ghcr.io/huggingface/text-embeddings-inference:cpu-ipex-1.6"


##################################
### MAX SEQ LEN QUESTION BEGIN ###
# would be nice to set the max token length however it is not yet implemented as a switch of the router:
# https://github.com/huggingface/text-embeddings-inference/issues/396
# solution: there is a max_seq_len in the file models\models--dunzhang--stella_en_1.5B_v5\blobs\5116148bff3b077908431eba25e9f8541bc17469
# however to make sure, now we use --auto-truncate to deal with longer sequences
#### MAX SEQ LEN QUESTION END #####
###################################


#####################################
### OUTUT DIMENSION QUESTION BEGIN ###
# the word_embedding_dimension is set to 1536 in models\models--dunzhang--stella_en_1.5B_v5\blobs\3e006833f60b3c3101dd5af977ef000b76cc9b7d, 
# however changing that does not change the output. There is nothing in the initially downloaded files we can change afaik 
### OUTUT DIMENSION QUESTION END ###
####################################


####################################################
### DOWNLOADED VS LOCAL MODEL USE QUESTION BEGIN ###
# After a model is being downloaded, we can modify the files e.g. above to modify the model itself.
# can we work on it with transformers maybe?
#### DOWNLOADED VS LOCAL MODEL USE QUESTION END ####
####################################################

docker stop $docker_container_name
docker rm $docker_container_name

# Check if anything is already listening on local port
if (Get-NetTCPConnection -LocalPort $localport -ErrorAction SilentlyContinue) {
    Write-Host "Error: Port $localport is already in use. Please stop the process using this port and try again." -ForegroundColor Red
    exit 1
}

# Run the Docker container
docker run -p ${localport}:$containerport -v "${volume}:/data" --name $docker_container_name $image `
    --model-id $model --default-prompt-name $prompt_name --port $containerport --hostname 0.0.0.0 --auto-truncate

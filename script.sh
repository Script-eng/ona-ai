# ======================================================================
# The Definitive Download Script for SadTalker
# ======================================================================

echo "--- Creating necessary directories ---"
mkdir -p ./SadTalker/checkpoints
mkdir -p ./SadTalker/gfpgan/weights

echo "\n--- Downloading Main SadTalker Models ---"
# These are the core models for audio processing and face reconstruction
wget -nc -P ./SadTalker/checkpoints https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/epoch_20.pth
wget -nc -P ./SadTalker/checkpoints https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/auido2pose_00140-model.pth
wget -nc -P ./SadTalker/checkpoints https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/auido2exp_00300-model.pth

echo "\n--- Downloading Renderer and Mapping Models ---"
wget -nc -P ./SadTalker/checkpoints https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/mapping_00109-model.pth.tar
wget -nc -P ./SadTalker/checkpoints https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/SadTalker_V1_256.safetensors

echo "\n--- Downloading 3D Morphable Model (BFM) ---"
wget -nc -P ./SadTalker/checkpoints https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/BFM_Fitting.zip
# The -o flag means overwrite without asking, which is safe here
unzip -o ./SadTalker/checkpoints/BFM_Fitting.zip -d ./SadTalker/checkpoints/

echo "\n--- Downloading Face Enhancer (GFPGAN) Models ---"
wget -nc -P ./SadTalker/gfpgan/weights https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

# These are additional dependencies for face detection and alignment used by GFPGAN
wget -nc -P ./SadTalker/gfpgan/weights https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth 

echo "\n\n✅✅✅ All essential models have been downloaded. ✅✅✅"
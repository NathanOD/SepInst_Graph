# SepInst Graph

## Installation

Create Python environnement:

```shell
conda create -n sepinst python=3.11 -y
conda activate sepinst
```

Install torch:

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install detectron2:

```shell
python -m pip install 'git+https://github.com/johnnynunez/detectron2.git'
```

Eeplace the model loading code in detectron2. You should specify your own detectron2 path.

```shell
cp -i tools/c2_model_loading.py path_to_detectron2/detectron2/checkpoint/c2_model_loading.py
# cp -i tools/c2_model_loading.py /home/travail/anaconda3/envs/occase/lib/python3.11/site-packages/detectron2/checkpoint/c2_model_loading.py
```

Install requirement packages:

```shell
pip install -r requirements.txt
```

Train model:

```shell
python tools/train_net.py --config-file configs/R50/config-sip-rcnn.yaml --num-gpus 1

python tools/train_net.py --config-file configs/Swin/config-sip-rcnn-swin.yaml --num-gpus 1
```

Test model:

```shell
python tools/train_net.py --config-file configs/R50/config-sip-rcnn.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS weights/RCNN-R50-SIP-50k.pth OUTPUT_DIR eval/eval-SIP-rcnn

python tools/train_net.py --config-file configs/R50/config-tcan-rcnn.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS weights/RCNN-R50-TCAN-70k.pth OUTPUT_DIR eval/eval-TCAN-rcnn

python tools/train_net.py --config-file configs/R50/config-cobo-rcnn.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS weights/RCNN-R50-COBO-40k.pth OUTPUT_DIR eval/eval-COBO-rcnn

python tools/train_net.py --config-file configs/R50/config-sip-g_r.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS weights/GCN-R50-SIP-40k.pth OUTPUT_DIR eval/eval-SIP-gcn

python tools/train_net.py --config-file configs/R50/config-tcan-g_r.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS weights/GCN-R50-TCAN-70k.pth OUTPUT_DIR eval/eval-TCAN-gcn

python tools/train_net.py --config-file configs/R50/config-cobo-g_r.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS weights/GCN-R50-COBO-60k.pth OUTPUT_DIR eval/eval-COBO-gcn
```

Visualize results:

```shell
python tools/demo.py --config-file configs/R50/config-sip-rcnn.yaml --input datasets/SIP-SEP_val/RGB/* --output results/results_SIP/ --opt MODEL.WEIGHTS weights/RCNN-R50-SIP-50k.pth

python tools/demo.py --config-file configs/R50/config-tcan-rcnn.yaml --input datasets/TCAN-D_val/RGB/* --output results/results_TCAN/ --opt MODEL.WEIGHTS weights/RCNN-R50-TCAN-70k.pth

python tools/demo.py --config-file configs/R50/config-cobo-rcnn.yaml --input datasets/COBO-SEP_val/RGB/* --output results/results_COBO/ --opt MODEL.WEIGHTS weights/RCNN-R50-COBO-40k.pth INPUT.FORMAT BGR
```

(create results folder to save all images)

Inference:

```shell
python tools/demo2.py --config-file configs/config-cobo.yaml --inference --input COBO135_U1G3/RGB/* --output-json inference/pred_COBO135_U1G3.json --opt MODEL.WEIGHTS weights/1S-R50-COBO-60k.pth
```

FLOPs and Parameters:

```shell
python tools/get_flops.py --config-file configs/R50/config-sip-g_r.yaml --tasks parameter flop
```

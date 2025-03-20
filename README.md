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
python tools/train_net.py --config-file configs/R50/config-sip.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS weights/1S-R50-SIP-30k.pth OUTPUT_DIR eval/eval-SIP

python tools/train_net.py --config-file configs/R50/config-tcan.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS weights/1S-R50-TCAN-80k.pth OUTPUT_DIR eval/eval-TCAN

python tools/train_net.py --config-file configs/R50/config-cobo.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS weights/1S-R50-COBO-60k.pth OUTPUT_DIR eval/eval-COBO

python tools/train_net.py --config-file config-OLD.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS weights/sip_oldconfig.pth OUTPUT_DIR eval/eval-SIP
```

Visualize results:

```shell
python tools/demo.py --config-file configs/config-sip.yaml --input datasets/SIP-SEP_val/RGB/* --output results/results_SIP/ --opt MODEL.WEIGHTS weights/1S-R50-SIP-30k.pth

python tools/demo.py --config-file configs/config-tcan.yaml --input datasets/TCAN-D_val/RGB/* --output results/results_TCAN/ --opt MODEL.WEIGHTS weights/1S-R50-TCAN-80k.pth

python tools/demo2.py --config-file configs/config-cobo.yaml --input datasets/COBO-SEP_val/RGB/* --output results/results_COBO/ --opt MODEL.WEIGHTS weights/1S-R50-COBO-60k.pth
```

(create results folder to save all images)

Inference:

```shell
python tools/demo2.py --config-file configs/config-cobo.yaml --inference --input COBO135_U1G3/RGB/* --output-json inference/pred_COBO135_U1G3.json --opt MODEL.WEIGHTS weights/1S-R50-COBO-60k.pth
```

FLOPs and Parameters:

```shell
python tools/get_flops.py --config-file configs/config.yaml --tasks parameter flop
```

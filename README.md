# Handwriting-Beautification


### Environment setup
```sh
conda create -n writebeau -y python=3.10
conda activate writebeau
pip install -r requirements.txt
pip install -e diffusers
```

### Train unconditional RectFlow on MNIST/EMNIST
```sh
. run.sh
```

<!--
### Train unconditional RectFlow on Flowers datatset
```sh
python train_unconditional.py \
  --dataset_name="huggan/flowers-102-categories" \
  --resolution=64 --center_crop --random_flip \
  --output_dir="rectflow-flowers-64" \
  --train_batch_size=128 \
  --num_epochs=100 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no
```
-->

#### Monitor training results
```sh
tensorboard --logdir rectflow-flowers-64
```



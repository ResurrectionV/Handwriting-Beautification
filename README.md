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


#### Classifer Usage
To train the classifier, you need to organize your dataset in a directory structure like this:
```example
printed_digits/
├── Comic_Sans/
│   ├── 0/
│   │   └── 0.png
│   ├── 1/
│   │   └── 1.png
│   ├── 2/
│   │   └── 2.png
│   └── ...
├── Didot/
│   ├── 0/
│   │   └── 0.png
│   ├── 1/
│   │   └── 1.png
│   └── ...
├── Helvetica/
│   ├── 0/
│   │   └── 0.png
│   ├── 1/
│   │   └── 1.png
│   └── ...

Open the Printed_Digits_Classifier.ipynb file.
Ensure the printed_digits/ dataset is prepared as described above
Run all cells in the notebook sequentially to:
-- Load the dataset.
-- Train the classifier.
-- Save the trained model as digit_classifier.pth.

To test the classifier, follow these steps:
-- Prepare a test dataset:
If using a custom dataset, organize it in the following structure:
hand_write_beauty_test/
├── 0/
│   ├── 0_1.png
│   ├── 0_2.png
│   └── ...
├── 1/
│   ├── 1_1.png
│   ├── 1_2.png
│   └── ...
└── ...
Then run the remaining cells in the notebook sequentially to test the classifer.
```

### Pretrained weights
https://huggingface.co/ernestchu/scriptly-yours

### Evaluation
WIP. See main.ipynb


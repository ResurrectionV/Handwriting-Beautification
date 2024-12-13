# Scriptly Yours


### Environment setup
```sh
conda create -n scriptlyyours -y python=3.10
conda activate scriptlyyours
pip install -r requirements.txt
pip install -e diffusers
```

### Train unconditional RectFlow on MNIST/EMNIST
```sh
. run.sh
```

### Pretrained weights
https://huggingface.co/ernestchu/scriptly-yours

### Evaluation
See main.ipynb


<!--
### Classifer Usage
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
-->

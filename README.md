# Teachable Machine Image Recognition Project

This project fulfills the assignment requirements:

1. Train an image-recognition model using Google Teachable Machine with at least two classes.
2. Export the trained model in TensorFlow/Keras format.
3. Use a Python script to load the model, accept an input image, and predict its class.
4. Evaluate the model and submit the Python script, exported model files, and a screenshot of the output.

## Suggested Classes

For a simple project, use:

- `Cat`
- `Dog`

You may use any two classes required by your instructor.

## 1. Install Anaconda

Install Anaconda, then open **Anaconda Prompt**.

Create a separate environment:

```bash
conda create -n teachable-machine python=3.11 -y
conda activate teachable-machine
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

## 2. Train the Model with Teachable Machine

Open Google Teachable Machine:

https://teachablemachine.withgoogle.com/

Choose:

**Image Project → Standard Image Model**

Create at least two classes, for example:

- Class 1: Cat
- Class 2: Dog

Add a good number of images to each class. Use different angles, positions, and backgrounds when possible.

Click:

**Train Model**

Then test the model using new images. Make sure the predictions are reasonable.

## 3. Export the Model

After training:

**Export Model → TensorFlow → Keras**

Download the model.

Place the exported files inside:

```text
models/
```

The project should contain files similar to:

```text
models/
├── keras_model.h5
└── labels.txt
```

## 4. Open the Project in VS Code

Open this folder in VS Code.

Select the Python interpreter for the Conda environment:

```text
teachable-machine
```

Then open the VS Code terminal and activate the environment if necessary:

```bash
conda activate teachable-machine
```

## 5. Test One Image

Put a test image anywhere in the project, for example:

```text
test_images/my_test.jpg
```

Run:

```bash
python predict.py test_images/my_test.jpg
```

Example output:

```text
==================================================
Teachable Machine Image Classification
==================================================
Input image : test_images/my_test.jpg
Predicted class: Cat
Confidence    : 97.35%
==================================================

All class probabilities:
- Cat: 97.35%
- Dog: 2.65%
```

Take a screenshot of your real output for the submission.

## 6. Evaluate the Model

Put new test images into folders whose names exactly match the class names in `labels.txt`.

Example:

```text
test_images/
├── Cat/
│   ├── cat_test_01.jpg
│   ├── cat_test_02.jpg
│   └── ...
└── Dog/
    ├── dog_test_01.jpg
    ├── dog_test_02.jpg
    └── ...
```

Then run:

```bash
python evaluate.py
```

The script reports accuracy for each class and overall accuracy.

## 7. Files to Submit to GitHub

Your repository should contain:

```text
teachable-machine-project/
├── models/
│   ├── keras_model.h5
│   └── labels.txt
├── test_images/
│   └── ...
├── predict.py
├── evaluate.py
├── requirements.txt
├── .gitignore
└── README.md
```

Also submit a screenshot showing the Python prediction output.

## Important

The `keras_model.h5` and `labels.txt` files cannot be created correctly until you train and export your own model from Teachable Machine. After you download them, copy them into the `models` folder and the Python scripts are ready to use.

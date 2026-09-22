# Fruit Identifier
A ResNet50-based image classifier that identifies 262 fruit species from a photo, fine-tuned on ImageNet-pretrained weights.

## Overview
- **Model:** ResNet50 (torchvision), pretrained on ImageNet, with a custom 3-layer classification head
- **Dataset:** 225,000+ images across 262 fruit classes
- **Frameworks:** PyTorch, Flask
- **Accuracy:** 76%

### Requirements
- Python 3.10+
- Python Libraries in requirements.txt

### Independent Web Application Coming Soon!

### Installation
```bash
git clone https://github.com/ovistefan/Fruit-Identifier.git
cd Fruit-Identifier/gui
pip install -r requirements.txt
```

### Model Weights
Trained weights(current_model) aren't included in this repo due to file size. Download them here:
https://huggingface.co/ovistefan/fruit-identifier-v5/tree/main

Place `current_model.pth` and `class_names.json` into `GUI` before running the app. These are found at the hugging face repositiory.

### Running the App
```bash
cd gui
python backend.py
```

This starts a local server which runs in the terminal. DO NOT CLOSE the terminal while the website is running.

To run the webite, simply open the html file in a browser while the server is running in the terminal. Upload images to run the identifier

## Training
Training code and experiments are in `notebooks/` and `src/`. Key details:
- Two phase training with LR MultiStep scheduling atop a frozen backbone.
- Random adjustments done to training images such as sharpness, rotation, blur and flip.

## License
This project's code is licensed under the MIT License — see `LICENSE`.
The fruit dataset and ImageNet pretrained weights are **not** covered by this license; check their original sources for usage terms.
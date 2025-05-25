# ViT–BERT Image Captioning

A simple image‐captioning pipeline that fine-tunes a Vision Transformer (ViT) encoder with a BERT decoder on the Flickr8k dataset, plus a standalone script to load the trained model and generate captions on new images.

---

## 📁 Repository Structure

```
.
├── image-captioning-fine-tune-vit-bert-model-flickr1k.ipynb  # Training notebook (for Flickr8k)
├── ViT-BERT_test_script.py                                   # Inference / testing script
└── README.md                                                 # This file
```

---

## 📥 Dataset

1. Download **Flickr8k** from Kaggle:  
   https://www.kaggle.com/datasets/adityajn105/flickr8k  
2. Extract to the location of your choice so that you have:  

```
/location/flickr8k/
├── Images/
└── captions.txt
```

---

## 🚀 Training

1. Place the Flickr8k dataset as described above.  
2. Open `image-captioning-fine-tune-vit-bert-model-flickr1k.ipynb` in Jupyter (Colab or local) change the locatioin of the training and testing dataset according to your location.  
3. Run **all cells**.  
4. The fine-tuned model will be saved as `model.safetensors` in your specified location.

---

## 🔍 Testing / Inference

1. Ensure `model.safetensors` is in correct location.  
2. Update the `test_image_path` variable in `ViT-BERT_test_script.py`, for example:  

```python
test_image_path = "/content/flickr8k/Images/2757779501_c41c86a595.jpg"
```

3. Run:

```bash
python ViT-BERT_test_script.py
```

4. The script will display the image and print its generated caption.

---

## 📊 Results

### Sample Captions

<table style="width: 80%; margin: 0 auto; border-collapse: collapse; text-align: center;">
  <tr>
    <th>Image</th>
    <th>Caption</th>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/54779ef1-c4dd-4d95-9916-69709470450c" alt="Dog" style="width: 200px; height: auto; display: block; margin: 0 auto;"></td>
    <td style="vertical-align: middle;">A white dog</td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/9a9adb84-faf1-4b4e-b78d-a0625ec05d90" alt="Girl" style="width: 200px; height: auto; display: block; margin: 0 auto;"></td>
    <td style="vertical-align: middle;">A smiling girl in front of a camera on a playground</td>
  </tr>
</table>


### Evaluation Metrics

* **BLEU Score**
* **METEOR**
* **CIDEr**

---

## ⚙️ Notes

* The model uses `VisionEncoderDecoderModel` from Hugging Face Transformers.
* Training notebook covers data preprocessing, model setup, and training loop.
* Testing script handles image loading, preprocessing, and caption generation.
* Although the notebook is named "flickr1k", it should be used with Flickr8k for consistency.
* We used 1000 images and lower number of epochs due to system system constraints you can utilize full dataset with higher epochs for impoved and better results.

---

## 🛠️ Requirements

```bash
pip install torch torchvision
pip install transformers
pip install datasets
pip install pillow
pip install matplotlib
pip install numpy
pip install pandas
pip install scikit-learn
pip install safetensors
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Feel free to open issues or pull requests for:
* Support for other datasets (e.g., MS COCO)
* Improvements to model architecture or hyperparameters
* Enhanced CLI or batch inference support

---

## 📄 License

This project is licensed under the MIT License.

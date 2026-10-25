# Pneumonia detection from chest X-rays - Streamlit app

Model: **VGG16 fine-tuned** (input 224x224 grayscale, test ROC-AUC 0.91); default decision threshold 0.32.

| Layer | File | Role |
|---|---|---|
| Frontend | `app.py` | Streamlit page: upload an image, show predicted class + probability |
| Backend | `inference.py` | image reading (DICOM/PNG/JPG), pre-processing, model loading, prediction |
| Model | `pneumonia_model.keras`, `model_config.json` | serialized network and its settings (threshold, input size) |
| Packaging | `Dockerfile`, `requirements.txt`, `.dockerignore` | container image |
| Codespaces | `.devcontainer/devcontainer.json`, `.streamlit/config.toml` | auto-forward port 8501 and start the app |

## Run with Docker
```bash
docker build -t pneumonia-app .
docker run --rm -p 8501:8501 pneumonia-app
# open http://localhost:8501
```

## Run in GitHub Codespaces
1. Create a GitHub repository and upload the *contents* of this folder (the model file is 62 MB, below the 100 MB GitHub limit).
2. On the repository page choose **Code -> Codespaces -> Create codespace on main**. The dev container installs the requirements and starts Streamlit.
3. Open the **Ports** tab, find port **8501**; its *Forwarded Address* (`https://<codespace-name>-8501.app.github.dev`) is the app URL. Set Visibility to *Public* to share it.
4. Upload a file from `sample_images/` and read the predicted class and probability.

Manual start inside the Codespace: `pip install -r requirements.txt && streamlit run app.py`

*Decision-support demo only - not a certified medical device.*

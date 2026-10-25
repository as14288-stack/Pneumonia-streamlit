# Frontend: Streamlit page. Upload a chest X-ray (DICOM / PNG / JPG) -> predicted class + probability.
import streamlit as st
import numpy as np
import inference

st.set_page_config(page_title='Pneumonia detection from chest X-rays', page_icon='🫁', layout='centered')

@st.cache_resource(show_spinner='Loading model ...')
def get_model():
    return inference.load_model()

cfg = inference.load_config()
st.title('🫁 Pneumonia detection from chest X-rays')
st.caption('Decision-support demo - NOT a medical device and not a substitute for a radiologist.')

with st.sidebar:
    st.header('About the model')
    st.write(f"**Network:** {cfg['model_name']}")
    st.write(f"**Input size:** {cfg['input_size']} x {cfg['input_size']} grayscale")
    st.write(f"**Test ROC-AUC:** {cfg['test_auc']:.2f}")
    thr = st.slider('Decision threshold', 0.05, 0.95, float(cfg['threshold']), 0.01,
                    help='Probability above which an image is flagged. Lower = fewer missed cases but more false alarms.')
    st.write(f"Default threshold {cfg['threshold']:.2f} was chosen on validation data to reach about {int(cfg['target_recall'] * 100)}% recall.")

file = st.file_uploader('Upload a chest X-ray', type=['dcm', 'png', 'jpg', 'jpeg'])
if file is not None:
    try:
        img = inference.read_image(file.getvalue(), file.name)
    except Exception as e:
        st.error(f'Could not read the file: {e}'); st.stop()
    col1, col2 = st.columns(2)
    col1.image(img, caption=f'{file.name} ({img.shape[1]}x{img.shape[0]})', clamp=True)
    with st.spinner('Analysing ...'):
        res = inference.predict(get_model(), img, cfg['input_size'], thr)
    flagged = res['probability'] >= thr
    with col2:
        st.subheader('Prediction')
        (st.error if flagged else st.success)(res['label'])
        st.metric('Probability of pneumonia', f"{res['probability'] * 100:.1f}%")
        st.progress(min(max(res['probability'], 0.0), 1.0))
        st.caption(f'Flagged when probability >= {thr:.2f}.')
else:
    st.info('Upload a .dcm, .png or .jpg chest X-ray to get a prediction. Example images are in the sample_images folder.')
st.divider()
st.caption('Trained on the RSNA Pneumonia Detection Challenge data. Results must be reviewed by a qualified clinician.')

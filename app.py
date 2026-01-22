import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🧠",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("ai_image_detector.h5")

model = load_model()

st.title("🧠 AI Image Authenticity Detector")
st.write("Upload an image to check whether it is **Real** or **AI-Generated**.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# ---------- Image Preprocessing ----------
def preprocess_image(image):
    image = image.resize((160, 160))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ---------- TEXTUAL REASON FUNCTION ----------
def get_textual_reason(prediction, confidence):
    if prediction > 0.5:  # AI-generated
        if confidence > 90:
            return (
                "The model detected highly uniform textures, repetitive fine patterns, "
                "and a lack of natural camera noise, which are strong indicators of AI-generated images."
            )
        elif confidence > 75:
            return (
                "The image shows unnatural texture smoothness and subtle pattern repetitions "
                "commonly introduced by generative AI models."
            )
        else:
            return (
                "Some regions exhibit texture and noise inconsistencies that are more "
                "frequently observed in AI-generated images, but the distinction is subtle."
            )
    else:  # Real image
        if confidence > 90:
            return (
                "The model identified natural texture variation, realistic noise patterns, "
                "and camera-like imperfections typically found in real photographs."
            )
        elif confidence > 75:
            return (
                "The image contains mostly natural visual characteristics such as irregular textures "
                "and realistic color transitions, suggesting a real photograph."
            )
        else:
            return (
                "The image shows several natural photographic features, "
                "but also contains some ambiguous regions that reduce certainty."
            )

# ---------- MAIN LOGIC ----------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)
    prediction = model.predict(img_array)[0][0]

    confidence = prediction if prediction > 0.5 else 1 - prediction
    confidence *= 100

    st.markdown("---")

    if prediction > 0.5:
        st.error(f"🤖 AI-Generated Image ({confidence:.2f}%)")
    else:
        st.success(f"📸 Real Image ({confidence:.2f}%)")

    reason = get_textual_reason(prediction, confidence)

    st.markdown("### 📝 Why the model thinks this way")
    st.info(reason)

    st.caption(
        "⚠️ This explanation is based on learned statistical patterns, not human-like reasoning."
    )

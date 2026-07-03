import streamlit as st
from PIL import Image
import pandas as pd
import time

from utils import predict

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Human vs AI Face Detection",
    page_icon="🤖",
    layout="wide"
)

# =====================================================
# CSS
# =====================================================

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
}

div[data-testid="metric-container"]{
    border:1px solid #E5E7EB;
    padding:15px;
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("🤖 Human vs AI")

    st.markdown("---")

    st.write("### Model")

    st.success("Xception CNN")

    st.write("### Input")

    st.info("Face Image (.jpg/.png/.jpeg)")

    st.write("### Output")

    st.warning("Human / AI")

    st.markdown("---")

    st.caption("Version 1.0")

# =====================================================
# HISTORY
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# TITLE
# =====================================================

st.title("🧠 Human vs AI Face Detection")

st.write(
"""
Deteksi apakah gambar wajah merupakan **Human** atau **AI Generated**
menggunakan model **Xception Deep Learning**.
"""
)

st.divider()

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded = st.file_uploader(

    "Upload Face Image",

    type=["jpg","jpeg","png"]

)

# =====================================================
# PREDICTION
# =====================================================

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.subheader("Prediction")

        with st.spinner("Predicting..."):

            start = time.time()

            label, confidence, prob = predict(image)

            elapsed = (time.time()-start)*1000

        if label == "Human":

            st.success("🟢 HUMAN")

        else:

            st.error("🔴 AI GENERATED")

        st.metric(

            "Confidence",

            f"{confidence:.2f}%"

        )

        st.progress(confidence/100)

        st.metric(

            "Inference Time",

            f"{elapsed:.2f} ms"

        )

        st.markdown("### Probability")

        human_prob = (1-prob)*100
        ai_prob = prob*100

        st.write("Human")

        st.progress(human_prob/100)

        st.write(f"{human_prob:.2f}%")

        st.write("AI")

        st.progress(ai_prob/100)

        st.write(f"{ai_prob:.2f}%")

        with st.expander("Prediction Details"):

            st.write(f"Raw Output : {prob:.6f}")

            st.write(f"Human Probability : {human_prob:.2f}%")

            st.write(f"AI Probability : {ai_prob:.2f}%")

    # ============================================

    st.session_state.history.append({

        "Image":uploaded.name,

        "Prediction":label,

        "Confidence (%)":round(confidence,2),

        "AI Probability":round(ai_prob,4),

        "Inference (ms)":round(elapsed,2)

    })

# =====================================================
# HISTORY
# =====================================================

if len(st.session_state.history)>0:

    st.divider()

    st.subheader("Prediction History")

    df = pd.DataFrame(st.session_state.history)

    st.dataframe(

        df,

        use_container_width=True

    )

    csv = df.to_csv(index=False).encode("utf-8")

    col1,col2=st.columns(2)

    with col1:

        st.download_button(

            "📥 Download History CSV",

            csv,

            "prediction_history.csv",

            "text/csv"

        )

    with col2:

        if st.button("🗑 Clear History"):

            st.session_state.history=[]

            st.rerun()

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
"""
Developed using **TensorFlow**, **Xception CNN**, and **Streamlit**.
"""
)

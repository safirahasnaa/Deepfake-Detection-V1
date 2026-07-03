from utils import predict

...

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    label, confidence, probability = predict(image)

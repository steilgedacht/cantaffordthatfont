import streamlit as st
from train import GoogleFontsClassifier
from PIL import Image, ImageOps
import torch
import numpy as np


def main():
    st.title("Google Fonts Classifier")
    st.write("For finding a similar Google Font")

    user_input = st.file_uploader("Bild hochladen oder einfügen", type=["png", "jpg", "jpeg"])

    # Load model and weights
    model = GoogleFontsClassifier(95)
    model.load_state_dict(torch.load("model/model_2025-07-13_14-01-49.pth", map_location=torch.device("cpu"), weights_only=True))
    model.eval()

    if user_input is not None:
        image = Image.open(user_input).convert("L")  # Convert to greyscale

        st.image(image, caption="Hochgeladenes Bild (Graustufen)", use_column_width=True)

        # Let user select a region of interest
        st.write("Wähle einen Bereich im Bild aus:")
        crop = st.slider("Bereich auswählen (x1, y1, x2, y2)", 0, min(image.size), (0, 0, image.size[0], image.size[1]))
        x1, y1, x2, y2 = crop
        cropped_img = image.crop((x1, y1, x2, y2))
        st.image(cropped_img, caption="Ausgewählter Bereich", use_column_width=True)

        # Preprocess for model (example: resize, normalize, to tensor)
        input_img = cropped_img.resize((224, 224))
        input_tensor = torch.from_numpy(np.array(input_img)).unsqueeze(0).unsqueeze(0).float() / 255.0

        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()

        st.write(f"Vorhergesagte Klasse: {pred}")

if __name__ == "__main__":
    main()
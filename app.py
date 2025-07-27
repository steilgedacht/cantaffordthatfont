import streamlit as st
from utils import GoogleFontsClassifier
from PIL import Image, ImageOps
import torch
import numpy as np
from dataloader import Datagenerator
from train import Config
import random
from streamlit_img_label import st_img_label
from streamlit_img_label.manage import ImageManager, ImageDirManager

def crop_image(image):
    # with st.form(key="crop_form", enter_to_submit=False):
    #     labels = st_img_label(image, box_color="red", key="crop_box")
    #     x1, y1, x2, y2 = map(int, labels[0]["rect"])
    #     cropped_img = image.crop((x1, y1, x2, y2))
    return image

def preprocess_image(image):
    # scale the image to a the height of 150px while keeping the aspect ratio
    image = image.resize((int( image.width * 150 / image.height), 150))

    image_np = np.array(image)

    # we check if the background is white or black and invert the image if it is white
    hist, bin_edges = np.histogram(image_np, bins=10)
    most_frequent_bins = np.argsort(hist)[::-1]
    if most_frequent_bins[0] < most_frequent_bins[1]:
        image_np = 255 - image_np

    # normalize the image between 0 and 255
    image_np_norm = (image_np-np.min(image_np))/(np.max(image_np)-np.min(image_np)) * 255

    # fit the image to a 150x700px
    padded_tensor = np.ones((150, 700))
    h = min(image_np_norm.shape[1], 700)
    padded_tensor[:, :h] = image_np_norm[:, :h]
    image = padded_tensor
    image = torch.from_numpy(image).float().unsqueeze(0)

    return image


def process_and_predict(image, model, dataloader):
    # st.image(image)

    # Let user select a region of interest
    cropped_img = crop_image(image)

    # Preprocess the image for the model
    input_tensor = preprocess_image(cropped_img)

    st.write("The processed image that the model will see:")
    st.image(input_tensor.squeeze(0).detach().numpy() / 255)

    with torch.no_grad():

        output = model(input_tensor)
        output = torch.nn.functional.softmax(output, dim=1)

        prediction = torch.topk(output, k=4, dim=1)

        st.header("Predictions:")
        
        columns = st.columns(3, vertical_alignment="center")

        for i, col in enumerate(columns):
            pred = prediction.indices[0][i].item()
            with col:
                st.write(dataloader.fonts_unique[pred])
                st.caption(f"{output[0,pred].item() * 100 :.3f}% Confidence")



def main():
    dataloader = Datagenerator(Config())

    st.title("Can't afford that font?")
    st.subheader("Find the most similar font on Google Fonts!")

    # load the style for the website
    with open("style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


    # Load model and weights
    model = GoogleFontsClassifier(1738)
    model.load_state_dict(torch.load("model/model_resnet_final_v1.pth", map_location=torch.device("cpu")))
    model.eval()

    # columns block
    col1, col2 = st.columns(2, vertical_alignment="center")
    with col1:    
        user_image = st.file_uploader("Upload or paste image", type=["png", "jpg", "jpeg", "bmp", "gif", "webp", "tiff"])
        if user_image is not None:
            st.session_state["upload_image"] = True
    with col2:
        if st.button("Try it out with a random image"):
            st.session_state["use_random_image"] = True

    # after the columns block
    if st.session_state.get("use_random_image", False):
        orginal_image = dataloader[random.randint(0, len(dataloader))]
        image = Image.fromarray(orginal_image[0].numpy()).convert("L")
        process_and_predict(image, model, dataloader)
        st.session_state["use_random_image"] = False

    if st.session_state.get("upload_image", False):
        if user_image is not None:
            image = Image.open(user_image).convert("L") 
            process_and_predict(image, model, dataloader)
            st.session_state["upload_image"] = False

if __name__ == "__main__":
    main()
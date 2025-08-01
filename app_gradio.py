import gradio as gr
from utils import GoogleFontsClassifier
from PIL import Image, ImageOps
import torch
import numpy as np
from dataloader import Datagenerator
from train import Config
import random
import gradio.themes.base

def crop_image(image):
    # Convert to numpy array
    img_np = np.array(image)
    # Threshold to binary image
    thresh = 240
    bin_img = img_np < thresh
    coords = np.argwhere(bin_img)
    if coords.size == 0:
        return image  # nothing to crop
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # slices are exclusive at the top
    cropped = img_np[y0:y1, x0:x1]
    return Image.fromarray(cropped)

def preprocess_image(image):
    image = image.resize((int(image.width * 150 / image.height), 150))
    image_np = np.array(image)
    hist, bin_edges = np.histogram(image_np, bins=10)
    most_frequent_bins = np.argsort(hist)[::-1]
    if most_frequent_bins[0] < most_frequent_bins[1]:
        image_np = 255 - image_np
    image_np_norm = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np)) * 255
    padded_tensor = np.ones((150, 700))
    h = min(image_np_norm.shape[1], 700)
    padded_tensor[:, :h] = image_np_norm[:, :h]
    image = padded_tensor
    image = torch.from_numpy(image).float().unsqueeze(0)
    return image

# Load model and dataloader once
dataloader = Datagenerator(Config())
model = GoogleFontsClassifier(1738)
model.load_state_dict(torch.load("model/model_resnet_final_v1.pth", map_location=torch.device("cpu")))
model.eval()

def process_and_predict(image, use_random_image):
    if use_random_image:
        orginal_image = dataloader[random.randint(0, len(dataloader))]
        image = Image.fromarray(orginal_image[0].numpy()).convert("L")
    else:
        if image is None:
            return None, "Bitte ein Bild hochladen oder Zufallsbild wählen.", None, None, None
        image = image.convert("L")
    cropped_img = crop_image(image)
    input_tensor = preprocess_image(cropped_img)
    processed_img = input_tensor.squeeze(0).detach().numpy() / 255

    with torch.no_grad():
        output = model(input_tensor)
        output = torch.nn.functional.softmax(output, dim=1)
        prediction = torch.topk(output, k=4, dim=1)
        pred_labels = [dataloader.fonts_unique[pred.item()] for pred in prediction.indices[0]]
        confidences = [f"{output[0, pred.item()].item() * 100:.3f}%" for pred in prediction.indices[0]]

    # Return processed image and predictions
    return (
        Image.fromarray((processed_img * 255).astype(np.uint8)),
        pred_labels[0], confidences[0],
        pred_labels[1], confidences[1],
        pred_labels[2], confidences[2],
        pred_labels[3], confidences[3]
    )

with gr.Blocks() as demo:
    gr.Markdown("# Can't afford that font?\nFinde die ähnlichste Google Font!")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Bild hochladen")
            random_btn = gr.Button("Zufallsbild ausprobieren")
        with gr.Column():
            processed_img = gr.Image(label="Vorverarbeitetes Bild", interactive=False)
    with gr.Row():
        pred1 = gr.Textbox(label="1. Vorhersage")
        conf1 = gr.Textbox(label="Konfidenz 1")
        pred2 = gr.Textbox(label="2. Vorhersage")
        conf2 = gr.Textbox(label="Konfidenz 2")
        pred3 = gr.Textbox(label="3. Vorhersage")
        conf3 = gr.Textbox(label="Konfidenz 3")
        pred4 = gr.Textbox(label="4. Vorhersage")
        conf4 = gr.Textbox(label="Konfidenz 4")

    def predict_from_upload(image):
        return process_and_predict(image, use_random_image=False)

    def predict_random(_):
        return process_and_predict(None, use_random_image=True)

    image_input.change(
        predict_from_upload,
        inputs=[image_input],
        outputs=[processed_img, pred1, conf1, pred2, conf2, pred3, conf3, pred4, conf4]
    )
    random_btn.click(
        predict_random,
        inputs=[random_btn],
        outputs=[processed_img, pred1, conf1, pred2, conf2, pred3, conf3, pred4, conf4]
    )

if __name__ == "__main__":
    demo.launch(theme=gradio.themes.base.Base())

import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torchvision.models import vgg19, VGG19_Weights
from torch import nn, optim

# Load the VGG model for feature extraction
class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.features = nn.Sequential(*list(vgg[:21]))  # Extract layers up to conv4_1

    def forward(self, x):
        return self.features(x)

# Define the style transfer function
def style_transfer(content_image, style_image, num_steps=500, style_weight=1e6, content_weight=1):
    # Preprocessing function
    def preprocess(image):
        preprocess_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.div(255)),  # Scale to [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess_transform(image).unsqueeze(0).to(device)

    # Postprocessing function
    def postprocess(tensor):
        tensor = tensor.squeeze().cpu().clone().detach()
        tensor = tensor.permute(1, 2, 0).numpy()
        tensor = np.clip(tensor, 0, 1)  # Ensure values are in range [0, 1]
        tensor = tensor * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        tensor = np.clip(tensor, 0, 1)  # Clamp again after de-normalizing
        return (tensor * 255).astype(np.uint8)

    # Load content and style images
    content_image = preprocess(Image.fromarray(content_image))
    style_image = preprocess(Image.fromarray(style_image))

    # Initialize generated image
    generated_image = content_image.clone().requires_grad_(True)

    # Load VGG model for feature extraction
    vgg = VGGFeatures().to(device).eval()

    # Define optimizer
    optimizer = optim.Adam([generated_image], lr=0.01)

    # Define loss functions
    mse_loss = nn.MSELoss()

    # Extract style features
    with torch.no_grad():
        style_features = vgg(style_image)
        content_features = vgg(content_image)

    # Compute Gram matrix for style features
    def gram_matrix(tensor):
        c, h, w = tensor.size()
        tensor = tensor.view(c, h * w)
        return torch.mm(tensor, tensor.t()) / (c * h * w)

    style_grams = [gram_matrix(f) for f in style_features]

    # Style transfer optimization loop
    for step in range(num_steps):
        optimizer.zero_grad()

        generated_features = vgg(generated_image)

        # Compute content loss
        content_loss = content_weight * mse_loss(generated_features[-1], content_features[-1])

        # Compute style loss
        style_loss = 0
        for gf, sg in zip(generated_features, style_grams):
            style_loss += mse_loss(gram_matrix(gf), sg)
        style_loss *= style_weight

        # Total loss
        total_loss = content_loss + style_loss

        # Backpropagate
        total_loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}/{num_steps}, Total Loss: {total_loss.item()}")

    # Postprocess and return the result
    return postprocess(generated_image)

# Fast Neural Style Transfer function
def fast_neural_style_transfer(content_image, style_model_path):
    # Preprocessing function
    def preprocess(image):
        preprocess_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.div(255)),  # Scale to [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess_transform(image).unsqueeze(0).to(device)

    # Postprocessing function
    def postprocess(tensor):
        tensor = tensor.squeeze().cpu().clone().detach()
        tensor = tensor.permute(1, 2, 0).numpy()
        tensor = tensor * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        tensor = np.clip(tensor, 0, 1)  # Clamp values to [0, 1]
        return (tensor * 255).astype(np.uint8)

    # Load the trained Fast Neural Style model
    print(style_model_path)
    print("!!!!!!!!!!!!!!!!!!!!!!!!")
    model = torch.jit.load(style_model_path).to(device).eval()

    # Preprocess content image
    content_image = preprocess(Image.fromarray(content_image))

    # Perform style transfer
    with torch.no_grad():
        output = model(content_image).cpu()

    # Postprocess and return the result
    return postprocess(output)

# Main function for style transfer on video
def process_video(input_video_path, style_image_path, output_video_path):
    # Load the style image
    style_image = cv2.imread(style_image_path)
    style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)
    style_image = cv2.convertScaleAbs(style_image, alpha=0.8, beta=0)  # Reduce brightness

    # Open video capture
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Open video writer
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply style transfer to each frame
        styled_frame = style_transfer(frame, style_image, style_weight=1e5, content_weight=1)

        # Write the styled frame to the output video
        out.write(cv2.cvtColor(styled_frame, cv2.COLOR_RGB2BGR))

    cap.release()
    out.release()
    print("Video processing complete.")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example usage
# process_video("input.mp4", "style.jpg", "output.mp4")

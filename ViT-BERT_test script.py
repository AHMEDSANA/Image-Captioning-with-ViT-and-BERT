import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor

def load_image_captioning_model(model_dir):
    """Load the saved image captioning model and its components"""
    # Load the model - use safetensors if available
    try:
        # Try loading the model directly from the directory
        model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    except:
        # If that fails, try loading using the model file structure
        model_path = model_dir
        if model_path.endswith('.safetensors'):
            # Get the directory containing the .safetensors file
            import os
            model_dir = os.path.dirname(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_dir)

    # Load the tokenizer and image processor
    decoder_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    # Ensure pad token is set
    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, decoder_tokenizer, image_processor, device

def generate_caption(image_path, model, decoder_tokenizer, image_processor, device):
    """Generate a caption for an input image"""
    # Load and process the image
    image = Image.open(image_path)

    # Display the image
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    # Process image for the model
    image = image.resize((224, 224))
    pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)

    # Generate caption
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values=pixel_values,
            decoder_start_token_id=decoder_tokenizer.cls_token_id,
            max_length=32
        )

    # Decode and clean up the caption
    caption = decoder_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    caption = caption.replace(".", "")

    print(f"Generated Caption: {caption}")
    return caption

# Main execution
if __name__ == "__main__":
    # Path to your saved model
    model_path = "/content/model.safetensors"  # Update this with your model path

    # Load the model and components
    model, decoder_tokenizer, image_processor, device = load_image_captioning_model(model_path)

    # Path to test image
    test_image_path = "/content/flickr8k/Images/2757779501_c41c86a595.jpg"  # Update this with your test image path

    # Generate caption
    caption = generate_caption(test_image_path, model, decoder_tokenizer, image_processor, device)
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import torch


def get_input_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image.")
    parser.add_argument('image_path', type=str, help='Path to the input image.')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint.')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes.')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to category-to-name mapping file.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference.')
    return parser.parse_args()

def load_checkpoint(filepath,device):
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)
    # Initialize the model architecture
     model = build_model(checkpoint['arch'], checkpoint['hidden_units'], checkpoint['dropout'], device)
        
    # Load model parameters
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    for param in model.parameters(): 
        param.requires_grad = False 
        
    return model


def process_image(image):
    image = Image.open(image)
    #Resize
    image.thumbnail((256, 256)) 
    #Center Crop to 224x224
    width, height = image.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))
    #image.show()
    # Convert to NumPy array & Normalize
    np_image = np.array(image) / 255.0
    
    # Normalize using ImageNet mean & std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Reorder dimensions for PyTorch 
    np_image = np_image.transpose((2, 0, 1))
    return np_image


def predict(image_path, model, topk=5, device="cpu"):
    processed_image = process_image(image_path)
    
    # Convert to a PyTorch tensor and add a batch dimension
    image_tensor = torch.from_numpy(processed_image).float()
    image_tensor = image_tensor.unsqueeze(0).to(device) 

    model.eval()
    model.to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Calculate probabilities
    probabilities = torch.softmax(outputs, dim=1)

    top_probs, top_indices = torch.topk(probabilities, topk)
    
    top_probs = top_probs.cpu().numpy().squeeze().tolist()
    top_indices = top_indices.cpu().numpy().squeeze().tolist()
    
    # Map indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}  # Invert class_to_idx
    top_classes = [idx_to_class[idx] for idx in top_indices]
    
    return top_probs, top_classes


def main():
    args = get_input_args()
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_checkpoint(args.checkpoint, device)

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Predict
    probs, classes = predict(args.image_path, model, device, args.top_k)
    class_names = [cat_to_name[str(cls)] for cls in classes]
    
    print("Top K Predictions:")
    for i in range(args.top_k):
        print(f"{class_names[i]}: {probs[i]:.4f}")

if __name__ == '__main__':
    main()



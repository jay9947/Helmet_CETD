import os
import argparse
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from helmet_detr import build_helmet_detr, HELMET_CLASSES

def get_args_parser():
    """
    Parse command line arguments for inference.
    """
    parser = argparse.ArgumentParser('Helmet DETR Inference', add_help=False)
    
    # Model parameters
    parser.add_argument('--checkpoint', default='', help='Path to checkpoint')
    parser.add_argument('--device', default='cuda', help='Device to use for inference')
    parser.add_argument('--threshold', default=0.7, type=float, help='Detection threshold')
    
    # Input/output parameters
    parser.add_argument('--input', required=True, help='Path to input image or folder')
    parser.add_argument('--output', default='./output', help='Path to output folder')
    
    return parser


def box_cxcywh_to_xyxy(x):
    """
    Convert bounding boxes from (cx, cy, w, h) to (x1, y1, x2, y2).
    """
    x_c, y_c, w, h = x
    return [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]


def rescale_bboxes(bboxes, img_size):
    """
    Rescale normalized bounding boxes to image size.
    """
    img_w, img_h = img_size
    scaled_bboxes = []
    for bbox in bboxes:
        scaled_bbox = box_cxcywh_to_xyxy(bbox.tolist())
        scaled_bbox[0] *= img_w
        scaled_bbox[1] *= img_h
        scaled_bbox[2] *= img_w
        scaled_bbox[3] *= img_h
        scaled_bboxes.append(scaled_bbox)
    return scaled_bboxes


def detect(model, image_path, device, threshold=0.7):
    """
    Perform detection on an image.
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # Get predictions
    pred_logits = outputs['pred_logits'][0]
    pred_boxes = outputs['pred_boxes'][0]
    
    # Filter predictions by confidence threshold
    probas = pred_logits.softmax(-1)
    keep = probas.max(-1).values > threshold
    
    # Extract predictions
    scores = probas[keep].max(-1).values
    labels = probas[keep].argmax(-1)
    boxes = pred_boxes[keep]
    
    # Rescale boxes to image size
    scaled_boxes = rescale_bboxes(boxes, image.size)
    
    return {
        'image': image,
        'scores': scores.tolist(),
        'labels': labels.tolist(),
        'boxes': scaled_boxes
    }


def visualize_prediction(result, output_path=None):
    """
    Visualize detection results.
    """
    image = result['image']
    scores = result['scores']
    labels = result['labels']
    boxes = result['boxes']
    
    # Convert PIL image to numpy array for plotting
    img_array = np.array(image)
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 9))
    
    # Display image
    ax.imshow(img_array)
    
    # Define colors for different classes
    colors = ['r', 'g', 'b', 'c', 'm']
    
    # Draw bounding boxes and labels
    for score, label, box in zip(scores, labels, boxes):
        # Get class name and color
        class_name = HELMET_CLASSES[label]
        color = colors[label % len(colors)]
        
        # Create rectangle patch
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        
        # Add patch to axes
        ax.add_patch(rect)
        
        # Add label
        label_text = f'{class_name}: {score:.2f}'
        ax.text(x1, y1, label_text, backgroundcolor=color, color='white', fontsize=10)
    
    # Remove axes
    plt.axis('off')
    
    # Show the plot or save to file
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.0)
        plt.close()
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()


def visualize_with_pil(result, output_path=None):
    """
    Visualize detection results using PIL.
    """
    image = result['image'].copy()
    scores = result['scores']
    labels = result['labels']
    boxes = result['boxes']
    
    # Create draw object
    draw = ImageDraw.Draw(image)
    
    # Try to get a font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    
    # Define colors for different classes
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]
    
    # Draw bounding boxes and labels
    for score, label, box in zip(scores, labels, boxes):
        # Get class name and color
        class_name = HELMET_CLASSES[label]
        color = colors[label % len(colors)]
        
        # Draw rectangle
        x1, y1, x2, y2 = [int(coord) for coord in box]
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        
        # Draw label background
        label_text = f'{class_name}: {score:.2f}'
        text_size = draw.textsize(label_text, font=font)
        draw.rectangle([(x1, y1), (x1 + text_size[0], y1 + text_size[1])], fill=color)
        
        # Draw text
        draw.text((x1, y1), label_text, fill=(255, 255, 255), font=font)
    
    # Save to file or return image
    if output_path:
        image.save(output_path)
        print(f"Saved visualization to {output_path}")
        return None
    else:
        return image


def process_folder(model, input_folder, output_folder, device, threshold=0.7):
    """
    Process all images in a folder.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process all images in the folder
    for filename in os.listdir(input_folder):
        # Check if file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Build input and output paths
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"detected_{filename}")
            
            # Detect and visualize
            try:
                result = detect(model, input_path, device, threshold)
                visualize_with_pil(result, output_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")


def main(args):
    """
    Main function.
    """
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Building model...")
    model, _ = build_helmet_detr(num_classes=len(HELMET_CLASSES), num_queries=20)
    model.to(device)
    
    # Load checkpoint
    if os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    else:
        print(f"No checkpoint found at {args.checkpoint}, using random weights!")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Check if input is a file or a folder
    if os.path.isfile(args.input):
        # Process single image
        print(f"Processing single image: {args.input}")
        result = detect(model, args.input, device, args.threshold)
        
        # Create output path
        output_path = os.path.join(args.output, f"detected_{os.path.basename(args.input)}")
        
        # Visualize result
        visualize_with_pil(result, output_path)
    
    elif os.path.isdir(args.input):
        # Process all images in the folder
        print(f"Processing all images in folder: {args.input}")
        process_folder(model, args.input, args.output, device, args.threshold)
    
    else:
        print(f"Input path {args.input} is neither a file nor a directory!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Helmet DETR Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

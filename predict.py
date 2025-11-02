import argparse
# Import necessary modules and functions
from functions import load_checkpoint, process_image, predict

def main():
    parser = argparse.ArgumentParser(description="Predict flower name from an image using a trained network.")
    parser.add_argument("input", type=str, help="Path to the input image")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint")
    parser.add_argument("--top_k", type=int, default=1, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, help="Path to JSON file mapping categories to real names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    args = parser.parse_args()

    # Load the model from checkpoint
    model = load_checkpoint(args.checkpoint)

    # Process the image
    image = process_image(args.input)
    # Predict the class
    probs, classes = predict(args.input, model, args.top_k, args.gpu)

    # Optionally map categories to real names
    if args.category_names:
        import json
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name.get(str(cls), cls) for cls in classes]

    print("Predicted Classes:", classes)
    print("Probabilities:", probs)

if __name__ == '__main__':
    main()

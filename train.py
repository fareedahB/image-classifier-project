import argparse
# Import necessary modules and functions
from functions import build_model, train_model, load_data, save_checkpoint
import torch 

def main():
    parser = argparse.ArgumentParser(description="Train a neural network on flower data.")
    parser.add_argument("data_directory", type=str, help="Path to the dataset")
    parser.add_argument("--save_dir", type=str, default="saved_models", help="Directory to save checkpoints")
    parser.add_argument("--arch", type=str, default="resnet50", help="Model architecture")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_layer_1_units", type=int, default=256, help="Number of neurons/units in first hidden layer")
    parser.add_argument("--hidden_layer_2_units", type=int, default=128, help="Number of neurons/units in second hidden layer")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    args = parser.parse_args()

    # Load the data
    trainloader, validloader, testloader = load_data(args.data_directory)
    # Determine device (use GPU if specified and available)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    # Build the model
    model = build_model(args.arch, args.hidden_layer_1_units, args.hidden_layer_2_units)
    model.to(device)  # Move model to GPU/CPU
    # Train the model
    optimizer = train_model(model, trainloader, validloader, args.epochs, args.learning_rate, args.gpu)
    # Save the checkpoint
    save_checkpoint(model, optimizer, args.epochs, trainloader.dataset.class_to_idx)
    
if __name__ == '__main__':
    main()

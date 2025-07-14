import argparse
import torch




def get_input_args():
    # Add arguments
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset of images.")
    parser.add_argument('data_dir', type=str, help='Path to the dataset directory.')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints.')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'resnet18'], help='Model architecture (vgg16 or resnet18).')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Number of hidden units in the classifier.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for the classifier.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training.')
    args = parser.parse_args()
    return args


def train_model(model, dataloader,valloader, criterion, optimizer, epochs, device="cpu"):
    model.to(device) 
    for e in range(epochs):
        #training loop 
        model.train()
        running_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad() 
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        train_loss = running_loss/len(dataloader)
        print(f"Epoch: {e+1} Training loss: {train_loss}")
        
        val_loss, val_accuracy = validate_model(model, valloader, criterion, device)
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy} , {val_accuracy * 100}%")

    return train_loss,val_loss,val_accuracy

def main():
    args = get_input_args()
    
    # Access the arguments
    epochs = args.epochs
    learning_rate = args.learning_rate
    dropout = args.dropout
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Print the arguments
    print(f"Training configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Save: {args.save_dir}")
    print(f"  Device: {device}")
    print(f" Data Dir: {args.data_dir} ")


    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

#     # Load data
#     trainloader, valloader, class_to_idx = load_data(args.data_dir, args.batch_size)

#     # Build model
#     model = build_model(args.arch, args.hidden_units, args.dropout, device)

#     # Define loss and optimizer
#     criterion = nn.NLLLoss()
#     optimizer = optim.Adam(model.classifier.parameters() if args.arch == 'vgg16' else model.fc.parameters(), lr=args.learning_rate)

#     # Train the model
#     train_model(model, trainloader, valloader, criterion, optimizer, args.epochs, device)

#     # Save checkpoint
#     save_checkpoint(model, class_to_idx, args.save_dir, args.arch, args.hidden_units, args.dropout, args.epochs, args.learning_rate)



if __name__ == '__main__':
    main()
    
    

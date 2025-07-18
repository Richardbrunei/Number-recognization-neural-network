import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from PIL import Image
import os # Import the os module for path operations
from PIL import ImageOps
from test_with_load import Net
from test_with_load import get_MNIST
from test_with_load import evaluate
from test_with_load import count_png_files




def main():
    print("program start")
    testercnt=count_png_files()
    net = Net()
    
    # Define a path for saving the model
    model_save_path = 'mnist_net.pth'

    # Check if a saved model exists
    loaded=False
    try:
        if os.path.exists(model_save_path):
            print(f"Loading pre-trained model from {model_save_path}...")
            net.load_state_dict(torch.load(model_save_path))
            print("Model loaded successfully.")
            loaded=True
        else:
            print("No pre-trained model found. ")
    except:
        print("Unsuccessful loading.")
        loaded=False
        
    inpt=input("Use test data? ").strip()
    if inpt.lower()=='y' or inpt.lower()=='yes':
        test_data = get_MNIST(is_train=False)
        for (n, (x, _)) in enumerate(test_data):
            if n > 10:
                break
            predict = torch.argmax(net.forward(x[0].view(-1, 28*28)))
            plt.figure(n)
            plt.imshow(x[0].view(28, 28))
            plt.title("prediction: " + str(int(predict)))
            plt.show()
    transform = transforms.ToTensor()

    
    for i in range(testercnt):
        imgpth=r'C:\Users\Liang\OneDrive\Documents\pytorch-tutorial-master\pytorch-tutorial-master\tester'+str(i)+".png";
        image=Image.open(imgpth)
        image = image.convert('L')
        image = image.resize((28,28), Image.LANCZOS)
        image_tensor = transform(image)
        input_batch = image_tensor.unsqueeze(0)

        #net.zero_grad()
        output=net.forward(input_batch.view(-1, 28*28))
        predict = torch.argmax(output)
        plt.figure(i)
        plt.imshow(image)
        plt.title("prediction: " + str(int(predict)))
        

        plt.show()

        

        


if __name__ == "__main__":
    main()

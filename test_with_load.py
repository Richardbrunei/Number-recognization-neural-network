import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps
import os # Import the os module for path operations

import glob

def count_png_files():
    # The ** allows for recursive search in subdirectories if recursive=True is set
    # For only the specified folder, use folder_path + '/*.png'
    png_files = glob.glob('tester*.png') 
    return len(png_files)

class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 64)
        self.fc5 = torch.nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.relu(self.fc4(x))
        x = torch.nn.functional.log_softmax(self.fc5(x), dim=1)
        return x


"""def get_MNIST(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)"""

def get_MNIST(is_train,invert_colors=False):
    transform_list = [transforms.ToTensor()]
    if invert_colors:
        transform_list.append(transforms.Lambda(lambda x: ImageOps.invert(transforms.ToPILImage()(x))))
        transform_list.append(transforms.ToTensor()) # Convert back to tensor after inversion

    to_tensor = transforms.Compose(transform_list)
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)


def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            outputs = net.forward(x.view(-1, 28*28))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total


def main():
    print("program start")
    testercnt=count_png_files()
    test_data = get_MNIST(is_train=False)
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
            print("No pre-trained model found. Training a new model...")
    except:
        print("Unsuccessful loading. Creating new model...")
        loaded=False
    print("initial accuracy:", evaluate(test_data, net))
    if loaded:
        inpt=input("More training? ").strip()
        if inpt.lower()=='y' or inpt.lower()=='yes':
            loaded=False
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    if not loaded:
        train_data = get_MNIST(is_train=True)
        train_data_inv=get_MNIST(is_train=True,invert_colors=True)
        inpt=int(input("How much? ").strip())
        for epoch in range(inpt):
            for (x, y) in train_data:
                net.zero_grad()
                output = net.forward(x.view(-1, 28*28))
                loss = torch.nn.functional.nll_loss(output, y)
                loss.backward()
                optimizer.step()
            for (x, y) in train_data_inv:
                net.zero_grad()
                output = net.forward(x.view(-1, 28*28))
                loss = torch.nn.functional.nll_loss(output, y)
                loss.backward()
                optimizer.step()
            print("epoch", epoch, "accuracy:", evaluate(test_data, net))
        
        # Save the trained model
        inpt=input("Save? ").strip()
        if inpt.lower()=='y' or inpt.lower()=='yes':
            print(f"Saving trained model to {model_save_path}...")
            torch.save(net.state_dict(), model_save_path)
            print("Model saved.")

    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28*28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction: " + str(int(predict)))
    plt.show()

    newtrain=[]
    newtrainidx=[]

    more=False

    inpt=input("User training? ").strip()
    if inpt.lower()=='y' or inpt.lower()=='yes':
        more=True
    
    for i in range(testercnt):
        imgpth=r'tester'+str(i)+".png";
        image=Image.open(imgpth)
        image = image.convert('L')
        image = image.resize((28,28), Image.LANCZOS)
        transform = transforms.ToTensor()
        image_tensor = transform(image)
        input_batch = image_tensor.unsqueeze(0)

        #net.zero_grad()
        output=net.forward(input_batch.view(-1, 28*28))
        predict = torch.argmax(output)
        plt.figure(i)
        plt.imshow(image)
        plt.title("prediction: " + str(int(predict)))
        

        plt.show()
        if not more:
            continue
        inpt=input("What should this be? ").strip()
        if (inpt in "0123456789") and len(inpt)==1:
            newtrain.append(inpt)
            newtrainidx.append(i)

    if more:
        for i in range(len(newtrain)):
            imgpth=r'tester'+str(newtrainidx[i])+".png";
            image=Image.open(imgpth)
            image = image.convert('L')
            image = image.resize((28,28), Image.LANCZOS)
            transform = transforms.ToTensor()
            image_tensor = transform(image)
            input_batch = image_tensor.unsqueeze(0)

            net.zero_grad()
            output=net.forward(input_batch.view(-1, 28*28))
            #predict = torch.argmax(output)
            target_tensor = torch.tensor(int(newtrain[i]), dtype=torch.long)
            loss = torch.nn.functional.nll_loss(output,target_tensor.unsqueeze(0) )
            loss.backward()
            optimizer.step()
        

        print("After new training:", evaluate(test_data, net))
    
        inpt=input("Save again? ").strip()
        if inpt.lower()=='y' or inpt.lower()=='yes':
            print(f"Saving trained model to {model_save_path}...")
            torch.save(net.state_dict(), model_save_path)
            print("Model saved.")

        


if __name__ == "__main__":
    main()

from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_model(no_epochs):

    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()

    loss_function = torch.nn.MSELoss()
    
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    accuracies = []
    train_loss = []
    train_accuracy = []

    for epoch_i in range(no_epochs):
        model.train()
        epoch_loss=0.0
        total_samples=0
        correct=0
        
        for idx, sample in enumerate(data_loaders.train_loader):
            input_val,label_val = sample['input'],sample['label']
            # sample['input'] and sample['label']
            optimizer.zero_grad()
            y_pred = model(input_val)
            label_val = label_val.view(-1, 1)
            y_pred = y_pred.view(-1, 1)
            loss = loss_function(y_pred,label_val)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            '''
            # Calculate accuracy
            _, predicted = torch.max(y_pred.data, 1)
            total_samples += label_val.size(0)
            correct_predictions += (predicted == label_val).sum().item()
            '''
            for o,l in zip(y_pred,label_val):
                pred = 0 if o.item()<.25 else 1
                correct = correct+1 if pred==l.item() else correct
                total_samples+=1
                
            
        epoch_loss /= len(data_loaders.train_loader)
        accuracy = (correct / total_samples)*100
        #plot
        train_loss.append(epoch_loss)
        train_accuracy.append(accuracy)
        
        #print("Training Loss:",epoch_loss,"Training Accuracy:",accuracy)
        
        
        test_loss,test_accuracy=model.evaluate(model, data_loaders.test_loader, loss_function)
        
        plt.plot(train_loss)
        plt.plot(test_loss)
        plt.title('1. Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.legend(['Train - loss', 'Test - loss'], loc='upper left')
        plt.show()
        
    #pickle.dump(model.state_dict(), open("saved/scaler.pkl", "wb")) 
    torch.save(model.state_dict(), 'saved/saved_model.pkl')
    



if __name__ == '__main__':
    no_epochs = 50
    train_model(no_epochs)




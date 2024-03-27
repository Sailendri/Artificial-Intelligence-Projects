import torch
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    def __init__(self,input_size=6,hidden_size=180,output_size=1):
        super(Action_Conditioned_FF,self).__init__()
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        self.input_to_hidden=nn.Linear(input_size, hidden_size)
        self.nonlinear_activation = nn.ReLU()
        self.hidden_to_output = nn.Linear(hidden_size,output_size)
        

    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        i_to_h = self.input_to_hidden(input)
        i_to_h_acti = self.nonlinear_activation(i_to_h)
        output = self.hidden_to_output(i_to_h_acti)
        return output


    def evaluate(self, model, test_loader, loss_function):
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.
        model_output=[]
        total_loss=0
        for idx, sample in enumerate(test_loader):
            label_val=sample['label']
            i=sample['input']
            y_pred=model.forward(i)
            #model_output.append(o)
            label_val = label_val.view(-1, 1)
            y_pred = y_pred.view(-1, 1)
            loss = loss_function(y_pred,label_val)
            total_loss+=loss.item()
            correct,total_samples=0,0
            for o,l in zip(y_pred,label_val):
                pred = 0 if o.item()<.25 else 1
                correct = correct+1 if pred==l.item() else correct
                total_samples+=1
        accuracy = (correct / total_samples)*100
        total_loss=total_loss/len(test_loader)
        print("Testing Accuracy",accuracy,"Testing Loss",total_loss)
        return total_loss,accuracy,test_loss,test_accuracy

def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()



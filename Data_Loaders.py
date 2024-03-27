import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,random_split


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')
# STUDENTS: it may be helpful for the final part to balance the distribution of your collected data

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
# STUDENTS: __len__() returns the length of the dataset
        return len(self.data)
        

    def __getitem__(self, idx):
        return_dict={}
        x_value = self.normalized_data[idx,:-1]
        y_value = self.normalized_data[idx,-1]
        if not isinstance(idx, int):
            idx = idx.item()
            
            return_dict['input'] = x_value.astype(dtype = np.float32)
            return_dict['label'] = np.array(y_value, dtype = np.float32)
            
            
        return return_dict
# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.

# +
class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
        
        train_data, test_data = train_test_split(self.nav_dataset , test_size=0.2, random_state=42)
        self.train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
        self.test_loader = DataLoader(test_data,batch_size=batch_size)
        
        
# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary




# -

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
 
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        a, b = sample['input'], sample['label']
        print("Train input:",a)
        print("Train Label:",b)
    for idx, sample in enumerate(data_loaders.test_loader):
        c, d = sample['input'], sample['label']
        print("Test input:",a)
        print("Test Label:",b)


if __name__ == '__main__':
    main()



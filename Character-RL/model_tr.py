import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import os

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        action = torch.unsqueeze(action, 0)
        reward = torch.unsqueeze(reward, 0)
        done = (done, )

        # 1. predicted Q values with current state
        shape_img = state.shape
        state = state.reshape(-1,shape_img[2], shape_img[0], shape_img[1])
        state = torch.FloatTensor(state)

        shape_img = next_state.shape
        next_state = next_state.reshape(-1,shape_img[2], shape_img[0], shape_img[1])
        next_state = torch.FloatTensor(next_state)

        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new
        # 2. Q_new = r + y * max(next_predicted Q value)
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

class Net(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 1, out_channels=12, kernel_size=(3,3), stride=(2,2))
        self.elu1 = torch.nn.ELU()
        self.conv2 = torch.nn.Conv2d(in_channels=12, out_channels=18, kernel_size=(3,3), stride=(2,2))
        self.elu2 = torch.nn.ELU()
        self.conv3 = torch.nn.Conv2d(in_channels=18, out_channels=24, kernel_size=(3,3), stride=(2,2))
        self.elu3 = torch.nn.ELU()
        self.conv4 = torch.nn.Conv2d(in_channels=24, out_channels=30, kernel_size=(3, 3), stride=(2, 2))
        self.elu4= torch.nn.ELU()
        self.conv5 = torch.nn.Conv2d(in_channels=30, out_channels=36, kernel_size=(3, 3), stride=(2, 2))
        self.elu5 = torch.nn.ELU()

        self.linear1 = torch.nn.Linear(7056, 2500)
        self.elu6 = torch.nn.ELU()
        self.linear2 = torch.nn.Linear(2500, 500)
        self.elu7 = torch.nn.ELU()
        self.linear3 = torch.nn.Linear(500, 100)
        self.elu8 = torch.nn.ELU()
        self.linear4 = torch.nn.Linear(100, output_size)


    def forward(self, x):
        x = self.conv1(x)
        x = self.elu1(x)
        x = self.conv2(x)
        x = self.elu2(x)
        x = self.conv3(x)
        x = self.elu3(x)
        x = self.conv4(x)
        x = self.elu4(x)
        x = self.conv5(x)
        x = self.elu5(x)
        x = x.reshape(-1, 7056)
        x = self.linear1(x)
        x = self.elu6(x)
        x = self.linear2(x)
        x = self.elu7(x)
        x = self.linear3(x)
        x = self.elu8(x)
        x = self.linear4(x)

        return x

    def save(self, file_name = 'model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(),file_name)

model = Net(input_size = (500,500,1), output_size = 5)
print(summary(model,input_size = (1,500,500)))
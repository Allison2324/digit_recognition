import torch
from time import time
from torch import nn, optim


class Model:
    def __init__(self, dataset, model_name, epochs, learning_rate, momentum):
        self.model = nn.Sequential(nn.Linear(28 * 28, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))
        self.dataset = dataset
        self.criterion = nn.NLLLoss()
        self.model_name = model_name
        self.epochs = epochs
        self.leaning_rate = learning_rate
        self.momentum = momentum

    def train(self):
        x = self.dataset.data["X"]
        y = self.dataset.data["Y"]

        x = x.view(x.shape[0], -1)
        output = self.model(x)
        loss = self.criterion(output, y)
        loss.backward()

        optimizer = optim.SGD(self.model.parameters(), lr=self.leaning_rate, momentum=self.momentum)
        time0 = time()
        for e in range(self.epochs):
            x = x.view(x.shape[0], -1)
            optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            optimizer.step()

            print("Epoch = {}; Training Time (in minutes) = {};".format(e, (time() - time0) / 60))

    def predict(self):
        x = self.dataset.data["X"]
        y = self.dataset.data["Y"]

        correct_count, all_count = 0, 0
        for i in range(len(y)):
            image = x[i].view(1, 784)
            with torch.no_grad():
                output = self.model(image)

            ps = torch.exp(output)
            probability = list(ps.numpy()[0])
            pred_label = probability.index(max(probability))
            true_label = y.numpy()[i]
            if true_label == pred_label:
                correct_count += 1
            all_count += 1

        print("Number Of Images Tested =", all_count)
        print("\nModel Accuracy = {}%".format((correct_count / all_count) * 100))

    def save_model(self):
        torch.save(self.model, "models/" + self.model_name)


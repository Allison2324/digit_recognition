import pandas as pd
import numpy as np
import torch


class Dataset:
    def __init__(self, filename):
        self.filename = filename
        self.data = self.read()

    def read(self):
        data = pd.read_csv(self.filename)
        x = torch.tensor(np.array(data.loc[:, data.columns != "label"])).type(torch.FloatTensor)
        y = torch.tensor(np.array(data["label"]))
        return {"X": x, "Y": y}

    def get_length(self):
        return len(self.data["X"])

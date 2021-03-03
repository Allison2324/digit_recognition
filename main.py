from model import *
from PIL import Image
from dataset import Dataset


def plot_digit(x_set, y_set, idx):
    img = x_set[idx].T.reshape(28, 28)
    plt.imshow(img, cmap='Greys', interpolation='nearest')
    plt.title('true label: %d' % y_set.T[idx])
    plt.show()


if __name__ == "__main__":
    dataset = Dataset("dataset/train.csv")
    model = Model(dataset, "1", 5000, 0.001, 0.9)
    model.train()
    model.predict()
    model.save_model()
    print("Work finished!")

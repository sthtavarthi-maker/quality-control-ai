import torch


def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print("Using GPU")

    else:

        device = torch.device("cpu")

        print("Using CPU")

    return device


if __name__ == "__main__":

    device = get_device()

    print("Selected device:", device)
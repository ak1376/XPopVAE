import math
import torch


def make_two_process_dataset(n_samples, input_length, noise_std=0.1):
    x_axis = torch.linspace(0, 1, input_length)

    signals = []
    labels = []

    for _ in range(n_samples):
        label = torch.randint(0, 2, (1,)).item()

        if label == 0:
            amplitude = torch.empty(1).uniform_(0.8, 1.2).item()
            frequency = torch.empty(1).uniform_(1.0, 3.0).item()
            phase = torch.empty(1).uniform_(0, 2 * math.pi).item()
            signal = amplitude * torch.sin(2 * math.pi * frequency * x_axis + phase)
        else:
            height = torch.empty(1).uniform_(0.8, 1.2).item()
            center = torch.empty(1).uniform_(0.2, 0.8).item()
            width = torch.empty(1).uniform_(0.03, 0.10).item()
            signal = height * torch.exp(-0.5 * ((x_axis - center) / width) ** 2)

        signal = signal + noise_std * torch.randn(input_length)
        signal = signal.unsqueeze(0)

        signals.append(signal)
        labels.append(label)

    X = torch.stack(signals)
    y = torch.tensor(labels)
    return X, y
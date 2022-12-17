#%%
import torch
import torch.optim as optim
import torchaudio
from torchmetrics.audio.sdr import SignalDistortionRatio
import torchaudio
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.optim as optim
import random
import glob
import random
import IPython


#%%

wavs = []
for file in glob.glob("wavs/*.wav"):
    wavs.append(file)
wav_tensors = []
for wav in wavs:
    wav_tensor, sample_rate = torchaudio.load(wav)
    wav_tensors.append(wav_tensor)


def get_chunk(start_time, waveform, sample_rate):
    # Specify the number of seconds for the window and the start time
    window_seconds = 2
    start_time_secs = start_time

    window_frames = int(sample_rate * window_seconds)
    start_time_frames = int(sample_rate * start_time_secs)

    # Get the label
    label = waveform[:, start_time_frames:start_time_frames + window_frames]

    # Downsample the label to 6kHz
    label = torchaudio.transforms.Resample(sample_rate, 6000)(label)

    # Get the target
    target = waveform[:, start_time_frames + window_frames:start_time_frames + 2 * window_frames]

    # Downsample the target to 6kHz
    target = torchaudio.transforms.Resample(sample_rate, 6000)(target)
    return label, target
#%%

USE_CUDA = True
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("Using device:", DEVICE)


# 1. Import necessary PyTorch libraries and modules
class Seq2SeqAudioModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Seq2SeqAudioModel, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim)
        self.mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=6000)
        self.relu = nn.ReLU()
        self.decoder = nn.LSTM(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.encoder(x)
        x = self.relu(x)
        x = self.relu(x)
        x, _ = self.decoder(x)
        return x

# 3. Define the model hyperparameters
input_dim = 12000
hidden_dim = 512
output_dim = 12000

# 4. Instantiate the model
model = Seq2SeqAudioModel(input_dim, hidden_dim, output_dim)

# 5. Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# 6. Train the model, checkpoint each 100 epochs
num_epochs = 20
checkpoint = 50
LOAD = False
if LOAD:
    model.load_state_dict(torch.load('model.pt'))



for epoch in range(num_epochs):
    optimizer.zero_grad()
    waveform, sample_rate = torchaudio.load(random.choice(wavs))
    label, target = get_chunk(random.randint(0, 120), waveform, sample_rate)
    output = model(label)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    # if we want the loss printed by tqdm, we need to use the following line
    print(f'Epoch: {epoch+1:3}/{num_epochs:3} loss: {loss.item():10.8f}')
    if epoch % checkpoint == 0:
        torch.save(model.state_dict(), 'model.pt')

# 7. Save the model
torch.save(model.state_dict(), 'model.pt')

#%%

# 8. Load the model
model.load_state_dict(torch.load('model.pt'))

waveform, sample_rate = torchaudio.load(wavs[0])

window_seconds = 2
window_frames = int(sample_rate * window_seconds)
start_time_secs = 100
start_time_frames = int(sample_rate * start_time_secs)
# truncate the waveform to 1 second
label = waveform[:, start_time_frames:start_time_frames + window_frames]
# downsampling to 6kHz
label = torchaudio.transforms.Resample(sample_rate, 6000)(label)

target = waveform[:, start_time_frames + window_frames:start_time_frames + 2 * window_frames]
target = torchaudio.transforms.Resample(sample_rate, 6000)(target)

def get_output(label, seconds_of_output):
    '''Feed the label to the model to get the output,
    then feed the output back to the model to get the next output
    and so on for the number of seconds specified in seconds_of_output
    '''
    outputs = []
    outputs.append(label)
    outputs.append(model(label))
    for i in range(seconds_of_output):
        outputs.append(model(outputs[-1]))
    return torch.cat(outputs, dim=1)
        
def plot_label_and_target(label, target):
    plt.figure(figsize=(12, 4))
    plt.plot(target[0].numpy(), label='Target')
    plt.plot(label[0].numpy(), label='Label')
    plt.legend()
    plt.show()



# Get the output
output = get_output(label, 5)
plot_label_and_target(label, output.detach())
torchaudio.save('output.wav', output.detach(), 6000)
# play the output
IPython.display.Audio('output.wav')

    
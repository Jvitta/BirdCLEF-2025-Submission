import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import IPython.display as ipd
from glob import glob
import pickle

df = pd.read_csv('/kaggle/input/birdclef-2025/train.csv')
fabio = df[df.author == 'Fabio A. Sarria-S'].copy()

print(f'We have {len(fabio)} Fabio\'s recordings in total')

N = 10
chunk_len = 0.05 # Chunk len in seconds

fig, ax = plt.subplots(nrows=N)
fig.set_size_inches((24, 3 * N))
for n in range(N):
    # Load the data
    rec = fabio.iloc[n]
    wav, sr = librosa.load(f'/kaggle/input/birdclef-2025/train_audio/{rec.filename}')

    # Calculate the sound power
    power = wav ** 2
    
    # Split the data into chunks and sum the energy in every chunk
    chunk = int(chunk_len * sr)
    
    pad = int(np.ceil(len(power) / chunk) * chunk - len(power))
    power = np.pad(power, (0, pad))
    power = power.reshape((-1, chunk)).sum(axis=1)

    t = np.arange(len(power)) * chunk_len
    ax[n].plot(t, 10 * np.log10(power))
    ax[n].set_xlim([0, 20])

level = -50
N = 1
result = fabio['filename'].to_frame()

for n, rec in fabio.iterrows():
    # Load the data
    wav, sr = librosa.load(f'/kaggle/input/birdclef-2025/train_audio/{rec.filename}')

    # Calculate the sound power
    power = wav ** 2
    
    # Split the data into chunks and sum the energy in every chunk
    chunk = int(chunk_len * sr)
    
    pad = int(np.ceil(len(power) / chunk) * chunk - len(power))
    power = np.pad(power, (0, pad))
    power = power.reshape((-1, chunk)).sum(axis=1)

    power_dB = 10 * np.log10(power)
    x = power_dB - level
    intersections = np.where(x[:-1] * x[1:] < 0)[0]
    a, b = intersections[:2]
    result.loc[result['filename'] == rec.filename, 'start'] = a * chunk_len
    result.loc[result['filename'] == rec.filename, 'stop'] = b * chunk_len

result.to_csv('fabio.csv', index=False)
display(result)

torch.set_num_threads(1)
model, (get_speech_timestamps, _, read_audio, _, _) = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')

df = pd.read_csv('/kaggle/input/birdclef-2025/train.csv')
df = df[df.collection == 'CSA'].copy()

author_map = {
    'Paula Caycedo-Rosales | Juan-Pablo López': 'Paula Caycedo-Rosales',
    'Eliana Barona-Cortés | Daniela García-Cobos': 'Eliana Barona-Cortés',
    'Ana María Ospina-Larrea | Daniela Murillo': 'Ana María Ospina-Larrea',
    'Alexandra Butrago-Cardona': 'Alexandra Buitrago-Cardona',
    'Eliana Barona- Cortés': 'Eliana Barona-Cortés',
    'Diego A Gómez-Morales': 'Diego A. Gomez-Morales',
}
author_map_func = lambda x: author_map[x] if x in author_map.keys() else x

df.author = df.author.map(author_map_func)
authors = sorted(df.author.unique())

# Here, I limit the output to 3 authors. Otherwise, the webpage becomes too heavy to load.
# If your are interested, please check the previous version of the notebook!
for author in authors[:3]:
    selection = df[df.author == author].copy()
    print(f'We have {len(selection)} CSA recordings by {author} in total')
    
    N = len(selection)
    chunk_len = 0.2 # Chunk len in seconds
    
    for n in range(N):
        # Load the data
        rec = selection.iloc[n]
        fname = f'/kaggle/input/birdclef-2025/train_audio/{rec.filename}'
        wav, sr = librosa.load(fname)
    
        # Calculate the sound power
        power = wav ** 2
        
        # Split the data into chunks and sum the energy in every chunk
        chunk = int(chunk_len * sr)
        
        pad = int(np.ceil(len(power) / chunk) * chunk - len(power))
        power = np.pad(power, (0, pad))
        power = power.reshape((-1, chunk)).sum(axis=1)

        speech_timestamps = get_speech_timestamps(torch.Tensor(wav), model)
        segmentation = np.zeros_like(wav)
        for st in speech_timestamps:
            segmentation[st['start']: st['end']] = 20
    
        fig = plt.figure(figsize=(24, 3))
        fig.suptitle(f'{rec.filename} by {rec.author}')
        
        t = np.arange(len(power)) * chunk_len
        plt.plot(t, 10 * np.log10(power), 'b')
        
        t = np.arange(len(segmentation)) / sr
        plt.plot(t, segmentation, 'r')        
        plt.show()
        
        display(ipd.Audio(fname))

files = sorted(glob('/kaggle/input/birdclef-2025/train_audio/*/*.ogg'))
voice_data = {}
for fname in files:
    wav = read_audio(fname)
    speech_timestamps = get_speech_timestamps(wav, model, return_seconds=True, threshold=0.4)
    if len(speech_timestamps):
        voice_data[fname] = speech_timestamps

        with open('train_voice_summary.txt', 'a') as f:
            f.write(f'{fname}\n')
            
with open('train_voice_data.pkl', 'wb') as f:
    pickle.dump(voice_data, f)

    from glob import glob
    import pickle

files = sorted(glob('/kaggle/input/birdclef-2025/train_soundscapes/*.ogg'))
voice_data = {}
for fname in files:
    wav = read_audio(fname)
    speech_timestamps = get_speech_timestamps(wav, model, return_seconds=True, threshold=0.4) # default threshold
    if len(speech_timestamps):
        voice_data[fname] = speech_timestamps

        with open('ss_voice_summary.txt', 'a') as f:
            f.write(f'{fname}\n')
            
with open('ss_voice_data.pkl', 'wb') as f:
    pickle.dump(voice_data, f)
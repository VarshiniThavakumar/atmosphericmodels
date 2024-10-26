import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import os
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D

proj_dir = Path("..") / "Data/r77-mini-data-fortnight"
print(proj_dir.joinpath("input"))

data_dir = proj_dir
input_dir = data_dir.joinpath("input")
fixed_input_dir = input_dir.joinpath("fixed")
temporal_input_dir = input_dir.joinpath("temporal")
target_dir = data_dir.joinpath("target")
fixed_target_dir = target_dir.joinpath("fixed")
temporal_target_dir = target_dir.joinpath("temporal")

files = list(temporal_target_dir.iterdir())


index_dir = Path("..")/"Index"
files_index = list(index_dir.iterdir())
indices = np.load(files_index[0]).squeeze()

#First hour of data
index = indices[0]
tt = np.load(files[index]).squeeze()
tt.shape

files_ti = list(temporal_input_dir.iterdir())
#all diff data
data = []

for j in range(360):
    index = indices[j]

    tt = np.load(files[index]).squeeze()
    tt = np.transpose(tt, (0, 2, 3, 1))[:, :, :, :64]  #shape: (144, 100, 3, 64)

    ti = np.load(files_ti[index]).squeeze()
    ti = np.transpose(ti, (0, 2, 1))[:, :, :64]  #shape: (144, 3, 64)

    #ensure ti has the same second dimension as tt
    ti_expanded = np.repeat(ti[:, None, :, :], tt.shape[1], axis=1)  #shape: (144, 100, 3, 64)

    #calculate diff and reshape
    diff_data = tt - ti_expanded  #shape: (144, 100, 3, 64)
    reshaped_tt = diff_data.reshape(-1, 3, 64)  #shape: (14400, 3, 64)

    data.append(reshaped_tt)


data = np.concatenate(data).reshape(-1, 3, 64)
data.shape

def normalise(vector):
    min_val = np.min(vector)
    max_val = np.max(vector)
    normalised_vector = (vector - min_val) / (max_val - min_val)
    return normalised_vector, min_val, max_val


training_data = normalise(data)[0]

files_fixed = list(fixed_input_dir.iterdir())
files_ft = list(fixed_target_dir.iterdir())
#std dev calculated for each grid using fxed target vectors, combined with fixed input vector to give three fixed variables for each grid
#fixed variables all (fixed input repeated 100 times for each grid)

cond = []

for j in range(360):
    index = indices[j]
    fi = np.load(files_fixed[index]).squeeze()
    ft = np.load(files_ft[index]).squeeze()

    stdev_orog = np.array([np.std(i) for i in ft]).reshape(-1,1)
    fixed_input = np.hstack((fi, stdev_orog))
    reshaped_fi = np.repeat(fixed_input, 100, axis=0)
    cond.append(reshaped_fi)


cond = np.concatenate(cond).reshape(-1, 3)
cond.shape

lsf = cond[:,0]
orog = cond[:,1]
stdev_orog = cond[:,2]

#all data
tt_all = np.stack([
    np.transpose(np.load(files[indices[i]]).squeeze(), (0, 2, 3, 1))[:,:,:,:64].reshape((14400, 3, 64), order = 'C')
    for i in range(360)
])

all_data = tt_all.reshape(-1,3,64)
all_data.shape

#has inversion
def has_inversion(temperature_profile, troposphere_height):
    gradient = np.gradient(temperature_profile[:troposphere_height])
    return np.any(np.array(gradient) > 0.0)

temp = all_data[:,2,:]

inversion = np.array([has_inversion(i, 40) for i in temp])
inversion[inversion == False].shape[0]

indices1 = np.where((inversion == False) & (lsf == 0))[0]
trainingdata1 = training_data[indices1]

indices2 = np.where((inversion == False) & (lsf == 1) & (orog < 0.06))[0]
trainingdata2 = training_data[indices2]

indices3 = np.where((inversion == False) & (lsf == 1) & (orog > 0.06))[0]
trainingdata3 = training_data[indices3]

indices4 = np.where((inversion == False) & (lsf < 1) & (lsf > 0) & (orog < 0.03))[0]
trainingdata4 = training_data[indices4]

indices5 = np.where((inversion == False) & (lsf < 1) & (lsf > 0) & (orog > 0.03))[0]
trainingdata5 = training_data[indices5]

indices6 = np.where((inversion == True) & (lsf == 0))[0]
trainingdata6 = training_data[indices6]

indices7 = np.where((inversion == True) & (lsf == 1) & (orog < 0.06))[0]
trainingdata7 = training_data[indices7]

indices8 = np.where((inversion == True) & (lsf == 1) & (orog > 0.06))[0]
trainingdata8 = training_data[indices8]

indices9 = np.where((inversion == True) & (lsf < 1) & (lsf > 0) & (orog < 0.03))[0]
trainingdata9 = training_data[indices9]

indices10 = np.where((inversion == True) & (lsf < 1) & (lsf > 0) & (orog > 0.03))[0]
trainingdata10 = training_data[indices10]

data_list = [trainingdata1, trainingdata2, trainingdata3,trainingdata4,trainingdata5,trainingdata6, trainingdata7, trainingdata8,trainingdata9,trainingdata10 ]
sum([i.shape[0] for i in data_list]) == data.shape[0]


for idx, trainingdata in enumerate(data_list, start = 1): 
    
    model = Unet1D(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 3
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = 64,
        timesteps = 100,
        objective = 'pred_v'
    )

    training_seq =  torch.from_numpy(training_data)

    trainer = Trainer1D(
        diffusion,
        dataset = training_seq,
        train_batch_size = 10, #set batch size here (take 100 samples, one grid)
        train_lr = 1e-4,
        train_num_steps = 1000,         # total training steps (1000)
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
    )
    trainer.train()

    # after a lot of training

    sampled_seq = diffusion.sample(batch_size = 1000)
    torch.save(sampled_seq, f"sampled_seq{idx}.pt") 

sampled_sequences_list = []

for i in range(1, 6):  
    sampled_seq = torch.load(f"sampled_seq{i}.pt")
    sampled_sequences_list.append(sampled_seq.cpu().numpy())

sampled_sequences = np.array(sampled_sequences_list)

folder = 'Samples_cond_diffs_0'
if not os.path.exists(folder):
    os.makedirs(folder)

file_path = os.path.join(folder, 'sample.npy')
np.save(file_path, sampled_sequences)

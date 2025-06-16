from statistics import median
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


out = torch.load('/home/mehran/motion-planning/MotionPlanning_UR5/train_data.pt')

data = []

for i in range(len(out['paths'])):
    data.append(out['paths'][i].numpy())

# print(data[0].shape)

group = ['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6']
length = []

mean = np.zeros((len(out['paths']),6))
for i in range(len(out['paths'])):
    mean[i,:] = np.mean(data[i],axis = 0)
    length.append(len(data[i]))
num_waypoints = np.sum(length)

print('maximum length = ', num_waypoints)

mean_pd = pd.DataFrame(mean, columns = group)

sns.set_theme(style="whitegrid")

fig, axs = plt.subplots(2, 3)
axs = axs.flatten()
for i in range(6):
        sns.violinplot(y=mean_pd[group[i]], ax=axs[i])
sns.despine()
plt.show()

med = np.zeros((len(out['paths']),6))
for i in range(len(out['paths'])):
    med[i,:] = np.median(data[i],axis = 0)

med_pd = pd.DataFrame(med, columns = group)

sns.set_theme(style="whitegrid")

fig, axs = plt.subplots(2, 3)
axs = axs.flatten()
for i in range(6):
        sns.violinplot(y=med_pd[group[i]], ax=axs[i])
sns.despine()
plt.show()

st = np.zeros((len(out['paths']),6))
for i in range(len(out['paths'])):
    st[i,:] = np.std(data[i],axis = 0)

st_pd = pd.DataFrame(st, columns = group)

sns.set_theme(style="whitegrid")

fig, axs = plt.subplots(2, 3)
axs = axs.flatten()
for i in range(6):
        sns.violinplot(y=st_pd[group[i]], ax=axs[i])
sns.despine()
plt.show()


joints_init = np.zeros((6,16000))

for i in range(len(data)):
    joints_init[:,i] = data[i][0]


m = 0
fig, axs = plt.subplots(2, 3)
fig.suptitle('Initial Configuration')
for i in range(2):
    for j in range(3):
        axs[i,j].hist(joints_init[m,:])
        m+=1

plt.show()


joints = np.zeros((6,num_waypoints))
m = 0
for i in range(len(data)):
    for j in range(len(data[i])):
        joints[:,m] = data[i][j]
        m+=1

k = 0
fig, axs = plt.subplots(2, 3)
fig.suptitle('All Configurations')
for i in range(2):
    for j in range(3):
        axs[i,j].hist(joints[k,:])
        k+=1

plt.show()   
'''
This file Visualises the ICESAR data
'''


import seaborn as sns
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
import numpy as np
from scipy import io
import matplotlib.pyplot as plt

labels = ['Open Water', 'Grey-white Ice', 'Level FY-Ice','Deformed FY-Ice', 'Nilas', 'Grey Ice']
color = ["#22181c", "#5dd9c1", "#ffe66d", "#e36397", "#8377d1", "#3b429f"]

sns.set_palette(sns.color_palette(color))

stats = np.load("/work/saloua/uncertainty-main/outputs/stats_icesar.npy")[:,:,:,:6]

filename = "/work/saloua/Datasets/ICESAR/"

gt = np.transpose(io.loadmat("".join([f'{filename}ICESAR_training_mask.mat']))['mask'])
y_true = gt.flatten().astype('float')

L = np.moveaxis(np.transpose(io.loadmat("".join([f'{filename}ICESAR_L_band.mat']))['L']), 0, -1)
D1 = L.shape[-1]
for i in range(D1):
    L = np.concatenate((L, stats[i,:,:,:]), axis = -1)

L_img = L.reshape((-1,L.shape[-1]))    
#hh = L[0,:,:]#(L[0,:,:]-np.min(L[0,:,:]))/(np.max(L[0,:,:]) - np.min(L[0,:,:]))
#hv = L[1,:,:]#(L[1,:,:]-np.min(L[1,:,:]))/(np.max(L[1,:,:]) - np.min(L[1,:,:]))
#vv = L[2,:,:]#(L[2,:,:]-np.min(L[2,:,:]))/(np.max(L[2,:,:]) - np.min(L[2,:,:]))
#L_img = np.moveaxis(np.concatenate((hh[None,:,:], hv[None,:,:], vv[None,:,:]), axis = 0), 0, -1).reshape((-1,3))

C = np.moveaxis(np.transpose(io.loadmat("".join([f'{filename}ICESAR_C_band.mat']))['C']), 0, -1)
D2 = C.shape[-1]
for i in range(D1,D2+D1):
    C = np.concatenate((C, stats[i,:,:,:]), axis = -1)

C_img = C.reshape((-1,C.shape[-1]))    

#hv = C[0,:,:] #(C[0,:,:]-np.min(C[0,:,:]))/(np.max(C[0,:,:]) - np.min(C[0,:,:]))
#hh = C[1,:,:] #(C[1,:,:]-np.min(C[1,:,:]))/(np.max(C[1,:,:]) - np.min(C[1,:,:]))
#C_img = np.moveaxis(np.concatenate((hv[None,:,:], hh[None,:,:]), axis = 0), 0, -1).reshape((-1,2))

glcm_feat = ["energy", "contrast", "correlation", "homogeneity", "dissimilarity", "ASM"]

cols = ["L1 [dB]", "L1-energy",  "L1-contrast", "L1-correlation", "L1-homogeneity", "L1-dissimilarity", "L1-ASM",
        "L2 [dB]", "L2-energy",  "L2-contrast", "L2-correlation", "L2-homogeneity", "L2-dissimilarity", "L2-ASM",
        "L3 [dB]", "L3-energy",  "L3-contrast", "L3-correlation", "L3-homogeneity", "L3-dissimilarity", "L3-ASM",
        "C1 [dB]", "C1-energy",  "C1-contrast", "C1-correlation", "C1-homogeneity", "C1-dissimilarity", "C1-ASM",
        "C2 [dB]", "C2-energy",  "C2-contrast", "C2-correlation", "C2-homogeneity", "C2-dissimilarity", "C2-ASM",
        "Classes"]

data = pd.DataFrame(data = {"L1 [dB]": L_img[y_true!=0,0], "L2 [dB]": L_img[y_true!=0,1], "L3 [dB]": L_img[y_true!=0,2], 
                            "L1-energy": L_img[y_true!=0,3],  "L1-contrast": L_img[y_true!=0,4], "L1-correlation": L_img[y_true!=0,5], "L1-homogeneity": L_img[y_true!=0,6], "L1-dissimilarity": L_img[y_true!=0,7], "L1-ASM": L_img[y_true!=0,8],
                            "L2-energy": L_img[y_true!=0,9],  "L2-contrast": L_img[y_true!=0,10], "L2-correlation": L_img[y_true!=0,11], "L2-homogeneity": L_img[y_true!=0,12], "L2-dissimilarity": L_img[y_true!=0,13], "L2-ASM": L_img[y_true!=0,14],
                            "L3-energy": L_img[y_true!=0,15],  "L3-contrast": L_img[y_true!=0,16], "L3-correlation": L_img[y_true!=0,17], "L3-homogeneity": L_img[y_true!=0,18], "L3-dissimilarity": L_img[y_true!=0,19], "L3-ASM": L_img[y_true!=0,20],
                            "C1 [dB]": C_img[y_true!=0,0], "C2 [dB]": C_img[y_true!=0,1], 
                            "C1-energy": C_img[y_true!=0,2],  "C1-contrast": C_img[y_true!=0,3], "C1-correlation": C_img[y_true!=0,4], "C1-homogeneity": C_img[y_true!=0,5], "C1-dissimilarity": C_img[y_true!=0,6], "C1-ASM": C_img[y_true!=0,7],
                            "C2-energy": C_img[y_true!=0,8],  "C2-contrast": C_img[y_true!=0,9], "C2-correlation": C_img[y_true!=0,10], "C2-homogeneity": C_img[y_true!=0,11], "C2-dissimilarity": C_img[y_true!=0,12], "C2-ASM": C_img[y_true!=0,13],
                            "Classes": y_true[y_true!=0]})
for i in range(len(labels)):
    idx = np.nonzero(np.asarray(data["Classes"]==i+1))[0]
    data["Classes"][idx] = labels[i]

images_dir = "outputs/icesar/images/"

for i in cols:
    pp = sns.kdeplot(
    data=data, x=i, hue="Classes",
    fill=True, common_norm=True, palette = sns.color_palette(color),
    alpha=.5, hue_order = labels)
    fig = pp.figure
    fig.subplots_adjust(top=0.93, wspace=0.3)
    #t = fig.suptitle('ICESAR-L Pairwise Plots, first band', fontsize=14)
    fig.savefig(f"{images_dir}GP_KdePlot_{i}.png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=500,)
    plt.close()

exit()

#! Prepare data for plots
idx1 = np.nonzero(y == 1)
idx2 = np.nonzero(y == 2)
idx3 = np.nonzero(y == 3)
idx4 = np.nonzero(y == 4)
idx5 = np.nonzero(y == 5)
idx6 = np.nonzero(y == 6)
x = min(len(idx1[0]), len(idx2[0]), len(idx3[0]), len(idx4[0]), len(idx5[0]), len(idx6[0]))
idx1 = idx1[0][:x]
idx2 = idx2[0][:x]
idx3 = idx3[0][:x]
idx4 = idx4[0][:x]
idx5 = idx5[0][:x]
idx6 = idx6[0][:x]

L_feat1 = pd.DataFrame(data = {"Open water": X1[idx1,0], "Grey-white ice": X1[idx2,0], "Level ice": X1[idx3,0], "Deformed ice": X1[idx4,0], "Nilas": X1[idx5,0], "Grey ice": X1[idx6,0]})
L_feat2 = pd.DataFrame(data = {"Open water": X1[idx1,1], "Grey-white ice": X1[idx2,1], "Level ice": X1[idx3,1], "Deformed ice": X1[idx4,1], "Nilas": X1[idx5,1], "Grey ice": X1[idx6,1]})
L_feat3 = pd.DataFrame(data = {"Open water": X1[idx1,2], "Grey-white ice": X1[idx2,2], "Level ice": X1[idx3,2], "Deformed ice": X1[idx4,2], "Nilas": X1[idx5,2], "Grey ice": X1[idx6,2]})
C_feat1 = pd.DataFrame(data = {"Open water": X2[idx1,0], "Grey-white ice": X2[idx2,0], "Level ice": X2[idx3,0], "Deformed ice": X2[idx4,0], "Nilas": X2[idx5,0], "Grey ice": X2[idx6,0]})
C_feat2 = pd.DataFrame(data = {"Open water": X2[idx1,1], "Grey-white ice": X2[idx2,1], "Level ice": X2[idx3,1], "Deformed ice": X2[idx4,1], "Nilas": X2[idx5,1], "Grey ice": X2[idx6,1]})

# store data feature as an attribute
L_feat1['feat'] = 'L1'   
L_feat2['feat'] = 'L2'
L_feat3['feat'] = 'L3'
C_feat1['feat'] = 'C1'
C_feat2['feat'] = 'C2'

# Concatenate datasets
data = pd.concat([L_feat1, L_feat2, L_feat3, C_feat1, C_feat2])
#! Pairwise plots
pp = sns.pairplot(L_feat1[labels], size=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))

fig = pp.fig
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('ICESAR-L Pairwise Plots, first band', fontsize=14)
fig.savefig("stats_plt_L1")

#! Violin plots
f, (ax) = plt.subplots(1, 1, figsize=(12, 4))
f.suptitle('Wine Quality - Sulphates Content', fontsize=14)

sns.violinplot(x="Open water", y="Grey-white ice", data=L_feat1,  ax=ax)
ax.set_xlabel("Wine Quality",size = 12,alpha=0.8)
ax.set_ylabel("Wine Sulphates",size = 12,alpha=0.8)
f.savefig("stats_plt_L1_violin")

exit()

print(data1.images.shape)
print(data2.images.shape)

# for i in range(6):
#     y0 = np.argwhere(y == i+1)
#     y1 = np.argwhere(data2.y_test == i+1)
#     print(len(y0)+len(y1))
#! classes 
'''
0: Open water 3836
1: Grey-white ice 17024
2: Level ice 22772
3: Deformed ice 10170
4: Nilas 20714
5: Grey ice 3748
'''
# y0 = np.argwhere(y == 1)
# y1 = np.argwhere(y == 2)
# y2 = np.argwhere(y == 3)
# y3 = np.argwhere(y == 4)
# y4 = np.argwhere(y == 5)
# y5 = np.argwhere(y == 6)

labels = ["Open water", "Grey-white ice", "Level ice", "Deformed ice", "Nilas", "Grey ice"]
#* Plot results in 3d
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
scatter = ax.scatter(X1[:,0], X1[:,1], X1[:,2], marker="o", c=y, s=25, edgecolor="k")
fig.savefig("ICESAR_L_data.png")

#* Plot results
fig, ax = plt.subplots(figsize=(10, 5))
scatter = ax.scatter(X1[:,0], X1[:,1], marker="o", c=y, s=25, edgecolor="k", label = labels)
a = scatter.legend_elements()
ax.legend(a, loc='upper left')
fig.savefig("ICESAR_L_data01.png")

#* Plot results
fig, ax = plt.subplots(figsize=(10, 5))
scatter = ax.scatter(X1[:,0], X1[:,2], marker="o", c=y, s=25, edgecolor="k", label = labels)
ax.legend(*scatter.legend_elements(), loc='upper left')
fig.savefig("ICESAR_L_data02.png")

#* Plot results
fig, ax = plt.subplots(figsize=(10, 5))
scatter = ax.scatter(X1[:,1], X1[:,2], marker="o", c=y, s=25, edgecolor="k", label = labels)
ax.legend(*scatter.legend_elements(), loc='upper left')
fig.savefig("ICESAR_L_data12.png")

#* Plot results
fig, ax = plt.subplots(figsize=(10, 5))
scatter = ax.scatter(X2[:,0], X2[:,1], marker="o", c=data2.y_test, s=25, edgecolor="k", label = labels)
ax.legend(*scatter.legend_elements(), loc='upper left')
fig.savefig("ICESAR_C_data.png")
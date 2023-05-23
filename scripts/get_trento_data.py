from cmath import nan
import slideio
import imageio
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import colors, colormaps, cm

from trento_config import (
            dataset,
            project_name,
            images_dir,
            outputs_dir,
            compatibility_matrix,
            color,
            labels,
        )

data_dir = "/work/saloua/uncertainty-main/datasets/"
#image = "TCGA-D8-A1JG-01Z-00-DX1.BA6D5CC7-3A9B-4D17-A86A-B159D345A216.svs"
#gt = "TCGA-D8-A1JG-DX1_xmin15677_ymin69205_.png"

# slide = slideio.open_slide(f"{data_dir}images/{image}",'SVS')
# num_scenes = slide.num_scenes

# print(num_scenes)
# scene = slide.get_scene(0)
# print(num_scenes, scene.name, scene.rect, scene.num_channels)
# print(slide.raw_metadata)

# image = scene.read_block()[69205:77602, 15677:22140,:]
# np.save(f"{data_dir}images/data.npy", np.asarray(image))
image, y = dataset.full_dataset

#print(max(image))
#print(min(image))

# y = imageio.v2.imread(f"{data_dir}masks/{gt}")
# y[y==0] = 0 # Outside Roi
# #y[y==1] = 21 #Tumor
# #y[y==4] = 22 # Necrosis
# #y[y==3] = 23 # Lymphocytic_infiltrate
# y[y==7] = 5 # Grandular secretions
# #y[y==2] = 26 # Stroma
# y[y==15] = 5 # Undetermined
# y[y==14] = 5 # Lymphatics
# y[y==18] = 5 # Blood_vessel
# y[y==11] = 5 # Immune_infiltrate
# y[y==12] = 5 # Mucoid

# np.save(f"{data_dir}masks/gt.npy", np.asarray(y))
#y = np.load(f"{data_dir}masks/gt.npy")
values = np.unique(y.ravel())
#color = cm.get_cmap("jet", len(values))

print(values)
#labels = list(map(str, values))#["0", "1", "2", "3", "4", "7", "11", "12", "14", "15", "18"]
patches = [mpatches.Patch(color=color[i], label=labels[i]) for i in range(len(values)) ]
plt.figure(dpi=500)
plt.imshow(
    y.reshape(dataset.shape),
    interpolation="nearest",
    cmap=colors.ListedColormap(color),
)
plt.axis("off")
plt.legend(handles=patches, loc=8, ncol = 4, fontsize='small', borderaxespad=-3.25)#, columnspacing = 1.35)
plt.savefig(
    f"{images_dir}trento_gt.eps",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)

# plt.figure(dpi=500)
# plt.imshow(image)
# plt.axis("off")
# plt.savefig(
#     f"{images_dir}test_im.eps",
#     bbox_inches="tight",
#     pad_inches=0,
#     dpi=500,
# )
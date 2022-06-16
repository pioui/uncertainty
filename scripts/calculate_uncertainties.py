import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import colors

from uncertainty.uncertainty_measurements import geometry_based_uncertainty, variance, shannon_entropy, semantic_based_uncertainty

print(os.listdir('outputs/'))

for project_name in os.listdir('outputs/'):
    if project_name == 'trento':
        dataset = 'trento'
        from trento_config import (
            dataset,
            project_name,
            images_dir,
            outputs_dir,
            heterophil_matrix,
            color
        )
    elif project_name == 'bcss':
        dataset = 'bcss'
        from bcss_config import (
            dataset,
            project_name,
            images_dir,
            outputs_dir,
            heterophil_matrix,
            color
        )

    else:
        continue

    print(project_name)

    X,y =dataset.full_dataset # 15107
    y_true = y.reshape(-1)

    acc_dict = []
    for file in os.listdir(os.path.join(outputs_dir)):
        print(file)
        if os.path.splitext(file)[-1].lower()=='.npy':
            model_name = file.split("_")[1].split(".")[0]

            y_pred_prob = np.load(os.path.join(outputs_dir,file))
            print(y_pred_prob.shape)
            y_pred_max_prob = y_pred_prob.max(1)
            y_pred = y_pred_prob.argmax(1)+1

            plt.figure(dpi=500)
            plt.imshow(y_pred.reshape(dataset.shape), interpolation='nearest', 
            cmap = colors.ListedColormap(color[1:])
            )
            plt.axis('off')
            plt.savefig(f"{images_dir}{model_name}_PREDICTIONS.png",bbox_inches='tight', pad_inches=0, dpi=500)

            plt.figure(dpi=500)
            plt.imshow(y_true.reshape(dataset.shape), interpolation='nearest', 
            cmap = colors.ListedColormap(color)
            )
            plt.axis('off')
            plt.savefig(f"{images_dir}{model_name}_GT.png",bbox_inches='tight', pad_inches=0, dpi=500)

            # plt.figure(dpi=500)
            # plt.imshow(X.reshape(500,1500,3), interpolation='nearest', 
            # # cmap = colors.ListedColormap(color[1:])
            # )
            # plt.axis('off')
            # plt.savefig(f"{images_dir}{model_name}_SCAN.png",bbox_inches='tight', pad_inches=0, dpi=500)

            
            plt.figure(dpi=500)
            plt.imshow(geometry_based_uncertainty(y_pred_prob).reshape(dataset.shape), cmap='coolwarm', 
            vmin=0, vmax=1
            )
            plt.axis('off')
            cbar = plt.colorbar(location='top')
            cbar.ax.tick_params(labelsize =8 )
            plt.savefig(f"{images_dir}{model_name}_GBU.png",bbox_inches='tight', pad_inches=0.1 ,dpi=500)

            plt.figure(dpi=500)
            plt.imshow(variance(y_pred_prob).reshape(dataset.shape), cmap='coolwarm', 
            vmin=0, vmax=1
            )
            plt.axis('off')
            cbar = plt.colorbar(location='top')
            cbar.ax.tick_params(labelsize =8 )
            plt.savefig(f"{images_dir}{model_name}_VARIANCE.png",bbox_inches='tight', pad_inches=0.1 ,dpi=500)

            plt.figure(dpi=500)
            plt.imshow(semantic_based_uncertainty(y_pred_prob, heterophil_matrix).reshape(dataset.shape), cmap='coolwarm', 
            vmin=0, vmax=1
            )
            plt.axis('off')
            cbar = plt.colorbar(location='top')
            cbar.ax.tick_params(labelsize =8 )
            plt.savefig(f"{images_dir}{model_name}_SBU.png",bbox_inches='tight', pad_inches=0.1 ,dpi=500)

            plt.figure(dpi=500)
            plt.imshow(shannon_entropy(y_pred_prob).reshape(dataset.shape), cmap='coolwarm', 
            vmin=0, vmax=1
            )
            plt.axis('off')
            cbar = plt.colorbar(location='top')
            cbar.ax.tick_params(labelsize =8 )
            plt.savefig(f"{images_dir}{model_name}_ENTROPY.png",bbox_inches='tight', pad_inches=0.1 ,dpi=500)



        



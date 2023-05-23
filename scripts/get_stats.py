'''
This script plot the violinplots
'''
import seaborn as sns
import numpy as np
import os
import pandas as pd
from uncertainty.compatibility_matrix import calculate_compatibility_matrix
import matplotlib.pyplot as plt
from uncertainty.uncertainty_measurements import (
    geometry_based_uncertainty,
    variance,
    shannon_entropy,
    semantic_based_uncertainty,
    FR_based_uncertainty,
)

# Create an array with the colors you want to use
#colors = ["#08415c", "#cc2936"]
#colors = ["#006e90", "#f18f01"]
colors = ["#087e8b", "#ff5a5f"]
#colors = ["#6bf8fa", "#ffc7f8"]
# Set your custom color palette
sns.set_palette(sns.color_palette(colors))


for project_name in os.listdir("outputs/"):
    if project_name == "trento":
        continue
        # from trento_config import (
        #     dataset,
        #     project_name,
        #     images_dir,
        #     outputs_dir,
        #     compatibility_matrix,
        #     compatibility_matrix1,
        #     color,
        #     labels,
        # )
        # location = "bottom"
        # orientation = "horizontal"
        # col = 6
        # borderaxespad =-2
        # columnspacing = 1
    elif project_name == "bcss":
        from bcss_config import (
            dataset,
            project_name,
            images_dir,
            outputs_dir,
            compatibility_matrix,
            color,
            labels,
        )
        location = "right"
        orientation = "vertical"
        col = 3
        borderaxespad =-3.25
        columnspacing = 1.25
    # elif project_name == "bcss_patched":
    #     from bcss_patched_config import (
    #         dataset,
    #         project_name,
    #         images_dir,
    #         outputs_dir,
    #         compatibility_matrix,
    #         color,
    #         labels,
    #     )
    #     location = "right"
    #     orientation = "vertical"
    #     col = 3
    #     borderaxespad =-3.25
    #     columnspacing = 1.25

    else:
        continue

    X, y = dataset.full_dataset  # 15107
    y_true = y.reshape(-1)
    C = np.unique(y_true)
    acc_dict = []
    for file in os.listdir(os.path.join(outputs_dir)):
        if os.path.splitext(file)[-1].lower() == ".npy":
            if 'RF' not in file:
                continue
            
            if 'new' in file:
                continue
            
            if 'test' in file:
                continue
        
            if 'clf' in file:
                continue
        
            if 'OPT' in file:
                if 'new' in file:
                    model_name = file.split("_")[-2] + '_OPT_new'
                elif 'glcm' in file:
                    model_name = file.split("_")[-3] + '_OPT_GLCM'
                else:
                    model_name = file.split("_")[-2] + '_OPT'
            else:
                model_name = file.split("_")[-1].split(".")[0]
            
            print(model_name)
            y_pred_prob = np.load(os.path.join(outputs_dir, file))#, allow_pickle = True)
            #if y_pred_prob.shape[1] == 7:
            #    y_pred_prob = y_pred_prob[:,:-1]/np.transpose(np.tile(np.sum(y_pred_prob, axis =1), (6,1)))
            y_pred_max_prob = y_pred_prob.max(1)
            y_pred = y_pred_prob.argmax(1) + 1
            #y_pred = y_pred[y_true!=0]
            #id_right = np.nonzero(y_true[y_true!=0]==y_pred[y_true!=0])
            id_wrong = np.nonzero(y_true[y_true!=0]!=y_pred[y_true!=0])
            
            GU = geometry_based_uncertainty(y_pred_prob)[y_true!=0]
            FR = FR_based_uncertainty(y_pred_prob)[y_true!=0]
            Var = variance(y_pred_prob)[y_true!=0]
            H = shannon_entropy(y_pred_prob)[y_true!=0]
            #SU_1 = semantic_based_uncertainty(y_pred_prob, compatibility_matrix)[y_true!=0]
            #SU_2 = semantic_based_uncertainty(y_pred_prob, compatibility_matrix1)[y_true!=0]
            #compatibility_matrix = calculate_compatibility_matrix(X, y, "wasserstein")[1:, 1:]
            #SU_W = semantic_based_uncertainty(y_pred_prob, compatibility_matrix)[y_true!=0]
            compatibility_matrix = calculate_compatibility_matrix(X, y, "energy")[1:, 1:]
            print('compatibility_matrix.shape', compatibility_matrix.shape)
            print('y_pred_prob.shape', y_pred_prob.shape)
            SU_E = semantic_based_uncertainty(y_pred_prob, compatibility_matrix)[y_true!=0]
            
            datax = np.concatenate((Var, FR, H, GU, SU_E))
            #uncertainties = [r"Var"] * GU.shape[0] + [r"$H$"] * GU.shape[0]+ [r"$GU$"] * GU.shape[0] + [r"$SU^\mathcal{H}$"] * GU.shape[0] + [r"$SU^\mathcal{S}$"] * GU.shape[0] #+ ["SU_W"] * GU.shape[0] 
            uncertainties = [r"Var"] * GU.shape[0] + [r"$GU_{2|FR}$"] * GU.shape[0] + [r"$GU_{1|KL}$"] * GU.shape[0]+ [r"$GU_{2|E}$"] * GU.shape[0] + [r"$SU_\mathcal{H}$"] * GU.shape[0]
            labels = np.zeros(GU.shape[0])
            #labels[id_right] = 0
            labels[id_wrong] = 1
            labels = np.tile(labels, 5)
            print("labels", labels.shape)
            classes = np.tile(y_true[y_true!=0], 5)
            print("classes", classes.shape)
            
            data = pd.DataFrame(data = {"Uncertainties": datax, "Measures": uncertainties, "Classification": labels, "classes": classes})
            
            
            idx = np.nonzero(np.asarray(data["Classification"]==1))[0]
            data["Classification"][idx] = "Wrong"
            idx = np.nonzero(np.asarray(data["Classification"]==0))[0]
            data["Classification"][idx] = "Right"

            #np.save(f"{outputs_dir}/images/{project_name}_{model_name}_violin.npy", data)
            
            for i in range(1,len(C)):
                #sns.plotting_context(font_scale=1.25)
                sns.set(font_scale = 1.5, rc={'axes.facecolor':'white'}) #, 'figure.facecolor':'white'
                sns.set_theme(style="white")

                pp = sns.violinplot(x="Measures", y="Uncertainties", hue="Classification", data=data[data.classes==i], hue_order= ["Right", "Wrong"], cut = 0, scale = 'area', palette = sns.color_palette(colors), split=True)
                fig = pp.figure
                fig.subplots_adjust(top=0.93, wspace=0.3)
                #t = fig.suptitle('ICESAR-L Pairwise Plots, first band', fontsize=14)
                fig.savefig(f"{images_dir}{model_name}_violinPlot_{i}.eps",
                    bbox_inches="tight",
                    pad_inches=0,
                    dpi=500,)
                plt.close()
                #ax = sns.violinplot(x="day", y="total_bill", hue="smoker",
                #        data=tips, palette="muted")

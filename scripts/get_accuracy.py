import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, cohen_kappa_score

for project_name in os.listdir("outputs/"):
    if project_name == "trento":
        continue
        # from trento_config import (
        #     dataset,
        #     project_name,
        #     images_dir,
        #     outputs_dir,
        #     compatibility_matrix,
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
    y = y.reshape(-1)
    y_true = y[y!=0]

    acc_dict = []
    for file in os.listdir(os.path.join(outputs_dir)):
        if 'test' in file:
            continue
        if 'RF' not in file:
            continue
        if os.path.splitext(file)[-1].lower() == ".npy":
            print(file.split("_")[-2].split("."))
            model_name = file.split("_")[-1].split(".")[0]
            print(file)
            y_pred_prob = np.load(os.path.join(outputs_dir, file))
            y_pred_max_prob = y_pred_prob.max(1)
            y_pred = y_pred_prob.argmax(1)[y!=0] + 1
            #print(y_true)
            #print(y_pred)
            print(project_name, model_name)
            AS = accuracy_score(y_true, y_pred)*100
            PS = precision_score(y_true, y_pred, average="micro") 
            CM = confusion_matrix(y_true, y_pred, normalize = "true")*100
            K = cohen_kappa_score(y_true, y_pred)*100
            np.set_printoptions(precision=2)
            print("{:.2f}".format(AS), "{:.2f}".format(PS), "{:.2f}".format(K))
            print(CM)
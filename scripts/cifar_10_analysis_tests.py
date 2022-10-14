import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from cifar10_config import (
    dataset,
    project_name,
    images_dir,
    outputs_dir,
    compatibility_matrix,
    color,
    labels
)  
model_name = "SVM"
# model_name = "vgg16_SVM"

from uncertainty.utils import print_latex_pm_table
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

y_gbu = np.load(f"{outputs_dir}uncertainty_npys/{project_name}_{model_name}_GBU.npy")[-10000:]
y_variance = np.load(f"{outputs_dir}uncertainty_npys/{project_name}_{model_name}_VARIANCE.npy")[-10000:]
y_entropy = np.load(f"{outputs_dir}uncertainty_npys/{project_name}_{model_name}_ENTROPY.npy")[-10000:]
y_sbu = np.load(f"{outputs_dir}uncertainty_npys/{project_name}_{model_name}_SBU_ENERGY_saloua.npy")[-10000:]
y_pred = np.load(f"{outputs_dir}uncertainty_npys/{project_name}_{model_name}_PREDICTIONS.npy")[-10000:]
y_true = np.load(f"{outputs_dir}uncertainty_npys/{project_name}_{model_name}_GT.npy")
# y_pred_prob = np.load(f"{outputs_dir}{project_name}_{model_name}.npy")

# print(y_true.shape)
# print(y_pred.shape)
# print(y_gbu.shape)
# print(y_variance.shape)
# print(y_sbu.shape)
# print(y_entropy.shape)
print(confusion_matrix(y_true, y_pred))
print(accuracy_score(y_true, y_pred) )

# plt.figure(dpi=500)
# plt.imshow(
#     y_sbu.reshape(dataset.shape),
#     cmap="coolwarm",
#     vmin=0, 
#     vmax=1
# )
# plt.axis("off")
# plt.savefig(
#     f"{images_dir}{model_name}_SBU_energy.png",
#     bbox_inches="tight",
#     pad_inches=0,
#     dpi=500,
# )

### MEAN AND STD TABLES

mean = np.zeros((10,10))
standard_deviation = np.zeros((10,10))
metrics = ["Entropy", "Variance (unit distance)", "Variance (energy distance)"]

for i,y,metric in zip(range(3),[y_entropy, y_gbu, y_sbu],metrics):
    for gt_label in range(10): # 10 labels
        gt_indeces = np.where(y_true == gt_label)
        y_pred_gt_label = y_pred[gt_indeces]
        y_gt_label = y[gt_indeces]

        print(y_gt_label.shape)
        for predicted_label in range(10):
            pred_indeces = np.where(y_pred_gt_label == predicted_label)
            y_gt_label_as_predicted_label = y_gt_label[pred_indeces]
            mean[gt_label,predicted_label] = np.mean(y_gt_label_as_predicted_label)
            standard_deviation[gt_label,predicted_label] = np.std(y_gt_label_as_predicted_label)

    print(f"{metric} - Mean and StD table:")
    print_latex_pm_table(mean, standard_deviation)


# mean_gbu = np.zeros((10,10))
# var_gbu = np.zeros((10,10))

# mean_bivariance = np.zeros((10,10))
# var_bivariance = np.zeros((10,10))      

# mean_entropy = np.zeros((10,10))
# var_entropy = np.zeros((10,10))      

# mean_sbu = np.zeros((10,10))
# var_sbu = np.zeros((10,10))    

# print("minmax", y_sbu.min(), y_sbu.max())
# for i,gt_label in zip(range (0,10000,1000),range(10)):
#     y_pred_class = y_pred[i:i+1000] # predictions of gt_label
#     y_gpu_class= y_gbu[i:i+1000]
#     y_variance_class= y_variance[i:i+1000]
#     y_entropy_class= y_entropy[i:i+1000]
#     y_sbu_class= y_sbu[i:i+1000]

#     for predicted_label in range(10):
#         # print("minmax", y_sbu_class.min(), y_sbu_class.max())

#         indices = np.where(y_pred_class == predicted_label) # the images with ground truth label predicted as predicted_label

#         mean_gbu[gt_label, predicted_label] = np.mean(y_gpu_class[indices])
#         var_gbu[gt_label, predicted_label] = np.std(y_gpu_class[indices])

#         mean_bivariance[gt_label, predicted_label] = np.mean(y_variance_class[indices])
#         var_bivariance[gt_label, predicted_label] = np.std(y_variance_class[indices])


#         mean_entropy[gt_label, predicted_label] = np.mean(y_entropy_class[indices])
#         var_entropy[gt_label, predicted_label] = np.std(y_entropy_class[indices])

#         mean_sbu[gt_label, predicted_label] = np.mean(y_sbu_class[indices])
#         var_sbu[gt_label, predicted_label] = np.std(y_sbu_class[indices])



# np.save(f"{outputs_dir}{project_name}_GBU_mean.npy", mean_gbu)
# np.save(f"{outputs_dir}{project_name}_GBU_var.npy", var_gbu)

# np.save(f"{outputs_dir}{project_name}_VARIANCE_mean.npy", mean_bivariance)
# np.save(f"{outputs_dir}{project_name}_VARIANCE_var.npy", var_bivariance)

# np.save(f"{outputs_dir}{project_name}_ENTROPY_mean.npy", mean_entropy)
# np.save(f"{outputs_dir}{project_name}_ENTROPY_var.npy", var_entropy)

# np.save(f"{outputs_dir}{project_name}_SBU_mean.npy", mean_sbu)
# np.save(f"{outputs_dir}{project_name}_SBU_var.npy", var_sbu)

# print("GBU")
# print_latex_pm_table(mean_gbu, var_gbu)
# print("VARIANCE")
# print_latex_pm_table(mean_bivariance, var_bivariance)
# print("ENTROPY")
# print_latex_pm_table(mean_entropy, var_entropy)
# print("SBU")
# print_latex_pm_table(mean_sbu, var_sbu)

# print(" \\\\\n".join([" & ".join(map(str,line)) for line in np.around(np.load(f"{outputs_dir}uncertainty_npys/{project_name}_energy_distance.npy"), decimals=2)]) )


# BARS OF PREDICTIONS WITH HIGH OR LOW UNCERTAINTY FOR EACH CLASS

for y,metric in zip([y_gbu, y_sbu, y_entropy, y_variance],["gbu", "sbu", "entropy", "variance"]):
    print("Metric", metric)
    limit = 1000

    print("Max 10% ")
    max_indices = np.argpartition(y, -limit)[-limit:]
    y_pred_max = y_pred[max_indices]
    y_true_max = y_true[max_indices]
    print(accuracy_score(y_true_max, y_pred_max) )
    cm = confusion_matrix(y_true_max, y_pred_max)
    # print(cm)
    for label in range(10):
        ax = plt.subplot(10,1,label+1)
        plt.bar(np.arange(10), height=cm[label,:], tick_label =labels, color = color[label], edgecolor = "black")
        plt.xticks(np.arange(10), labels)
        plt.text(-1, 50, f"Ground Truth = {label}", size = 'small')
        plt.xlim(-1, 10)
        plt.ylim(0, 80)
        for spine,i in zip(plt.gca().spines.values(),range(4)):
            if i == 2: spine.set_visible(True)
            else: spine.set_visible(False)
        plt.yticks(color='w')
        plt.tick_params(axis = 'y', left = False, right = False, top=False, labelbottom=False )        

    plt.savefig(f"{images_dir}{project_name}_{model_name}_{metric}_high_bars.png")
    plt.show()

    print("Min 10% ")
    min_indices = np.argpartition(y, limit)[:limit]
    y_pred_min = y_pred[min_indices]
    y_true_min = y_true[min_indices]
    print(accuracy_score(y_true_min, y_pred_min) )
    cm = confusion_matrix(y_true_min, y_pred_min)
    for label in range(10):
        ax = plt.subplot(10,1,label+1)
        plt.bar(np.arange(10), height=cm[label,:], tick_label =labels, color = color[label], edgecolor = "black")
        plt.xticks(np.arange(10), labels)
        plt.text(-1, 50, f"Ground Truth = {label}", size = 'small')
        plt.xlim(-1, 10)
        plt.ylim(0, 80)
        for spine,i in zip(plt.gca().spines.values(),range(4)):
            if i == 2: spine.set_visible(True)
            else: spine.set_visible(False)
        plt.yticks(color='w')
        plt.tick_params(axis = 'y', left = False, right = False, top=False, labelbottom=False )        

    plt.savefig(f"{images_dir}{project_name}_{model_name}_{metric}_low_bars.png")
    plt.show()


# TOTAL BARS OF PREDICTIONS WITH HIGH OR LOW UNCERTAINTY



plot_color = ['lightblue', 'orange','lightgreen', ]
edge_color = ['blue', 'darkorange','green', ]
metrics = ["Entropy", "Variance (unit distance)", "Variance (energy distance)"]
legend_elements = [Patch(facecolor=plot_color[i], edgecolor=edge_color[i],
                         label=metrics[i]) for i in range(3)]


for i,y,metric in zip(range(3),[y_entropy, y_gbu, y_sbu],metrics):
    limit = 1000
    max_indices = np.argpartition(y, -limit)[-limit:]
    y_pred_max = y_pred[max_indices]
    y_true_max = y_true[max_indices]

    print(f"Max  10%  - {metric}")
    cm = confusion_matrix(y_true_max, y_pred_max)
    # print(cm)
    plt.subplot(4,1,i+1)
    if i == 0:
        plt.legend(handles = legend_elements, loc='upper center', ncol=3, frameon=False, fontsize = 6 )


    plt.bar(np.arange(10), height=np.sum(cm,axis = 0), tick_label =labels, color = plot_color[i], alpha = 0.6, edgecolor = edge_color[i])
    plt.xlim(-1, 10)
    plt.ylim(0, 300)
    for spine,i in zip(plt.gca().spines.values(),range(4)):
        if i == 2: spine.set_visible(True)
        else: spine.set_visible(False)
    plt.yticks(color='w')
    plt.xticks(color='w')

    plt.tick_params(axis = 'y', left = False, right = False, top=False, labelbottom=False )    

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.005)
plt.xticks(color='black')  
plt.xticks(np.arange(10), labels, size = 'xx-small')
plt.savefig(f"{images_dir}{project_name}_{model_name}_sum_high_bars.png",
                bbox_inches="tight",
                pad_inches=0,
                dpi=500,)

plt.show()


for i,y,metric in zip(range(3),[y_entropy, y_gbu, y_sbu],metrics):
    # print("Metric", metric)
    limit = 1000
    max_indices = np.argpartition(y, limit)[:limit]
    y_pred_max = y_pred[max_indices]
    y_true_max = y_true[max_indices]

    print(f"Min 10%  - {metric}")
    print(accuracy_score(y_true_max, y_pred_max) )

    cm = confusion_matrix(y_true_max, y_pred_max)
    # print(cm)
    plt.subplot(4,1,i+1)
    if i == 0:
        plt.legend(handles = legend_elements, loc='upper center', ncol=3, frameon=False, fontsize = 6 )


    plt.bar(np.arange(10), height=np.sum(cm,axis = 0), tick_label =labels, color = plot_color[i], alpha = 0.6, edgecolor = edge_color[i])
    plt.xlim(-1, 10)
    plt.ylim(0, 300)
    for spine,i in zip(plt.gca().spines.values(),range(4)):
        if i == 2: spine.set_visible(True)
        else: spine.set_visible(False)
    plt.yticks(color='w')
    plt.xticks(color='w')

    plt.tick_params(axis = 'y', left = False, right = False, top=False, labelbottom=False )    

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.005)
plt.xticks(color='black')  
plt.xticks(np.arange(10), labels, size = 'xx-small')
plt.savefig(f"{images_dir}{project_name}_{model_name}_sum_low_bars.png",
                bbox_inches="tight",
                pad_inches=0,
                dpi=500,)

plt.show()



# for i,y,metric in zip(range(3),[y_entropy, y_gbu, y_sbu],metrics):
#     # print("Metric", metric)
#     limit = 1000
#     max_indices = np.argpartition(y, -limit)[-limit:]
#     y_pred_max = y_pred[max_indices]
#     y_true_max = y_true[max_indices]
#     # print(np.unique(y_pred_max))
#     # print(np.unique(y_true_max))
    

#     print(f"Max - {metric}")
#     print(accuracy_score(y_true_max, y_pred_max) )
#     # print(precision_score(y_true_max, y_pred_max, average = 'macro'))
#     # print(recall_score(y_true_max, y_pred_max, average = 'macro') )
#     cm = confusion_matrix(y_true_max, y_pred_max)
#     # print(cm)
#     plt.subplot(4,1,i+1)
#     if i == 0:
#         plt.legend(handles = legend_elements, loc='upper center', ncol=3, frameon=False, fontsize = 6 )


#     plt.bar(np.arange(10), height=np.sum(cm,axis = 0), tick_label =labels, color = plot_color[i], alpha = 0.6, edgecolor = edge_color[i])
#     plt.xlim(-1, 10)
#     plt.ylim(0, 300)
#     for spine,i in zip(plt.gca().spines.values(),range(4)):
#         if i == 2: spine.set_visible(True)
#         else: spine.set_visible(False)
#     plt.yticks(color='w')
#     plt.xticks(color='w')

#     plt.tick_params(axis = 'y', left = False, right = False, top=False, labelbottom=False )    

# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.005)
# plt.xticks(color='black')  
# plt.xticks(np.arange(10), labels, size = 'xx-small')
# plt.savefig(f"{images_dir}{project_name}_{model_name}_sum_high_bars.png",
#                 bbox_inches="tight",
#                 pad_inches=0,
#                 dpi=500,)

# plt.show()


# for i,y,metric in zip(range(3),[y_sbu, y_entropy, y_variance],metrics):
#     print("Metric", metric)
#     limit = 1000
#     min_indices = np.argpartition(y, limit)[:limit]
#     y_pred_min = y_pred[min_indices]
#     y_true_min = y_true[min_indices]

#     print("Min")
#     print(accuracy_score(y_true_min, y_pred_min) )
#     print(precision_score(y_true_min, y_pred_min, average = 'macro') )
#     print(recall_score(y_true_min, y_pred_min, average = 'macro') )
#     cm = confusion_matrix(y_true_min, y_pred_min)
#     print(cm)
#     plt.subplot(4,1,i+1)
#     if i == 0:
#         plt.legend(handles = legend_elements, loc='upper center', ncol=3, frameon=False, fontsize = 6 )


#     plt.bar(np.arange(10), height=np.sum(cm,axis = 0), tick_label =labels, color = plot_color[i], alpha = 0.7, edgecolor = edge_color[i])
#     plt.xlim(-1, 10)
#     plt.ylim(0, 500)
#     for spine,i in zip(plt.gca().spines.values(),range(4)):
#         if i == 2: spine.set_visible(True)
#         else: spine.set_visible(False)
#     plt.yticks(color='w')
#     plt.xticks(color='w')

#     plt.tick_params(axis = 'y', left = False, right = False, top=False, labelbottom=False )    

# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.005)
# plt.xticks(color='black')  
# plt.xticks(np.arange(10), labels, size = 'xx-small')
# plt.savefig(f"low_bars",
#                 bbox_inches="tight",
#                 pad_inches=0,
#                 dpi=500,)



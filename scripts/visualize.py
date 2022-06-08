

plt.figure(dpi=500)
plt.matshow(m_confusion_matrix, cmap="coolwarm")
plt.xlabel("True Labels")
plt.xticks(np.arange(0,N_LABELS,1), range(1,len(labels)))
plt.ylabel("Predicted Labels")
plt.yticks(np.arange(0,N_LABELS,1), range(1,len(labels)))
for k in range (len(m_confusion_matrix)):
    for l in range(len(m_confusion_matrix[k])):
        plt.text(k,l,str(m_confusion_matrix[k][l]), va='center', ha='center', fontsize='small') #trento
plt.savefig(f"{images_dir}{dataset}_SVM_test_CONFUSION_MATRIX.png",bbox_inches='tight', pad_inches=0.2, dpi=500)
       
plt.figure(dpi=500)
plt.imshow(y_pred.reshape(SHAPE), interpolation='nearest', cmap = colors.ListedColormap(color[1:]))
plt.axis('off')
plt.savefig(f"{images_dir}{dataset}_SVM_PREDICTIONS.png",bbox_inches='tight', pad_inches=0, dpi=500)

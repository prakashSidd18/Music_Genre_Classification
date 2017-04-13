import itertools
import matplotlib.pyplot as plt
import numpy as np
labels = {
    0: 'blues'     ,
    1: 'classical' ,
    2: 'country'   ,
    3: 'disco'     ,
    4: 'hiphop'    ,
    5: 'jazz'      ,
    6: 'metal'     ,
    7: 'pop'       ,
    8: 'reggae'    ,
    9: 'rock'      ,
}

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #classes = [labels[i] for i in classVals]
    cvals = []
    for i in range(0,len(classes)):
        cvals.append(labels.get(classes[i]))
    classes = cvals
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

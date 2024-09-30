import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import svm
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.ensemble import BalancedRandomForestClassifier
import os
import pickle

def training_model(method, max_distance):

    path2data = f'../data/svc/distance_cases/{str(max_distance).zfill(3)}/data_{max_distance}m.npz'
    data = np.load(path2data, allow_pickle=True)
    X_train = data['X_train']
    y_train = data['y_train']

    # Define the model and the hyperparameters to be optimized
    if method == 'SVC':

        model = svm.SVC(class_weight='balanced', random_state=42)

        param_grid = [
    {'C': [0.1, 1, 10], 'kernel': ['linear']},  # Grid for linear kernel
    {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['rbf']}  # Grid for other kernels
]

    elif method == 'BRF':

        model = BalancedRandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            sampling_strategy='all',
            replacement=True,
            bootstrap=False
        )

        param_grid = {
            'n_estimators': [100, 500, 1000],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

    # Define the cross-validation strategy
    stratified_kfold = StratifiedKFold(n_splits=4)
    grid_search = GridSearchCV(model, param_grid, cv=stratified_kfold, scoring='balanced_accuracy', n_jobs=-1, verbose=2)

    # Fit the model
    grid_search.fit(X_train, y_train)

    return grid_search

def save_model_and_data(model, cm, balanced_accuracy, max_dist, species, path2save, method):
    """
    Save the model and confusion matrix data to disk.

    Parameters:
    - model: Trained model.
    - cm: Normalized confusion matrix.
    - balanced_accuracy: Balanced accuracy score.
    - max_dist: Maximum distance used for clustering.
    - species: Array of species labels.
    """
    model_filename = os.path.join(path2save, f'model_{method}_{max_dist}m.pkl')
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)

    data_dict = {
        'confusion_matrix': cm,
        'balanced_accuracy': balanced_accuracy,
        'distance': max_dist,
        'species': species,
    }

    data_filename = os.path.join(path2save, f'cm_{method}_{max_dist}m.pkl')
    with open(data_filename, 'wb') as f:
        pickle.dump(data_dict, f)

def evaluate_model(grid_search, method, max_distance):

    path2data = f'../data/svc/distance_cases/{str(max_distance).zfill(3)}/'
    data = np.load(path2data + f'data_{max_distance}m.npz', allow_pickle=True)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    best_model = grid_search.best_estimator_

    # save the parameters of the best model in a txt file
    with open(f'{path2data}best_{method}.txt', 'w') as f:
        f.write(str(grid_search.best_params_))

    y_pred = best_model.predict(X_test)
    balanced_accuracy_test = balanced_accuracy_score(y_test, y_pred)

    y_pred_train = best_model.predict(X_train)
    balanced_accuracy_train = balanced_accuracy_score(y_train, y_pred_train)

    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    cm = cm / cm.sum(axis=1)[:, np.newaxis]

    # Save the model and confusion matrix data
    save_model_and_data(best_model, cm, balanced_accuracy_test, max_distance, np.unique(y_test), path2data, method)

    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap='Blues', vmin=0, vmax=1)

    # Add colorbar with fixed range from 0 to 1
    cbar = fig.colorbar(cax)
    cbar.set_label('Proportions')

    # Annotate the confusion matrix
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white' if val > 0.5 else 'black')

    # Set labels
    ax.set_xticks(np.arange(len(np.unique(y_test))))
    ax.set_yticks(np.arange(len(np.unique(y_test))))
    ax.xaxis.set_ticks_position('bottom')  # Move x-axis ticks to the bottom
    ax.xaxis.set_label_position('bottom')  # Move x-axis label to the bottom
    ax.set_xticklabels(np.unique(y_test))
    ax.set_yticklabels(np.unique(y_test))
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    # Rotate the tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    plt.title('$BA_{test}$ = %.2f, $BA_{train}$ = %.2f' % (balanced_accuracy_test, balanced_accuracy_train))
    plt.savefig(path2data + 'confusion_matrix_' + method + '.png')
    plt.close()
    #plt.show()

# PARAMETERS
METHOD = 'BRF'
MAX_DISTANCE = 50

# RUN THE FUNCTIONS
grid_search = training_model(METHOD, MAX_DISTANCE)
evaluate_model(grid_search, METHOD, MAX_DISTANCE)
print('Done!')
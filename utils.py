import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(X, y, model, steps=1000, cmap='Paired'):
    """
    Plots the decision boundary for a model and overlays the scatter points of the dataset.
    """
    cmap = plt.get_cmap(cmap)

    # Define region of interest by data limits
    xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
    ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # Make predictions across region of interest
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        outputs = model(inputs)
        # Assuming binary classification with one output node or two output nodes
        if outputs.shape[1] > 1:
            _, labels = torch.max(outputs, 1)
        else:
            labels = (outputs > 0.5).int()
        labels = labels.cpu().numpy()

    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)

    # Get predicted labels on training data and plot
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, lw=0)

    return fig, ax

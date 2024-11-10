def eigenface_project(images, xPixel=48, yPixel=48, cum=0.9):
    """
    This function project a data set onto its eigen-vectors
    Args:
        images: the input data set
        cum: the minimum cumulative variance to consider
        xPixel, yPixel: the number of pixels on both axises
    Returns:
        images_project: the projected data set
    """
    import numpy as np
    import pandas as pd
    # eigenfaces and eigen vectors
    # 1. Reshape the data 
    # images = df.drop(columns=df.columns[-1]).sample(n=3000, axis='rows').values # remove the class column and sample 3000 images.
    image_shape = (xPixel, yPixel)
    images = images.reshape(-1, *image_shape)

    # 2. Calculate mean face and subtract the mean
    mean_face = np.mean(images, axis=0)
    centered_images = images - mean_face

    # 3. Calculate covariance matrix of for faces
    covariance_matrix = np.cov(centered_images.reshape(images.shape[0], -1).T)

    # 4. Eigenvalue decomposition and select the top eigenfaces (e.g., top 10)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # 5. Calculate the cumulative variance
    total_variance = np.sum(eigenvalues)  # Total variance of the data
    explained_variance = [(i / total_variance) for i in sorted(eigenvalues, reverse=True)]
    cumulative_variance = np.cumsum(explained_variance)

    # 6. select the eigenvectors whose cumulative variance exceed 0.9
    k = np.argmax(cumulative_variance > 0.9)
    selected_eigenvectors = eigenvectors[:, :k]

    images_project = np.dot(centered_images.reshape(images.shape[0], -1), selected_eigenvectors)
    return images_project, selected_eigenvectors
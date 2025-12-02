from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import numpy as np


# covariance_type: 'full' | 'tied' | 'diag' | 'spherical'
def fit_DPGMM(embeddings: list[list[int]], max_components=100, covariance_type ='full'):
    embeddings = np.asarray(embeddings)

    model = BayesianGaussianMixture(
        n_components=max_components,
        covariance_type=covariance_type,
        weight_concentration_prior_type='dirichlet_process',
        max_iter=500,
        init_params='kmeans'
    )

    model.fit(embeddings)    
    labels = model.predict(embeddings)
    
    return model, labels


def fit_GMM(embeddings: list[list[int]], n_components: int, covariance_type='full') :

    embeddings = np.asarray(embeddings)

    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    gmm.fit(embeddings)

    labels = gmm.predict(embeddings)
    return gmm, labels
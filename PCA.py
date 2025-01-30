import matplotlib.pyplot as plt
import pickle
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import itertools


def preprocess_PCA(Db: pd.DataFrame):
    """
    This function applies the standard scaling of the input Dataset.
    """
    scaler = StandardScaler()
    Db = pd.DataFrame(scaler.fit_transform(Db).T, index=Db.columns)
    return Db


def run_PCA(Db: pd.DataFrame):
    """
    This function preprocesses and runs the PCA function on the Dataset.

    Input:
        - Db:   The raw input Dataset

    Outputs:
        - pca:                      The fitted PCA object.
        - principal_components:     The PCA-transformed data (output of pca.fit_transform).
        - nb_relevant_features:     Number of relevant components to plot.
    """

    Db = preprocess_PCA(Db)

    pca = PCA(n_components=len(Db.columns))
    principal_components = pca.fit_transform(Db)

    nb_relevant_features = 1
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    while cumsum[nb_relevant_features - 1] < 0.8:
        nb_relevant_features += 1
    return pca, principal_components, nb_relevant_features


def plot_explained_variance_pca(pca: PCA, nb_relevant_features: int):
    """
    Plots the explained variance.

    Inputs:
        - pca:                      The fitted PCA object.
        - nb_relevant_features:     Number of relevant components to plot.
    """
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = len(pca.explained_variance_ratio_)

    # Plot all components with color distinction
    plt.figure(figsize=(10, 6))

    # Plot relevant components in one color
    plt.plot(
        range(1, nb_relevant_features + 1),
        explained_variance[:nb_relevant_features],
        marker="o",
        linestyle="-",
        color="b",
        label="Relevant Components",
    )

    # Plot non-relevant components in a different color
    plt.plot(
        range(nb_relevant_features + 1, num_components + 1),
        explained_variance[nb_relevant_features:],
        marker="o",
        linestyle="-",
        color="gray",
        label="Other Components",
    )

    # Add labels and title
    plt.axhline(
        0.8, color="r", linestyle="--", label="80% Explained Variance Threshold"
    )
    plt.title("Cumulative Explained Variance by Principal Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.show()


def plot_pca_distribution(
    principal_components,
    components: list,
    categories=None,
):
    """
    Plots the 2D or 3D distribution of data points in the PCA-reduced space.

    Inputs:
        - principal_components:     The transformed data from PCA.
        - components:               Index of the components to plot.
        - categories:               List of categories name from reference File. If None, the categories will not be distinguished on the PCA graphs
    """

    if len(components) == 2:
        if categories is not None:
            # If weld is provided, create a scatter plot with different colors for each weld type
            categories_types = set(categories)
            for type in categories_types:
                mask = categories == type
                plt.scatter(
                    principal_components[mask][:, components[0]],
                    principal_components[mask][:, components[1]],
                    alpha=0.7,
                    label=type,
                )
                plt.xlabel(f"PC{components[0]}")
                plt.ylabel(f"PC{components[1]}")
                plt.title(f"PCA Distribution (PC{components[0]} vs PC{components[1]})")
                plt.legend()
        else:
            # 2D Scatter plot for 2 principal components
            plt.scatter(
                principal_components[:, components[0]],
                principal_components[:, components[1]],
                alpha=0.7,
            )
            plt.xlabel(f"PC{components[0]}")
            plt.ylabel(f"PC{components[1]}")
            plt.title(f"PCA Distribution (PC{components[0]} vs PC{components[1]})")
    else:
        if categories is not None:
            # If weld is provided, create a scatter plot with different colors for each weld type
            categories_types = set(categories)
            ax = plt.axes(projection="3d")

            for type in categories_types:
                mask = categories == type
                ax.scatter3D(
                    principal_components[mask][:, components[0]],
                    principal_components[mask][:, components[1]],
                    principal_components[mask][:, components[2]],
                    alpha=0.7,
                    label=type,
                )
                ax.set_xlabel(f"PC{components[0]}")
                ax.set_ylabel(f"PC{components[1]}")
                ax.set_zlabel(f"PC{components[2]}")
                ax.set_title(
                    f"PCA Distribution (PC{components[0]}, PC{components[1]}, PC{components[2]})"
                )
                ax.legend()
        else:
            # 3D Scatter plot for 3 principal
            ax = plt.axes(projection="3d")
            ax.scatter3D(
                principal_components[:, components[0]],
                principal_components[:, components[1]],
                principal_components[:, components[2]],
                alpha=0.7,
            )
            ax.set_xlabel(f"PC{components[0]}")
            ax.set_ylabel(f"PC{components[1]}")
            ax.set_zlabel(f"PC{components[2]}")
            ax.set_title(
                f"PCA Distribution (PC{components[0]}, PC{components[1]}, PC{components[2]})"
            )
    plt.show()


def generate_tuples(pca: PCA, nb_relevant_features: int, graphs3d: bool):
    """
    This function generates the 2 or 3-tuples corresponding to the indexes of the components to plot.
    We limited the indexes to a max value of 10 for combinatory reasons.

    Inputs:
        - pca:                      The fitted PCA object.
        - nb_relevant_features:     Number of relevant components to visualize.
        - graphs3d:                 Set to True if you want to plot 3d PCA graphs
    """

    if graphs3d:
        tuples = list(itertools.combinations(range(min(nb_relevant_features, 10)), 3))
        tuples.sort(
            key=lambda x: pca.explained_variance_ratio_[x[0]]
            + pca.explained_variance_ratio_[x[1]]
            + pca.explained_variance_ratio_[x[2]],
            reverse=True,
        )

    else:
        tuples = list(itertools.combinations(range(min(nb_relevant_features, 10)), 2))
        tuples.sort(
            key=lambda x: pca.explained_variance_ratio_[x[0]]
            + pca.explained_variance_ratio_[x[1]],
            reverse=True,
        )

    return tuples


def plot_PCA(
    pca: PCA,
    principal_components,
    nb_relevant_features: int,
    categories=None,
    graphs3d: bool = False,
):
    """
    For each 2 or 3-tuple of principal components under the top nb_relevant_features,
    this function plots the PCA distribution.

    Inputs:
        - pca:                      The fitted PCA object.
        - principal_components:     The PCA-transformed data (output of pca.fit_transform).
        - nb_relevant_features:     Number of relevant components to visualize.
        - categories:               List of categories name from reference File. If None, the categories will not be distinguished on the PCA graphs
        - graphs3d:                 Set to True if you want to plot 3d PCA graphs
    """
    # # Generate all pairs of components within the relevant features
    tuples = generate_tuples(pca, nb_relevant_features, graphs3d)

    # Plot each pair
    for t in tuples:
        # Plot the PCA distribution for the current pair of components (i, j)
        plot_pca_distribution(
            principal_components,
            components=t,
            categories=categories,
        )


def PCA_on_embeddings(
    embeddings_path: str, product_path: str = None, graphs3d: bool = False
) -> None:
    """
    This function computes a PCA on the embeddings and displays the explained variance graph and the principal components 2D or 3D graphs

    Inputs:
        - embeddings_path:      The path to the embeddings csv file
        - product_path:         The path to the reference csv file containing categories. If None, categories will not be distinguished on the PCA graphs
        - graphs3d:             Set to True if you want to plot 3d PCA graphs
    """
    try:
        Db = pd.read_csv(embeddings_path)
    except:
        try:
            with open(embeddings_path, "rb") as f:
                Db = pd.DataFrame(pickle.load(f))
        except:
            raise ValueError("The given file is neither a .csv nor a .pkl file")

    Db = Db.reindex(sorted(Db.columns), axis=1)

    if product_path is not None:
        categories = pd.read_csv(product_path)

        mask = categories["MMC"] == ""

        for col in Db.columns:
            mask = mask | (categories["MMC"] == col.split(".")[0])

        categories = categories[mask]
        categories_list = categories["Product_BusinessUnitDesc"]
    else:
        categories_list = None

    pca, principal_components, nb_relevant_features = run_PCA(Db)
    plot_explained_variance_pca(pca, nb_relevant_features)
    plot_PCA(
        pca,
        principal_components,
        nb_relevant_features,
        categories_list,
        graphs3d,
    )


if __name__ == "__main__":
    PCA_on_embeddings("DAM_embeddings.csv", "data\product_list.csv", graphs3d=False)

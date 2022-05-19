"""
This is where I keep some of the most helpful functions I use all the time
"""


"""
---------------------------------------------------------------------------
This function returns a list of n_features based on an input set of features and a label.
"""
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import itertools

def rfeSelctor(features, label, n_features):
    # create the RFE model and select 10 attributes
    rfc = RandomForestClassifier()
    rfe = RFE(rfc, n_features_to_select=n_features)
    rfe = rfe.fit(features, label)

    # summarize the selection of the attributes
    feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), features.columns)]
    selected_features = [v for i, v in feature_map if i==True]

    return selected_features
"""
---------------------------------------------------------------------------
"""

"""
----------------------------------------------------------------------------
"""
import matplotlib.pyplot as plt
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = f"images\{fig_id}{fig_extension}"
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
"""
----------------------------------------------------------------------------
"""
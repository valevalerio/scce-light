# scce-light

[![GitHub tests](https://github.com/valevalerio/scce-light/actions/workflows/tests.yml/badge.svg)](https://github.com/valevalerio/scce-light/actions/workflows/tests.yml) [![GitLab tests](https://gitlab.com/tango-ecosystem/tango-library/utils/scce-light/badges/master/pipeline.svg?key_text=GitLab%20tests&key_width=80)](https://gitlab.com/tango-ecosystem/tango-library/utils/scce-light/-/pipelines) ![Code size](https://img.shields.io/github/languages/code-size/valevalerio/scce-light) <!--[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)--> [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![IEEE](https://img.shields.io/badge/IEEE-11570111-00629B.svg)](https://ieeexplore.ieee.org/document/11570111)


Lightweight version of [SC-CE](https://github.com/clarapunzi/SC-CE), Selective Classification via Counterfactual Explanations:
a model agnostic rejection-based abstention method for classifiers.

**Why is it lightweight:** It has no submodules and no pre-computed artifacts. The only dependencies are `numpy`, `scipy`, `pandas` and `scikit-learn`. 

<!-- ![NumPy](https://img.shields.io/badge/numpy-013243.svg?style=flat&logo=numpy&logoColor=white&labelColor=555) ![scipi](https://img.shields.io/badge/scipy-8CAAE6.svg?style=flat&logo=scipy&logoColor=white&labelColor=555) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white&labelColor=555) [![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white&labelColor=555)](https://scikit-learn.org/)
-->
## Intuition
The main idea is that a classifier abstains when a small change to the input, found by a counterfactual generator, is enough to flip its prediction. 
- Data focused
- Model agnostic
- Easy to use

<p align="center">
  <img src="Drawings/figures/decision_boundary.gif" alt="Decision boundary of an MLP evolving during training on a two-moons dataset" width="45%">
  &nbsp;
  <img src="Drawings/figures/Growing_Circle.gif" alt="A circle grows around a sample until it reaches the closest point predicted with the opposite class" width="45%">
</p>

A classifier learns a decision boundary (left). For a given instance, a sphere grows around it until it reaches a point predicted with the other class: its closest counterfactual (right). If that distance is small, a small change can flip the prediction, so the classifier should abstain. scce-light implements this idea with the Growing Spheres counterfactual generator. The confidence of the model is not used at all to decide whether to abstain.

## Pipeline

| Module | Content |
|---|---|
| `scce_light/growing_spheres.py` | `GrowingSpheres`: self-contained port of the [SC-CE fork](https://github.com/valevalerio/growingspheres) (multiple counterfactuals per instance, sparsification), seeded and vectorised |
| `scce_light/distances.py` | `euclidean`, `aggregate` (`min` / `mean` / `max`) |
| `scce_light/pipeline.py` | `generate_counterfactuals`, `cf_distances` |
| `scce_light/rejectors.py` | `CFDistRejector` (SC-CE, empirical or Gamma quantiles), `PlugInRule` (confidence baseline) |
| `scce_light/metrics.py` | coverage, non-rejected accuracy, classification quality, rejection quality, `evaluate` |

## Pip install
To install the repo as a package, run either of the following commands:
```bash
pip install git+https://github.com/valevalerio/scce-light.git   # no clone needed
# or, after cloning/downloading anywhere:
pip install -e path/to/scce-light
```

## Quick start
```bash
jupyter notebook demo.ipynb
```
```python
from scce_light import GrowingSpheres, CFDistRejector, generate_counterfactuals, cf_distances

# 1. Load your dataset
X, y = classification_dataset()  # e.g. sklearn.datasets.make_classification
# Split into train, calibration and test sets
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.7, random_state=0)
X_cal, X_test, y_cal, y_test = train_test_split(X_dev, y_dev, test_size=0.4, random_state=0)

# 2. Train the model
model = MLPClassifier()
model.fit(X_train, y_train)

# 3. Instantiate the Counterfactual Generator
gs = GrowingSpheres(model.predict, n_in_layer=200, first_radius=0.05, decrease_radius=2.0, random_state=0)

# 4. Generate counterfactuals and compute distances
d_cal = cf_distances(X_cal, generate_counterfactuals(gs, X_cal, num_cf=32), how='min')
d_test = cf_distances(X_test, generate_counterfactuals(gs, X_test, num_cf=32), how='min')

# 5. Calibrate the rejector
rejector = CFDistRejector(model).calibrate(d_cal, coverages=[0.9, 0.8, 0.7])

# 6. Make predictions on the test set
accepted = rejector.accept(d_test, coverage=0.8)   # True = predict, False = abstain
```
The **`accepted` boolean mask** holds the instances that the model predicts on. You can use it to compute the non-rejected accuracy or other metrics. For example, to compute the accuracy on the test set and the non-rejected accuracy:
```
# 7. Evaluate the classifier equipped with the rejector
y_pred = model.predict(X_test)
print(f"Accuracy {accuracy_score(y_test, y_pred)}")
print(f"Non-rejected accuracy {accuracy_score(y_test[accepted], y_pred[accepted])}")
```
<!-- There should be also a explain why part, but it is not embedded in the rejector. -->

---
The demo notebook runs the whole pipeline on `make_classification` with an MLP. With the defaults it takes about 2-3 minutes on a laptop, most of it spent generating 32 counterfactuals for each of 1,200 instances. Lower `NUM_CF` for a quicker run.


## Differences from the full SC-CE code

- Only Growing Spheres. The full repository also has DiCE, LORE and ILS, as well as latent-space distances.
- Only the Euclidean distance; `cf_distances` accepts any `distance(cfs, x)` callable.
- No dataset loaders, nested cross-validation, Lipschitz MLP.
- Rejectors expose `accept(scores, coverage)`, a boolean mask for a given coverage, instead of `qband` indices.

## Citation
If you find this work useful, please cite the original paper:
```bibtex
@article{bonsignori2026scce,
  author  = {Bonsignori, V. and Punzi, C. and Pellungrini, R. and Giannotti, F.},
  title   = {``I know that I don't know... and I explain why'' Robust abstention via counterfactual explanations},
  journal = {IEEE Access},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3705102}
}
```

<!-- The animations are generated by [Drawings/counterfactuals.ipynb](Drawings/counterfactuals.ipynb). -->
# Methodology

The CSV is validated without dropping or filling records. Features are study hours, attendance,
sleep, and extra classes encoded as Yes=1/No=0. The target is final score; IDs are excluded.

Rows are split 80/20 with `random_state=42` before preprocessing. Means and population standard
deviations are fitted only on training features and reused for test and interactive inputs. A zero
training standard deviation is replaced by a scale of one, leaving that feature centred and safe.

The NumPy model minimises MSE using batch gradient descent. It is compared on the untouched test
set with scikit-learn linear regression and a baseline that always predicts the training target
mean. MAE is average absolute error, RMSE emphasises larger errors, and R² compares the model with a
constant-mean reference. Raw predictions are not clipped.

The inferential analysis predefines a two-sided Welch independent t-test comparing `extra_classes`
Yes and No. It reports a Yes-minus-No difference, Welch degrees of freedom, p-value, and a 95%
confidence interval at alpha 0.05. It assumes independent observations. A failure to reject H0 is
not proof of equality, and neither the test nor correlations establish causality.

All conclusions are limited to a supplied synthetic educational sample. The project uses one split,
does not tune hyperparameters, and is unsuitable for decisions about real students.


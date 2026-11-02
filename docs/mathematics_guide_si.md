# Student Performance Math Lab — සිංහල ගණිත මාර්ගෝපදේශය

මෙය synthetic educational dataset එකක් භාවිත කරන ඉගෙනුම් project එකකි. සැබෑ සිසුන් ගැන causal
නිගමන සඳහා මෙය භාවිත නොකළ යුතුය.

## Vectors සහ Matrices

එක් සිසුවෙකුගේ `[study hours, attendance, sleep, extra classes]` යනු shape `(4,)` vector එකකි.
උදාහරණයට `[10, 80, 7, 1]`. සිසුන් තුන්දෙනෙක් rows ලෙස තැබූ විට `X` shape `(3, 4)` matrix එකක්
ලැබේ. `X.T` shape `(4, 3)` යි. `[1,2] + [3,4] = [4,6]`; `2[1,2]=[2,4]`.
Dot product උදාහරණය: `[1,2]·[3,4]=1×3+2×4=11`. මේවා `linear_algebra.py` තුළ ඇත.

Prediction formula එක `X @ weights + bias`. `X:(n,4)`, `weights:(4,)`, output `(n,)` වේ.
Demo weights ඉගෙනගත් weights නොවේ.

## Mean, Median, Mode, Variance සහ Standard Deviation

`[2,2,4]` සඳහා mean `8/3`, median `2`, mode `2`. Population variance එක deviations squared වල
සාමාන්‍යය (`ddof=0`) යි. Sample variance එක denominator `n-1` (`ddof=1`) භාවිත කරයි. Standard
deviation යනු variance හි square root එකයි. `descriptive.py` manual core calculations library
results සමඟ පැහැදිලිව සම්බන්ධ කරයි. Mode කිහිපයක් තිබිය හැකි බව අමතක නොකරන්න.

## Probability

100 දෙනාගෙන් 60 pass නම් `P(Pass)=60/100=0.60`. Attendance≥80 සිසුන් 40 දෙනාගෙන් 30 pass
නම් `P(Pass | Attendance≥80)=30/40=0.75`. මේවා model confidence නොව empirical probabilities ය.
Conditioning group හි denominator zero නම් result `None` වේ. `probability.py` බලන්න.

## Normal Distribution සහ Z-score

Normal curve එක bell shape distribution එකකි. `z=(x-mean)/std`; mean 60, std 10, score 80 නම්
`z=2`. Histogram, fitted density, Q-Q plot සහ z-scores `distributions.py` සහ
`statistics_plots.py` තුළ ඇත. Score 0–100 අතර bounded නිසා normal විය යුතුම නැත. std zero නම්
z-scores zero ලෙස ආරක්ෂිතව ලබා දෙයි.

## Covariance සහ Correlation

Covariance variables දෙක එකට වෙනස්වන දිශාව පෙන්වයි; එයට combined units ඇත. Pearson
correlation `-1` සිට `1` දක්වා unitless අගයකි. උදාහරණයට study hours වැඩිවන විට scores වැඩිවීම
positive correlation විය හැක. එය causation නොවේ. `relationships.py` IDs ඉවත් කර sample covariance
සහ correlation matrices ගණනය කරයි. Constant column එකක correlation undefined වේ.

## Hypothesis Testing

`H0`: Yes සහ No population means සමානයි. `H1`: වෙනස්. `hypothesis_testing.py` two-sided Welch
t-test භාවිත කරන්නේ variances සමාන යැයි බල නොකරයි. උදාහරණයට p=0.03 සහ alpha=0.05 නම්
`reject H0`; p=0.20 නම් `fail to reject H0`. දෙවැනි ප්‍රකාශය H0 සත්‍ය බව ඔප්පු නොකරයි. p-value
යනු H0 සත්‍ය වීමේ probability එක නොවේ. Confidence interval එක Yes−No mean difference සඳහාය.

## Calculus, MSE සහ Gradients

`f(x)=x²` නම් derivative `f'(x)=2x`; x=3 දී slope 6. Regression සඳහා
`errors = X@weights+bias-y`, `MSE=mean(errors²)`. Shapes: `X:(n,p)`, `y:(n,)`, `weights:(p,)`.
Gradient එක `2/n * X.T @ errors`; bias gradient `2*mean(errors)`. `calculus.py` analytical result
central finite differences සමඟ check කරයි.

Gradient Descent පියවර: (1) weights initialise කරන්න, (2) predictions ගන්න, (3) MSE ගන්න,
(4) gradients ගන්න, (5) `weight -= learning_rate*gradient`, (6) නැවත කරන්න. මෙය
`models/linear_regression.py` තුළය. Learning rate ඉතා විශාල නම් loss diverge විය හැක.

## සාමාන්‍ය වැරදි

- Split කිරීමට පෙර scaling කිරීමෙන් test-data leakage ඇති කිරීම.
- `student_id` feature එකක් කිරීම හෝ pass/fail label එක input කිරීම.
- Matrix shapes නොගැළපීම සහ `X.T` අමතක කිරීම.
- Sample සහ population variance මිශ්‍ර කිරීම.
- Correlation හෝ significant p-value එක causation ලෙස පැවසීම.
- Regression predictions 0–100ට silently clip කර metrics වෙනස් කිරීම.

## කෙටි interview ප්‍රශ්න

**Q: Training statistics පමණක් scaling සඳහා ඇයි?** A: Test information training වෙත leak වීම
වැළැක්වීමට. **Q: RMSE සහ MAE වෙනස?** A: RMSE විශාල errors වැඩි ලෙස penalise කරයි.
**Q: Gradient check එකේ අරමුණ?** A: analytical derivative එක numerical approximation එකට
ගැළපේද බැලීම. **Q: R² negative විය හැකිද?** A: ඔව්; model එක mean baseline එකට වඩා නරක නම්.
**Q: p>0.05 කියන්නේ groups සමානද?** A: නැහැ; H0 reject කිරීමට ප්‍රමාණවත් evidence නැති බව පමණයි.


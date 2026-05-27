# Traffic Demand Prediction — Model & Submission

## Project overview
- Task: Predict `demand` for test rows. Index column name: `Index`.
- Required submission file: `submission.csv` with two columns: `Index`, `demand` (shape 41778 × 2).

## Model
- Model type: `RandomForestRegressor` (scikit-learn)
- Main hyperparameters used:
  - `n_estimators=200`
  - `max_depth=14`
  - `min_samples_split=2`
  - `min_samples_leaf=1`
  - `random_state=42`

## Preprocessing
- Numeric columns: missing values filled with column mean (train mean used for test fill).
- Categorical columns: missing values filled with string `'Unknown'`, then label-encoded. Encoders are fit on combined train+test in `create_submission.py` to avoid unseen labels.
- Features scaled with `StandardScaler` before training.

## Training details
- Training script: `model.py` (prints training R² and basic metrics).
- Submission script: `create_submission.py` (trains same model and writes `submission.csv`).
- Random seed: `42` (used for model reproducibility).

## Achieved metrics (on training set)
- Train R²: 0.941317 (reported in `model.py`)
- Score used by scoreboard: `max(0, 100 * R²) = 94.13`
- MAE: 0.0255
- RMSE: 0.0344

Notes:
- These metrics are calculated on the training set because `test.csv` does not contain ground-truth `demand` labels in this workspace. The official leaderboard/test score must be computed by the competition platform using held-out true labels.

## Submission
1. Run the submission generator (this trains the same model and creates `submission.csv`):

```bash
python3 create_submission.py
```

2. `create_submission.py` will write `submission.csv` with columns `Index,demand` and shape `(41778, 2)`.

3. Verify before uploading:

```bash
head -n1 submission.csv
wc -l submission.csv   # should be 41779 (header + 41778 rows)
```

## Reproducibility
- To save the trained model after training, add code like:

```python
import joblib
joblib.dump(model, 'model.joblib')
```

- To load later:

```python
model = joblib.load('model.joblib')
```

## Environment
- Minimal dependencies are listed in `requirements.txt`.

## Files included
- `model.py` — training script and metrics printout
- `create_submission.py` — script that trains and writes `submission.csv`
- `submission.csv` — generated predictions (ready for upload)
- `README.md` — this file
- `requirements.txt` — minimal environment file

If you want, I can also save the trained model as `model.joblib` and create a `submission_package.zip` containing the recommended files.
# ml_proj
# ml-classifier-demo
a classifier trained on a small tabular dataset using scikit-learn.

It's simple end-to-end machine learning workflow using scikit-learn — generates a dataset, trains a Random Forest classifier through a preprocessing + model pipeline, evaluates performance, and saves the trained pipeline.

## Project Structure
simple-ml-project/
├── data/
│ └── dataset.csv # Generated dataset
├── save_dataset.py # Generates dataset.csv from sklearn's Iris data
├── train.py # Loads data, builds pipeline, trains, evaluates, saves model
├── requirements.txt # Python dependencies
├── model.pkl # Trained pipeline (generated after running train.py)
└── README.md



## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Generate the dataset:
```bash
   python save_dataset.py
```

2. Train the pipeline:
```bash
   python train.py
```

This will:
1. Load the dataset from `data/dataset.csv`
2. Split it into training/test sets
3. Build a scikit-learn `Pipeline` chaining feature scaling and a Random Forest classifier
4. Train the pipeline
5. Print accuracy and a classification report
6. Save the trained pipeline as `model.pkl`

## Example Output

Accuracy: 1.0
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30

Pipeline saved to model.pkl


## Why a Pipeline?

This project uses a scikit-learn `Pipeline` to chain feature scaling (`StandardScaler`) and model training (`RandomForestClassifier`) into a single reproducible object. This ensures consistent preprocessing at both training and inference time, and avoids accidentally applying a differently-fit scaler or leaking test data into preprocessing.

Note: `StandardScaler` has minimal impact on tree-based models like Random Forest, but is included here to demonstrate proper pipeline structure for preprocessing + model chaining.

## Loading the Trained Pipeline

```python
import joblib
pipeline = joblib.load("model.pkl")
prediction = pipeline.predict(new_raw_data)  # scaling happens automatically
```

## Tech Stack
- Python
- scikit-learn
- pandas

## License
MIT
# 🌲 Forest Cover Type Classification (TensorFlow + Keras)

This project uses **Deep Learning** to predict the dominant forest cover type for a given 30x30 meter cell, based solely on cartographic variables such as elevation, slope, aspect, and distance to hydrology/roadways.  
The dataset originates from the **U.S. Forest Service Region 2** and represents areas in the Roosevelt National Forest of northern Colorado.

---

## Project Overview

The goal is to build a **multi-class classification model** that identifies which of the following seven tree types dominates a given area:

1. Spruce/Fir  
2. Lodgepole Pine  
3. Ponderosa Pine  
4. Cottonwood/Willow  
5. Aspen  
6. Douglas-fir  
7. Krummholz  

Using TensorFlow and Keras, this project explores **data preprocessing**, **feature scaling**, **model training**, and **performance evaluation** to accurately classify forest cover types.

---

## 🧩 Key Features

- **Data Preprocessing Pipeline**  
  - Standardization of numerical columns using `StandardScaler`  
  - Retention of binary one-hot columns (Wilderness Area & Soil Type)  
  - Modular preprocessing class for reusability  

- **Neural Network Model**  
  - Built with TensorFlow/Keras  
  - Layers: Dense + BatchNormalization + Dropout  
  - Activation: ReLU and Softmax for multi-class output  
  - EarlyStopping and ReduceLROnPlateau for optimized convergence  

- **Performance Metrics**  
  - Accuracy, Balanced Accuracy  
  - Confusion Matrix & Classification Report (precision, recall, F1)  
  - Validation accuracy up to **~0.89** on held-out data  

- **Model Persistence**  
  - Saved trained model in `.keras` format  
  - Includes utility for individual prediction  

---

## 🧠 Technologies Used

| Category | Tools / Libraries |
|-----------|------------------|
| Language | Python 3.10+ |
| Framework | TensorFlow, Keras |
| Data Handling | Pandas, NumPy |
| Preprocessing | Scikit-learn |
| Visualization | Matplotlib |
| Environment | VS Code / Jupyter / Terminal |

---

## ⚙️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/forest-cover-type-tf.git
   cd forest-cover-type-tf
   ```

2. **Set up a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add the dataset**
   - Place your `cover_data.csv` file in the project root.  
   *(File is excluded from GitHub for size and licensing reasons.)*

5. **Run training**
   ```bash
   python cover_type_classifier.py
   ```

6. **Saved Model**
   - The trained model will be saved automatically under `saved_model/cover_type_model.keras`.

---

## 📈 Results

| Metric | Value |
|--------|--------|
| Train Accuracy | ~0.86 |
| Validation Accuracy | ~0.89 |
| Test Accuracy | ~0.88–0.89 |
| Balanced Accuracy | ~0.85 |

Model achieved stable generalization across all 7 cover types with improved minority-class recall using balanced class weights.

---

## 🔍 Example Prediction

```python
row_df = X_test.iloc[[0]]
x = prep.transform(row_df)
proba = model.predict(x)
pred_idx = int(proba.argmax())
print("Predicted:", target_names[pred_idx])
```

---

## 📂 Project Structure

```
forest-cover-type-tf/
│
├── cover_type_classifier.py    # Main training script
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Ignored files
├── saved_model/                # Saved model (ignored by Git)
└── cover_data.csv              # Dataset (not committed)
```

---

## Author

**Patrick Dalington**  
*Software & Machine Learning Engineer*  
- Focus: Deep Learning, Predictive Modeling, and Intelligent Systems  
- Passionate about continuous learning and applied AI  
- [LinkedIn](https://www.linkedin.com/in/patrick-olumba/) | [GitHub](https://github.com/patdal1810)

---

## Future Improvements

- Experiment with **XGBoost** and **Random Forest** for comparison  
- Implement **feature importance visualization**  
- Add **streamlit dashboard** for live predictions  
- Integrate **TensorBoard** for training visualization  

---

## 📝 License

This project is open-source under the **MIT License**. Feel free to fork and experiment.

---

> “Knowledge compounds when applied.”  
> The more we learn, the better we build. 🌲

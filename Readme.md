# 🧠 Digit Recognizer using Logistic Regression PCA

This project is a **Digit Recognition Web App** that uses a
**Logistic Regression** trained on the **MNIST dataset**.
The user uploads a handwritten digit image, and the model predicts a
number from 0--9 with a corresponding historical or fun fact.

------------------------------------------------------------------------

## 🚀 Features

-   Upload handwritten digit image (`.jpg`, `.png`, etc.).
-   Loader animation during prediction.
-   Dimensionality reduction using **PCA (Principal Component
    Analysis)**.
-   Flask backend for prediction.
-   Beautiful frontend built with **Tailwind CSS**.
-   Hosted backend on **Render** and frontend on **GitHub Pages**.

------------------------------------------------------------------------

## 🧩 How It Works

### 1️⃣ Preprocessing

-   The uploaded image is converted to grayscale.
-   Resized to 28×28 pixels to match MNIST format.
-   Normalized (pixel values scaled between 0 and 1).

### 2️⃣ Feature Extraction using PCA

-   PCA is applied on the flattened image data to reduce dimensionality
    while preserving maximum variance.
-   This step makes the model faster and less prone to noise.

### 3️⃣ Logistic Regression Prediction

- The preprocessed image (flattened to 784 features) is first **scaled** using `MinMaxScaler()` to bring all pixel values between 0 and 1.  
- Then, **Principal Component Analysis (PCA)** reduces the dimensionality of the image data (for example, from 784 → 150 components), preserving the most significant variance.  
- The reduced feature vector is passed to the **Logistic Regression** classifier, which learns linear decision boundaries between digits (0–9).  
- The model outputs the **predicted digit** with the highest confidence score.

### 4️⃣ Historical Fact

Each digit (0--9) has a historical or fun fact stored in a `facts.json`
file.\
For example:

``` json
{
  "0": "Zero was first used in India by Aryabhata around the 5th century.",
  "1": "The number one represents unity and the beginning of all numbers.",
  "2": "Binary code, the foundation of computers, is based on 0s and 2s.",
  "3": "The number three symbolizes harmony and balance in many cultures."
}
```

------------------------------------------------------------------------

## 🧠 Model Architecture

``` python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

-   **Loss Function:** `categorical_crossentropy`\
-   **Optimizer:** `adam`\
-   **Metrics:** `accuracy`

------------------------------------------------------------------------


## ⚙️ Tech Stack

  Layer          Technology
  -------------- ----------------------------------------------
  **Frontend**   HTML, Tailwind CSS, JavaScript
  **Backend**    Flask (Python)
  **ML Model**   Logistic Regression, PCA (Scikit-learn)
  **Hosting**    Render (Backend), GitHub Pages (Frontend)

------------------------------------------------------------------------

## 🌍 Deployment

### 🔹 Backend (Render)

1.  Push your Flask app to GitHub.

2.  Go to [Render](https://render.com).

3.  Create a new **Web Service** and connect your repo.

4.  In "Start Command":

    ``` bash
    gunicorn app:app
    ```

5.  Deploy.

### 🔹 Frontend (GitHub Pages)

1.  Push your frontend files to a GitHub repo.
2.  Go to **Settings → Pages → Deploy from Branch**.
3.  Choose `main` and root `/` directory.
4.  Visit your GitHub Pages link
    (e.g. `https://username.github.io/Digit-Classifier/`).

------------------------------------------------------------------------

## 🧮 Example Prediction Flow

1.  User uploads a handwritten digit (e.g., "7").
2.  Image is sent to Flask backend (`/predict`).
3.  Flask processes the image → runs PCA → LR predicts digit.
4.  Flask returns a response:

``` json
{
  "digit": 7,
  "fact": "Seven is considered a lucky number in many cultures."
}
```

5.  Frontend displays the predicted number with its fun fact.

------------------------------------------------------------------------

## 🧪 Local Testing

``` bash
cd backend
pip install -r requirements.txt
python app.py
```

Then open `frontend/index.html` in your browser.

------------------------------------------------------------------------

## 📜 License

MIT License © 2025 Sayandip Paul

------------------------------------------------------------------------

## 💡 Credits

-   **Dataset:** MNIST by Yann LeCun
-   **Frameworks:**  Flask
-   **Design:** Tailwind CSS
-   **Developer:** Sayandip Paul

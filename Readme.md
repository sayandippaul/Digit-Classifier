# 🧠 Digit Recognizer using CNN & PCA

This project is a **Digit Recognition Web App** that uses a
**Convolutional Neural Network (CNN)** trained on the **MNIST dataset**.
The user uploads a handwritten digit image, and the model predicts a
number from 0--9 with a corresponding historical or fun fact.

------------------------------------------------------------------------

## 🚀 Features

-   Upload handwritten digit image (`.jpg`, `.png`, etc.).\
-   Loader animation during prediction.\
-   CNN model built using **TensorFlow/Keras**.\
-   Dimensionality reduction using **PCA (Principal Component
    Analysis)**.\
-   Flask backend for prediction.\
-   Beautiful frontend built with **Tailwind CSS**.\
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

### 3️⃣ CNN Prediction

-   The processed image is fed into a **Convolutional Neural Network
    (CNN)** trained on MNIST.
-   CNN layers extract spatial features (edges, shapes, etc.) and output
    a probability distribution across digits 0--9.
-   The digit with the highest probability is the final prediction.

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

## 🧱 Project Structure

    Digit-Classifier/
    │
    ├── backend/
    │   ├── app.py                # Flask server
    │   ├── model.pkl             # Trained CNN + PCA model
    │   ├── facts.json            # Fun facts for digits
    │   └── requirements.txt
    │
    ├── frontend/
    │   ├── index.html            # User interface
    │   ├── style.css             # Tailwind CSS
    │   └── script.js             # Fetch API and interaction logic
    │
    ├── dataset/
    │   ├── mnist_train.csv
    │   └── mnist_test.csv
    │
    └── README.md

------------------------------------------------------------------------

## ⚙️ Tech Stack

  Layer          Technology
  -------------- ----------------------------------------------
  **Frontend**   HTML, Tailwind CSS, JavaScript
  **Backend**    Flask (Python)
  **ML Model**   CNN (TensorFlow + Keras), PCA (Scikit-learn)
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

1.  User uploads a handwritten digit (e.g., "7").\
2.  Image is sent to Flask backend (`/predict`).\
3.  Flask processes the image → runs PCA → CNN predicts digit.\
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

-   **Dataset:** MNIST by Yann LeCun\
-   **Frameworks:** TensorFlow, Flask\
-   **Design:** Tailwind CSS\
-   **Developer:** Sayandip Paul

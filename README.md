AI Image Authenticity Detector

This project detects whether an uploaded image is a real photograph or an AI-generated image using a deep learning–based convolutional neural network. The system provides a confidence score and a human-readable explanation for transparency and trust.

Step-by-Step Procedure to Run the Project
Step 1: Clone the Repository

Clone the project repository and navigate into the project folder.

git clone <repository-link>
cd AI-Image-Authenticity-Detector

Step 2: Create and Activate a Virtual Environment (Recommended)
Windows
python -m venv venv
venv\Scripts\activate

Linux / macOS
python3 -m venv venv
source venv/bin/activate

Step 3: Install Required Dependencies

Install all required Python libraries using the requirements file.

pip install -r requirements.txt

Step 4: Prepare the Dataset

Download the CIFAKE: Real and AI-Generated Images Dataset from Kaggle and organize it in the following structure:

dataset/
├── real/
│   ├── image1.jpg
│   └── ...
└── fake/
    ├── image1.jpg
    └── ...


Ensure that both real and fake folders contain a balanced number of images.

Step 5: Train the Model

Run the training script to train the deep learning model.

python train_model.py


After training, the file ai_image_detector.h5 will be generated in the project directory.

Step 6: Run the Streamlit Application

Start the Streamlit web application using the following command:

python -m streamlit run app.py

Step 7: Use the Application

Upload an image (.jpg, .png, or .jpeg)

The model predicts whether the image is Real or AI-Generated

View the confidence score and textual explanation

Step 8: Stop the Application

Press Ctrl + C in the terminal to stop the Streamlit server.

# Emotion Recognition System - Flask Backend and React+Vite Frontend

## Instructions to Run Flask Backend

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/emotion-recognition-system.git
    cd emotion-recognition-system
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
    If you get scripts are disabled error, use this command: 
        ```bash
        Set-ExecutionPolicy Unrestricted -Scope Process
        ```
3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set the Flask application environment variable:(OPTIONAL)**
    ```bash
    export FLASK_APP=app.py  # On Windows use `set FLASK_APP=app.py`
    export FLASK_ENV=development  # On Windows use `set FLASK_ENV=development`
    ```

5.  **Run the Flask application:** *NOT REQUIRED*
    ```bash
    flask run
    ```
    If it doesn't run try:
        flask --app main run

6. **Access the application:**
    Open your web browser and go to `http://127.0.0.1:5000/`.


## Additional Notes

- Ensure you have Python and pip installed on your system.
- If you encounter any issues, check the Flask documentation or the project's issue tracker for help.


## Instructions to Run React + Vite Frontend

1. **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```

2. **Install the required dependencies:**
    ```bash
    npm install
    ```

3. **Run the Vite development server:**
    ```bash
    npm run dev
    ```

4. **Access the application:**
    Open your web browser and go to the URL provided in the terminal, typically `http://localhost:3000/`.

## Full Stack Instructions

1. **Run the Flask backend:**
    Follow the instructions in the "Instructions to Run Flask Backend" section above.

2. **Run the React + Vite frontend:**
    Follow the instructions in the "Instructions to Run React + Vite Frontend" section above.

3. **Access the full application:**
    Open your web browser and go to the frontend URL, typically `http://localhost:3000/`. The frontend will communicate with the backend running on `http://127.0.0.1:5000/`.

## Additional Notes

- Ensure you have Node.js and npm installed on your system for the frontend.
- If you encounter any issues, check the Vite documentation or the project's issue tracker for help.
- Make sure both the backend and frontend are running simultaneously for the full application to work correctly.

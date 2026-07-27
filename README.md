# Insurance Prediction App

This project is a web application that predicts medical insurance charges based on user input such as age, sex, BMI, number of children, smoking status, and region. It uses a trained regression model served through a FastAPI backend and a simple HTML front end.

## Features

- Predict insurance charges from a web form
- Expose a JSON API for programmatic predictions
- Health check endpoint for deployment monitoring
- Ready for containerized deployment

## Tech Stack

- Python
- FastAPI
- Jinja2
- Pandas
- PyCaret
- Uvicorn

## Project Structure

- app.py - FastAPI application and prediction logic
- templates/ - HTML templates for the web UI
- static/ - CSS and static assets
- requirements.txt - Python dependencies
- Procfile - Deployment command for hosting platforms
- Dockerfile - Container build configuration

## Installation

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

> Note: PyCaret is required to load the trained model. If needed, install it separately in your environment.

## Run Locally

Start the app with:

```bash
uvicorn app:app --reload
```

Then open the following in your browser:

- http://127.0.0.1:8000/ - Web interface
- http://127.0.0.1:8000/docs - Interactive API docs
- http://127.0.0.1:8000/health - Health check endpoint

## API Usage

### Predict via form
Use the web interface at the home page.

### Predict via JSON API
Send a POST request to:

```bash
curl -X POST "http://127.0.0.1:8000/predict_api" \
  -H "Content-Type: application/json" \
  -d '{"age": 30, "sex": "male", "bmi": 28.5, "children": 2, "smoker": "no", "region": "southwest"}'
```

## Deployment

This project can be deployed using Docker or hosting platforms such as Render. A Procfile and Dockerfile are included for container-based deployment.

## License

This project is intended for educational and demonstration purposes.

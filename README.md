# Computed Tomography Exam Scheduler

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-orange.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg?logo=docker)](https://www.docker.com/)

This repository contains the supporting code for the article *"Optimizing CT scheduling in the Emergency Department: development and simulation of a machine learning scheduling tool."*

It provides a simple demonstration of how predictions from machine learning techniques can help balance the radiology workload on weekdays.

---

## Try the App Online

You can try the application directly online without any installation by visiting the following link:

**👉 [https://computedtomographyallocation.onrender.com](https://computedtomographyallocation.onrender.com)**

---

## Key Features

- **Dynamic Forms**: Generates forms based on the number of patients and the selected balancing criterion.
- **Exam Customization**: Allows selection of scan type, department, and patient/exam details.
- **Optimization**: An algorithm to balance CT scan and radiologists' workload.
- **Visualization**: Compares classic vs. optimized scheduling, using both expected results and Monte Carlo simulations.

---

## Local Installation and Setup (with Docker)

To run the application on your computer, follow these steps:

1.  **Install Docker Desktop**:  
    Download it from [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop).

2.  **Clone the Repository**:
    Open a terminal and run the following command:
    ```bash
    git clone https://github.com/SaraCambiaghi/ComputerTomographyAllocation.git
    ```
    Alternatively, you can download the repository as a ZIP file and unzip it.

3.  **Build the Docker Image**:  
    Make sure Docker Desktop is running. Open your terminal, navigate to the project folder, and run:
    ```bash
    docker build -t ct-scheduler-app .
    ```
    *(You can replace `ct-scheduler-app` with a name of your choice.)*

4.  **Run the Container**:
    Run the Docker container with the following command:
    ```bash
    docker run -p 5000:5000 ct-scheduler-app
    ```

5.  **Access the Application**:  
    Open your browser and go to [http://localhost:5000](http://localhost:5000).

---

## Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: CatBoost
- **Optimization**: Python-MIP
- **Containerization**: Docker

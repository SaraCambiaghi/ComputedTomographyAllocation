# CT Exam Scheduler

This repository accompanies the article *"Computed Tomography Allocation in an Emergency Department: Comparison of Machine Learning and Deep Learning Methods as Support Tools."*

It provides a toy example demonstrating how predictions made with machine learning techniques can assist in balancing radiology workload across weekdays.

## Features

- Dynamic form generation based on the number of patients and the selected balancing criterion  
- Selectable scan types, departments, and patient/exam details  
- Basic optimization to balance radiology workload  
- Visualization of classic vs. optimized scheduling, using both expected and simulated outcomes via Monte Carlo methods  

## Getting Started

1. Download and install Docker Desktop:  
   👉 [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)

2. Build the Docker image:  
   Start Docker Desktop, open your terminal, and run:
   ```bash
   docker build -t my-app .
(You can replace my-app with any name you prefer for the image.)

3. Run the container and start the app:
     ```bash
     docker run -p 5000:5000 my-app
4. Open your browser and go to:
   
   👉 [http://localhost:5000](http://localhost:5000)




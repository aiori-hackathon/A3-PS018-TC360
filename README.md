# HAPS Link-Quality Prediction Using a Documented Channel Model

## Project Overview

HAPS (High Altitude Platform Station) is a communication platform positioned in the stratosphere at an altitude of approximately 20 km.

This project develops a link-quality prediction system that analyses and predicts the Signal-to-Noise Ratio (SNR) between a HAPS platform and ground receivers under different propagation and weather conditions.

The system combines ITU-R propagation models, link budget calculations, and a Random Forest Regression model to predict communication link quality.

## Problem Statement

To develop a HAPS link-quality prediction system using documented channel models and machine learning to evaluate communication performance under varying distances and rain conditions.

## Project Objectives

- Analyse HAPS-to-ground communication link quality.
- Calculate propagation losses using ITU-R recommendations.
- Evaluate received signal power and SNR.
- Study the impact of rain attenuation on link performance.
- Predict SNR using a Random Forest Regression model.
- Visualise link-quality variations through an interactive dashboard.

## Methodology

1. Simulate different ground receiver trajectories.
2. Calculate propagation losses using ITU-R channel models.
3. Perform link budget calculations.
4. Generate a synthetic dataset for different environmental conditions.
5. Train and evaluate a Random Forest Regression model.
6. Predict SNR and display the results through a Streamlit dashboard.

## Technologies and Models Used

- Python
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- ITU-R P.525
- ITU-R P.528
- ITU-R P.838
- Random Forest Regression

## Key Features

- Interactive HAPS link-quality dashboard.
- Link budget and SNR calculations.
- Clear-sky and rain-condition analysis.
- Received power and SNR visualisations.
- Machine learning-based SNR prediction.
- Model evaluation using MAE, RMSE and R².
- Feature importance analysis.
- Dataset and result visualisation.

## Working Demo

[Watch the Working Demo](https://youtu.be/UyWDJKvVWig)

## Team Introduction Video

[Watch the Team Introduction Video](https://youtu.be/rTSsb8g734A)

## Team Details

- **Team Name:** Team RVTS
- **Team ID:** A3-PS018-TC360
- **Project Title:** HAPS Link-Quality Prediction Using a Documented Channel Model
- **Institution:** K.S. School of Engineering and Management

## Repository Structure

- `src/` – Project source code.
- `data/` – Dataset resources.
- `.streamlit/` – Streamlit configuration.
- `requirements.txt` – Required Python packages.

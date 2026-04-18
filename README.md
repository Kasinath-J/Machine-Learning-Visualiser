# Machine Learning Visualiser

Visualising basic Machine Learning algorithms from scratch, inspired by Andrew Ng's Machine Learning course. 

This repository implements fundamental Machine Learning models using pure mathematics (via `numpy`) and visually plots the model's learning process in real-time (via `matplotlib`). With these tools, you can actually watch the gradient descent algorithm adjust its parameters and converge towards lower cost functions.

## Multi Linear Regression
<img height="400" src="https://github.com/Kasinath-J/Machine-Learning-Visualiser/blob/main/assets/multiLinearRegression.gif">

## Multi Logistic Regression
<img height="400" src="https://github.com/Kasinath-J/Machine-Learning-Visualiser/blob/main/assets/multiLinearClassification.gif">

## Included Models

### 1. Linear Regression
- Includes Simple Linear Regression and Multiple Linear Regression.
- 3D interactive plots that allow you to visualize the regression plane fitting against multi-dimensional data points.

### 2. Classification (Logistic Regression)
- Implementations of Logistic Regression for binary classification.
- Interactive plots that draw decision boundaries.
- 3D surface animations showcasing the sigmoid surface adapting to separate data classes efficiently.

## Prerequisites

To run these visualisations, you need Python 3 installed. It is recommended to create a virtual environment to manage dependencies locally.

### Setup Instructions

1. **Create and Activate a Virtual Environment:**
   ```bash
   # On macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   This project relies on `numpy` for pure mathematical operations (matrix multiplication, cost function mapping) and `matplotlib` for 2D/3D visualizations.
   ```bash
   pip install numpy matplotlib
   ```

3. **Run the Visualisers:**
   Navigate into the respective folders and run the python files. A live matplotlib window will open tracking the gradient descent convergence.
   ```bash
   # Example: Running the 3D Classification visualizer
   python3 "Classification/5.plot.py"

   # Example: Running the Multiple Linear Regression visualizer
   python3 "linear regression/5.1.Mulit Linear Regression with plot.py"
   ```

## Requirements
- `numpy`
- `matplotlib`

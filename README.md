Chess Outcome Prediction with XGBoost
Overview
This project implements a machine learning model to predict the outcomes of chess games ("Black Wins," "Draw," or "White Wins") based on board positions. Leveraging the XGBoost algorithm, it processes a large dataset of evaluated chess positions from Lichess, extracts meaningful features (e.g., king safety, pawn structure, piece activity), and trains a classifier to make accurate predictions. The model is designed to handle complex middlegame and endgame scenarios, offering a practical application of gradient boosting in strategic games.
Features

Data Processing: Streams and decompresses Lichess's Zstandard-compressed JSON Lines dataset (lichess_db_eval.jsonl.zst) to load up to 10 million positions.
Feature Engineering: Extracts 784-dimensional feature vectors, including board encoding, king safety (pawn shield, enemy attacks, mobility), pawn structure (passed, isolated, doubled pawns), and piece activity (mobility, center control).
Model Training: Utilizes XGBoost with customizable hyperparameters (e.g., max_depth=8, n_estimators=100, learning_rate=0.1) to classify game outcomes.
Evaluation: Provides accuracy and a detailed classification report (precision, recall, F1-score) on a test set.
Prediction: Outputs predicted outcomes and approximate evaluation scores (e.g., "+2.0" for White advantage) for new FEN positions.

Requirements

Python 3.x
Libraries:

numpy
pandas
chess
zstandard
json
scikit-learn
xgboost
os
sys
time


Dataset: Lichess evaluation database (lichess_db_eval.jsonl.zst) available from database.lichess.org.

Installation

Clone the repository:
git clone https://github.com/BwoahMeow/Chess_Evaluator.git
cd chess-outcome-prediction

Usage
Run the script to train the model and predict outcomes:

In Spyder: Open lichess_chess_gbm_with_features.py and press F5.
Command Line: Navigate to the project directory and run: python lichess_chess_gbm_with_features.py
The script will:

Load and process the dataset (up to 10 million positions).
Train the XGBoost model.
Display accuracy, classification report, and a prediction for the example FEN.
Save the model as chess_xgb_model.json and log the total execution time.

To predict a custom position, modify the new_fen variable (e.g., "rnbqkbnr/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKBNR w KQkq - 0 1").

Development

Author: Shubham Singhal
Created: Saturday, August 23, 2025
Last Updated: Thursday, September 18, 2025

Improvements:

Add cross-validation or early stopping to enhance model robustness.
Implement batch processing for larger datasets to manage memory.
Extend feature set (e.g., material balance, piece-square tables).
Automate retraining with a scheduling script.

"""
Created on Sat Aug 23 2025

@author: Shubham Singhal
"""
import numpy as np
import pandas as pd
import chess
import zstandard as zstd
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import os 
import sys
import time

start_time = time.time()
# Set random seed for reproducibility
np.random.seed(42)

# Function to compute king safety features
def compute_king_safety(board, color):
    king_square = board.king(color)
    if king_square is None:
        return 0, 0, 0

    # Pawn shield
    pawn_shield = 0
    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)
    pawn_squares = [
        chess.square(king_file + i, king_rank + (1 if color == chess.WHITE else -1))
        for i in [-1, 0, 1] if 0 <= king_file + i <= 7
    ]
    for square in pawn_squares:
        if 0 <= square <= 63 and board.piece_at(square) == chess.Piece(chess.PAWN, color):
            pawn_shield += 1

    # Enemy attacks near king
    attack_squares = [
        chess.square(king_file + i, king_rank + j)
        for i in [-1, 0, 1] for j in [-1, 0, 1]
        if 0 <= king_file + i <= 7 and 0 <= king_rank + j <= 7
    ]
    enemy_attacks = sum(
        len(board.attackers(not color, square)) for square in attack_squares if 0 <= square <= 63
    )

    # King mobility
    king_mobility = len([move for move in board.legal_moves if move.from_square == king_square])

    return pawn_shield, enemy_attacks, king_mobility

# Function to compute pawn structure features
def compute_pawn_structure(board, color):
    passed_pawns = 0
    isolated_pawns = 0
    doubled_pawns = 0
    files_with_pawns = [0] * 8

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece == chess.Piece(chess.PAWN, color):
            file_idx = chess.square_file(square)
            files_with_pawns[file_idx] += 1

            # Passed pawn check
            is_passed = True
            for rank in range(chess.square_rank(square) + (1 if color == chess.WHITE else -1), (8 if color == chess.WHITE else -1), (1 if color == chess.WHITE else -1)):
                for adj_file in [file_idx - 1, file_idx, file_idx + 1]:
                    if 0 <= adj_file <= 7:
                        if board.piece_at(chess.square(adj_file, rank)) == chess.Piece(chess.PAWN, not color):
                            is_passed = False
                            break
                if not is_passed:
                    break
            if is_passed:
                passed_pawns += 1

            # Isolated pawn check
            is_isolated = not any(files_with_pawns[adj_file] for adj_file in [file_idx - 1, file_idx + 1] if 0 <= adj_file <= 7)
            if is_isolated:
                isolated_pawns += 1

    # Doubled pawns
    doubled_pawns = sum(count - 1 for count in files_with_pawns if count > 1)

    return passed_pawns, isolated_pawns, doubled_pawns

# Function to compute piece activity and center control
def compute_piece_activity(board, color):
    central_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    center_control = 0
    piece_mobility = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == color and piece.piece_type != chess.PAWN and piece.piece_type != chess.KING:
            # Mobility: count attacked squares
            piece_mobility += len(board.attacks(square))
            # Center control
            if any(square in board.attacks(central_square) for central_square in central_squares):
                center_control += 1

    return piece_mobility, center_control

# Function to extract features from a board position
def extract_features(board):
    # Basic board encoding (12 piece types x 64 squares)
    piece_map = board.piece_map()
    board_features = np.zeros(12 * 64)
    for square, piece in piece_map.items():
        piece_idx = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
        board_features[square * 12 + piece_idx] = 1

    # King safety features
    white_pawn_shield, white_enemy_attacks, white_king_mobility = compute_king_safety(board, chess.WHITE)
    black_pawn_shield, black_enemy_attacks, black_king_mobility = compute_king_safety(board, chess.BLACK)

    # Pawn structure features
    white_passed, white_isolated, white_doubled = compute_pawn_structure(board, chess.WHITE)
    black_passed, black_isolated, black_doubled = compute_pawn_structure(board, chess.BLACK)

    # Piece activity and center control
    white_mobility, white_center_control = compute_piece_activity(board, chess.WHITE)
    black_mobility, black_center_control = compute_piece_activity(board, chess.BLACK)

    # Combine features
    features = np.concatenate([
        board_features,
        [white_pawn_shield, white_enemy_attacks, white_king_mobility,
         black_pawn_shield, black_enemy_attacks, black_king_mobility,
         white_passed, white_isolated, white_doubled,
         black_passed, black_isolated, black_doubled,
         white_mobility, white_center_control,
         black_mobility, black_center_control]
    ])
    return features

# Function to stream Lichess evaluated positions
def load_lichess_data(zst_path, max_positions= 10000):
    if not os.path.exists(zst_path):
        raise FileNotFoundError(f"File not found: {zst_path}")
    X, y = [], []
    try:
        with open(zst_path, 'rb') as ifh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(ifh) as reader:
                buffer = ""
                i = 0
                while i < max_positions:
                    chunk = reader.read(8192)  # Read 8KB chunks
                    if not chunk:  # End of file
                        break
                    try:
                        buffer += chunk.decode('utf-8', errors='ignore')
                        lines = buffer.split('\n')
                        for line in lines[:-1]:
                            if line.strip():
                                try:
                                    position = json.loads(line)
                                    fen = position['fen']
                                    board = chess.Board(fen)
                                    features = extract_features(board)

                                    # Label based on Stockfish evaluation
                                    evals = position['evals'][0]['pvs'][0]
                                    cp = evals.get('cp', 0)
                                    mate = evals.get('mate', None)
                                    if mate is not None:
                                        label = 2 if mate > 0 else 0
                                    else:
                                        label = 2 if cp > 200 else 0 if cp < -200 else 1

                                    X.append(features)
                                    y.append(label)
                                    i += 1
                                    if i >= max_positions:
                                        break
                                except (json.JSONDecodeError, KeyError) as e:
                                    print(f"Error processing line {i}: {e}")
                        buffer = lines[-1]
                    except UnicodeDecodeError as e:
                        print(f"Error decoding chunk at position {i}: {e}")
                        break
    except Exception as e:
        print(f"Error reading file {zst_path}: {e}")
        raise
    return np.array(X), np.array(y)

# Load data (replace with actual path to .zst file)
zst_path = r"D:\lichess_db_eval.jsonl.zst"  # Update with your file path
X, y = load_lichess_data(zst_path, max_positions= 100000)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and configure XGBoost classifier
xgb_classifier = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    max_depth=8,
    learning_rate=0.1,
    n_estimators=100,
    eval_metric='mlogloss',
    random_state=42
)

# Train the model
xgb_classifier.fit(X_train, y_train)

# Make predictions on test set
y_pred = xgb_classifier.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Black Wins', 'Draw', 'White Wins']))

# Example: Predict for a new position
new_fen = "4r1k1/ppp1rpp1/2np3p/5qb1/3P4/2P2N2/PP3PPP/R1BQR1K1 b - - 3 19"
board = chess.Board(new_fen)
new_features = extract_features(board)
prediction = xgb_classifier.predict([new_features])
class_names = ['Black Wins', 'Draw', 'White Wins']
print(f"\nPredicted outcome for position {new_fen}: {class_names[prediction[0]]}")

# Save the model
xgb_classifier.save_model('chess_xgb_model.json')

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")


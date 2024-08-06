import csv
import requests
import time
import numpy as np
import pandas as pd
import chess
import chess.pgn
from io import StringIO

# Add headers
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
    'Referer': 'https://www.chess.com/'
}

# Function to fetch JSON data from a URL
def fetch_json(url):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

# Function to clean the PGN
def clean_pgn(pgn):
    moves = pgn.split('\n\n')[1]
    moves = moves.replace('{[%clk ', '')
    moves = ' '.join(moves.split()).replace('} ', '')
    moves = moves.split(' ')[:-1]  # remove the game result (e.g., "1-0", "0-1", "1/2-1/2")
    
    # Remove timestamps
    cleaned_moves = []
    for move in moves:
        if ']' in move:
            move = move.split(']')[1]
        cleaned_moves.append(move)
        
    return ' '.join(cleaned_moves)

# Function to process a single game
def process_game(game):
    if 'pgn' not in game:
        return None, None, None

    pgn = game['pgn']
    white_player = game['white']['username']
    black_player = game['black']['username']
    white_result = game['white']['result']
    black_result = game['black']['result']

    # Determine the result for Hikaru
    if white_player.lower() == 'hikaru':
        hikaru_color = 'white'
        if white_result == 'win':
            hikaru_result = 'win'
        elif white_result in ['resigned', 'timeout', 'checkmated']:
            hikaru_result = 'lose'
        elif white_result == 'agreed':
            hikaru_result = 'draw'
        else:
            hikaru_result = 'lose'
    else:
        hikaru_color = 'black'
        if black_result == 'win':
            hikaru_result = 'win'
        elif black_result in ['resigned', 'timeout', 'checkmated']:
            hikaru_result = 'lose'
        elif black_result == 'agreed':
            hikaru_result = 'draw'
        else:
            hikaru_result = 'lose'

    trimmed_pgn = clean_pgn(pgn)
    return hikaru_result, hikaru_color, trimmed_pgn

# Function to fetch all games from a list of archive URLs
def fetch_all_games(archive_urls):
    all_games = []
    for url in archive_urls:
        try:
            games = fetch_json(url)['games']
            all_games.extend(games)
            print(f"Fetched {len(games)} games from {url}")
        except Exception as e:
            print(f"Failed to fetch games from {url}: {e}")
        # Sleep to avoid hitting rate limits
        time.sleep(1)
    return all_games

# Function to write games to a CSV file
def write_games_to_csv(games, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Result', 'Color', 'Moves'])
        for game in games:
            hikaru_result, hikaru_color, moves = process_game(game)
            if hikaru_result and hikaru_color and moves:
                writer.writerow([hikaru_result, hikaru_color, moves])

# Function to convert board to features
def board_to_features(board):
    features = np.zeros((8, 8, 12), dtype=np.float32)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            features[i // 8, i % 8, piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)] = 1
    return features

# Function to convert model prediction back to move
def predict_move(board, model, move_encoder):
    features = board_to_features(board)
    features = np.expand_dims(features, axis=0)
    preds = model.predict(features)
    sorted_indices = np.argsort(preds[0])[::-1]  # Indices of moves sorted by probability
    
    for move_idx in sorted_indices:
        move_uci = [uci for uci, idx in move_encoder.items() if idx == move_idx][0]
        move = chess.Move.from_uci(move_uci)
        if move in board.legal_moves:
            return move
    
    return None  # If no legal move is found

# Function to play a game between the model and Stockfish
def play_game_against_stockfish(model, move_encoder, stockfish_path, stockfish_level=1):
    board = chess.Board()
    stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = predict_move(board, model, move_encoder)
        else:
            result = stockfish.play(board, chess.engine.Limit(time=0.1))
            move = result.move
        
        if move is None:
            print("No legal move found.")
            break
        board.push(move)
    
    stockfish.quit()
    return board.result()

# Function to calculate Elo rating based on game results
def calculate_elo(model, move_encoder, stockfish_path, num_games=10, stockfish_level=1):
    wins, losses, draws = 0, 0, 0

    for _ in range(num_games):
        result = play_game_against_stockfish(model, move_encoder, stockfish_path, stockfish_level)
        if result == '1-0':
            wins += 1
        elif result == '0-1':
            losses += 1
        else:
            draws += 1
    
    print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")

    # Basic Elo calculation assuming Stockfish has a fixed rating (e.g., 2400 at level 1)
    stockfish_rating = 2400
    K = 32  # K-factor in Elo rating calculation
    score = (wins + 0.5 * draws) / num_games
    expected_score = 1 / (1 + 10 ** ((stockfish_rating - 1500) / 400))  # Assuming initial rating of 1500 for model
    new_rating = 1500 + K * (score - expected_score)

    print(f"Estimated Elo rating for the model: {new_rating}")

# Function to play a game between two Stockfish instances
def play_game_stockfish_vs_stockfish(stockfish_path, stockfish_level=1):
    board = chess.Board()
    stockfish1 = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    stockfish2 = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    stockfish1.configure({"Skill Level": stockfish_level})
    stockfish2.configure({"Skill Level": stockfish_level})

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            result = stockfish1.play(board, chess.engine.Limit(time=0.1))
        else:
            result = stockfish2.play(board, chess.engine.Limit(time=0.1))
        board.push(result.move)
    
    stockfish1.quit()
    stockfish2.quit()
    return board.result()

# Function to calculate Elo rating based on game results
def calculate_stockfish_elo(stockfish_path, num_games=50, stockfish_level=1):
    wins, losses, draws = 0, 0, 0

    for _ in range(num_games):
        result = play_game_stockfish_vs_stockfish(stockfish_path, stockfish_level)
        if result == '1-0':
            wins += 1
        elif result == '0-1':
            losses += 1
        else:
            draws += 1
    
    print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")

    # Basic Elo calculation assuming Stockfish has a fixed rating (e.g., 2400 at level 1)
    stockfish_rating = 2400
    K = 32  # K-factor in Elo rating calculation
    score = (wins + 0.5 * draws) / num_games
    expected_score = 1 / (1 + 10 ** ((stockfish_rating - 2400) / 400))  # Assuming Stockfish plays against itself
    new_rating = 2400 + K * (score - expected_score)

    print(f"Estimated Elo rating for 2400 rated Stockfish as a verification: {new_rating}")

# Function to calculate and print statistics
def calculate_and_print_stats(df):
    # Calculate overall statistics
    total_games = len(df)
    win_rate = len(df[df['Result'] == 'win']) / total_games
    draw_rate = len(df[df['Result'] == 'draw']) / total_games
    loss_rate = len(df[df['Result'] == 'lose']) / total_games

    # Calculate statistics by color
    white_games = df[df['Color'] == 'white']
    black_games = df[df['Color'] == 'black']

    white_win_rate = len(white_games[white_games['Result'] == 'win']) / len(white_games)
    white_draw_rate = len(white_games[white_games['Result'] == 'draw']) / len(white_games)
    white_loss_rate = len(white_games[white_games['Result'] == 'lose']) / len(white_games)

    black_win_rate = len(black_games[black_games['Result'] == 'win']) / len(black_games)
    black_draw_rate = len(black_games[black_games['Result'] == 'draw']) / len(black_games)
    black_loss_rate = len(black_games[black_games['Result'] == 'lose']) / len(black_games)

    # Print overall statistics
    print(f"Total games: {total_games}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Draw rate: {draw_rate:.2%}")
    print(f"Loss rate: {loss_rate:.2%}")

    # Print statistics by color
    print("\nAs White:")
    print(f"Total games as White: {len(white_games)}")
    print(f"Win rate: {white_win_rate:.2%}")
    print(f"Draw rate: {white_draw_rate:.2%}")
    print(f"Loss rate: {white_loss_rate:.2%}")

    print("\nAs Black:")
    print(f"Total games as Black: {len(black_games)}")
    print(f"Win rate: {black_win_rate:.2%}")
    print(f"Draw rate: {black_draw_rate:.2%}")
    print(f"Loss rate: {black_loss_rate:.2%}")

    return (win_rate, draw_rate, loss_rate, white_win_rate, white_draw_rate, white_loss_rate, black_win_rate, black_draw_rate, black_loss_rate)
# Hikaru Chess Model

This project builds a chess model based on Hikaru Nakamura's games. The model can analyze and predict moves and has an estimated Elo rating of 1500, verified by Stockfish. Users can customize this model to include games from other players, combinations of players, or any other custom data.

## Features

- Fetches and processes chess games from Chess.com.
- Trains a neural network to predict moves based on the processed data.
- Evaluates the trained model using Stockfish to estimate its Elo rating.
- Provides functionality to customize the model for different players.

## Prerequisites

- Python 3.6 or higher
- Git
- TensorFlow
- Stockfish (not included in the repo)

## Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/hikaru-chess-model.git
cd hikaru-chess-model
```

### Set Up the Environment

Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### Build Stockfish

Stockfish is not included in the repository. You can build it from source:

1. Download the source code from the [Stockfish GitHub page](https://github.com/official-stockfish/Stockfish).
2. Follow the instructions for your platform to build the executable.

For macOS (Apple Silicon):

```bash
cd stockfish-master/src
make -j profile-build COMP=clang ARCH=apple-silicon
```

For other platforms, refer to the [Stockfish Wiki](https://github.com/official-stockfish/Stockfish/wiki/Compiling-from-source).

### Configuration

Ensure the path to the Stockfish executable is correctly set in your Jupyter notebook or Python script:

```python
stockfish_path = "/path/to/your/stockfish/stockfish"
```

## Usage

### Fetch and Process Games

Run the provided Jupyter notebook to fetch, process, and analyze the chess games.

### Training the Model

The notebook includes steps to train a neural network based on Hikaru Nakamura's games. The model can be trained with custom data by modifying the fetching and processing steps.

### Evaluating the Model

The model is evaluated using Stockfish to estimate its Elo rating. You can use the provided functions to play games against Stockfish and calculate the Elo rating.

## Customization

You can customize the model by modifying the data fetching, processing, and training steps.
Try fetching games from different players, combining games from multiple players, or training a custom model based on your own dataset.

### Using Different Players

To customize the model for different players, modify the `archives_url` to fetch games from a different Chess.com player:

```python
archives_url = "https://api.chess.com/pub/player/another_player/games/archives"
```

### Combining Games

You can combine games from multiple players by fetching and processing their archives separately and then merging the datasets.

### Training a Custom Model

Use the provided functions and notebook to train a custom model based on your dataset. Adjust the neural network architecture and training parameters as needed.

## Model Evaluation

The current V2 model is estimated to have an Elo rating of 1500, verified through matches against Stockfish. Users can build their own version of Stockfish and test their custom models.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome!

## Acknowledgments

- Thanks to Chess.com for providing the game data.
- Thanks to the Stockfish developers for their powerful chess engine.
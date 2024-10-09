import pandas as pd
import numpy as np
import chess
import chess.pgn
import os




def bitmap_encoding(board):
    """ Convert a chess.Board to a NumPy array with piece representations. """
    numpy_board = np.zeros(shape=(8,8))#np.full((8, 8), 0, dtype=object)  # Start with an empty board filled with 0

    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        piece_symbol = 1 if piece.symbol()==piece.symbol().lower() else -1  # fill pieces position with 1 and -1
        numpy_board[row][col] = piece_symbol

    return numpy_board.reshape(-1)


def get_game_matrix(game, num_moves=50):
    numpy_game = np.zeros(shape=(num_moves, 64))  # np.full((num_moves,8, 8), 0, dtype=object)

    board = game.board()

    numpy_game[0] = bitmap_encoding(board)

    for num, move in enumerate(game.mainline_moves()):
        if num + 1 == num_moves:
            break
        board.push(move)
        numpy_game[num + 1] = bitmap_encoding(board)

    return numpy_game


def encoded_all_game(dir_path, save_dir_path, memmap_shape=(300000, 64, 64)):
    pgn_files = os.listdir(dir_path)
    games_tensor = np.memmap(os.path.join(save_dir_path,"games_tensor.dat"), dtype='float64', mode='w+', shape=memmap_shape)
    current_index = 0
    target = np.zeros(memmap_shape[0])

    print("encoding ......")

    for num, pgn_file in enumerate(pgn_files):

        with open(os.path.join(dir_path, pgn_file)) as PGN_file:

            while True:
                game = chess.pgn.read_game(PGN_file)

                if game is None:
                    break

                if game.headers["Termination"] in ['Abandoned', 'Rules infraction']:
                    continue

                result = str(game.headers["Result"])
                if result == '0-1':
                    target[current_index] = 0

                elif result == '1-0':
                    target[current_index] = 1

                else:
                    target[current_index] = 2

                games_tensor[current_index] = get_game_matrix(game, num_moves=memmap_shape[1])

                current_index += 1

                if not current_index % 100000:
                    print(f"{current_index}/{memmap_shape[0]} is done")
                    print("encoding ......")

                if current_index >= memmap_shape[0]:
                    break

        if current_index >= memmap_shape[0]:
            break

    print("encoding is done!")
    games_tensor.flush()
    np.save(os.path.join(save_dir_path,"target.npy"), target)


def split_data(save_dir_path,random_seed=42,shape=(300000, 64, 64)):

    print("spliting data process started .....")
    np.random.seed(random_seed)
    games_tensor = np.memmap(os.path.join(save_dir_path,"games_tensor.dat"), dtype='float64', mode='r', shape=shape)
    target = np.load(os.path.join(save_dir_path,"target.npy"))

    indexes = np.arange(games_tensor.shape[0])
    np.random.shuffle(indexes)

    train_index = indexes[:int(games_tensor.shape[0] * 0.8)]
    temp_indexes = indexes[int(games_tensor.shape[0] * 0.8):]
    val_index = temp_indexes[:temp_indexes.shape[0] // 2]
    test_index = temp_indexes[temp_indexes.shape[0] // 2:]

    train_games_tensor = np.memmap(os.path.join(save_dir_path,"train_games_tensor.dat"), dtype='float64', mode='w+',
                                   shape=(train_index.shape[0], shape[1], shape[2]))
    train_target = np.zeros(shape=(train_index.shape[0]))

    for index in range(train_games_tensor.shape[0]):
        train_games_tensor[index] = games_tensor[train_index[index]]
        train_target[index] = target[train_index[index]]

    train_games_tensor.flush()
    np.save(os.path.join(save_dir_path,"train_target.npy"), train_target)

    val_games_tensor = np.memmap(os.path.join(save_dir_path,"val_games_tensor.dat"), dtype='float64', mode='w+',
                                 shape=(val_index.shape[0], shape[1], shape[2]))
    val_target = np.zeros(shape=(val_index.shape[0]))

    for index in range(val_games_tensor.shape[0]):
        val_games_tensor[index] = games_tensor[val_index[index]]
        val_target[index] = target[val_index[index]]

    train_games_tensor.flush()
    np.save(os.path.join(save_dir_path,"val_target.npy"), val_target)

    test_games_tensor = np.memmap(os.path.join(save_dir_path,"test_games_tensor.dat"), dtype='float64', mode='w+',
                                  shape=(test_index.shape[0],shape[1], shape[2]))
    test_target = np.zeros(shape=(test_index.shape[0]))

    for index in range(val_games_tensor.shape[0]):
        test_games_tensor[index] = games_tensor[test_index[index]]
        test_target[index] = target[test_index[index]]

    train_games_tensor.flush()
    np.save(os.path.join(save_dir_path,"test_target.npy"), val_target)

    print("spliting is done!")

    with open(os.path.join(save_dir_path,"dataset_info.txt"), "w") as f:
        text = f"train shape {train_games_tensor.shape}"
        text += f"\n val shape {val_games_tensor.shape}"
        text += f"\n test shape {test_games_tensor.shape}"

        f.write(text)
        print(text)


def main(dir_Path,save_dir_path,random_seed=42,shape=(300000, 64, 64)):
    encoded_all_game(dir_path=dir_Path,save_dir_path=save_dir_path,memmap_shape=shape)
    split_data(save_dir_path=save_dir_path,random_seed=random_seed,shape=shape)

if __name__=="__main__":
    dir_Path = "../data/PGN"
    save_path = "../data/numpy_objects"
    random_seed=42
    shape=(300000, 64, 64)


    main(dir_Path, save_path, random_seed, shape)


import random
import os
import pandas as pd

def player(prev_play, opponent_history=[]):
    if not hasattr(player, "our_moves"):
        player.our_moves = []
        player.opp_moves = []

    opponent_history.append(prev_play)
    
    # Save opponent move if it's valid (not the first empty "")
    if prev_play in ["R", "P", "S"]:
        player.opp_moves.append(prev_play)

    move = random.choice(["R", "P", "S"])
    player.our_moves.append(move)

    # Print moves after 5 opponent moves
    if len(player.opp_moves) % 5 == 0 and len(player.opp_moves) >= 5:
        update_dataset(player.our_moves[:5], player.opp_moves)
        player.our_moves = player.our_moves[5:]
        player.opp_moves.clear()
    return move

def update_dataset(our, opp):

    #Label Encoding
    res = get_result(our, opp)
    our = encode_move(our)
    opp = encode_move(opp)

    #testing
    print("Updating dataset with our moves:", our, "and opponent moves:", opp, "and results:", res)

    # Dataset check
    if not os.path.exists("data.csv"):
        df = pd.DataFrame(columns=["our_move", "opp_move", "result", "opp_next_move"])
        df.to_csv("data.csv", index=False)

    df = pd.read_csv("data.csv")
    new_data = pd.DataFrame({
        "our_move": our,
        "opp_move": opp,
        "result": res,
        # "opp_next_move": opp[1:] + [""]  
        })
    
    # Append new data and save
    df = pd.concat([df, new_data], ignore_index=True)
    df["opp_next_move"] = df["opp_move"].shift(-1)
    df.to_csv("data.csv", index=False)

    pass


def encode_move(our):
    # Define encoding rules
    encoding_map = {'R': -1, 'S': 0, 'P': 1}
    
    # Apply encoding to each move in the list
    encoded_moves = [encoding_map[move] for move in our]
    
    return encoded_moves





def get_result(our, opp):
    win_map = {('R', 'S'), ('S', 'P'), ('P', 'R')}
    
    results = []

    for o, op in zip(our, opp):
        if o == op:
            results.append(0)  # Draw
        elif (o, op) in win_map:
            results.append(1)  # Win
        else:
            results.append(-1)  # Loss
    
    return results

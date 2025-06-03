import random
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pickle


def player(prev_play, opponent_history=[]):
    if not hasattr(player, "our_moves"):
        player.our_moves = []
        player.opp_moves = []


    ###############################---------------DAtaset Creation-----------------###############################
    # opponent_history.append(prev_play)

    # if prev_play in ["R", "P", "S"]:
    #     player.opp_moves.append(prev_play)
    

    # move = random.choice(["R", "P", "S"])
    # player.our_moves.append(move)

    # if len(player.opp_moves) % 5 == 0 and len(player.opp_moves) >= 5:
    #     update_dataset(player.our_moves[:5], player.opp_moves)
    #     player.our_moves = player.our_moves[5:]
    #     player.opp_moves.clear()
    # final_move = move




    ###############################---------------MODEL PLAY-----------------###############################
    # if not prev_play:
    #     move = random.choice(["R", "P", "S"])
    #     print("First Move:", move)
    #     player.our_moves.append(move)
    #     return move 
    # last_move = player.our_moves[-1]
    # final_move = ask_model(last_move, prev_play)




    ###############################---------------Algorithm PLAY-----------------###############################
    # final_move = counter_abbey(prev_play)
    final_move = abbey_evil(prev_play)




    return final_move

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



def cal_winRate(res):
    filtered_res = [r for r in res if r != 0]
    win_count = filtered_res.count(1)
    winrate = win_count / len(filtered_res) if filtered_res else 0
    return winrate



def train_model():
    df = pd.read_csv("data.csv")
    df = df.dropna()
    # df = df.drop_duplicates()
    X = df.iloc[:,0:-1]
    y = df.iloc[:,-1]
    # X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=43, test_size=.2)
    mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500)
# Train the model
    mlp.fit(X, y)
    return mlp

def decode_move(encoded_moves):
    # Define decoding rules
    decoding_map = {-1: 'R', 0: 'S', 1: 'P'}
    
    # Apply decoding to each encoded move in the list
    decoded_moves = [decoding_map[move] for move in encoded_moves]
    
    return decoded_moves




def ask_model(last_move, prev_play):
    inputs = []   
    result = get_result(last_move, prev_play)
    inputs.append(last_move)
    inputs.append(prev_play)
    inputs = encode_move(inputs)
    inputs.append(result[0])


    print("****************************",decode_move(inputs[:-1]), result)

    model = pickle.load(open("Rock_Scissor_Paper/Decision_RPC", 'rb'))
    move = model.predict(np.array(inputs).reshape(1, -1))[0]
    p_move = decode_move([move])[0]
    print("Predicted Move:", p_move)


    ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}
    final_move = ideal_response[p_move]
    print("Response:", final_move)
    player.our_moves.append(final_move)
    return final_move









def counter_abbey(prev_opponent_play,
                  my_history=[],
                  opponent_history=[],
                  cycle_position=[0]):
    
    # Initialize on first move
    if not prev_opponent_play:
        my_history.clear()
        opponent_history.clear()
        cycle_position[0] = 0
        return 'R'
    
    # Track histories
    opponent_history.append(prev_opponent_play)
    
    # Strategy 1: Anti-frequency analysis with controlled randomness
    # Create patterns that will mislead Abbey's frequency counter
    
    # Use a mixed strategy that changes behavior every few moves
    move_num = len(my_history)
    
    if move_num < 2:
        # Start with simple moves to establish initial pattern
        my_move = 'P'
    elif move_num < 10:
        # Create false patterns early to poison Abbey's data
        pattern = ['R', 'P', 'S', 'R', 'P', 'S', 'P', 'R']
        my_move = pattern[move_num - 2]
    else:
        # Main strategy: Predict what Abbey will predict and counter it
        
        # Simulate Abbey's prediction process
        if len(my_history) >= 2:
            last_two_mine = "".join(my_history[-2:])
            
            # Abbey looks at potential plays based on my last move
            my_last = my_history[-1]
            potential_plays = [
                my_last + "R",
                my_last + "P", 
                my_last + "S",
            ]
            
            # Simulate Abbey's frequency counting
            # We need to track what Abbey thinks our patterns are
            play_counts = {"RR": 0, "RP": 0, "RS": 0, "PR": 0, "PP": 0, "PS": 0, "SR": 0, "SP": 0, "SS": 0}
            
            # Count our actual patterns
            for i in range(len(my_history) - 1):
                pattern = my_history[i] + my_history[i + 1]
                if pattern in play_counts:
                    play_counts[pattern] += 1
            
            # Find what Abbey would predict
            sub_counts = {k: play_counts[k] for k in potential_plays if k in play_counts}
            
            if sub_counts and max(sub_counts.values()) > 0:
                abbey_prediction = max(sub_counts, key=sub_counts.get)[-1]
                
                # Abbey will play the counter to this prediction
                abbey_counters = {'P': 'S', 'R': 'P', 'S': 'R'}
                abbey_move = abbey_counters[abbey_prediction]
                
                # We counter Abbey's counter
                our_counter = {'S': 'R', 'P': 'S', 'R': 'P'}
                my_move = our_counter[abbey_move]
            else:
                # Fallback: use anti-pattern strategy
                cycle_moves = ['R', 'P', 'S', 'P', 'R', 'S']
                my_move = cycle_moves[cycle_position[0] % len(cycle_moves)]
                cycle_position[0] += 1
        else:
            my_move = 'S'
    
    # Add some controlled unpredictability to avoid being counter-predicted
    if move_num > 20 and move_num % 7 == 0:
        # Occasionally break pattern
        import random
        my_move = random.choice(['R', 'P', 'S'])
    
    my_history.append(my_move)
    return my_move



def abbey_evil(prev_opponent_play,
          opponent_history=[],
          play_order=[{
              "RR": 0,
              "RP": 0,
              "RS": 0,
              "PR": 0,
              "PP": 0,
              "PS": 0,
              "SR": 0,
              "SP": 0,
              "SS": 0,
          }]):

    if not prev_opponent_play:
        prev_opponent_play = 'R'
    opponent_history.append(prev_opponent_play)

    last_two = "".join(opponent_history[-2:])
    if len(last_two) == 2:
        play_order[0][last_two] += 1

    potential_plays = [
        prev_opponent_play + "R",
        prev_opponent_play + "P",
        prev_opponent_play + "S",
    ]

    sub_order = {
        k: play_order[0][k]
        for k in potential_plays if k in play_order[0]
    }

    prediction = max(sub_order, key=sub_order.get)[-1:]

    ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}
    return ideal_response[prediction]
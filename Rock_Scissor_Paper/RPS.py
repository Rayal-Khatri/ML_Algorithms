import random
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pickle
from collections import Counter

# maps a move to the move that beats it
COUNTER = {"R": "P", "P": "S", "S": "R"}


def player(prev_play, opponent_history=[]):
    if not hasattr(player, "our_moves"):
        player.our_moves = []
        player.opp_moves = []
        player.rounds_played = 0
        player.use_quincy = True
        player.counter = 0

    ###############################---------------DAtaset Creation-----------------###############################
    # if not prev_play:
    #     opponent_history.append("R")
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
    # final_move = abbey_evil(prev_play)
    # final_move = quincy_evil(prev_play)
    # final_move = kris_evil(prev_play)

    bot_name = detect_bot(opponent_history)
    # print(f"Detected Bot: {bot_name}")
    
    # Call the respective bot's strategy function
    if bot_name == "kris":
        final_move = kris_evil(prev_play)
    elif bot_name == "quincy":
        # print("Using Quincy Evil Strategy")
        final_move = quincy_evil(prev_play)
    elif bot_name == "mrugesh":
        final_move = kris_evil(prev_play)
    elif bot_name == "abbey":
        final_move = abbey_evil(prev_play)
    else:
        # If bot is unknown, use random move
        final_move = random.choice(["R", "P", "S"])
    
    player.our_moves.append(final_move)
    opponent_history.append(prev_play)
    
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





def detect_bot(your_history, opp_history):
    n = len(opp_history)
    if n < 12:
        return "Unknown"  # wait until at least 20 rounds

    # 1) Check Quincy: fixed 5‐move cycle ["R","R","P","P","S"]
    cycle = ["R", "R", "P", "P", "S"]
    is_quincy = True
    for i, move in enumerate(opp_history):
        if move != cycle[i % 5]:
            is_quincy = False
            break
    if is_quincy:
        print("Detected Quincy Bot")
        return "quincy"

    # 2) Check Kris: opp_move[i] == counter(your_move[i-1]) for all i ≥ 1
    is_kris = True
    for i in range(1, n):
        expected = COUNTER.get(your_history[i - 1], None)
        if opp_history[i] != expected:
            is_kris = False
            break
    if is_kris:
        print("Detected Kris Bot")
        return "kris"

    # 3) Check Mrugesh: for i ≥ 10,
    #    opp_move[i] == counter(most_frequent(your_history[i-10 : i]))
    is_mrugesh = True
    for i in range(10, n):
        window = your_history[i - 10 : i]
        most_common = Counter(window).most_common(1)[0][0]
        if opp_history[i] != COUNTER[most_common]:
            is_mrugesh = False
            break
    if is_mrugesh:
        print("Detected Mrugesh Bot")
        return "mrugesh"

    # 4) Otherwise treat as Abbey
    print("Detected Abbey Bot")
    return "abbey"






def get_result(our, opp):
    win_map = {('R', 'S'), ('S', 'P'), ('P', 'R')}
    
    results = []

    for o, op in zip(our, opp):
        if o == op:
            results.append(0)  # Draw
        elif (o == "P" and op == "R") or (o == "R" and op == "S") or (o == "S" and op == "P"):
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



def abbey_evil(prev_opponent_play,
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
 
                cycle_moves = ['R', 'P', 'S', 'P', 'R', 'S']
                my_move = cycle_moves[cycle_position[0] % len(cycle_moves)]
                cycle_position[0] += 1
        else:
            my_move = 'S'

    if move_num > 20 and move_num % 7 == 0:
        # Occasionally break pattern
        import random
        my_move = random.choice(['R', 'P', 'S'])
    
    my_history.append(my_move)
    return my_move




def quincy_evil(prev_play, history=[]):
    
    # The underlying pattern in quincy's bot
    base_pattern = ["R", "R", "P", "P", "S"]
    
    # Record the opponent's (quincy's) previous move if it's valid.
    if prev_play in ("R", "P", "S"):
        history.append(prev_play)
        
    # The round number here is the number of moves observed plus 
    # one for the move we are about to play.
    round_no = len(history) + 1

    # Try to determine the phase offset. For a given offset (0-4),
    # check if every observed move fits the pattern:
    # predicted_move at round i should be base_pattern[(i + offset) % 5].
    possible_offsets = []
    for offset in range(5):
        valid = True
        for i, move in enumerate(history, start=1):
            if move != base_pattern[(i + offset) % 5]:
                valid = False
                break
        if valid:
            possible_offsets.append(offset)
    
    if possible_offsets:
        # Use the first valid offset.
        offset = possible_offsets[0]
        # Predict opponent's next move.
        predicted_move = base_pattern[(round_no + offset) % 5]
    else:
        # Fallback: if we don't have a confident offset yet,
        # simply use the opponent's last move as a (naive) proxy for prediction.
        predicted_move = history[-1] if history else "R"
    
    # Determine our winning move:
    # Paper beats Rock, Scissors beats Paper, and Rock beats Scissors.
    counter_moves = {"R": "P", "P": "S", "S": "R"}
    return counter_moves[predicted_move]





def kris_evil(prev_opponent_play, my_last_move=[None]):
    if my_last_move[0] is None:
        my_last_move[0] = 'R'  
        return 'R' 
    
    kris_response = {'P': 'S', 'R': 'P', 'S': 'R'}
    kris_will_play = kris_response[my_last_move[0]]
    

    our_counter = {'P': 'S', 'R': 'P', 'S': 'R'}
    our_move = our_counter[kris_will_play]

    my_last_move[0] = our_move
    
    return our_move
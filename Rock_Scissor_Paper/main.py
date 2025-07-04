# This entrypoint file to be used in development. Start by reading README.md
from RPS_game import play, mrugesh, abbey, quincy, kris, human, random_player
from RPS import player
from unittest import main

play(player, quincy, 100)
play(player, abbey, 100)
play(player, kris, 100)
play(player, mrugesh, 100)

# Uncomment line below to play interactively against a bot: 
# play(human, player, 14, verbose=True)

# Uncomment line below to play against a bot that plays randomly:
# play(human, player, 11)



# Uncomment line below to run unit tests automatically
# main(module='test_module', exit=False)
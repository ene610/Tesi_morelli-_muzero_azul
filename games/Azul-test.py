import datetime
import os
import random
import gym
import numpy
import torch
import pandas as pd
import numpy as np
from .abstract_game import AbstractGame




class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (1, 15, 12)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(180))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 5  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "expert"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = True
        self.max_moves = 30  # Maximum number of moves if game is not finished before
        self.num_simulations = 10  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 100  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        
        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 1  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 2  # Number of channels in policy head
        self.resnet_fc_reward_layers = []  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = []  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = []  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 8
        self.fc_representation_layers = [512]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [512]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [512]  # Define the hidden layers in the reward network
        self.fc_value_layers = [512]  # Define the hidden layers in the value network
        self.fc_policy_layers = [512]  # Define the hidden layers in the policy network

        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 600 * 1000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 128  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 1  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.005  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 1000 

        ### Replay Buffer
        self.replay_buffer_size = 15000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 10  # Number of game moves to keep for every batch element
        self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
        self.PER = False  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = True

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Azul()

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """

        #TODO modifica per farlo entrare subito nel while
        pit = int(input(f"Enter the pit to play for the player {self.to_play()}: "))
        color = int(input(f"Enter the color to play for the player {self.to_play()}: "))
        row = int(input(f"Enter the row to play for the player {self.to_play()}: "))

        action_numerical = self.env.game.from_tuple_action_to_action(pit, color, row)
        
        while action_numerical not in [action_numerical for action in self.legal_actions()]:
            pit = int(input(f"Enter the pit to play for the player {self.to_play()}: "))
            color = int(input(f"Enter the color to play for the player {self.to_play()}: "))
            row = int(input(f"Enter the row to play for the player {self.to_play()}: "))
            action_numerical = self.env.game.from_tuple_action_to_action(pit ,color ,row)
        
        return int(action_numerical)
        

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        action_pit_choice, action_tile_type, action_column_choice = self.env.game.from_action_to_tuple_action(action_number)
        return f"Play-> pit: {action_pit_choice}, tile_color: {action_tile_type}, column : {action_column_choice}"
        #return f"Play column {action_number + 1}"

class Azul_game():

    def __init__(self):

        self.p1_score = 0
        self.p2_score = 0

        self.player_turn = "P1"
        self.initial_player = "P1"
        #refattorizza nome
        self.new_first_player = True

        self.board_p1 = np.zeros((5, 5), dtype=int)
        self.board_p2 = np.zeros((5, 5), dtype=int)

        self.rows_p1 = self.initialize_rows()
        self.rows_p2 = self.initialize_rows()

        self.penalty_row_p1 = [0, 0, 0, 0, 0, 0, 0]
        self.penalty_row_p2 = [0, 0, 0, 0, 0, 0, 0]

        self.create_drawing_pit()

        self.gameover = False
        self.is_done_phase = False

        self.inserted_tile_in_column_for_action = 0
        self.penality_for_action = 0


    def initialize_rows(self):

        first_row = np.zeros(1, dtype=int)
        second_row = np.zeros(2, dtype=int)
        third_row = np.zeros(3, dtype=int)
        fourth_row = np.zeros(4, dtype=int)
        fifth_row = np.zeros(5, dtype=int)

        # return [first_row , second_row , third_row , fourth_row , fifth_row]
        return [[0], [0, 0], [0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0, 0]]

    def create_drawing_pit(self):
        pit_collection = []

        #
        self.new_first_player = True
        self.player_turn = self.initial_player
        self.is_done_phase = False

        for i in range(5):
            pit = [0, 0, 0, 0, 0]

            for j in range(4):
                generated_tile_type = np.random.randint(0, 5)
                pit[generated_tile_type] = pit[generated_tile_type] + 1

            pit_collection.append(pit)

        discard_pit = [0, 0, 0, 0, 0]
        pit_collection.append(discard_pit)

        self.penalty_row_p1 = [0, 0, 0, 0, 0, 0, 0]
        self.penalty_row_p2 = [0, 0, 0, 0, 0, 0, 0]
        self.drawing_pit =  pit_collection

    def valid_move(self, player, pit_choice, tile_type, column_choice):

        drawed_tile = self.drawing_pit[pit_choice][tile_type]

        if(drawed_tile == 0):
            return False

        if (player == "P1"):
            rows = self.rows_p1
            scoreboard = self.board_p1
        else:
            rows = self.rows_p2
            scoreboard = self.board_p2
        if(column_choice != 5):
            position = ((tile_type + column_choice) % 5)
            if (scoreboard[column_choice][position] == 1):
                return False
            if (rows[column_choice][0] == 0 or rows[column_choice][0] == tile_type + 1):
                return True
            return False

        else: return True

    def take_tile_from_pit(self, pit_choice, tile_type, player):

        drawed_tiles = self.drawing_pit[pit_choice][tile_type]
        self.drawing_pit[pit_choice][tile_type] = 0

        if not self.new_first_player:
            self.new_first_player = False
            self.initial_player = player

        if (pit_choice != 5):
            for i in range(5):
                discarded_tile = self.drawing_pit[pit_choice][i]
                self.drawing_pit[pit_choice][i] = 0
                self.drawing_pit[5][i] = self.drawing_pit[5][i] + discarded_tile

        return drawed_tiles

    def insert_tiles_in_column(self, tile_type, drawed_tiles, column_choice, player):
        number_of_drawed_tile = drawed_tiles

        if (player == "P1"):
            rows = self.rows_p1
            scoreboard = self.board_p1
        else:
            rows = self.rows_p2
            scoreboard = self.board_p2

        if (column_choice == 5):
            self.insert_tiles_in_penalty_column(number_of_drawed_tile, player)
            return
        
        self.inserted_tile_in_column_for_action =  0
        
        for i in range(column_choice + 1):

                if (rows[column_choice][i] == 0 and number_of_drawed_tile > 0):
                   
                    rows[column_choice][i] = tile_type + 1
                    self.inserted_tile_in_column_for_action = self.inserted_tile_in_column_for_action + 1
                    number_of_drawed_tile = number_of_drawed_tile - 1

        #mette le rimanenti nella penality column
        self.insert_tiles_in_penalty_column(number_of_drawed_tile, player)

    def insert_tiles_in_penalty_column(self, number_of_tiles, player):
        
        if (player == "P1"):
            penality_row = self.penalty_row_p1
        else:
            penality_row = self.penalty_row_p2

        self.penality_for_action = 0

        #se si riempie la penality column allora vanno scartate le tessere
        for i in range(7):
            if (penality_row[i] == 0 and number_of_tiles > 0):
                penality_row[i] = 1

                if (i < 2):
                    self.penality_for_action = self.penality_for_action + 1                    
                elif (i < 5):
                    self.penality_for_action = self.penality_for_action + 2                    
                else:
                    self.penality_for_action = self.penality_for_action + 3
                    
                
                number_of_tiles = number_of_tiles - 1

    def play_turn(self, player, pit_choice, tile_type, column_choice):
        
        self.penality_for_action = 0

        #controlla se è una mossa valida
        if(self.valid_move(player, pit_choice, tile_type, column_choice)):

            drawed_tiles = self.take_tile_from_pit(pit_choice, tile_type,player)

            if(pit_choice == 5 and self.new_first_player):
                self.initial_player = player
                self.new_first_player = False
                #aggiunge 1 penalità
                self.insert_tiles_in_penalty_column(1, player)

            # inserisci tile nella colonna specificata del player

            if (column_choice != 5):
                    self.insert_tiles_in_column(tile_type, drawed_tiles, column_choice, player)
            else:
                # se 5 allora inserisci direttamente nella colonna penalità
                self.insert_tiles_in_penalty_column(drawed_tiles, player)

            if self.player_turn == "P1":
                self.player_turn = "P2"
            else:
                self.player_turn = "P1"

            return True

        return False

    def clear_row(self, index_row, player):

        if (player == "P1"):
            rows = self.rows_p1
        else:
            rows = self.rows_p2

        for i in range(index_row + 1):
            rows[index_row][i] = 0
        
    def add_tile_to_scoreboard(self, tile, player, index_row):

        if (player == "P1"):
            scoreboard = self.board_p1
        else:
            scoreboard = self.board_p2

        index_column = ((tile - 1 + index_row) % 5)
        scoreboard[index_row][index_column] = 1

        # cambia nome metodi in adiacent
        score_row = self.compute_row_point(index_row, index_column, player)
        score_column = self.compute_column_point(index_row, index_column, player)

        # 1 sta per l'inserimento della piastrella
        return 1 + score_row + score_column

    def compute_row_point(self, index_row, index_column, player):
        if (player == "P1"):
            scoreboard = self.board_p1
        else:
            scoreboard = self.board_p2
        score = 0
        i = 0
        for elem in scoreboard[index_row]:
            if (i < index_column):
                if (elem):
                    score = score + 1
                else:
                    score = 0
            else:
                if (elem):
                    score = score + 1
                else:
                    break
            i = i + 1
        return score - 1

    def compute_column_point(self, index_row, index_column, player):
        if (player == "P1"):
            scoreboard = self.board_p1
        else:
            scoreboard = self.board_p2
        point = 0
        i = 0
        for row in scoreboard:

            if (i < index_row):
                if (row[index_column]):
                    point = point + 1
                else:
                    point = 0
            else:
                if (row[index_column]):
                    point = point + 1
                else:
                    break

            i = i + 1
        return point - 1

    def calculate_penality(self, player):
        if (player == "P1"):
            penality_row = self.penalty_row_p1
        else:
            penality_row = self.penalty_row_p2
        penality = 0

        for i in range(7):
            if (penality_row[i]):
                if (i < 2):
                    penality = penality - 1
                elif (i < 5):
                    penality = penality - 2
                else:
                    penality = penality - 3

        self.update_score(player, penality)

    def update_score(self, player, points):

        if (player == "P1"):
            if (self.p1_score + points > 0):
                self.p1_score = self.p1_score + points
            else:
                self.p1_score = 0
        else:
            if (self.p2_score + points > 0):
                self.p2_score = self.p2_score + points
            else:
                self.p2_score = 0

    def calculate_score(self, player):
        
        # calcolo dello score
        if (player == "P1"):
            rows = self.rows_p1

        else:
            rows = self.rows_p2

        index_row = 0
        for row in rows:
            first_elem = row[0]
            completed_row = True

            for elem in row:
                if (elem != first_elem or elem == 0):
                    completed_row = False
                    break

            if (completed_row):
                self.clear_row(index_row, player)
                partial_score = self.add_tile_to_scoreboard(first_elem, player, index_row)
                self.update_score(player, partial_score)

            index_row = index_row + 1
        # calculate penality
        self.calculate_penality(player)    

    def valid_actions(self,player):
        valid_actions = []
        for i in range(6):
            for j in range(5):
                for k in range(6):
                    if(self.valid_move(player, i, j, k)):
                        valid_actions.append([i,j,k])
        return valid_actions

    def is_turn_done(self):
        for pit in self.drawing_pit:
            for tile_type in pit:
                if(tile_type != 0):
                    self.is_done_phase = False
                    return

        self.is_done_phase = True
        return

    def is_game_done(self):
        #controlla la board p1
        self.is_turn_done()
        if self.is_done_phase:
            for row in self.board_p1:
                completed_tiles_in_a_row = 0
                for tile in row:
                    completed_tiles_in_a_row = completed_tiles_in_a_row + tile
                if(completed_tiles_in_a_row == 5):
                    self.gameover = True
                    return

            # controlla la board p2
            for row in self.board_p2:
                completed_tiles_in_a_row = 0
                for tile in row:
                    completed_tiles_in_a_row = completed_tiles_in_a_row + tile
                if (completed_tiles_in_a_row == 5):
                    self.gameover = True
                    return

            self.gameover = False
            return
        else : return False

    def compute_final_points(self):

        def row_completed_score(scoreboard):

            row_completed = 0
            for row in scoreboard:
                n_tile_in_a_row = 0

                for tile in row :
                    if(tile == 1):
                        n_tile_in_a_row = n_tile_in_a_row +1
                if(n_tile_in_a_row == 5):
                    row_completed = row_completed + 1
            return row_completed * 2

        def column_completed_score(scoreboard):

            column_completed = 0
            cumulative = [0,0,0,0,0]

            for row in scoreboard:
                cumulative = cumulative + row

            for elem in cumulative:
                if(elem == 5):
                    column_completed = column_completed + 1

            return column_completed * 5

        def tile_completed_score(scoreboard):

            tile_completed = 0
            tile_array = [0,0,0,0,0]

            for i in range(5):
                for j in range(5):
                    if(scoreboard[i][j] == 1):
                        tile_array[(i+j) % 5] = tile_array[(i+j) % 5] + 1

            for elem in tile_array:
                if(elem == 5):
                    tile_completed = tile_completed + 1

            return tile_completed *7


        #calcola per P1
        scoreboard = self.board_p1
        self.p1_score = self.p1_score + row_completed_score(scoreboard) + column_completed_score(scoreboard) + tile_completed_score(scoreboard)
        #calcola per P2
        scoreboard = self.board_p2
        self.p2_score = self.p2_score + row_completed_score(scoreboard) + column_completed_score(scoreboard) + tile_completed_score(scoreboard)

    def action_analisys(self, player, tile_type, column_choice): # TODO refactor name (tipo analisi qualcosa)
        
        def compute_expected_row_points(row_array, column_choice, tile_type, board):

            expected_row_points = 0

            row_choice = board[column_choice]
            index_of_inserted_element = (tile_type + column_choice) % 5
            flag = False

            for i in range(5):
                if i == index_of_inserted_element:
                    expected_row_points = expected_row_points + 1
                else:
                    if row_choice[i] == 1 :
                        expected_row_points = expected_row_points + 1
                    else:
                        if flag : expected_row_points = 0    
                        else : expected_row_points
                             

            return expected_row_points

        def compute_expected_column_points(row_array, column_choice, tile_type, board):
            
            flag = False
            expected_column_points_for_action = 0

            for i in range(5):
                if i == column_choice:
                    expected_column_points_for_action = expected_column_points_for_action + 1
                    flag = True
                else : 
                    if board[i][tile_type] != 0:
                        expected_column_points_for_action = expected_column_points_for_action + 1
                    else:
                        #controlla che se la row  è piena e che quindi verrà considerata successivamente come piazzata
                        for tile in row_array[i]:
                            if tile == 0:
                                if not flag : expected_column_points_for_action = 0
                                else: return  expected_column_points_for_action
                            else : expected_column_points_for_action = expected_column_points_for_action + 1
            return expected_column_points_for_action
            
        def action_analisys_row(board, column_choice):
            count_tiles_in_row = 0

            for tile in board[column_choice]:
                if tile != 0:
                    count_tiles_in_row = count_tiles_in_row + 1

            return count_tiles_in_row

        def action_analisys_column(board, tile_type):
            
            cumulative = np.array([0,0,0,0,0])
            i = 1
            for row in board:
                np.array(row) * i
                cumulative = np.add(cumulative, row)
                i + 1
                    
            return cumulative[tile_type]       

        def action_analisys_color(board, tile_type):

            tile_completed = 0
            tile_array = [0, 0, 0, 0, 0]
            i = 1

            for i in range(5):
                for j in range(5):
                    if(board[i][j] == 1):
                        tile_array[(i+j) % 5] = tile_array[(i+j) % 5] + 1 * i
                i = i + 1

            return tile_array[tile_type]
            
        row, column, color, expected_row_points, expected_column_point = 0, 0, 0, 0, 0

        if player == "P1" :

            board = self.board_p1
            row_array = self.rows_p1

        else :

            board = self.board_p2
            row_array = self.rows_p2
        
        if column_choice == 5:
            return row, column, color, expected_row_points, expected_column_point
        
        count_tiles_in_column = 0

        for tile in row_array[column_choice]:
            if tile == (tile_type + 1):
                count_tiles_in_column = count_tiles_in_column + 1
        
        if count_tiles_in_column == (column_choice + 1):
            expected_row_points = compute_expected_row_points(row_array, column_choice, tile_type, board)
            expected_column_point = compute_expected_column_points(row_array, column_choice, tile_type, board)
       
        row = count_tiles_in_column + action_analisys_row(board, column_choice)
        column = count_tiles_in_column + action_analisys_column(board, tile_type)
        color = count_tiles_in_column + action_analisys_color(board, tile_type)

        return row, column, color, expected_row_points, expected_column_point
        
        def action_analisys_row(board, column_choice):
            count_tiles_in_row = 0

            for tile in board[column_choice]:
                if tile != 0:
                    count_tiles_in_row = count_tiles_in_row + 1

            return count_tiles_in_row

        def action_analisys_column(board, tile_type):
            
            cumulative = np.array([0,0,0,0,0])
            i = 1
            for row in board:
                np.array(row) * i
                cumulative = np.add(cumulative, row)
                i + 1
                    
            return cumulative[tile_type]

        def action_analisys_color(board, tile_type):

            tile_completed = 0
            tile_array = [0, 0, 0, 0, 0]
            i = 1

            for i in range(5):
                for j in range(5):
                    if(scoreboard[i][j] == 1):
                        tile_array[(i+j) % 5] = tile_array[(i+j) % 5] + 1 * i
                i = i + 1

            return tile_array[tile_type]

    def from_action_to_tuple_action(self,action):

        action_pit_choice = 0
        action_tile_type = 0
        action_column_choice = 0

        if action < 6 :
            action_pit_choice = action
            return action_pit_choice, action_tile_type, action_column_choice

        if action < 30 :
            action_pit_choice = action % 6
            action_tile_type = int(action / 6)
            return action_pit_choice, action_tile_type, action_column_choice

        action_pit_choice = action % 6
        action_tile_type = int(( action % (6 * 5)) / 6)
        action_column_choice = int(action / (6 * 5))
        
        return action_pit_choice, action_tile_type, action_column_choice

    def from_tuple_action_to_action(self,action_pit_choice ,action_tile_type ,action_column_choice):

        a = action_pit_choice
        b = action_tile_type
        c = action_column_choice

        return action_pit_choice + action_tile_type * 6 + action_column_choice * 6 * 5

    def print_table(self):

        print(f"P1:{self.p1_score}")
        print(self.board_p1)
        print(f"row_p1:{self.rows_p1}")
        print(f"penality:{self.penalty_row_p1}")
        print("=" * 20)
        print(f"P2:{self.p2_score}")
        print(self.board_p2)
        print(f"row_p2:{self.rows_p2}")
        print(f"penality:{self.penalty_row_p2}")
        print("=" * 20)
        print(self.drawing_pit)
        print("=" * 20)

    def game_to_string(self):

        board_str = ""
        board_str += f"P1:{self.p1_score}" + "\n"
        board_str += f"{self.board_p1}" + "\n"
        board_str += f"row_p1:{self.rows_p1}" + "\n"
        board_str += f"penality:{self.penalty_row_p1}" + "\n"
        board_str += "=" * 20 + "\n"
        board_str += f"P2:{self.p2_score}" + "\n"
        board_str += f"{self.board_p2}" + "\n"
        board_str += f"row_p2:{self.rows_p2}" + "\n"
        board_str += f"penality:{self.penalty_row_p2}" + "\n"
        board_str += "=" * 20 + "\n"
        board_str += f"{self.drawing_pit}" + "\n"
        board_str += "=" * 20 + "\n"

        return board_str

class Azul:

    def __init__(self):

        self.game = Azul_game()
        self.player = 0 if self.game.play_turn == "P1" else 1
        
        #ToDO board azul
        self.board = self.board_to_obs()

    def board_to_obs(self):

        # processo per p1 ----------------------------------------------
        lst_penality_p1_and_score = self.game.penalty_row_p1 + [0, self.game.p1_score]

        np_penality_p1_and_score = np.array(lst_penality_p1_and_score)
        np_penality_p1_and_score.resize(5, 2)
        pd_penality_p1_and_score = pd.DataFrame(np_penality_p1_and_score)

        pd_row_p1 = pd.DataFrame(self.game.rows_p1).fillna(0)
        row_p1_with_penality_and_score = pd.concat([pd_row_p1, pd_penality_p1_and_score], axis=1).to_numpy().astype(int)

        #stesso processo per p2 -----------------------------------------
        lst_penality_p2_and_score = self.game.penalty_row_p2 + [0, self.game.p2_score]

        np_penality_p2_and_score = np.array(lst_penality_p2_and_score)
        np_penality_p2_and_score.resize(5, 2)
        pd_penality_p2_and_score = pd.DataFrame(np_penality_p2_and_score)

        pd_row_p2 = pd.DataFrame(self.game.rows_p2).fillna(0)
        row_p2_with_penality_and_score = pd.concat([pd_row_p2, pd_penality_p2_and_score], axis=1).to_numpy().astype(int)

        ###------------------------------------------------------------####
        #common pit

        pd_drawing_pit = pd.DataFrame(self.game.drawing_pit)
        pd_drawing_pit_trapsposte = pd_drawing_pit.transpose()
        #pd_drawing_pit_trapsposte.loc[len(pd_drawing_pit_trapsposte)] = 0
        zero_col =[7,8,9,10,11,12]
        for colum in zero_col:
            pd_drawing_pit_trapsposte[colum] = 0
        complete_board_common_pit = pd_drawing_pit_trapsposte.to_numpy()
        

        #-------------------------#
        
        #complete_board = []
        complete_board_p1 = np.concatenate([self.game.board_p1,row_p1_with_penality_and_score], axis=1)
        complete_board_p2 = np.concatenate([self.game.board_p2,row_p2_with_penality_and_score], axis=1)

        complete_board_players =  np.concatenate([complete_board_p1,complete_board_p2], axis=0)
        #print(complete_board_common_pit)  
        #print(complete_board_players)   
        complete_board =  np.concatenate([complete_board_players,complete_board_common_pit], axis=0)
        #print(complete_board.shape)
        return [complete_board]

    def to_play(self):
        return 0 if self.game.player_turn == "P1" else 1

    def reset(self):

        if self.game.gameover :
            self.game = Azul_game()
        else :
            self.game.create_drawing_pit()
        
        self.game = Azul_game()
        #ToDO board azul 
        self.board = self.board_to_obs()
        self.player = 0 if self.game.play_turn == "P1" else 1 #cambia in to_play
        return self.get_observation()

    def step(self, action):
        
        #traduci azione da 14 a [1,2,3]
        action_pit_choice , action_tile_type, action_column_choice = self.game.from_action_to_tuple_action(action)
        
        #controlla azione sia valida
        #valid_move = self.game.valid_move("P1", action_pit_choice, action_tile_type, action_column_choice)

        #fai azione
        player = self.game.player_turn
        self.game.play_turn(player, action_pit_choice, action_tile_type, action_column_choice)

        row_analisys, column_analisys, color_analisys, expected_row_points, expected_column_point = self.game.action_analisys(player, action_tile_type, action_column_choice)

        self.game.is_turn_done()
        self.game.is_game_done()

        #controlla se è finito il turno
        #controlla se è finita la partita

        if(self.game.is_done_phase):

            self.game.calculate_score("P1")
            self.game.calculate_score("P2")

            if(self.game.gameover):
                self.game.compute_final_points()
    
        #reward
        #prossimo player

        done = self.have_winner()

        # if self.game.is_done_phase :
        #    reward = self.game.p1_score if self.game.play_turn == "P1" else self.game.p2_score
        # else : 
        #    reward = 0
        
        # penality = self.game.penality_for_action
        # column_complete_reward = expected_column_point
        # row_complete_reward = expected_row_points
        # placed_tile_reward = self.game.inserted_tile_in_column_for_action
        
        # #TODO fai tutto assieme
        # placed_tile_reward = placed_tile_reward / (penality + 1)
        
        # #TODO rifattorizza come objective
        # row_reward = 2 * row_analisys / (5 * (action_column_choice + 1))
        # column_reward = 5 * (column_analisys / 15)
        # color_reward = 7 * (color_analisys / 15)
        # reward =  placed_tile_reward + row_reward + column_reward + color_reward
        
        #eward = 10 - self.game.penality_for_action
        #nuovo reward --- da provare su 300000 ---
        #TODO cancella calculate Score sopra
        # # reward = 0
        # # remain_actions = 0

        # # for pit in self.game.drawing_pit:
        # #      for tile in pit :
        # #          if tile != 0 :
        # #              remain_actions = remain_actions + 1
        
        # # if remain_actions < 3 :

        # #     if self.game.player_turn == "P1":
        # #         self.game.calculate_score("P1")
        # #         reward =  self.game.p1_score
        # #     else:
        # #         self.game.calculate_score("P2")
        # #         reward =  self.game.p2_score
            
        # # #print(self.game.player_turn, self.game.is_done_phase,reward)

        

        ###########################################
        #reward riempimento celle

        if self.game.player_turn == "P1":
            placed_tile_reward = self.game.inserted_tile_in_column_for_action
            penality = self.game.penality_for_action
            reward = placed_tile_reward - penality
            
        else:
            reward = 0

        #reward = placed_tile_reward - penality
        ##########################
        self.player = self.to_play()

        #self.player = 1 if self.game.play_turn == "P1" else 0

        return self.get_observation(), reward, done

    def get_observation(self):  
        return self.board_to_obs()

    def legal_actions(self):
        
        legal = []
        legal_action_tuple = self.game.valid_actions("P1" if self.player == 0 else "P2")
        for elem in legal_action_tuple:
          action = self.game.from_tuple_action_to_action(elem[0],elem[1],elem[2]) 
          legal.append(action)
        return legal

    def have_winner(self):
        return self.game.is_done_phase

    def expert_action(self):

        #random_player
        actions = []
        actions = self.game.valid_actions("P2")
        random.shuffle(actions)
        tuple_action = actions.pop()
        action = self.game.from_tuple_action_to_action(tuple_action[0],tuple_action[1],tuple_action[2])

        return action

    def render(self):
        self.game.print_table()
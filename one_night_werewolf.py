import copy
from itertools import permutations
"""
This is a program for implementing cfr algorithms on the game One Night Werewolf. 
The game is a 3 player game, where each player is dealt one of the possible roles, and the remaining 2 roles are in the middle.
Players are either bad (werewolf), in which case they win if they are not voted off at the end, good in which case they win if they vote of the werewolf, or if there is no werewolf, if they dont vote anyone off.

The structure of the game is a tree of nodes, where each node (implicitly through its location in the tree) relates to a history of actions by every player up to the current point.
Each node also has 'hidden' states, which represent the information that the current player could have learnt at the start of the node (e.g. they know they were the troublemaker, but decided not to swap).

The game progresses in a series of sub-levels.
First, each player gets to decide if they are ready to vote, and if they have information they want to claim (4 possibilities).
Then, for each player who has information they want to claim, each player makes a claim about information they have (11 possibilities).
This then repeats MAX_LEVEL number of times. e.g. if MAX_LEVEL = 0, each player can only make one information claim. if MAX_LEVEL = 3 , then each player could make up to 4 information claims (which may be contradictory).

Finally, every player votes based on all the information everyone provided, which resolves the game.
"""
#TODO: Implement external sampling (dont fully recurse opponent nodes, just sample one of their actions randomly)
global ROLES, PLAYER_COUNT, MAX_LEVEL, STARTS, TOTAL_NODES, ADMIN_NODES
ROLES = ["werewolf", "villager", "troublemaker", "insomniac"] #2x villager
PLAYER_COUNT = 3 # players 0 1 and 2 
MAX_LEVEL = 0
STARTS = ["werewolf", "villager", "troublemaker no swap", "troublemaker swap", "insomniac insomniac", "insomniac werewolf", "insomniac villager"]
TOTAL_NODES = 0
ADMIN_NODES = 0 
class Node():
    def __init__(self, sequence, level, gs):
        global TOTAL_NODES
        self.gs = gs 
        self.sequence = sequence # [[Order of players],current location in order]
        self.level = level
        self.children = [] # children[i] = children if pre-game permutation x (e.g. dealt werewolf, troublmaker and swapped). Children[i][j] = [node, probability of going to node given pre-game permutation]
        TOTAL_NODES += 1
        if TOTAL_NODES % 10000 == 0:
            print(TOTAL_NODES)
    def gen_children(self):
        pass 
    def generate_correlations(self):
        """
        Root node function to deal with correlation between start states
        """
        pass
    def cfr(self, prev_grid = -1):
        """
        Perform one CFR traversal of this node.
        
        Computes:
        - (i,j,k)-state EVs: EV when current=i, next=j, after=k
        - I-state EVs: EV for current player's type i (marginalized over j,k)
        - Action EVs: EV of each action for each I-state
        
        Args:
            prev_grid: Frequency grid from parent node
        """    
        # Initialize at root
        if prev_grid == -1:
            prev_grid = self.generate_correlations()
        
        # ev_grid[0][i][j][k] = [(i,j,k)-state data]
        # ev_grid[0][i][j][k][0] = {action_node: [freq, ev]}
        # ev_grid[0][i][j][k][1] = (i,j,k)-state EV
        # ev_grid[0][i][j][k][2] = conditional prob from parent
        # ev_grid[1][i] = total frequency for I-state i (for normalization)
        ev_grid = [{}, {i:0 for i in STARTS}]

        # Propagate frequencies from parent to current node
        for start_idx, i in enumerate(STARTS):
            ev_grid[0][i] = {}
            for j in STARTS:
                ev_grid[0][i][j] = {}
                for k in STARTS:
                    # Get frequency of (i,j,k) state from parent
                    # Parent's (k,i,j) maps to current's (i,j,k) due to player rotation
                    conditional_prob = prev_grid[0][k][i][j][0][self][0]
                    
                    # Compute action frequencies for this (i,j,k) state
                    action_dict = {}
                    for action in self.children[start_idx]:
                        action_node = action[0]
                        strategy_prob = action[1]  # Current strategy for I-state i
                        
                        # Frequency = strategy_prob * incoming_frequency
                        freq = strategy_prob * conditional_prob
                        action_dict[action_node] = [freq, 0.0]  # [frequency, EV]
                    
                    ev_grid[0][i][j][k] = [
                        action_dict,
                        0.0,               # (i,j,k)-state EV (computed later)
                        conditional_prob   # Incoming frequency
                    ]
                    
                    # Accumulate total frequency for I-state i
                    ev_grid[1][i] += conditional_prob

        
        # Recurse on all children
        for action in self.children[0]:
            action[0].cfr(ev_grid)

        # Backpropagate: compute (i,j,k)-state EVs from children
        for i in ev_grid[0]:
            for j in ev_grid[0][i]:
                for k in ev_grid[0][i][j]:
                    ijk_ev = 0.0
                    total_freq = 0.0
                    
                    for action_node, action_data in ev_grid[0][i][j][k][0].items():
                        # Get child's EV (with rotated indices: i,j,k → j,k,i)
                        child_ev = action_node.ev_grid[0][j][k][i][1]
                        action_data[1] = child_ev
                        
                        # Accumulate weighted EV
                        freq = action_data[0]
                        ijk_ev += child_ev * freq
                        total_freq += freq
                    
                    # (i,j,k)-state EV is weighted average over actions
                    if total_freq > 0:
                        ev_grid[0][i][j][k][1] = ijk_ev / total_freq
        
        # Compute I-state EVs and action EVs (marginalized over j,k)
        strategy_evs = {}
        
        for i in STARTS:
            i_state_ev = 0.0
            action_evs = {action[0]: [0.0, 0.0] for action in self.children[0]}
            
            for j in STARTS:
                for k in STARTS:
                    if k not in ev_grid[0][i][j]:
                        continue
                    
                    for action_node, action_data in ev_grid[0][i][j][k][0].items():
                        freq = action_data[0]
                        action_ev = action_data[1]
                        
                        # Normalized frequency for I-state EV
                        if ev_grid[1][i] > 0:
                            normalized_freq = freq / ev_grid[1][i]
                        else:
                            normalized_freq = 0.0
                        
                        # Accumulate I-state EV (marginalized over j,k and actions)
                        i_state_ev += normalized_freq * action_ev
                        
                        # Accumulate action EV (marginalized over j,k only)
                        action_evs[action_node][0] += action_ev * freq
                        action_evs[action_node][1] += freq
            
            # Normalize action EVs
            for action_node in action_evs:
                if action_evs[action_node][1] > 0:
                    action_evs[action_node][0] /= action_evs[action_node][1]
            
            # Store: [I-state EV, {action_node: [action_EV, total_freq]}]
            strategy_evs[i] = [i_state_ev, action_evs]
        # Store results
        self.ev_grid = ev_grid
        self.strategy_evs = strategy_evs

    def mccfr(self):
        pass

class giveInfo(Node):
    """
    Making information claims (can either be true or lie)
    """
    def __init__(self, type, sequence, level, sub_level, gs):
        self.type = type
        self.sub_level = sub_level
        super().__init__( sequence, level, gs)

    def gen_children(self):
        possible = []
        for i in ROLES:
            possible.append("claim dealt "+i) # 4 

        # for i in ROLES:
        #     possible.append("claim is currently "+i) # This may be unneeded (should be logically deducible?)
        
        possible.append("retract dealt role claim")
        
        possible.append("claim troublemaker swap") # claiming troublemaker and to have swapped. Because 3 player, implicitly gives the players who were swapped
        possible.append("claim troublemaker no swap")
        # 7
        for i in ROLES:
            if i != "troublemaker":
                possible.append("claim insomniac "+i)
        # 10

        # for i in range(PLAYER_COUNT): #Not needed? deducible? 
        #     if i != self.sequence[0][self.sequence[1]]:
        #         possible.append("claim voting for "+str(i))
        # #12
        new_seq = copy.deepcopy(self.sequence)
        new_seq[1] += 1 
        new_level = self.level
        if new_seq[1] >= len(new_seq[0]): # this was the final node in the subsequence
            
            new_seq = [[0,1,2],0]
            new_gamestate = copy.deepcopy(self.gs)
            new_gamestate["has claim"][0] = 0
            new_gamestate["has claim"][1] = 0
            new_gamestate["has claim"][2] = 0
            new_level += 1 

            base_class = AdminAction
            if self.level > MAX_LEVEL:
                base_class = gameOver
        else:
            base_class = giveInfo
            new_gamestate = copy.deepcopy(self.gs)

        for i in possible:
            self.children.append(base_class(i, new_seq, new_level, self.sub_level + 1, new_gamestate))
        #11
        if len(self.children) > 1:
            children2 = [] 
            honesty_factor = 10 # incentivise initial probabilities to make you tell the truth
            for start in STARTS:
                start_children = []
                if " " in start:
                    dealt_role = start.split(" ")[0]
                else:
                    dealt_role = start 
                if dealt_role == "troublemaker":
                    h_factor = honesty_factor/2 # 2 potential claims which are true (dealt troublemaker always true, and either troublemaker swap or troublemaker no swap true)
                elif dealt_role == "insomniac":
                    h_factor = honesty_factor/2
                else:
                    h_factor = honesty_factor
                for i in self.children:
                    if i.type == "claim dealt " + dealt_role or i.type == "claim " + start:
                        likelihood = (1+h_factor)/(len(self.children) + honesty_factor) # rescale likelihoods 
                    else:
                        likelihood = 1/(len(self.children)+ honesty_factor)
                    start_children.append([i, likelihood])
                children2.append(start_children)
            self.children = children2

            for i in self.children[0]: # the node at self.children[0][0] = the node at self.children[1][0], just a different assigned probability
                i[0].gen_children() 
        else:
            self.children = [[self.children[0],1]]
            self.children[0][0].gen_children()
class terminalInfo(giveInfo):
    """
    Player has finished with the information they currently want to give, ready to concede to next player 
    """
    def __init__(self, type, sequence, level, sub_level, gs):
        super().__init__(type, sequence,  level, sub_level, gs)

    def gen_children(self):
        new_seq = copy.deepcopy(self.sequence)
        new_seq[1] += 1 
        if new_seq[1] >= len(new_seq[0]): # this was the final node in the subsequence
            if self.level > MAX_LEVEL:
                self.children = [[gameOver(self.sequence, self.level, self.sub_level, self.gs),1]]
            else:
                new_seq = [[0,1,2],0]
                new_gamestate = copy.deepcopy(self.gs)
                new_gamestate["has claim"][0] = 0
                new_gamestate["has claim"][1] = 0
                new_gamestate["has claim"][2] = 0
                self.children = [[AdminAction("finished info loop", new_seq, self.level + 1, 0, new_gamestate),1]]
        else:
            self.children = [[giveInfo("conceded to " + str(new_seq[0][new_seq[1]]), new_seq, self.level, 0, self.gs), 1]]
        
        
        self.children[0][0].gen_children()

class AdminAction(Node):
    """
    Actions which doesnt involve making explicit claims and where the order of execution (e.g. player 1 then 2 ...) shouldnt matter
    """
    def __init__(self, type, sequence, level, sub_level, gs):
        self.type = type
        self.sub_level = sub_level
        global ADMIN_NODES
        ADMIN_NODES += 1 
        print(ADMIN_NODES, "AN", level)
        super().__init__(sequence, level, gs)
    
    def gen_children(self):
        """possible.append("set happy to vote")"""
        #TODO: change this to options: Have claim + happy to vote. Have claim + ¬happy to vote. ¬Have claim + happy to vote. ¬Have claim + ¬happy to vote. All should have terminal admin as children
        #TODO: Get rid of terminal admin and just directly point to next player node 
        new_seq = copy.deepcopy(self.sequence)
        new_seq[1] += 1 
        if new_seq[1] >= len(new_seq[0]): # this was the final node in the subsequence
            for_seq = [] 
            for i in range(PLAYER_COUNT):
                if self.gs["has claim"][i] == 1:
                    for_seq.append(i)
            
            # first nodes where don't want to make claim
            new_game_state = copy.deepcopy(self.gs)
            new_game_state["has claim"][self.sequence[1]] = 0
            if len(for_seq) == 0:
                if self.level > MAX_LEVEL: # If done reached stack limit and nobody has claims then end
                    self.children.append(gameOver("end", self.sequence, self.level, self.sub_level, copy.deepcopy(new_game_state)))
                else:
                    new_seq = [[0,1,2],0] # ascending order default for admin actions as order shouldnt matter 
                    self.children.append(AdminAction("no claims; conceded to player 0",new_seq, self.level+1, 0, copy.deepcopy(new_game_state)))
            else:
                self.children = [giveInfo("claim loop started; conceded to player " + str(for_seq[0]), [for_seq, for_seq[0]], self.level+1, 0, self.gs)]

            # Now nodes where do want to make a claim 
            for_seq.append(2) # This must be player num 2
            new_game_state["has claim"][self.sequence[1]] = 1
            new_game_state["ready to vote"][self.sequence[1]] = 1 
            self.children.append(giveInfo("has claim and ready to vote. claim loop started; conceded to player " + str(for_seq[0]), [for_seq, for_seq[0]], self.level+1, 0, copy.deepcopy(new_game_state)))
            new_game_state["ready to vote"][self.sequence[1]] = 0
            self.children.append(giveInfo("has claim and not ready to vote. claim loop started; conceded to player " + str(for_seq[0]), [for_seq, for_seq[0]], self.level+1, 0, copy.deepcopy(new_game_state)))
        else:
            new_game_state = copy.deepcopy(self.gs)
            new_game_state["ready to vote"][self.sequence[1]] = 1 
            new_game_state["has claim"][self.sequence[1]] = 1
            self.children.append(AdminAction("set happy to vote, has claim", new_seq, self.level, self.sub_level, copy.deepcopy(new_game_state)))
            new_game_state["has claim"][self.sequence[1]] = 0
            self.children.append(AdminAction("set happy to vote, hasnt claim", new_seq, self.level, self.sub_level, copy.deepcopy(new_game_state)))

            
            new_game_state["ready to vote"][self.sequence[1]] = 0 
            new_game_state["has claim"][self.sequence[1]] = 1
            self.children.append(AdminAction("not happy to vote, has claim", new_seq, self.level, self.sub_level, copy.deepcopy(new_game_state)))
            new_game_state["has claim"][self.sequence[1]] = 0
            self.children.append(AdminAction("not happy to vote, hasnt claim", new_seq, self.level, self.sub_level, copy.deepcopy(new_game_state)))

            
        #4 
        children2 = [] 
        for start in STARTS:
            start_children = []
            for i in self.children:
                likelihood = 1/(len(self.children))
                start_children.append([i, likelihood])
            children2.append(start_children)
        self.children = children2

        for i in self.children[0]:
            i[0].gen_children() 
            

class vote(Node):
    def __init__(self, sequence, level, sub_level, gs, player, type = "started"):
        self.type = type
        self.player = player 
        super().__init__( sequence, level, gs)
    
    def gen_children(self):
        for i in range(PLAYER_COUNT):
            if i != self.sequence[1]:
                self.children.append(["player " + str(self.player) + " voted for "+str(i), i])
        
        children2 = [] 
        for start in STARTS:
            start_children = []
            for i in self.children:
                likelihood = 1/(len(self.children))
                start_children.append([i, likelihood])
            children2.append(start_children)
        self.children = children2

        return self.children

class gameOver(Node): 
    def __init__(self, type, sequence, level, sub_level, gs):
        self.type = type
        self.sub_level = sub_level
        super().__init__( sequence, level, gs)

        self.player0_decision = vote(self.sequence, self.level, self.sub_level, self.gs, 0).gen_children()
        self.player1_decision = vote(self.sequence, self.level, self.sub_level, self.gs, 1).gen_children()
        self.player2_decision = vote(self.sequence, self.level, self.sub_level, self.gs, 2).gen_children()
    
    def gen_children(self):
        return 
    
gs = {}
gs["has claim"] = [0,0,0]
gs["ready to vote"] = [0,0,0]
start = AdminAction("start", [[0,1,2],0], 0, 0, gs)
start.gen_children()

#4150000

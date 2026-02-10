import copy
from itertools import permutations
import GUI
import time
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
global ROLES, PLAYER_COUNT, MAX_LEVEL, STARTS, TOTAL_NODES, ADMIN_NODES, CALCULATED_EV_1 , CALCULATED_EV_2
ROLES = ["werewolf", "villager", "troublemaker", "insomniac"] #2x villager
PLAYER_COUNT = 3 # players 0 1 and 2 
MAX_LEVEL = 0
STARTS = ["werewolf", "villager", "troublemaker no swap", "troublemaker swap", "insomniac insomniac", "insomniac werewolf", "insomniac villager"]
TOTAL_NODES = 0
ADMIN_NODES = 0 
CALCULATED_EV_1 = 0
CALCULATED_EV_2 = 0
class Node():
    def __init__(self, sequence, level, gs, parent=None):
        global TOTAL_NODES
        self.parent = parent
        self.gs = gs 
        self.sequence = sequence # [[Order of players],current location in order]
        self.level = level
        self.children = [] # children[i] = children if pre-game permutation x (e.g. dealt werewolf, troublmaker and swapped). Children[i][j] = [node, probability of going to node given pre-game permutation]
        self.regrets = -1
        self.nodelocked = True
        TOTAL_NODES += 1
        if TOTAL_NODES % 10000 == 0:
            print(TOTAL_NODES)
    def gen_children(self):
        pass 

    def generate_correlations(self):
        """
        Generate initial probability distribution over (i,j,k) START configurations.
        
        Game setup: 5 roles total (werewolf, troublemaker, insomniac, villager, villager)
        3 roles dealt to players, 2 remain in center.
        
        Returns:
            grid[0][i][j][k][0][self][0] = P(i,j,k configuration)
        """
        
        grid = [{}, {i: 0 for i in STARTS}]
        total_prob = 0.0
        
        # First pass: compute raw probabilities
        for i in STARTS:
            grid[0][i] = {}
            for j in STARTS:
                grid[0][i][j] = {}
                for k in STARTS:
                    prob = self.get_config_probability(i, j, k)
                    
                    grid[0][i][j][k] = [
                        {self: [prob, 0.0]},
                        0.0,
                        prob
                    ]
                    total_prob += prob
        
        # Second pass: normalize probabilities to sum to 1
        if total_prob > 0:
            for i in STARTS:
                for j in STARTS:
                    for k in STARTS:
                        normalized_prob = grid[0][i][j][k][0][self][0] / total_prob
                        grid[0][i][j][k][0][self][0] = normalized_prob
                        grid[0][i][j][k][2] = normalized_prob
                        grid[1][i] += normalized_prob
        
        return grid

    def get_config_probability(self, i, j, k):
        """
        Compute probability of (i,j,k) START configuration.
        Returns 0 if invalid, 0.5 if 2 villagers, 1.0 otherwise.
        
        Args:
            i, j, k: START values for current, next, and after players
        
        Returns:
            Probability weight (0 if invalid)
        """
        
        roles = [i, j, k]
        
        # Extract dealt roles (what was originally dealt before night actions)
        dealt_roles = []
        for role in roles:
            if "insomniac" in role:
                dealt_roles.append("insomniac")
            elif "troublemaker" in role:
                dealt_roles.append("troublemaker")
            else:
                dealt_roles.append(role)
        
        # Count each dealt role
        role_counts = {}
        for role in dealt_roles:
            role_counts[role] = role_counts.get(role, 0) + 1
        
        # Rule: If 2 of the same non-villager role, probability = 0
        for role, count in role_counts.items():
            if role != "villager" and count > 1:
                return 0.0
        
        # Rule: At most 2 villagers (we only have 2 in the deck)
        if role_counts.get("villager", 0) > 2:
            return 0.0
        
        # Check troublemaker consistency
        has_troublemaker = "troublemaker" in dealt_roles
        swap_occurred = any("troublemaker swap" in role for role in roles)
        no_swap = any("troublemaker no swap" in role for role in roles)
        
        if has_troublemaker:
            # Must have exactly one swap status
            if not (swap_occurred or no_swap):
                return 0.0
            if swap_occurred and no_swap:
                return 0.0
        else:
            # Can't have swap info without troublemaker
            if swap_occurred or no_swap:
                return 0.0
        
        # Check insomniac consistency
        for idx, role in enumerate(roles):
            if "insomniac" in role:
                # Parse what insomniac observed
                parts = role.split()
                if len(parts) < 2:
                    return 0.0  # Must have observation
                
                observed = " ".join(parts[1:])
                
                if swap_occurred:
                    # Rule: if "troublemaker swap" in i,j,k and one is insomniac,
                    # then it must be "insomniac <the other non-troublemaker role>"
                    
                    # Find the other non-troublemaker role that was dealt
                    other_role = None
                    for idx2, r in enumerate(dealt_roles):
                        if idx2 != idx and r != "troublemaker" and r != "insomniac":
                            other_role = r
                            break
                    
                    if other_role is None:
                        return 0.0  # No valid swap target
                    
                    if observed != other_role:
                        return 0.0  # Insomniac should observe the swapped role
                
                else:
                    # Rule: else (no swap), if i,j,k is insomniac,
                    # then it must be "insomniac insomniac"
                    if observed != "insomniac":
                        return 0.0
        
        # Rule: If there is one villager, 2 combinations of that so twice as likely
        if role_counts.get("villager", 0) == 1:
            return 2
        
        # Otherwise, probability = 1.0 (will be normalized)
        return 1.0
    
    def main_cfr(self):
        self.get_evs()
        self.compute_regrets()
        self.update_strategy()

    def get_evs(self, prev_grid = -1):
        """
        Perform one CFR traversal of this node.
        
        Computes:
        - (i,j,k)-state EVs: EV when current=i, next=j, after=k
        - I-state EVs: EV for current player's type i (marginalized over j,k)
        - Action EVs: EV of each action for each I-state
        
        Args:
            prev_grid: Frequency grid from parent node
        """    
        
        global CALCULATED_EV_1, CALCULATED_EV_2
        # Initialize at root
        if prev_grid == -1:
            prev_grid = self.generate_correlations()
        
        # ev_grid[0][i][j][k] = [(i,j,k)-state data]
        # ev_grid[0][i][j][k][0] = {action_node: [freq, ev]}
        # ev_grid[0][i][j][k][1] = (i,j,k)-state EV
        # ev_grid[0][i][j][k][2] = conditional prob from parent
        # ev_grid[1][i] = total frequency for I-state i (for normalization)
        t1 = time.time_ns()
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
                        strategy_prob = action[1]  # Current strategy for I-state i (current action prob)
                        
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

        t2 = time.time_ns()-t1
        CALCULATED_EV_1 += 1 
        if CALCULATED_EV_1 % 1 == 0:
            print("Calculated evs 1", CALCULATED_EV_1)
            print("Node - first half", t2)
        # Recurse on all children
        for action in self.children[0]:
            action[0].get_evs(ev_grid)
        t1 = time.time_ns()
        # Backpropagate: compute (i,j,k)-state EVs from children
        for i in ev_grid[0]:
            for j in ev_grid[0][i]:
                for k in ev_grid[0][i][j]:
                    ijk_ev = 0.0
                    total_freq = 0.0
                    
                    bad = self.get_bad(i,j,k)
                    for action_node, action_data in ev_grid[0][i][j][k][0].items():
                        # Get child's EV (with rotated indices: i,j,k → j,k,i)
                        
                        child_ev = action_node.ev_grid[0][j][k][i][1]
                        if bad == 0: # This player is bad
                            child_ev = -child_ev
                        elif bad == 1: # The next player is bad, so EV = -EV of next player 
                            child_ev = -child_ev

                        action_data[1] = child_ev
                        
                        # Accumulate weighted EV
                        freq = action_data[0]
                        ijk_ev += child_ev * freq
                        total_freq += freq
                    
                    # (i,j,k)-state EV is weighted average over actions
                    if total_freq > 0:
                        ev_grid[0][i][j][k][1] = ijk_ev / total_freq # Strategy EV given IJK
        
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
        
        t2 = time.time_ns()-t1
        CALCULATED_EV_2 += 1 
        if CALCULATED_EV_2 % 1 == 0:
            print("Calculated evs 2", CALCULATED_EV_2)
            print("Node - second half", t2)

    def compute_regrets(self):
        if self.nodelocked:
            return
        if self.regrets == -1: #Initialise
            self.regrets = {} 
            for infoset in STARTS:
                self.regrets[infoset] = {} # [strategy cumulative regret, {each action cumulative regret}]
                for action in self.children[0]:
                    self.regrets[infoset][action[0]] = 0


        for infoset in STARTS:
            infoset_ev = self.strategy_evs[infoset][0]
            action_evs = self.strategy_evs[infoset][1]
            for action in self.children[0]:
                inst_regret = action_evs[action[0]][0] - infoset_ev
                self.regrets[infoset][action[0]] += inst_regret

        for child in self.children[0]:
            child[0].compute_regrets()

    def update_strategy(self):
        if self.nodelocked:
            return
        for index,infoset in enumerate(STARTS):
            pos_sum = 0 
            for action in self.children[index]:
                pos_sum += max(0,self.regrets[infoset][action[0]])

            if pos_sum > 0:
                for action in self.children[index]:
                    action[1] = max(0,self.regrets[infoset][action[0]])/pos_sum
            else:
                # Uniform strategy
                num_actions = len(self.children[index])
                for action in self.children[index]:
                    action[1] = 1.0 / num_actions
            

        
        for child in self.children[0]:
            child[0].update_strategy()

    def get_bad(self,i,j,k):
        trip = [i,j,k]
        if "werewolf" not in trip:
            return "none"
        
        if "troublemaker swap" in trip:
            for idx, role in enumerate(trip):
                if role != "werewolf" and role != "troublemaker swap":
                    return idx
        elif "werewolf" in trip:
            return trip.index("werewolf")
    
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
            honesty_factor = 1000 # incentivise initial probabilities to make you tell the truth
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
                self.children = [giveInfo("has no claim, claim loop started; conceded to player " + str(for_seq[0]), [copy.deepcopy(for_seq), 0], self.level+1, 0, self.gs)]

            # Now nodes where do want to make a claim 
            for_seq.append(2) # This must be player num 2
            new_game_state["has claim"][self.sequence[1]] = 1
            new_game_state["ready to vote"][self.sequence[1]] = 1 
            self.children.append(giveInfo("has claim and ready to vote. claim loop started; conceded to player " + str(for_seq[0]), [for_seq, 0], self.level+1, 0, copy.deepcopy(new_game_state)))
            new_game_state["ready to vote"][self.sequence[1]] = 0
            self.children.append(giveInfo("has claim and not ready to vote. claim loop started; conceded to player " + str(for_seq[0]), [for_seq, 0], self.level+1, 0, copy.deepcopy(new_game_state)))
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
                start_children.append([i[0], likelihood, i[1]])
            children2.append(start_children)
        self.children = children2

        return self.children

class gameOver(Node): 
    def __init__(self, type, sequence, level, sub_level, gs):
        self.type = type
        self.sub_level = sub_level
        super().__init__(sequence, level, gs)

        self.player0 = vote(self.sequence, self.level, self.sub_level, self.gs, 0)
        self.player1 = vote(self.sequence, self.level, self.sub_level, self.gs, 1)
        self.player2 = vote(self.sequence, self.level, self.sub_level, self.gs, 2)
        self.player_list = [self.player0, self.player1, self.player2]

        self.player0_decision = self.player0.gen_children()
        self.player1_decision = self.player1.gen_children()
        self.player2_decision = self.player2.gen_children()
    
    def gen_children(self):
        return 
    
    def get_evs(self, prev_grid):
        
        global CALCULATED_EV_1, CALCULATED_EV_2
        ev_grid = [{}, {i:0 for i in STARTS}]

        # Propagate frequencies from parent to current node
        t1 = time.time_ns()
        strategy_evs = {0:{}, 1:{}, 2:{}}

        for start_idx, i in enumerate(STARTS):
            if i not in strategy_evs[0]:
                strategy_evs[0][i] = [[0,0],{0:0,1:0}]
                        
            ev_grid[0][i] = {}
            for j in STARTS:
                if j not in strategy_evs[1]:
                    strategy_evs[1][j] = [[0,0],{0:0,1:0}]
                    
                ev_grid[0][i][j] = {}
                for k in STARTS:
                    if k not in strategy_evs[2]:
                        strategy_evs[2][k] = [[0,0],{0:0,1:0}]
                    conditional_prob = prev_grid[0][k][i][j][0][self][0]
                    prob = self.get_probs(i, j, k) #likelihood of each player being voted out given this i,j,k state 
                    bad = self.get_bad(i,j,k)
                    state_ev = 0 
                    
                    for voted_out in prob:
                        if voted_out == bad:
                            state_ev += prob[voted_out]
                        else:
                            state_ev -= prob[voted_out]
                    individual_vote_outcomes = []

                    # EVs for each action
                    for player_num in range(PLAYER_COUNT):
                        outcome = [] 
                        prob = self.get_probs(i,j,k,[player_num,1,0]) #likelihood of each player being voted out given this i,j,k state, and that player <player_num> votes for their smallest neighbour (used for EV if action is taken pure)
                        s_ev = 0 
                        for voted_out in prob:
                            if voted_out == bad:
                                s_ev += prob[voted_out]
                            else:
                                s_ev -= prob[voted_out]
                        outcome.append(s_ev*conditional_prob)
                        strategy_evs[player_num][[i,j,k][player_num]][1][0] += s_ev*conditional_prob

                        prob = self.get_probs(i,j,k,[player_num,0,1]) #likelihood of each player being voted out given this i,j,k state, and that player <player_num> votes for their smallest neighbour (used for EV if action is taken pure)
                        s_ev = 0
                        for voted_out in prob:
                            if voted_out == bad:
                                s_ev += prob[voted_out]
                            else:
                                s_ev -= prob[voted_out]
                        outcome.append(s_ev*conditional_prob)
                        strategy_evs[player_num][[i,j,k][player_num]][1][1] += s_ev*conditional_prob

                        individual_vote_outcomes.append(copy.deepcopy(outcome))

                    state_ev = state_ev*conditional_prob

                    strategy_evs[0][i][0][0] += state_ev
                    strategy_evs[0][i][0][1] += conditional_prob

                    strategy_evs[1][j][0][0] += state_ev
                    strategy_evs[1][j][0][1] += conditional_prob

                    strategy_evs[2][k][0][0] += state_ev
                    strategy_evs[2][k][0][1] += conditional_prob

                    ev_grid[0][i][j][k] = [copy.deepcopy(individual_vote_outcomes), state_ev, conditional_prob]

        self.ev_grid = ev_grid
        self.strategy_evs = strategy_evs
        t2 = time.time_ns()-t1
        print("gameover, ", t2)
        CALCULATED_EV_1 += 1 
        CALCULATED_EV_2 += 1 
        if CALCULATED_EV_1 % 10000 == 0:
            print("Calculated evs 1", CALCULATED_EV_1)
        if CALCULATED_EV_2 % 10000 == 0:
            print("Calculated evs 2", CALCULATED_EV_2)
    def compute_regrets(self):
        for player_num, player in enumerate(self.player_list):
            if player.regrets == -1: #Initialise
                player.regrets = {} 
                for infoset in STARTS:
                    player.regrets[infoset] = {} # [strategy cumulative regret, {each action cumulative regret}]
                    for idx, action in enumerate(player.children[0]):
                        player.regrets[infoset][idx] = 0 # [infoset][who to vote out]


            for infoset in STARTS:
                infoset_ev = self.strategy_evs[player_num][infoset][0]
                prob = infoset_ev[1]
                infoset_ev = infoset_ev[0]/prob # Normalise to conditional probs
                action_evs = self.strategy_evs[player_num][infoset][1]
                for idx, action in enumerate(player.children[0]):
                    inst_regret = action_evs[idx]/prob - infoset_ev
                    player.regrets[infoset][idx] += inst_regret

    def update_strategy(self):
        for player in self.player_list:
            player.update_strategy()


    def get_probs(self,i,j,k, modify=[-1,0,0]):
        probs = {0: 0.0, 1: 0.0, 2: 0.0, "none":0}
        
        votes0 = self.player0_decision[STARTS.index(i)]
        votes0 = [1,votes0[0][1], votes0[1][1]]
        votes1 = self.player1_decision[STARTS.index(j)]
        votes1 = [votes1[0][1], 1, votes1[1][1]]
        votes2 = self.player2_decision[STARTS.index(k)]
        votes2 = [votes2[0][1], votes2[1][1], 1]

        if modify[0] == 0:
            votes0 = [1,modify[1],modify[2]]
        if modify[0] == 1:
            votes1 = [modify[1],1,modify[2]]
        if modify[0] == 2:
            votes2 = [modify[1],modify[2],1]

        p_total = 0 
        for player in range(PLAYER_COUNT):
            probs[player] = votes0[player]*votes1[player]*votes2[player] # prob other 2 vote for this player. This player is set as voting for itself (even though that isnt the case) to make the maths easier and more general.
            p_total += probs[player]
        probs["none"] = 1-p_total
        return probs
    


def test1():

    gs = {}
    gs["has claim"] = [0,0,0]
    gs["ready to vote"] = [0,0,0]
    start = AdminAction("start", [[0,1,2],0], 0, 0, gs)
    grid = start.generate_correlations()
    base_p = grid[0]["werewolf"]["villager"]["villager"][2] # p
    p = grid[0]["werewolf"]["villager"]["insomniac insomniac"][2] # 2p
    if abs(p - 2*base_p) > 0.001:
        return False
    p = grid[0]["werewolf"]["villager"]["insomniac werewolf"][2] # 0
    if p != 0:
        return False
    p = grid[0]["werewolf"]["troublemaker swap"]["insomniac werewolf"][2] # p
    if abs(p - base_p) > 0.001:
        return False
    p = grid[0]["werewolf"]["werewolf"]["insomniac werewolf"][2] # 0
    if p != 0:
        return False
    p = grid[0]["villager"]["villager"]["insomniac werewolf"][2] # 0
    if p != 0:
        return False
    p = grid[0]["villager"]["villager"]["insomniac insomniac"][2] # p
    if abs(p - base_p) > 0.001:
        return False
    p = grid[0]["villager"]["troublemaker swap"]["insomniac villager"][2] # 2p
    if abs(p - 2*base_p) > 0.001:
        return False
    return True
    
print(test1())

gs = {}
gs["has claim"] = [0,0,0]
gs["ready to vote"] = [0,0,0]
start = AdminAction("start", [[0,1,2],0], 0, 0, gs)
start.gen_children()
start.get_evs()
# for i in range(100):
#     start.main_cfr()
app = GUI.TreeViewer(start, STARTS)
app.mainloop()
#calculate ev - 32/s ~7000s = 10hrs
# Gameover:      16525900 to 62733100ns
# Node 1st half: 1507100 to 4019400
# Node 2nd half: 6348300 to 22617100
# ~~ .01s average node

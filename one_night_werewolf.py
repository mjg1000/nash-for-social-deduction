import copy
from itertools import permutations
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

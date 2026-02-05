global ROLES
ROLES = ["werewolf", "villager", "troublemaker", "insomniac"]
class Node():
    def __init__(self, player_num, level):
        self.type = "null"
        self.player_num = player_num
        self.level = level
        self.children = [] # children[i] = children if was dealt role x. Children[i][j] = [node, probability]

    def gen_children(self):
        pass 

class giveInfo(Node):
    def __init__(self, player_num, level, sub_level):
        self.sub_level = sub_level
        super.__init__(self, player_num, level)

    @classmethod
    def createChild(cls, type, sub_type = "-1", terminal=False):
        baseClass = terminalInfo if terminal else giveInfo

        if type == "claim dealt":
            if sub_type == ""
class terminalInfo(giveInfo):
    def __init__(self, player_num, level, sub_level):
        super.__init__(self, player_num, level, sub_level)
def createInfoChild():

class AdminAction(Node):
    pass

class terminalAdmin(AdminAction):
    pass

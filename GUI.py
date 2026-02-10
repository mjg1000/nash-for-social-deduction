import tkinter as tk
from tkinter import ttk
import json
global STATES 
STATES = ["good", "bad", "medium"]
class Node:
    def __init__(self, t, data, parent=None):
        self.type = t
        self.data = data
        self.parent = parent
        self.children = [[] for i in STATES]

    def add_child(self, child):
        child.parent = self
        for idx,i in enumerate(STATES):
            self.children[idx].append([child, 1])
def build_sample_tree():
    root = Node("root", {"name": "Root node"})

    a = Node("a", {"value": 10, "v2":10})
    b = Node("b", {"value": 20, "v2":10})
    c = Node("c", {"value": 30, "v2":10})

    root.add_child(a)
    root.add_child(b)
    a.add_child(c)

    return root
import tkinter as tk
from tkinter import ttk
import json
class TreeViewer(tk.Tk):
    def __init__(self, root_node, states=STATES):
        super().__init__()

        self.title("Tree Viewer")
        self.geometry("600x500")
        
        self.current_node = root_node

        # Configure root grid
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Node title
        self.node_label = ttk.Label(self, font=("Arial", 14, "bold"))
        self.node_label.grid(row=0, column=0, pady=10, sticky="n")

        # Data box
        self.data_table = ttk.Treeview(
            self,
            columns=("key", "value"),
            show="headings"
        )
        self.data_table.heading("key", text="Key")
        self.data_table.heading("value", text="Value")

        self.data_table.grid(
            row=1, column=0,
            padx=10, pady=5,
            sticky="nsew"
        )

        # Make columns resize
        self.data_table.column("key", width=150, anchor="w")
        self.data_table.column("value", width=300, anchor="w")



        # Navigation frame
        self.button_frame = ttk.Frame(self)
        self.button_frame.grid(row=2, column=0, pady=10, sticky="ew")

        self.button_frame.grid_columnconfigure(0, weight=1)
        
        self.states = states

        self.render_node()

    def render_node(self):
        node = self.current_node

        self.node_label.config(text=f"Node: {node.type}")

        # self.data_box.delete("1.0", tk.END)
        # self.data_box.insert(tk.END, json.dumps(node.data, indent=2))
        # Clear table
        for row in self.data_table.get_children():
            self.data_table.delete(row)
        
        if node.children == []:
            columns = ["voted for " + str(i) for i in range(2)]
        else:
            columns = [i[0].type for i in node.children[0]]
        columns.insert(0,"State")
        self.data_table["columns"] = columns
        
        for col in columns:
            self.data_table.heading(col, text=col)
            self.data_table.column(col, anchor="w", stretch=True)

        # Insert rows
        if node.children == []: # gameover 
            for p_idx, player in enumerate(node.player_list):
                for idx, child_list in enumerate(player.children):
                    vals = [self.states[idx]+" " + str(p_idx)]
                    for child in child_list:
                        vals.append(child[1])

                    self.data_table.insert("", "end", values=vals)
        else:
            for idx, child_list in enumerate(node.children):
                vals = [self.states[idx]]
                for child in child_list:
                    vals.append(child[1])

                self.data_table.insert("", "end", values=vals)

        # Clear navigation buttons
        for widget in self.button_frame.winfo_children():
            widget.destroy()

        row = 0

        if node.parent:
            ttk.Button(
                self.button_frame,
                text="⬅ Parent",
                command=lambda: self.go_to(node.parent)
            ).grid(row=row, column=0, pady=2, sticky="w")
            row += 1

        ttk.Label(self.button_frame, text="Children").grid(
            row=row, column=0, pady=5, sticky="w"
        )
        row += 1

        if node.children != []:
            for child2 in node.children[0]:
                child = child2[0]
                ttk.Button(
                    self.button_frame,
                    text=child.type,
                    command=lambda c=child: self.go_to(c, self.current_node)
                ).grid(row=row, column=0, pady=2, sticky="w")
                row += 1

    def go_to(self, node, parent=None):
        if parent:
            node.parent = parent
        self.current_node = node
        self.render_node()


if __name__ == "__main__":
    root = build_sample_tree()
    app = TreeViewer(root)
    app.mainloop()

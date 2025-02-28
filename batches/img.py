import graphviz

# Create a new directed graph with JPEG output format.
dot = graphviz.Digraph(comment='Neuromodulation Process', format='jpeg')
dot.attr(rankdir='TB', size="8,5")  # Top-to-Bottom layout

# Node 1: Memorized Tensors
dot.node('A', 'Model Memorizes:\nAvg. W Change\nTop-k Ranking', shape='box')

# Node 2: Training Sample Received
dot.node('B', 'Training Sample', shape='box')
dot.edge('A', 'B', label='Start Training')

# Node 3: Update Tensors & Check Conditions
dot.node('C', 'Update Tensors\n& Check Conditions', shape='box')
dot.edge('B', 'C')

# Decision Node: Are conditions satisfied?
dot.node('D', 'Top-k?\n& Exceeds Avg?', shape='diamond')
dot.edge('C', 'D')

# Node 4A: Neuromodulation (if Yes)
dot.node('E', 'Neuromodulate:\nReduce LR (Top-k)\nIncrease LR (Others)', shape='box')
dot.edge('D', 'E', label='Yes')

# Node 4B: No Neuromodulation (if No)
dot.node('G', 'No Neuromodulation', shape='box')
dot.edge('D', 'G', label='No')

# Node 5: Apply Neuromodulation / Proceed Training (merging both branches)
dot.node('F', 'Apply Neuromodulation\nProceed Training', shape='box')
dot.edge('E', 'F')
dot.edge('G', 'F')

# Render the diagram to a JPEG file and view it.
dot.render('neuromodulation_process_diagram', view=True)

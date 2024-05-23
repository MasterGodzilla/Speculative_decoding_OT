from eagle.model.choices import *
from eagle.model.utils_c import *

tree = binary5

tree_buffer = generate_tree_buffers(tree)

print (tree_buffer)
# latest quad tree is here
import matplotlib.pyplot as plt
from shapely.geometry.polygon import Polygon

# node class
class node:
    def __init__(self, bounding_box):
        self.bounding_box=bounding_box  # Bounding box for this node
        self.data=[]  # List of data points associated with this node
        self.children=[None, None, None, None]  # Four child nodes (NW, NE, SW, SE)
        self.bounding_boxes=[]
# quad tree class
class quadtree:
    def __init__(self, bounding_box, capacity):
        self.bounding_box = bounding_box  # Represents the overall spatial bounds
        self.capacity = capacity  # Maximum number of data points a node can hold before splitting
        # self.nodes = []  # Array to save all node objects
        self.root = None  # The root node of the quadtree

    # inserting a new node in the tree
    def insert(self, x, y, data):
        if self.root is None:
            self.root = node(self.bounding_box)
            # self.nodes.append(self.root)
        self._insert_recursive(self.root, x, y, data)

    def _insert_recursive(self, current_node, x, y, data):
        if current_node.children[0] is not None:
            for i in range(4):
                if (x >= current_node.children[i].bounding_box[0] and
                    x < current_node.children[i].bounding_box[2] and
                    y >= current_node.children[i].bounding_box[1] and
                    y < current_node.children[i].bounding_box[3]):
                    self._insert_recursive(current_node.children[i], x, y, data)
                    return

        current_node.data.append((x, y, data))
        if len(current_node.data) > self.capacity:
            self._split_node(current_node)

    def _split_node(self, current_node):
        x0, y0, x1, y1 = current_node.bounding_box
        x_mid = (x0 + x1) / 2
        y_mid = (y0 + y1) / 2

        current_node.children[0] = node([x0, y0, x_mid, y_mid])
        current_node.children[1] = node([x_mid, y0, x1, y_mid])
        current_node.children[2] = node([x0, y_mid, x_mid, y1])
        current_node.children[3] = node([x_mid, y_mid, x1, y1])

        for data_point in current_node.data:
            x, y, data = data_point
            for i in range(4):
                if (x >= current_node.children[i].bounding_box[0] and
                    x < current_node.children[i].bounding_box[2] and
                    y >= current_node.children[i].bounding_box[1] and
                    y < current_node.children[i].bounding_box[3]):
                    self._insert_recursive(current_node.children[i], x, y, data)
        current_node.data = []

        # Append the child nodes to the Quadtree's nodes array
        # self.nodes.extend(current_node.children)

    # save bounding boxes of the leaf level nodes. Depth first approach
    def get_all_bounding_boxes(self):
        boxes=[]
        levels=[]
        level=0
        self.levels=0
        if self.root:
            self._get_bounding_boxes_recursive(self.root, boxes, levels, level)
        self.bounding_boxes=boxes
        self.bounding_box_levels=levels

    def _get_bounding_boxes_recursive(self, current_node, boxes, levels, level):
        if current_node.children[0] is not None:
            # If the current node has children, traverse the children
            for child_index in [2, 3, 0, 1]:
                self._get_bounding_boxes_recursive(current_node.children[child_index], boxes, levels, level+1)
        # elif current_node.data:
        else:
            # If the current node is in leaf level, add its bounding box to the list
            boxes.append(current_node.bounding_box)
            levels.append(level)
            if self.levels<level:
                self.levels=level

    # Function to plot the Quadtree and data points with different colors for each level
    def _plot_quadtree(self, node, ax, level):
        if node is not None:
            x0, y0, x1, y1 = node.bounding_box
            colormap = plt.get_cmap("tab20")
            color = colormap(level*4)
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color=color))
            for data_point in node.data:
                x, y, data = data_point
                ax.plot(x, y, 'ro', markersize=5)
            for child in node.children:
                self._plot_quadtree(child, ax, level+1)
    
    # plot method
    def plot_quadtree(self):
        # Create a plot
        fig, ax = plt.subplots(figsize=(18, 18))
        # ax.set_xlim(0, 100)
        # ax.set_ylim(0, 100)
        self._plot_quadtree(self.root, ax, 0)

        # Display the plot
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("Quadtree with Data Points")
        plt.show()

    # convert the quad tree cells into a list of polyogns
    def convertQTToPolys(self):
        wkts=[]
        for i in range(len(self.bounding_boxes)):
            box=self.bounding_boxes[i]
            poly2=Polygon([(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])])
            wkts.append(poly2)
        return wkts
 
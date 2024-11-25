from typing import List, Dict, Any

class Node:
    def __init__(self, name: str, **params: Any):
        self.name = name
        self.children: List[Node] = []
        self.params = params
        self.params['weight_bounds'] = params.get('weight_bounds', (0, 1))

    def add_child(self, child_node) -> None:
        self.children.append(child_node)

    def __repr__(self):
        return f"Node({self.name}, weight_bounds={self.params['weight_bounds']})"


class Tree:
    def __init__(self, root_name: str):
        self.root = Node(root_name)
        self.nodes: Dict[str, Node] = {root_name: self.root}

    def insert(self, parent_name: str, child_name: str, **params: Any) -> bool:
        parent_node = self.nodes.get(parent_name)
        if parent_node:
            child_node = Node(child_name, **params)
            parent_node.add_child(child_node)
            self.nodes[child_name] = child_node 
            return True
        return False

    def draw(self) -> None:
        lines = self._build_tree_string(self.root, '')
        print('\n'.join(lines))

    def _build_tree_string(self, node: Node, prefix: str, is_tail: bool = True) -> List[str]:
        lines = [f"{prefix}{'`-- ' if is_tail else '|-- '}{node.name}"]
        prefix += '    ' if is_tail else '|   '
        child_count = len(node.children)
        for i, child in enumerate(node.children):
            is_last_child = i == child_count - 1
            lines.extend(self._build_tree_string(child, prefix, is_last_child))
        return lines

    def get_all_nodes(self) -> List[Node]:
        return list(self.nodes.values())
    
    def get_all_nodes_name(self) -> List:
        return [k.name for k in list(self.nodes.values())][1:]
    
    def get_leaf_nodes(self) -> List[str]:
        """리프 노드(자식이 없는 노드)만 이름으로 반환"""
        leaf_nodes = [node.name for node in self.nodes.values() if not node.children]
        return leaf_nodes
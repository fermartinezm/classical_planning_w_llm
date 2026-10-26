from typing import Set, List, Optional, FrozenSet

from task import Operator


class TreeNode:
    """A node in the successor generator tree."""
    def __init__(self):
        self.atom: Optional[str] = None
        self.yes_child: Optional[TreeNode] = None
        self.no_child: Optional[TreeNode] = None
        self.applicable_operators: List[Operator] = []

    def __repr__(self):
        return f"<TreeNode atom={self.atom}, ops={len(self.applicable_operators)}>"


class SuccessorGenerator:
    """
    A tree-based successor generator for efficiently finding applicable operators.
    """
    def __init__(self, operators: Optional[List[Operator]] = None):
        # The root node doesn't test any specific atom initially.
        # It acts as the entry point.
        self._root = TreeNode()
        for operator in operators or []:
            self.add_operator(operator)

    def add_operator(self, operator: Operator):
        self._insert_recursive(self._root, operator, set(operator.preconditions))

    def _insert_recursive(self, node: TreeNode, operator: Operator, remaining_preconditions: Set[str]) -> None:
        assert node

        # Base case: All preconditions processed, add operator to current node's list.
        if not remaining_preconditions:
            node.applicable_operators.append(operator)
            return

        # Case 1: Current node is empty (needs an atom test assigned)
        if node.atom is None:
            test_atom = remaining_preconditions.pop()
            node.atom = test_atom
            node.yes_child = TreeNode()
            node.no_child = TreeNode()
            self._insert_recursive(node.yes_child, operator, remaining_preconditions)

        # Case 2: Current node tests an atom in the precondition. Recurse down 'yes'.
        elif node.atom in remaining_preconditions:
            assert node.yes_child
            remaining_preconditions.remove(node.atom)
            self._insert_recursive(node.yes_child, operator, remaining_preconditions)

        # Case 3: The current node tests an atom that is not in the remaining preconditions.
        else:
            assert node.no_child
            self._insert_recursive(node.no_child, operator, remaining_preconditions)

    def get_applicable_operators(self, state: FrozenSet[str]) -> List[Operator]:
        #applicable_ops = []
        #self._collect_applicable_recursive(self._root, state, applicable_ops)
        #return applicable_ops
        return self.get_applicable_operators_iterative(state)

    def _collect_applicable_recursive(self, node: Optional[TreeNode], state: FrozenSet[str], result_list: List[Operator]):
        result_list.extend(node.applicable_operators)

        if node.atom in state and node.yes_child:
            self._collect_applicable_recursive(node.yes_child, state, result_list)
        if node.no_child:
            self._collect_applicable_recursive(node.no_child, state, result_list)

    def get_applicable_operators_iterative(self, state: FrozenSet[str]) -> List[Operator]:
        """
        Iteratively compute the list of applicable operators for the given state
        by traversing the tree using a stack.
        """
        applicable_ops = []
        stack: List[TreeNode] = []

        if self._root:
            stack.append(self._root)

        while stack:
            current_node = stack.pop()
            applicable_ops.extend(current_node.applicable_operators)

            if current_node.atom in state and current_node.yes_child:
                stack.append(current_node.yes_child)

            if current_node.no_child:
                stack.append(current_node.no_child)

        return applicable_ops

    def dump(self, node: Optional[TreeNode] = None, level: int = 0):
        if node is None:
            node = self._root
        indent = "  " * level
        print(f"{indent}Node: {node.atom}, Ops: {node.applicable_operators}")
        if node.yes_child:
            print(f"{indent}Yes Child:")
            self.dump(node.yes_child, level + 1)
        if node.no_child:
            print(f"{indent}No Child:")
            self.dump(node.no_child, level + 1)

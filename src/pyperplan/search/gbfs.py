import heapq
import logging

from task import Task


class SearchNode:
    __slots__ = ("state", "parent", "action")

    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action

    def extract_solution(self):
        solution = []
        while self.parent is not None:
            solution.append(self.action)
            self = self.parent
        solution.reverse()
        return solution


def make_root_node(initial_state):
    return SearchNode(initial_state, None, None)


def make_child_node(parent_node, action, state):
    return SearchNode(state, parent_node, action)


# Helper function for creating ordered nodes for the priority queue
def ordered_node_greedy_best_first(node, h, node_tiebreaker):
    """
    Creates an ordered search node for GBFS (order: h).
    Primary key is h, secondary is the tiebreaker.
    """
    return (h, node_tiebreaker, node)


def gbfs_early_goal_test(task: Task, heuristic):
    """
    Searches for a plan using a Greedy Best-First Search with an early goal test.

    This version is optimized for GBFS by:
    - Prioritizing nodes solely based on heuristic value 'h'.
    - Using a 'closed list' (set) to track visited states, preventing cycles
      and redundant expansions.
    - Not storing or comparing path costs ('g' values) for node selection.
    - Implementing an early goal test: checks if a successor node satisfies
      the goal condition immediately upon generation.

    @param task: The planning task instance.
    @param heuristic: A callable heuristic function (node -> h_value).
                      Must return float('inf') for dead ends.
    @returns: A solution path (list of operators) or None if unsolvable.
    """
    open_list = []  # Priority queue (min-heap)
    closed_list = set()  # Stores visited states to prevent cycles/re-expansion
    node_tiebreaker = 0  # Ensures FIFO for nodes with equal h-value

    root_node = make_root_node(task.initial_state)

    # Calculate initial heuristic value
    initial_h = heuristic(root_node)

    if initial_h == float("inf"):
        logging.info("Initial state is a dead end according to the heuristic.")
        return None # Cannot reach goal from initial state

    # Check if the initial state is the goal state (part of early goal testing)
    if task.goal_reached(root_node.state):
         logging.info("Initial state is a goal state.")
         return root_node.extract_solution() # Empty plan

    # Push the root node onto the open list
    heapq.heappush(open_list, ordered_node_greedy_best_first(root_node, initial_h, node_tiebreaker))
    logging.info(f"Initial h value: {initial_h:.2f}")

    best_h = initial_h # Track the best h value seen so far (lowest)
    max_open_size = 1
    expansions = 0

    def print_statistics():
        logging.info(f"{expansions} Nodes expanded")
        logging.info(f"Max open list size: {max_open_size}")

    while open_list:
        # Pop the node with the lowest h-value
        h, _tie, current_node = heapq.heappop(open_list)
        current_state = current_node.state
        #print(f"Expanding node with h={h:.2f} and state={current_state}")

        # Logging for progress
        if h < best_h:
            best_h = h
            logging.info(f"Found new best h: {best_h:.2f} after {expansions} expansions")

        # Check if state was already visited/expanded
        # This prevents cycles and redundant work on states already processed.
        if current_state in closed_list:
            continue # Skip if already expanded

        # Add state to the closed list *before* expansion
        closed_list.add(current_state)
        expansions += 1

        # --- Successor Generation ---
        for op, successor_state in task.get_successor_states(current_state):

            # Check if successor state has already been expanded.
            # If we find the goal early, we don't care if it was closed,
            # but for non-goal states, we avoid reprocessing closed ones.
            if successor_state in closed_list:
                continue # Skip if already expanded

            # Create the successor node object
            successor_node = make_child_node(current_node, op, successor_state)

            # Check if the newly generated successor state is a goal state.
            if task.goal_reached(successor_state):
                logging.info(f"Early goal reached via operator {op.name}.")
                # Note: The expansion count reflects nodes fully expanded *before*
                # finding this goal successor. The goal node itself isn't expanded.
                print_statistics()
                return successor_node.extract_solution()

            # Calculate heuristic for the successor
            h_successor = heuristic(successor_node)

            # Check for dead ends indicated by the heuristic
            if h_successor == float("inf"):
                logging.debug(f"Successor state via {op.name} is a dead end (h=inf). Skipping.")
                continue # Don't add dead ends to the open list

            # Add the valid, non-goal, non-closed successor to the open list
            node_tiebreaker += 1
            heapq.heappush(open_list, ordered_node_greedy_best_first(successor_node, h_successor, node_tiebreaker))

        max_open_size = max(max_open_size, len(open_list))

        # Optional: Log progress periodically
        # if expansions % 1000 == 0:
        #    logging.debug(f"Expansions: {expansions}, Open list size: {len(open_list)}, Best h: {best_h:.2f}")


    # Loop finished because open list is empty without finding the goal
    logging.info("Open list is empty. Task unsolvable.")
    print_statistics()
    return None

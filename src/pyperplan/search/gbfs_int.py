import heapq
import logging

from search import searchspace
from task import Task


# Helper function for creating ordered nodes for the priority queue
def ordered_node_greedy_best_first(node, h, node_tiebreaker):
    """
    Creates an ordered search node for GBFS (order: h).
    Primary key is h, secondary is the tiebreaker.
    """
    return (h, node_tiebreaker, node)


def gbfs_early_goal_test_int(task: Task, heuristic):
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
    facts = sorted(task.facts)
    atom_to_index = {atom: i for i, atom in enumerate(facts)}

    def encode_state(state):
        return frozenset(atom_to_index[atom] for atom in state)

    def decode_state(encoded_state):
        return frozenset(facts[id] for id in encoded_state)

    open_list = []  # Priority queue (min-heap)
    closed_list = set()  # Stores visited states to prevent cycles/re-expansion
    node_tiebreaker = 0  # Ensures FIFO for nodes with equal h-value

    root_node = searchspace.make_root_node(task.initial_state)

    # Calculate initial heuristic value
    initial_h = heuristic(root_node)

    if initial_h == float("inf"):
        logging.info("Initial state is a dead end according to the heuristic.")
        return None # Cannot reach goal from initial state

    # Check if the initial state is the goal state (part of early goal testing)
    if task.goal_reached(root_node.state):
         logging.info("Initial state is a goal state.")
         return root_node.extract_solution() # Empty plan

    root_node.state = encode_state(root_node.state) # Encode the initial state

    # Push the root node onto the open list
    heapq.heappush(open_list, ordered_node_greedy_best_first(root_node, initial_h, node_tiebreaker))
    logging.info(f"Initial h value: {initial_h:.2f}")

    best_h = initial_h # Track the best h value seen so far (lowest)
    expansions = 0

    while open_list:
        # Pop the node with the lowest h-value
        h, _tie, current_node = heapq.heappop(open_list)
        encoded_state = current_node.state
        decoded_state = decode_state(encoded_state)
        current_node.state = decoded_state # Decode the state for processing

        # Logging for progress
        if h < best_h:
            best_h = h
            logging.info(f"Found new best h: {best_h:.2f} after {expansions} expansions")

        # Check if state was already visited/expanded
        # This prevents cycles and redundant work on states already processed.
        if encoded_state in closed_list:
            continue # Skip if already expanded

        # Add state to the closed list *before* expansion
        closed_list.add(encoded_state)
        expansions += 1

        # --- Successor Generation ---
        for op, successor_state in task.get_successor_states(decoded_state):

            # Check if successor state has already been expanded.
            # If we find the goal early, we don't care if it was closed,
            # but for non-goal states, we avoid reprocessing closed ones.
            if encode_state(successor_state) in closed_list:
                continue # Skip if already expanded

            # Create the successor node object
            successor_node = searchspace.make_child_node(current_node, op, successor_state)

            # Check if the newly generated successor state is a goal state.
            if task.goal_reached(successor_state):
                logging.info(f"Early goal reached via operator {op.name}.")
                # Note: The expansion count reflects nodes fully expanded *before*
                # finding this goal successor. The goal node itself isn't expanded.
                logging.info(f"{expansions} Nodes expanded")
                return successor_node.extract_solution()

            # Calculate heuristic for the successor
            h_successor = heuristic(successor_node)

            # Check for dead ends indicated by the heuristic
            if h_successor == float("inf"):
                logging.debug(f"Successor state via {op.name} is a dead end (h=inf). Skipping.")
                continue # Don't add dead ends to the open list

            successor_node.state = encode_state(successor_state)

            # Add the valid, non-goal, non-closed successor to the open list
            node_tiebreaker += 1
            heapq.heappush(open_list, ordered_node_greedy_best_first(successor_node, h_successor, node_tiebreaker))

        # Optional: Log progress periodically
        # if expansions % 1000 == 0:
        #    logging.debug(f"Expansions: {expansions}, Open list size: {len(open_list)}, Best h: {best_h:.2f}")


    # Loop finished because open list is empty without finding the goal
    logging.info("Open list is empty. Task unsolvable.")
    logging.info(f"{expansions} Nodes expanded")
    return None

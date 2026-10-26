# Remove unused import
# from fnmatch import fnmatch # Not used in the final code
from heuristics.heuristic_base import Heuristic
from collections import deque
import math

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle potential non-string or malformed input gracefully
        return []
    # Split by space, filter out empty strings resulting from multiple spaces
    return [part for part in fact[1:-1].split() if part]

# BFS function to find shortest path distance
def bfs_distance(start_loc, end_loc, graph):
    """
    Computes the shortest path distance between two locations in the graph.

    Args:
        start_loc: The starting location string.
        end_loc: The target location string.
        graph: Adjacency list representation of the location graph {loc: {adj_loc1, ...}}.

    Returns:
        The shortest distance (number of edges) or float('inf') if unreachable.
    """
    if start_loc == end_loc:
        return 0

    # Ensure start and end locations are valid nodes in the graph
    if start_loc not in graph or end_loc not in graph:
        # This might happen if a box or goal is in a location not defined in adjacent facts
        # Treat as unreachable for heuristic purposes
        return float('inf')

    queue = deque([(start_loc, 0)])
    visited = {start_loc}

    while queue:
        current_loc, dist = queue.popleft()

        # Check if current_loc is in the graph (should be handled by initial check, but safety)
        if current_loc not in graph:
             continue

        for neighbor in graph[current_loc]:
            if neighbor not in visited:
                visited.add(neighbor)
                if neighbor == end_loc:
                    return dist + 1
                queue.append((neighbor, dist + 1))

    return float('inf') # Target is unreachable

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the shortest
    path distances for each misplaced box to its corresponding goal location.
    The distance is calculated on the graph of all locations defined by
    'adjacent' facts, ignoring the robot's position and other boxes as obstacles.
    This is a relaxation of the problem and is non-admissible.

    # Assumptions
    - The goal specifies the final location for each box using '(at ?box ?location)' facts.
      There is a one-to-one mapping between boxes and goal locations based on these facts.
    - The underlying location structure forms an undirected graph defined by 'adjacent' facts.
    - The shortest path distance on this graph is a reasonable estimate of the
      minimum number of pushes required for a single box in isolation.
    - All boxes mentioned in the goal are present in the initial state and subsequent states.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Builds an undirected graph of locations based on the 'adjacent' static facts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Check if the current state satisfies all goal conditions. If yes, the heuristic is 0.
    2. If not a goal state, parse the current state to find the location of each box
       that is specified in the goal.
    3. Initialize a total distance counter to 0.
    4. For each box identified in the goal:
       a. Get its current location from the parsed state.
       b. Get its target goal location from the pre-calculated goal mapping.
       c. If the box's current location is different from its goal location:
          i. Compute the shortest path distance between the box's current location
             and its goal location using a Breadth-First Search (BFS) on the
             location graph built during initialization.
          ii. If the BFS indicates that the goal location is unreachable from the
              box's current location on the graph, this implies a potential
              deadlock or unsolvable situation for this box. Assign an infinite
              cost for this state, as it's likely not on a path to the solution.
          iii. Add the computed shortest distance to the total distance counter.
    5. The final heuristic value is the accumulated total distance. This value
       represents a lower bound on the number of push actions required if each
       box could be moved independently along the shortest path to its goal,
       ignoring the robot's movement constraints and other boxes as obstacles.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and building the
        location graph from static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the location graph from adjacent facts
        self.location_graph = {}
        for fact in static_facts:
            parts = get_parts(fact)
            # Check if parts is not empty and the predicate is 'adjacent'
            if parts and parts[0] == 'adjacent':
                # Ensure there are enough parts for loc1, loc2, and direction
                if len(parts) >= 3:
                    _, loc1, loc2, *rest = parts # Use *rest to handle optional direction parameter if needed, though PDDL implies 4 parts
                    self.location_graph.setdefault(loc1, set()).add(loc2)
                    self.location_graph.setdefault(loc2, set()).add(loc1) # Graph is undirected

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Check if parts is not empty and the predicate is 'at'
            if parts and parts[0] == 'at':
                # Assuming goal facts are (at box location)
                if len(parts) >= 3:
                    box, location = parts[1], parts[2]
                    self.goal_locations[box] = location
                # else: Handle malformed goal fact? Assume valid PDDL.

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        (pushes) to get all boxes to their goal locations.
        """
        state = node.state

        # Check if goal is reached (heuristic is 0 iff goal is reached)
        # The goal only specifies box locations.
        if self.goals <= state:
             return 0

        # Find current box locations for the boxes specified in the goal
        box_locations = {}
        # Pre-populate box_locations with None or a marker for goal boxes not found in state
        # This helps identify if a goal box is unexpectedly missing (shouldn't happen in valid states)
        for box in self.goal_locations:
             box_locations[box] = None # Initialize location as unknown

        # Iterate through state facts to find actual box locations
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            # We only care about 'at' facts for objects that are goal boxes
            if predicate == 'at' and len(parts) >= 3:
                 obj = parts[1]
                 if obj in self.goal_locations: # Check if this object is one of the goal boxes
                    box_locations[obj] = parts[2] # Store its current location

        total_distance = 0

        # Sum distances for all misplaced boxes
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box) # Get the current location

            # If a goal box was not found in the state facts, it's an invalid state
            # or indicates a problem. Return infinity.
            if current_loc is None:
                 return float('inf')

            # If the box is not at its goal location
            if current_loc != goal_loc:
                dist = bfs_distance(current_loc, goal_loc, self.location_graph)

                if dist == float('inf'):
                    # If a box cannot reach its goal location on the full graph,
                    # the state is likely unsolvable or in a deadlock relative to that box.
                    # Return infinity to prune this branch.
                    return float('inf')

                total_distance += dist

        # If we reached here, it means the state is not the goal state (checked at the start),
        # but all goal boxes were found and their sum of distances is calculated.
        # This sum should be > 0 if the state is not the goal.
        # The only way total_distance could be 0 here is if goal_locations was empty,
        # which would mean the goal was empty, and the initial check `self.goals <= state`
        # would have returned 0 already.
        # So, total_distance > 0 if not a goal state and solvable (based on this heuristic).

        return total_distance

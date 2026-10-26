# Import necessary modules
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, maybe raise an error or return None/empty list
        return []
    return fact[1:-1].split()

# Helper function for BFS on the location graph
def bfs_distance(graph, start, goal):
    """
    Find the shortest path distance between two locations in the graph.
    Returns -1 if the goal is unreachable.
    """
    if start == goal:
        return 0

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, current_dist = queue.popleft()

        # Use .get for safety in case a location from state isn't in the graph
        for neighbor in graph.get(current_loc, []):
            if neighbor not in visited:
                visited.add(neighbor)
                if neighbor == goal:
                    return current_dist + 1
                queue.append((neighbor, current_dist + 1))

    return -1 # Goal unreachable


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing the
    shortest path distances for each box from its current location to its
    goal location. This is a relaxation that ignores the robot's position,
    other boxes, and the need for clear paths and robot access to push.

    # Assumptions
    - The problem involves moving boxes to specific goal locations.
    - The grid structure and connectivity are defined by `adjacent` facts.
    - Each box has a unique goal location assigned in the problem definition.
    - The graph defined by `adjacent` facts is connected for relevant locations.

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task's goal conditions.
    - Builds an undirected graph representing the connectivity between locations
      based on the `adjacent` static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is the goal state using the task's goal conditions. If yes, the heuristic is 0.
    2. Identify the current location of each box from the state facts.
    3. Initialize the total heuristic value to 0.
    4. For each box that has a goal location defined:
       a. Determine its assigned goal location.
       b. Find its current location in the state.
       c. If the box is not at its goal location:
          i. Calculate the shortest path distance between the box's current
             location and its goal location using BFS on the pre-computed
             location graph. This distance represents the minimum number of
             pushes required for this box in a relaxed environment.
          ii. If the goal location is unreachable from the box's current location
              in the graph, return infinity, as this state is likely a dead end.
          iii. Add this distance to the total heuristic value.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the location graph from static facts.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are not affected by actions.
        static_facts = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal facts are typically (at box location)
            if parts and parts[0] == "at" and len(parts) == 3:
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location

        # Build the graph from adjacent facts.
        # The graph is undirected for distance calculation.
        self.graph = {}
        for fact in static_facts:
            parts = get_parts(fact)
            # Adjacent facts are (adjacent loc1 loc2 direction)
            if parts and parts[0] == "adjacent" and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                if loc2 not in self.graph:
                    self.graph[loc2] = []
                # Add edges in both directions for undirected graph
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1)

        # Remove duplicates from adjacency lists (BFS handles this implicitly,
        # but cleaning the graph is good practice)
        for loc in self.graph:
            self.graph[loc] = list(set(self.graph[loc]))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # If the state is the goal state, the heuristic is 0.
        if self.goals <= state:
             return 0

        # Find current locations of boxes
        current_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) == 3:
                 obj_name = parts[1]
                 # Only track objects that are goals (i.e., boxes)
                 if obj_name in self.goal_locations:
                      current_locations[obj_name] = parts[2]

        total_cost = 0  # Initialize action cost counter.

        # Calculate cost for each box not at its goal
        for box, goal_location in self.goal_locations.items():
            # Ensure the box exists in the current state (it always should in Sokoban)
            if box in current_locations:
                current_location = current_locations[box]

                if current_location != goal_location:
                    # Calculate shortest path distance for the box
                    # This is the minimum number of pushes required for this box
                    # in a relaxed problem where it can move freely.
                    dist = bfs_distance(self.graph, current_location, goal_location)

                    # If goal is unreachable for a box in the graph, return infinity
                    # This state is likely a dead end.
                    if dist == -1:
                        return float('inf') # Indicate unsolvable state

                    total_cost += dist
            # else: # Box not found in state? Should not happen in Sokoban.
            #     # This case implies a malformed state or problem, treat as unsolvable.
            #     return float('inf')

        # The heuristic is the sum of distances for all misplaced boxes.
        # If all boxes are at their goals, total_cost will be 0, which is handled
        # by the initial goal check.

        return total_cost

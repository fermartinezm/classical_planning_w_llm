# Required imports
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and remove outer parentheses
    if isinstance(fact, str) and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    return [] # Return empty list for invalid input

# BFS function to calculate shortest path distance on the grid graph
def bfs_distance(graph, start, end):
    """
    Calculates the shortest path distance between two nodes in a graph using BFS.
    Assumes graph nodes are location strings.
    """
    if start == end:
        return 0

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc == end:
            return dist

        # Check if current_loc is in the graph keys before accessing neighbors
        if current_loc in graph:
            for neighbor_loc, _ in graph[current_loc]:
                if neighbor_loc not in visited:
                    visited.add(neighbor_loc)
                    queue.append((neighbor_loc, dist + 1))

    # If BFS completes without finding the end, it's unreachable
    return float('inf')

# Domain-dependent heuristic class for Sokoban
class sokobanHeuristic:
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the
    shortest path distances for each box from its current location to its
    goal location. The distances are calculated on the grid graph defined
    by adjacency facts, ignoring other objects as obstacles.

    # Assumptions
    - Each box has a unique goal location specified in the task goals.
    - The grid structure is fully defined by the 'adjacent' facts, and
      adjacency is symmetric (if A is adjacent to B, B is adjacent to A).
    - The heuristic is non-admissible; it does not guarantee a lower bound
      on the actual cost.

    # Heuristic Initialization
    - The constructor extracts the goal location for each box from the task's
      goal conditions.
    - It builds an adjacency list representation of the grid graph from the
      static 'adjacent' facts, ensuring symmetric connections.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify the current location of each box from the state.
    2.  Initialize the total heuristic value to 0.
    3.  For each box:
        a.  Determine its goal location (stored during initialization).
        b.  Determine its current location from the state.
        c.  If the box is not at its goal location:
            i.  Calculate the shortest path distance from the box's current
                location to its goal location using Breadth-First Search (BFS)
                on the pre-computed adjacency graph. This BFS considers the
                structure of the grid but ignores dynamic obstacles like the
                robot or other boxes.
            ii. If the goal is unreachable for this box on the grid, the state
                is likely a dead end or unsolvable from this point; return
                infinity.
            iii. Add the calculated distance to the total heuristic value.
    4.  Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the grid graph.

        Args:
            task: An object representing the planning task, containing goals and static facts.
                  Assumed to have attributes 'goals' (frozenset of goal fact strings)
                  and 'static' (frozenset of static fact strings).
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the adjacency graph from 'adjacent' facts.
        # Graph maps location string -> list of (adjacent_location_string, direction_string)
        self.adjacency_graph = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'adjacent':
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                
                # Add edge loc1 -> loc2
                if loc1 not in self.adjacency_graph:
                    self.adjacency_graph[loc1] = []
                self.adjacency_graph[loc1].append((loc2, direction))

                # Assuming adjacency is symmetric, add the reverse edge loc2 -> loc1
                # We don't strictly need the opposite direction string for BFS,
                # but we add the reverse adjacency.
                if loc2 not in self.adjacency_graph:
                    self.adjacency_graph[loc2] = []
                # The direction from loc2 to loc1 is the opposite of the direction from loc1 to loc2
                # We could determine the opposite direction string, but BFS only needs the neighbor.
                # Let's just add the neighbor without determining the opposite direction string explicitly here.
                self.adjacency_graph[loc2].append((loc1, None)) # Direction is not used by BFS

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal facts are typically (at boxX loc_Y_Z)
            if parts and parts[0] == 'at' and parts[1].startswith('box'):
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node: An object representing a state in the search space.
                  Assumed to have an attribute 'state' (frozenset of fact strings).

        Returns:
            An estimated cost (non-negative integer or float('inf')) to reach the goal.
        """
        state = node.state

        # Find current box locations.
        current_box_locations = {}
        # Iterate through the frozenset of facts in the state
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at' and parts[1].startswith('box'):
                box, location = parts[1], parts[2]
                current_box_locations[box] = location

        total_box_distance = 0

        # Calculate the sum of distances for boxes not at their goals.
        # We iterate through the goal locations to ensure we consider all boxes
        # that need to reach a specific goal.
        for box, goal_loc in self.goal_locations.items():
            current_loc = current_box_locations.get(box)

            # If a box required by the goal is not found in the current state,
            # this state is inconsistent with the problem definition.
            # Returning infinity signals this is likely not a path to a valid goal.
            if current_loc is None:
                 return float('inf')

            if current_loc != goal_loc:
                # Calculate shortest path distance for the box to its goal.
                # We use the full adjacency graph, ignoring other objects as obstacles.
                dist = bfs_distance(self.adjacency_graph, current_loc, goal_loc)

                if dist == float('inf'):
                    # Box cannot reach its goal on the grid graph. This state
                    # represents a dead end for this box.
                    return float('inf')

                total_box_distance += dist

        # The heuristic is the sum of box-to-goal distances.
        # This is 0 if and only if all boxes are at their goals.
        return total_box_distance

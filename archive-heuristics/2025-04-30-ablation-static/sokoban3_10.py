from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact string."""
    # Example: "(at box1 loc_3_5)" -> ["at", "box1", "loc_3_5"]
    return fact[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing the
    shortest path distances for each box from its current location to its
    specific goal location. The distance is measured in the number of grid steps
    (equivalent to the minimum number of pushes required if the robot could
    magically be in the correct position for each push).

    # Assumptions
    - Each box has a specific goal location.
    - The cost of moving a box one step (one push action) is the primary cost.
    - Robot movement costs and potential deadlocks are ignored for simplicity
      and efficiency, making the heuristic non-admissible but potentially
      effective for greedy search.
    - The locations form a grid graph defined by the 'adjacent' predicates.
    - All relevant locations (initial box/robot, goals, adjacent locations)
      are expected to be part of the connected component defined by 'adjacent' facts
      where movement is possible.

    # Heuristic Initialization
    - Parses the goal conditions to map each box to its specific goal location.
    - Builds a graph of locations based on the 'adjacent' static facts.
    - Precomputes the shortest path distance from each goal location to all
      other reachable locations using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    1. For a given state, identify the current location of each box.
    2. For each box, determine its specific goal location (precomputed during initialization).
    3. If a box is not at its goal location, look up the precomputed shortest
       path distance from its current location to its goal location in the
       location graph. This distance represents the minimum number of pushes
       required for that box to reach its goal, ignoring obstacles and robot position.
    4. Sum the minimum push distances for all boxes that are not yet at their goals.
    5. The total sum is the heuristic value for the state. If any box is in a
       location from which its goal is unreachable within the grid, the heuristic
       returns infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the location graph from static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box
        self.box_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal facts are typically (at box loc)
            if parts[0] == "at" and len(parts) == 3 and parts[1].startswith("box"):
                 box, loc = parts[1:]
                 self.box_goals[box] = loc

        # Build the location graph from adjacent facts
        self.location_graph = {}
        all_locations = set() # Collect all locations mentioned in adjacent facts

        for fact in static_facts:
             parts = get_parts(fact)
             # Adjacent facts are typically (adjacent loc1 loc2 dir)
             if parts[0] == "adjacent" and len(parts) == 4:
                 l1, l2, _ = parts[1:]
                 all_locations.add(l1)
                 all_locations.add(l2)

        # Initialize graph with all found locations
        for loc in all_locations:
            self.location_graph[loc] = []

        # Add edges
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "adjacent" and len(parts) == 4:
                l1, l2, _ = parts[1:]
                # Add bidirectional edges
                self.location_graph[l1].append(l2)
                self.location_graph[l2].append(l1)

        # Precompute shortest path distances from each goal location
        self.goal_distances = {}
        # Use set of values to avoid recomputing BFS for the same goal location
        for goal_loc in set(self.box_goals.values()):
            # Ensure goal_loc is actually in the graph before running BFS
            # If a goal is not in the graph, it's unreachable from any location in the graph.
            # The BFS result will reflect this (distances will be inf for graph nodes).
            # If the goal_loc itself is not a node in the graph, the BFS won't run meaningfully.
            # Let's assume valid instances where goal locations are part of the grid graph.
            if goal_loc in self.location_graph:
                 self.goal_distances[goal_loc] = self._bfs(goal_loc)
            else:
                 # Handle case where a goal location is not part of the adjacent graph.
                 # This implies it's unreachable from any location defined by adjacent facts.
                 # The only way a box can be at this goal is if it starts there.
                 # If a box needs to reach this goal and isn't there, it's impossible.
                 # We can represent this by having distances be infinity for any location
                 # other than the goal itself.
                 self.goal_distances[goal_loc] = {loc: float('inf') for loc in self.location_graph}
                 self.goal_distances[goal_loc][goal_loc] = 0 # Distance from goal to itself is 0


    def _bfs(self, start_node):
        """
        Performs BFS starting from start_node to find distances to all reachable locations
        within the graph.
        Returns a dictionary mapping location to its distance from start_node.
        Includes all nodes from the graph, with infinity for unreachable ones.
        Assumes start_node is a key in self.location_graph.
        """
        distances = {node: float('inf') for node in self.location_graph}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current = queue.popleft()

            # current must be in self.location_graph keys if start_node was
            for neighbor in self.location_graph[current]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

        return distances


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Find current locations of all boxes
        current_box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3 and parts[1].startswith("box"):
                box, loc = parts[1:]
                current_box_locations[box] = loc

        total_distance = 0

        # Sum distances for each box to its goal
        for box, goal_loc in self.box_goals.items():
            current_loc = current_box_locations.get(box)

            # If a box is not found in the state, something is wrong with the state representation.
            # Assuming valid states where all objects exist and are located.
            if current_loc is None:
                 # This should not happen in a valid state
                 # Return infinity as the state is likely invalid or unsolvable
                 return float('inf')

            # If the box is not at its goal, add the distance
            if current_loc != goal_loc:
                # Look up the precomputed distance.
                # Use .get() defensively in case goal_loc wasn't in self.goal_distances
                # (e.g., if goal_loc was not in the graph initially).
                # The inner .get() handles the case where current_loc is not reachable
                # from goal_loc (distance is inf).
                distance = self.goal_distances.get(goal_loc, {}).get(current_loc, float('inf'))

                # If any box is in a location unreachable from its goal, the state is
                # likely unsolvable. Return infinity.
                if distance == float('inf'):
                    return float('inf')

                total_distance += distance

        return total_distance

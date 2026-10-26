# Required imports
from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in a module named heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper functions from examples
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at-robot loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't try to zip unequal length lists if pattern is longer than fact parts
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the shortest
    path distances for each box to its goal location and adding the shortest
    path distance from the robot to the closest box that is not yet at its goal.
    This combination aims to prioritize states where boxes are closer to their
    goals and the robot is positioned to interact with a box that needs moving.

    # Assumptions
    - Each box has a unique goal location specified in the task goals.
    - The grid connectivity is defined by the 'adjacent' predicates.
    - Shortest path distances are calculated on the undirected graph derived
      from these adjacencies.
    - The heuristic does not explicitly detect or penalize deadlocks (e.g.,
      boxes pushed into corners or against other immovable objects).
    - The cost of moving the robot and pushing a box are implicitly combined
      in the distance calculations.

    # Heuristic Initialization
    - Extract the goal location for each box from the task goals.
    - Build an undirected graph representing the grid connectivity from the
      static 'adjacent' facts.
    - Pre-calculate shortest path distances between all pairs of locations
      reachable within this graph using BFS. Store these distances for quick lookup.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Find Current Locations:** Determine the current location of the robot
        and each box by iterating through the state facts.
    2.  **Identify Off-Goal Boxes:** For each box specified in the task goals,
        check if its current location in the state matches its goal location.
        Create a list of boxes that are not currently at their goals.
    3.  **Check for Goal State:** If the list of off-goal boxes is empty, the
        current state is a goal state, and the heuristic value is 0.
    4.  **Calculate Box-Goal Distances:** For each box that is not at its goal,
        look up the pre-calculated shortest path distance from its current
        location to its goal location using the graph distances. Sum these
        distances. If any box's goal is unreachable from its current location,
        return infinity (or a very large number) as the state is likely unsolvable.
    5.  **Calculate Robot-to-Box Distance:** Find the shortest path distance
        from the robot's current location to the current location of each
        off-goal box. Determine the minimum of these distances. If the robot
        cannot reach any off-goal box, return infinity.
    6.  **Combine Components:** The total heuristic value is the sum of the
        total box-goal distances (from step 4) and the minimum robot-to-closest-box
        distance (from step 5). This value estimates the remaining effort
        considering both the box movements required and the robot's need to
        reach a box.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goals and building the location graph.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # Build the location graph from adjacent facts.
        self.graph = {}
        # Collect all locations mentioned in adjacent facts
        all_locations_in_graph = set()
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                self.graph.setdefault(loc1, set()).add(loc2)
                self.graph.setdefault(loc2, set()).add(loc1) # Assume adjacency is symmetric for distance
                all_locations_in_graph.add(loc1)
                all_locations_in_graph.add(loc2)

        # Pre-calculate all-pairs shortest paths.
        self.distances = {}
        # Only compute distances for nodes that are part of the connected graph
        for start_node in self.graph:
             self.distances[start_node] = self._bfs(start_node)

        # For any location that might be in the goals but not in the graph
        # (e.g., an isolated goal location), add it to distances with no reachable nodes.
        # This ensures get_distance doesn't fail when looking up a goal location.
        for goal_loc in self.goal_locations.values():
             if goal_loc not in self.distances:
                 self.distances[goal_loc] = {}


    def _bfs(self, start_node):
        """
        Perform BFS from a start node to find distances to all reachable nodes.
        Returns a dictionary {node: distance}.
        """
        distances = {start_node: 0}
        queue = deque([start_node])
        visited = {start_node}

        while queue:
            current_node = queue.popleft()

            # Check if current_node exists in the graph (it should if it came from queue)
            if current_node in self.graph:
                for neighbor in self.graph[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)

        return distances

    def get_distance(self, loc1, loc2):
        """
        Get the pre-calculated shortest distance between two locations.
        Returns float('inf') if loc1 or loc2 are not in the graph or if loc2 is
        unreachable from loc1.
        """
        # Check if start location is in our pre-calculated distances map
        if loc1 not in self.distances:
             # This location was not part of the graph defined by adjacent facts.
             # It might be an isolated location in the initial state or a state fact.
             # Distance to anywhere else is infinite unless loc1 == loc2.
             return 0 if loc1 == loc2 else float('inf')

        # Check if target location is reachable from start location
        if loc2 not in self.distances[loc1]:
             # loc2 is not reachable from loc1 in the graph.
             return float('inf')

        return self.distances[loc1][loc2]


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Find current robot location
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
                break
        # If robot_location is None, the state is malformed, return inf
        if robot_location is None:
             return float('inf')


        # Find current box locations for all boxes mentioned in goals
        current_box_locations = {}
        # Iterate through state facts to find locations of all boxes
        for fact in state:
            if match(fact, "at", "box*", "*"):
                box, location = get_parts(fact)[1:]
                current_box_locations[box] = location


        # Identify off-goal boxes
        off_goal_boxes = []
        # Iterate through boxes that have a goal location
        for box, goal_loc in self.goal_locations.items():
            current_loc = current_box_locations.get(box) # Get current location, None if box not found

            # A box is off-goal if it's supposed to be at goal_loc but isn't currently there.
            # This includes cases where the box isn't found in the state facts at all
            # (shouldn't happen in valid states, but defensive check), or is found
            # but at a different location.
            if current_loc != goal_loc:
                 # If the box is supposed to be at a goal but isn't there, it's off-goal.
                 # We assume boxes mentioned in goals are present in the state at some location.
                 # If current_loc is None, it implies a malformed state or the box was removed,
                 # which is not possible in this domain. We proceed assuming current_loc is valid
                 # for any box in self.goal_locations.keys().
                 off_goal_boxes.append(box)


        # If all boxes are at their goals, heuristic is 0
        if not off_goal_boxes:
            return 0

        # Calculate sum of box-goal distances
        sum_box_goal_dist = 0
        for box in off_goal_boxes:
            current_loc = current_box_locations[box] # current_loc should not be None here based on assumption
            goal_loc = self.goal_locations[box]
            dist = self.get_distance(current_loc, goal_loc)

            if dist == float('inf'):
                 # If any box cannot reach its goal, the state is likely a deadlock
                 # or unsolvable from here. A high heuristic value is appropriate.
                 return float('inf')

            sum_box_goal_dist += dist

        # Calculate minimum robot-to-box distance for off-goal boxes
        min_robot_box_dist = float('inf')
        for box in off_goal_boxes:
            box_loc = current_box_locations[box] # box_loc should not be None here
            dist = self.get_distance(robot_location, box_loc)
            # If robot cannot reach a box, the state is likely unsolvable
            if dist == float('inf'):
                 return float('inf')
            min_robot_box_dist = min(min_robot_box_dist, dist)

        # Total heuristic value
        # This combines the effort for boxes to reach goals and robot to reach a box.
        heuristic_value = sum_box_goal_dist + min_robot_box_dist

        return heuristic_value

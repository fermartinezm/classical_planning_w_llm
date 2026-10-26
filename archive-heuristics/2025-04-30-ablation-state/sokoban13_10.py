from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact.strip()[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by summing the shortest path distances for each box to its goal location
    and adding the minimum shortest path distance from the robot to any location
    adjacent to a box that is not yet at its goal.

    # Assumptions
    - The environment is represented as a graph where locations are nodes
      and `adjacent` predicates define edges.
    - All movements (robot or box push) have a cost of 1.
    - The shortest path distance between locations is a reasonable estimate
      of the minimum number of moves/pushes required to traverse that distance.
    - The heuristic does not explicitly account for dynamic obstacles (other
      boxes or the robot occupying locations) in the distance calculations.
    - The heuristic does not explicitly detect or penalize deadlocks.

    # Heuristic Initialization
    1. Parse the goal conditions to identify the target location for each box.
    2. Collect all unique locations mentioned in the initial state and static facts.
    3. Parse the static facts (`adjacent` predicates) to build the graph of locations.
       Adjacency is assumed to be symmetric.
    4. Compute the shortest path distance between all pairs of locations using BFS.
       These distances are stored in a dictionary for quick lookup during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes from the state.
    2. Identify which boxes are not yet at their goal locations by comparing
       current box locations with the goal locations stored during initialization.
    3. If all boxes are at their goals, the heuristic is 0.
    4. Calculate the sum of shortest path distances for each box that is not
       at its goal, from its current location to its goal location. This provides
       a base estimate of the minimum number of pushes required for the boxes.
       If any box cannot reach its goal location in the static graph, the state
       is likely unsolvable, and the heuristic returns infinity.
    5. Identify all locations that are adjacent (in the static graph) to any box
       that is not at its goal. These are potential target locations for the robot
       to position itself to initiate a push.
    6. Calculate the minimum shortest path distance from the robot's current location
       to any of the target locations identified in step 5. This estimates the
       robot's effort to reach a position where it can push a box towards its goal.
       If the robot cannot reach any such location, the state is likely unsolvable,
       and the heuristic returns infinity.
    7. The total heuristic value is the sum from step 4 (box distances) and step 6
       (minimum robot distance to a push-adjacent location).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        location graph, and precomputing all-pairs shortest paths.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # 1. Parse goal conditions to find box goals
        self.box_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                obj, location = args
                # Assuming objects with 'at' goals are boxes
                self.box_goals[obj] = location

        # 2. Collect all unique locations from initial state and static facts
        all_locations = set()

        # Locations from initial state
        for fact in task.initial_state:
             if match(fact, "at-robot", "*"):
                 all_locations.add(get_parts(fact)[1])
             elif match(fact, "at", "*", "*"):
                 # Assuming the second argument of 'at' is always a location for boxes
                 all_locations.add(get_parts(fact)[2])
             elif match(fact, "clear", "*"):
                 all_locations.add(get_parts(fact)[1])

        # Locations from static facts (adjacent)
        for fact in self.static_facts:
             if match(fact, "adjacent", "*", "*", "*"):
                 l1, l2, _ = get_parts(fact)
                 all_locations.add(l1)
                 all_locations.add(l2)

        self.locations = list(all_locations) # Store as list

        # 3. Build the location graph from adjacent facts
        self.adj = {loc: [] for loc in self.locations}

        for fact in self.static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                l1, l2, _ = get_parts(fact)
                # Add bidirectional edges, but only if both locations were collected
                if l1 in self.adj and l2 in self.adj:
                    self.adj[l1].append(l2)
                    self.adj[l2].append(l1) # Assuming adjacency is symmetric

        # 4. Compute all-pairs shortest paths using BFS
        self.distances = {loc: {} for loc in self.locations}
        for start_loc in self.locations:
            self._bfs(start_loc)

    def _bfs(self, start_loc):
        """Perform BFS from start_loc to compute distances to all reachable locations."""
        q = deque([(start_loc, 0)])
        visited = {start_loc}
        self.distances[start_loc][start_loc] = 0

        # Ensure start_loc is in adj, otherwise it's an isolated node
        if start_loc not in self.adj:
             # Mark all other locations as unreachable
             for loc in self.locations:
                 if loc != start_loc:
                     self.distances[start_loc][loc] = float('inf')
             return

        while q:
            current_loc, dist = q.popleft()

            for neighbor in self.adj.get(current_loc, []): # Use .get for safety
                if neighbor not in visited:
                    visited.add(neighbor)
                    self.distances[start_loc][neighbor] = dist + 1
                    q.append((neighbor, dist + 1))

        # Fill in unreachable locations with infinity
        for loc in self.locations:
             if loc not in self.distances[start_loc]:
                 self.distances[start_loc][loc] = float('inf')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify current locations
        robot_loc = None
        box_locations = {} # {box_name: location}

        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
            elif match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)
                # Only consider objects that are boxes and have goals
                if obj in self.box_goals:
                     box_locations[obj] = loc

        # Ensure robot location is found (should always be the case in a valid state)
        if robot_loc is None:
             return float('inf') # Indicates an invalid state

        # 2. Identify boxes not at goals
        boxes_to_move = {
            box: loc for box, loc in box_locations.items()
            if box in self.box_goals and loc != self.box_goals[box]
        }

        # 3. Goal state check
        if not boxes_to_move:
            return 0 # All boxes are at their goals

        total_heuristic = 0

        # 4. Sum of box-to-goal distances
        for box, current_box_loc in boxes_to_move.items():
            goal_box_loc = self.box_goals[box]

            # Check if locations exist and path exists
            if current_box_loc not in self.distances or goal_box_loc not in self.distances[current_box_loc]:
                 # Should not happen if all locations were collected correctly, but safety check
                 return float('inf')
            
            dist = self.distances[current_box_loc][goal_box_loc]

            if dist == float('inf'):
                 # Box cannot reach its goal location in the static graph
                 return float('inf')

            total_heuristic += dist

        # 5. Identify potential robot target locations (adjacent to boxes to move)
        target_robot_locs = set()
        for box, current_box_loc in boxes_to_move.items():
             if current_box_loc in self.adj: # Ensure the box location is in the graph
                 for neighbor in self.adj[current_box_loc]:
                      target_robot_locs.add(neighbor)

        # If no adjacent locations found for any box (e.g., box is isolated or graph is weird), this might be unsolvable
        if not target_robot_locs:
             # This could happen if a box is in a location with no adjacent facts
             return float('inf') # Cannot push any box

        # 6. Calculate minimum robot-to-adjacent-location distance
        min_robot_adj_dist = float('inf')

        if robot_loc in self.distances: # Ensure robot location is in the graph
            for target_loc in target_robot_locs:
                 if target_loc in self.distances[robot_loc]: # Ensure target is reachable from robot
                     dist = self.distances[robot_loc][target_loc]
                     min_robot_adj_dist = min(min_robot_adj_dist, dist)

        # If robot cannot reach any target adjacent location
        if min_robot_adj_dist == float('inf'):
             return float('inf')

        # 7. Return total heuristic
        total_heuristic += min_robot_adj_dist

        return total_heuristic

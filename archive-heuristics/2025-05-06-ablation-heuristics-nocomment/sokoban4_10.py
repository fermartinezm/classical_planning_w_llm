from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function
def get_parts(fact):
    """Splits a PDDL fact string into its components."""
    return fact[1:-1].split()

# BFS function
def bfs(start_node, graph):
    """
    Performs Breadth-First Search from a start node on the graph
    to find shortest path distances to all reachable nodes.
    Returns a dictionary {node: distance}.
    """
    distances = {node: float('inf') for node in graph}
    if start_node in graph:
        distances[start_node] = 0
        queue = deque([start_node])
        while queue:
            current_node = queue.popleft()
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

# Heuristic class
class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        two main components:
        1. The sum of shortest path distances for each misplaced box to its
           corresponding goal location. This estimates the minimum number of
           pushes required for the boxes.
        2. The shortest path distance from the robot's current location to
           the location of the closest misplaced box. This estimates the
           robot's travel cost to engage with a box.
        The distances are computed on the graph defined by the 'adjacent'
        predicates in the PDDL domain, representing the traversable locations.
        This heuristic is non-admissible but aims to guide the search effectively
        by prioritizing states where boxes are closer to their goals and the
        robot is positioned to interact with a box.

    Assumptions:
        - The PDDL instance defines a graph of locations via 'adjacent' predicates.
        - Each box in the goal state has a unique target location.
        - The 'adjacent' predicates define symmetric connections (if A is
          adjacent to B, B is adjacent to A), allowing for an undirected
          graph for distance calculations.
        - The goal state only contains 'at' predicates for boxes.
        - All locations mentioned in initial state and goal state are part
          of the graph defined by 'adjacent' facts, or are isolated nodes.

    Heuristic Initialization:
        1. Parse the goal state to create a mapping from each box object
           to its target goal location (`self.box_goals`).
        2. Parse the static facts ('adjacent' predicates) to build an
           undirected graph of locations (`self.graph`). This graph includes
           all locations mentioned in 'adjacent' facts, plus any locations
           from the initial state or goal state that were not in 'adjacent'
           facts (as isolated nodes).
        3. Compute all-pairs shortest path distances between all locations
           that are keys in `self.graph` using BFS starting from each such
           location. Store these distances in `self.all_pairs_distances`.
           This precomputation allows for efficient distance lookups during
           the search.

    Step-By-Step Thinking for Computing Heuristic:
        1. Given a state (a frozenset of facts):
        2. Identify the robot's current location by finding the fact
           '(at-robot ?l)'. If not found, return infinity (invalid state).
        3. Identify the current location of each box by finding facts
           '(at ?b ?l)' for each box object known from the goal state.
           If any box from the goal is not found in the state, return
           infinity (invalid state).
        4. Determine which boxes are currently misplaced (i.e., not at their
           target goal location).
        5. Initialize the heuristic value `h` to 0.
        6. For each box that is misplaced:
           a. Get its current location and its target goal location.
           b. Look up the precomputed shortest path distance between the
              box's current location and its goal location using
              `self.all_pairs_distances`.
           c. If the distance is infinity (goal is unreachable from
              current box location within the graph), the state is
              unsolvable; return infinity.
           d. Add this distance to `h`. This component estimates the
              minimum number of pushes needed for the boxes.
        7. If there are any misplaced boxes:
           a. Find the minimum shortest path distance from the robot's
              current location to the location of any of the misplaced boxes.
              Look up distances using `self.all_pairs_distances`.
           b. If this minimum distance is infinity (robot cannot reach any
              misplaced box within the graph), the state is unsolvable;
              return infinity.
           c. Add this minimum distance to `h`. This component estimates the
              robot's travel cost to get to a box it needs to push.
        8. Return the calculated value `h`. If no boxes are misplaced, `h`
           will be 0, correctly identifying a goal state.
    """
    def __init__(self, task):
        # 1. Parse goal state for box goals
        self.box_goals = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                box, location = parts[1], parts[2]
                self.box_goals[box] = location

        # 2. Build graph from adjacent facts and collect all relevant locations
        self.graph = {}
        all_relevant_locations = set()

        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                if loc1 not in self.graph:
                    self.graph[loc1] = set()
                if loc2 not in self.graph:
                    self.graph[loc2] = set()
                self.graph[loc1].add(loc2)
                self.graph[loc2].add(loc1) # Assume symmetric connectivity
                all_relevant_locations.add(loc1)
                all_relevant_locations.add(loc2)

        # Add any locations from goals or initial state that are not in adjacent facts
        for goal_loc in self.box_goals.values():
             all_relevant_locations.add(goal_loc)
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] in ['at-robot', 'at'] and len(parts) >= 2:
                  loc = parts[-1]
                  all_relevant_locations.add(loc)

        # Ensure all relevant locations are keys in the graph dictionary,
        # even if they have no neighbors (isolated). This allows BFS to be
        # called for these nodes and correctly report distances (0 to self, inf to others).
        for loc in all_relevant_locations:
             if loc not in self.graph:
                  self.graph[loc] = set()

        # 3. Compute all-pairs shortest paths
        self.all_pairs_distances = {}
        # Run BFS from every location that is a key in the graph dictionary
        for loc in self.graph.keys():
            self.all_pairs_distances[loc] = bfs(loc, self.graph)


    def __call__(self, node):
        state = node.state

        # 1. Get robot location
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
                break
        if robot_loc is None:
             # Robot location not found, invalid state
             return float('inf')

        # 2. Get box locations and identify misplaced boxes
        box_locations = {}
        misplaced_boxes = []
        # Check for each box we expect in the goal
        for box in self.box_goals:
             found = False
             for fact in state:
                  parts = get_parts(fact)
                  if parts[0] == 'at' and len(parts) == 3 and parts[1] == box:
                       loc = parts[2]
                       box_locations[box] = loc
                       if loc != self.box_goals[box]:
                            misplaced_boxes.append(box)
                       found = True
                       break # Found location for this box, move to next box
             if not found:
                  # A box from the goal state is not found in the current state
                  return float('inf') # Invalid state

        # 3. Calculate heuristic
        h = 0

        # Component 1: Sum of box-goal distances
        for box in misplaced_boxes: # Only consider misplaced boxes
            current_loc = box_locations[box]
            goal_loc = self.box_goals[box]

            # Look up the precomputed distance. If start or end is not in the
            # precomputed map (meaning it wasn't a key in self.graph), or if
            # the end is unreachable from the start, the distance will be inf.
            dist = self.all_pairs_distances.get(current_loc, {}).get(goal_loc, float('inf'))

            if dist == float('inf'):
                 return float('inf') # Unsolvable state (goal unreachable from box)
            h += dist

        # Component 2: Add robot-to-closest-misplaced-box distance
        if misplaced_boxes:
            min_robot_dist = float('inf')
            for box in misplaced_boxes:
                box_loc = box_locations[box]
                # Distance from robot current location to box current location
                dist = self.all_pairs_distances.get(robot_loc, {}).get(box_loc, float('inf'))

                min_robot_dist = min(min_robot_dist, dist)

            if min_robot_dist == float('inf'):
                return float('inf') # Robot cannot reach any misplaced box

            h += min_robot_dist

        return h

import re
import math
from collections import defaultdict, deque

def parse_fact(fact_str):
    """
    Parses a PDDL fact string into a tuple.
    e.g., '(at box1 loc_4_4)' becomes ('at', 'box1', 'loc_4_4')
    """
    # Remove surrounding parentheses and split by space
    parts = fact_str.strip("()").split()
    return tuple(parts)

class sokobanHeuristic:
    """
    Domain-dependent heuristic for the Sokoban planning domain.

    Summary:
    Estimates the cost to reach the goal state by summing the shortest path
    distances for each box to its respective goal location and adding the
    shortest path distance for the robot to reach any location from which
    it can initiate a push action on any box. This heuristic is designed
    for greedy best-first search and is not admissible.

    Assumptions:
    - The problem graph defined by 'adjacent' facts represents the traversable
      locations and connections.
    - Each box has a unique goal location specified in the task goals.
    - The heuristic calculation for robot movement and push positions
      ignores the 'clear' predicate for simplicity and efficiency (it assumes
      the robot can always reach an adjacent location and a push destination
      exists based purely on static adjacency, not current state occupancy).
      This makes the heuristic non-admissible but faster to compute.
    - The goal state consists solely of boxes being at their target locations.
      (Based on example PDDL, robot location is not part of the goal).

    Heuristic Initialization:
    1. Parses the goal facts from the task to create a mapping from each box
       to its target goal location (`self.box_goals`).
    2. Parses the static 'adjacent' facts to build:
       - An undirected graph (`self.adj_graph`) representing connectivity
         between locations for shortest path calculations.
       - Directed adjacency lists (`self.adj_dirs`, `self.rev_adj_dirs`)
         including direction information, used to identify potential pushing
         positions.
    3. Collects all unique location names from static facts, initial state,
       and goals (`self.all_locations`).
    4. Computes all-pairs shortest path distances between all locations
       using Breadth-First Search (BFS) on the undirected graph (`self.adj_graph`).
       These distances are stored in `self.location_distances` and represent
       the minimum number of 'move' actions (or pushes, ignoring robot
       positioning constraints) between locations.

    Step-By-Step Thinking for Computing Heuristic:
    1. Given a state, first check if it is a goal state using `self.task.goal_reached(state)`.
       If true, the heuristic value is 0.
    2. Parse the current state facts to identify the current location of the
       robot (`robot_loc`) and each box (`box_locations`).
    3. Calculate the total box distance (`total_box_distance`): For each box,
       find its goal location from `self.box_goals`. Get the precomputed
       shortest path distance from the box's current location to its goal
       location using `self.get_distance`. Sum these distances. If any box
       cannot reach its goal (distance is `math.inf`), the state is likely
       unsolvable, and the heuristic returns `math.inf`.
    4. Identify potential pushing positions: A location `rloc` is a potential
       pushing position for a box at `bloc` if `rloc` is adjacent to `bloc`
       in some direction `dir`, AND `bloc` is adjacent to some `floc` in the
       *same* direction `dir`. This forms a collinear sequence `rloc -> bloc -> floc`.
       These positions are identified using the precomputed `self.adj_dirs`
       and `self.rev_adj_dirs`.
    5. Calculate the minimum robot distance to a valid pushing position
       (`min_robot_to_valid_push_pos`): Find the shortest path distance from
       the robot's current location (`robot_loc`) to *any* location `rloc`
       that was identified as a potential pushing position for *any* box
       that is not yet at its goal location.
    6. If no such valid pushing position exists for any box (meaning no box
       can be pushed in any direction from any adjacent location), and the
       total box distance is greater than 0 (meaning goals are not met),
       the state is likely unsolvable, and the heuristic returns `math.inf`.
    7. If `min_robot_to_valid_push_pos` is still `math.inf` after checking
       all boxes and potential push positions, it means the robot cannot
       reach any location from which a box can be pushed. If goals are not
       met, the state is unsolvable, and the heuristic returns `math.inf`.
    8. Otherwise, the heuristic value is the sum of the `total_box_distance`
       and the `min_robot_to_valid_push_pos`.
    """
    def __init__(self, task):
        self.task = task
        self.box_goals = {}
        self.all_locations = set()
        self.adj_graph = defaultdict(list) # Undirected graph for distances
        self.adj_dirs = defaultdict(list) # Directed graph with directions (l1 -> l2, dir)
        self.rev_adj_dirs = defaultdict(list) # Reversed directed graph (l2 -> l1, dir)
        self.location_distances = {} # Stores shortest path distances between locations

        # 1. Extract box-goal mappings
        for goal_fact_str in task.goals:
            pred_args = parse_fact(goal_fact_str)
            if pred_args[0] == 'at':
                box, goal_loc = pred_args[1:]
                self.box_goals[box] = goal_loc

        # 2. Build adjacency graphs and collect all locations
        for static_fact_str in task.static:
            pred_args = parse_fact(static_fact_str)
            if pred_args[0] == 'adjacent':
                l1, l2, dir = pred_args[1:]
                # Undirected graph for distance calculation
                self.adj_graph[l1].append(l2)
                self.adj_graph[l2].append(l1)
                # Directed graphs with directions for push position logic
                self.adj_dirs[l1].append((l2, dir))
                self.rev_adj_dirs[l2].append((l1, dir))
                # Collect all locations
                self.all_locations.add(l1)
                self.all_locations.add(l2)

        # Add locations from initial state and goals that might not be in static adjacent facts
        # (e.g., isolated locations, though unlikely in typical Sokoban)
        for fact_str in task.initial_state:
             pred_args = parse_fact(fact_str)
             if pred_args[0] in ('at-robot', 'at'):
                 self.all_locations.add(pred_args[-1]) # Last argument is location
        for goal_fact_str in task.goals:
             pred_args = parse_fact(goal_fact_str)
             if pred_args[0] == 'at':
                 self.all_locations.add(pred_args[-1]) # Last argument is location

        # 3. Precompute all-pairs shortest paths
        for start_loc in self.all_locations:
            self.location_distances[start_loc] = self._bfs(start_loc, self.adj_graph)

    def _bfs(self, start_node, graph):
        """Helper function to perform BFS from a start node."""
        distances = {start_node: 0}
        queue = deque([start_node])
        while queue:
            current_node = queue.popleft()
            dist = distances[current_node]
            if current_node in graph:
                for neighbor in graph[current_node]:
                    if neighbor not in distances:
                        distances[neighbor] = dist + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Helper to get precomputed distance, returns infinity if no path."""
        if loc1 not in self.location_distances or loc2 not in self.location_distances[loc1]:
             # If loc2 is not reachable from loc1 in the static graph
             return math.inf
        return self.location_distances[loc1][loc2]

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.
        """
        # 1. Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        robot_loc = None
        box_locations = {} # box -> current_location

        # 2. Parse current state
        for fact_str in state:
            pred_args = parse_fact(fact_str)
            if pred_args[0] == 'at-robot':
                robot_loc = pred_args[1]
            elif pred_args[0] == 'at':
                box, loc = pred_args[1:]
                box_locations[box] = loc
            # We ignore 'clear' facts for this heuristic's distance calculations

        # 3. Calculate sum of box-goal distances
        total_box_distance = 0
        unreachable_goal = False
        for box, goal_loc in self.box_goals.items():
            current_loc = box_locations.get(box)
            if current_loc is None:
                 # This state is missing a box location, likely invalid.
                 unreachable_goal = True
                 break

            dist = self.get_distance(current_loc, goal_loc)
            if dist == math.inf:
                 unreachable_goal = True
                 break
            total_box_distance += dist

        if unreachable_goal:
             return math.inf # At least one box cannot reach its goal

        # 4. & 5. Calculate minimum robot distance to a valid pushing position for any box
        min_robot_to_valid_push_pos = math.inf
        found_any_valid_push_pos = False

        for box, current_loc in box_locations.items():
             # If box is already at goal, no need to consider pushing it for the robot cost part
             if self.box_goals.get(box) == current_loc:
                 continue

             # Find locations `rloc` adjacent to `current_loc`
             if current_loc in self.rev_adj_dirs:
                 for rloc, dir1 in self.rev_adj_dirs[current_loc]: # rloc is adjacent to current_loc in dir1
                     # Check if current_loc is adjacent to some floc in the same direction dir1
                     if current_loc in self.adj_dirs:
                         for floc, dir2 in self.adj_dirs[current_loc]:
                             if dir2 == dir1:
                                 # Found rloc -> current_loc -> floc collinear path
                                 # rloc is a valid pushing position for box at current_loc
                                 robot_dist = self.get_distance(robot_loc, rloc)
                                 min_robot_to_valid_push_pos = min(min_robot_to_valid_push_pos, robot_dist)
                                 found_any_valid_push_pos = True
                                 # Optimization: Once we find *a* valid push position for *this* rloc/current_loc pair,
                                 # we don't need to check other flocs for the same dir1.
                                 # break # This break is incorrect, need to find min over all rloc for this box

        # 6. Handle case where no valid push position is found
        if not found_any_valid_push_pos:
             # If goals are not met (total_box_distance > 0) and no box can be pushed, it's unsolvable.
             # If total_box_distance is 0, goal_reached check should have returned 0.
             # So if we are here and total_box_distance > 0, it's unsolvable.
             if total_box_distance > 0:
                 return math.inf
             # If total_box_distance is 0, it implies all boxes are at goals,
             # which contradicts the initial goal_reached check returning False.
             # This case should ideally not be reached in a valid problem state sequence.
             # Returning 0 here might mask an issue, but returning inf is also wrong
             # if the goal is actually met. The initial check is the definitive one for h=0.
             # If we reach here and total_box_distance is 0, it means goal_reached was false
             # but all boxes are at goals. This heuristic doesn't consider robot goal location.
             # Assuming standard Sokoban where only box locations matter for the goal,
             # this implies an inconsistency or non-standard goal. Let's rely on the initial check.
             # If total_box_distance > 0 and no push pos, it's unsolvable.
             pass # Keep infinity if not found_any_valid_push_pos and total_box_distance > 0

        # 7. Handle case where robot cannot reach any valid push position
        if min_robot_to_valid_push_pos == math.inf:
             # If goals are not met (total_box_distance > 0) and robot cannot reach any push pos, it's unsolvable.
             # Similar logic as above. If total_box_distance is 0, goal_reached should have returned 0.
             if total_box_distance > 0:
                 return math.inf
             # If total_box_distance is 0, and robot cannot reach any push pos, but goals are met (contradiction).
             # Rely on initial goal_reached check. If we are here, total_box_distance > 0.
             return math.inf # Robot cannot reach any pushable box

        # 8. Calculate final heuristic value
        return total_box_distance + min_robot_to_valid_push_pos


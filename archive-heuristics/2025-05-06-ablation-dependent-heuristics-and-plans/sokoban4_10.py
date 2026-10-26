from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic # Uncomment if using the framework

# Utility functions (can be placed outside the class)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string or invalid format gracefully
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj1 loc1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Define a minimal Heuristic base class if not provided by the framework
# This allows the code to be runnable standalone for testing the heuristic logic
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Heuristic base class not provided")


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the total number of actions (pushes and robot movements)
    required to reach the goal state. It sums, for each box not at its goal,
    the shortest path distance from the box's current location to its goal
    (estimating the number of pushes needed for that box) and the shortest
    path distance for the robot to reach a position from which it can make
    the first push towards that box's goal.

    # Assumptions
    - The grid structure and connectivity are defined solely by the `adjacent` facts.
    - Distances are computed as shortest paths on the graph defined by `adjacent` facts.
    - The cost of moving a box is estimated by its shortest path distance to the goal,
      ignoring potential blockages by other boxes or walls not represented by
      missing `adjacent` facts.
    - The robot's movement cost is estimated by the distance to reach a valid
      pushing position for the first step of the box's shortest path towards the goal.
    - The heuristic sums costs for each box independently, ignoring potential
      synergies (robot moving towards one box helps another) or conflicts
      (boxes blocking each other, robot needing to move around boxes).
    - The heuristic does not explicitly detect or penalize dead-end states (e.g., a box pushed
      into a corner with no goal access), although unreachable goals or push positions
      will result in an infinite heuristic value.

    # Heuristic Initialization
    The initialization phase precomputes static information from the task:
    - Identifies all unique locations from `adjacent` facts.
    - Builds an adjacency graph (`graph`) where nodes are locations and edges
      represent adjacency with direction.
    - Builds a reverse adjacency graph (`rev_graph`) used to find locations
      adjacent *to* a given location.
    - Builds a map (`robot_pos_for_push`) that, for a given box move (from_loc, to_loc),
      provides the required robot location to perform that push.
    - Computes all-pairs shortest path distances (`dist`) between all locations
      using Breadth-First Search (BFS) on the `graph`.
    - Extracts the goal location for each box from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot and each box present in the state.
    2. Initialize the total heuristic cost to 0.
    3. For each box that has a specified goal location in the task and is not currently at that goal location:
        a. Get the box's current location (`curr_loc`) and its goal location (`goal_loc`).
        b. Calculate the shortest path distance (`box_dist`) from `curr_loc` to `goal_loc`
           using the precomputed distances (`self.dist`). If the goal is unreachable,
           return infinity (or a very large number) as the heuristic value.
        c. Add `box_dist` to the total cost. This represents the minimum number of
           push actions needed for this box if the path were clear.
        d. Find potential next locations (`next_loc`) for the box: these are neighbors
           of `curr_loc` in the graph that are exactly one step closer to `goal_loc`
           according to the precomputed distances (`self.dist`).
        e. Initialize a variable `min_robot_dist_to_push_pos` to infinity.
        f. For each potential `next_loc`:
            i. Determine the required robot location (`r_loc`) from which the robot
               must push the box from `curr_loc` to `next_loc`. This required location
               is looked up in the precomputed `self.robot_pos_for_push` map using
               the tuple `(curr_loc, next_loc)` as the key.
            ii. If a required robot location `r_loc` is found:
                - Calculate the shortest path distance from the robot's current location
                  to this required `r_loc` using `self.dist`.
                - Update `min_robot_dist_to_push_pos` with the minimum distance found so far.
        g. If a valid push position towards the goal was found for the box (i.e.,
           `min_robot_dist_to_push_pos` is not infinity), add `min_robot_dist_to_push_pos`
           to the total cost. If no such push position exists (e.g., the box is
           in a local optimum or stuck), the heuristic implicitly becomes infinity
           if `min_robot_dist_to_push_pos` remains infinity.
    4. Return the accumulated total cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by precomputing graph information and goal locations."""
        self.goals = task.goals
        static_facts = task.static

        self.locations = set()
        self.graph = {} # location -> list[(neighbor_location, direction)]
        # rev_graph[l2] stores locations l1 such that (adjacent l1 l2 d)
        self.rev_graph = {} # location -> list[(prev_location, direction)]

        # 1. Collect locations and build graph/rev_graph
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'adjacent' and len(parts) == 4:
                l1, l2, d = parts[1], parts[2], parts[3]
                self.locations.add(l1)
                self.locations.add(l2)
                self.graph.setdefault(l1, []).append((l2, d))
                self.rev_graph.setdefault(l2, []).append((l1, d))

        # Ensure all locations mentioned in goals are included, even if isolated
        for goal in self.goals:
             if match(goal, "at", "*", "*"):
                 _, box, location = get_parts(goal)
                 self.locations.add(location)
             elif get_parts(goal)[0] == 'and':
                 # Simple split for (and (f1) (f2) ...) format
                 goal_body_str = goal[4:-1].strip()
                 fact_strings = goal_body_str.replace(')(', ') (')
                 fact_strings = fact_strings.split(' (')
                 fact_strings = ['(' + fs for fs in fact_strings if fs]

                 for fact in fact_strings:
                     if match(fact, "at", "*", "*"):
                         _, box, location = get_parts(fact)
                         self.goal_map[box] = location


        # 2. Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_node in list(self.locations): # Iterate over a copy as locations might be added in BFS
            self.dist[start_node] = self._bfs(start_node)

        # 3. Build robot_pos_for_push map
        # robot_pos_for_push[(bloc, floc)] = rloc if (adjacent rloc bloc dir) and (adjacent bloc floc dir)
        self.robot_pos_for_push = {}
        for bloc in self.locations:
            # Find all locations floc reachable from bloc by a push, and the required rloc
            # Iterate through neighbors of bloc (potential floc)
            if bloc in self.graph:
                for floc, push_dir in self.graph[bloc]:
                     # Find locations rloc adjacent to bloc such that (adjacent rloc bloc push_dir)
                     # These are locations l1 in rev_graph[bloc] where the direction from l1 to bloc is push_dir
                     if bloc in self.rev_graph:
                         for rloc, dir_from_rloc_to_bloc in self.rev_graph[bloc]:
                             if dir_from_rloc_to_bloc == push_dir:
                                 self.robot_pos_for_push[(bloc, floc)] = rloc
                                 # Assuming only one rloc exists for a given bloc and push_dir in valid PDDL

        # 4. Extract goal locations for each box
        self.goal_map = {}
        for goal in self.goals:
            # Goals can be complex, but in Sokoban instances, they are typically (at box location)
            # We assume goals are conjunctions of (at box location)
            if match(goal, "at", "*", "*"):
                 _, box, location = get_parts(goal)
                 self.goal_map[box] = location
            # Handle potential 'and' goals
            elif get_parts(goal)[0] == 'and':
                 # Simple split for (and (f1) (f2) ...) format
                 goal_body_str = goal[4:-1].strip()
                 fact_strings = goal_body_str.replace(')(', ') (')
                 fact_strings = fact_strings.split(' (')
                 fact_strings = ['(' + fs for fs in fact_strings if fs]

                 for fact in fact_strings:
                     if match(fact, "at", "*", "*"):
                         _, box, location = get_parts(fact)
                         self.goal_map[box] = location


    def _bfs(self, start_node):
        """Performs BFS from start_node to find shortest distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.locations}
        if start_node not in self.locations:
             # Start node is not in the graph of locations defined by adjacent facts
             # This might happen if the initial state has an object/robot in an isolated location
             # In this case, no movement is possible from here. Distances remain inf.
             return distances

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node is in graph (handle potential nodes with no outgoing edges)
            if current_node in self.graph:
                for neighbor_node, _ in self.graph[current_node]:
                    # Ensure neighbor_node is a known location
                    if neighbor_node in self.locations and distances[neighbor_node] == float('inf'):
                        distances[neighbor_node] = distances[current_node] + 1
                        queue.append(neighbor_node)

        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify current robot and box locations
        robot_loc = None
        box_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts

            predicate = parts[0]
            if predicate == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            elif predicate == 'at' and len(parts) == 3: # (at box location)
                box, loc = parts[1], parts[2]
                box_locations[box] = loc

        # Ensure robot location is found and is a known location
        if robot_loc is None or robot_loc not in self.locations:
             # Robot location is missing or invalid, cannot move
             return float('inf')


        # 2. Initialize total heuristic cost
        total_cost = 0

        # 3. For each box not at its goal
        for box, goal_loc in self.goal_map.items():
            curr_loc = box_locations.get(box) # Get current location

            # If a box mentioned in the goal is not in the current state, something is wrong.
            # Or maybe it implies the box was removed? Assume valid states.
            if curr_loc is None:
                 # This shouldn't happen in standard Sokoban problem instances
                 # where objects persist. Treat as an unsolvable state.
                 return float('inf')

            if curr_loc == goal_loc:
                continue # Box is already at its goal

            # a. Calculate box distance (estimated pushes)
            # Check if curr_loc and goal_loc are in the precomputed distances map
            if curr_loc not in self.dist or goal_loc not in self.dist[curr_loc]:
                 # Goal location unreachable from current box location
                 return float('inf') # Box is stuck or goal is unreachable

            box_dist = self.dist[curr_loc][goal_loc]
            total_cost += box_dist

            # b. Find required robot location for the first push towards the goal
            min_robot_dist_to_push_pos = float('inf')
            found_push_pos_towards_goal = False

            # Find neighbors of curr_loc that are one step closer to goal_loc
            potential_next_steps = []
            if curr_loc in self.graph:
                for neighbor_loc, direction in self.graph[curr_loc]:
                    # Check if neighbor_loc is a known location and reachable from curr_loc
                    if neighbor_loc in self.locations and self.dist[curr_loc].get(neighbor_loc, float('inf')) == 1:
                        # Check if neighbor_loc is one step closer to the goal
                        if neighbor_loc in self.dist and goal_loc in self.dist[neighbor_loc] and self.dist[neighbor_loc][goal_loc] == box_dist - 1:
                             potential_next_steps.append(neighbor_loc)

            # For each potential next step, find the required robot location and its distance
            for next_loc in potential_next_steps:
                # The required robot location r_loc is where robot_pos_for_push[(curr_loc, next_loc)] points
                required_robot_loc = self.robot_pos_for_push.get((curr_loc, next_loc))

                if required_robot_loc is not None:
                    # Check if required_robot_loc is a known location and reachable by the robot
                    if robot_loc in self.dist and required_robot_loc in self.dist[robot_loc]:
                        robot_dist_to_push_pos = self.dist[robot_loc][required_robot_loc]
                        min_robot_dist_to_push_pos = min(min_robot_dist_to_push_pos, robot_dist_to_push_pos)
                        found_push_pos_towards_goal = True

            # Add the minimum distance for the robot to reach a valid first push position towards the goal
            # If no valid push position towards the goal was found, min_robot_dist_to_push_pos remains inf.
            # If box_dist was > 0 but no push_pos towards goal was found, this state is likely bad.
            # Adding inf here makes the total_cost inf, pruning this state.
            if found_push_pos_towards_goal:
                 total_cost += min_robot_dist_to_push_pos
            elif box_dist > 0: # Box needs moving, but cannot move towards goal via shortest path
                 return float('inf') # Treat as a dead end or very costly state


        # Heuristic is 0 if and only if all goal conditions (box locations) are met.
        # If total_cost is finite, it's >= 0.
        # If total_cost is 0, it means box_dist was 0 for all boxes, and min_robot_dist_to_push_pos was 0
        # (which happens trivially if box_dist is 0). So all boxes are at goal.
        # If total_cost is > 0, at least one box is not at goal or robot needs to move.
        return total_cost

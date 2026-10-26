import collections
import math

# Assume the Task class is available from the provided code-file-task
# from task import Task # Assuming task.py exists and contains the Task class

class sokobanHeuristic:
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing the minimum
    number of pushes required for each box to reach its goal location, plus
    the minimum number of robot moves required to reach a position from which
    it can make the first push towards the goal for any box that needs moving.
    Distances are calculated based on the static adjacency graph, ignoring
    dynamic obstacles (other boxes, robot) for efficiency.

    Assumptions:
    - The location names are strings following the pattern 'loc_row_col' (though parsing doesn't rely on this structure, only adjacency).
    - The adjacency relations define a connected graph for all relevant locations.
    - The goal specifies a unique target location for each box using `(at box_name loc_name)` facts.
    - The PDDL fact strings are in the format '(predicate arg1 arg2 ...)'.

    Heuristic Initialization:
    1. Parse the goal facts from `task.goals` to create a mapping from each box
       object name to its target goal location name (`self.box_goals`).
    2. Parse the static `adjacent` facts from `task.static` to build the
       location graph representation:
       - `self.locations`: A set of all unique location names.
       - `self.adj_list`: An adjacency list mapping each location name to a list
         of location names it is adjacent to.
       - `self.adj_map_dir`: A mapping from a location name to a dictionary,
         where keys are direction names ('up', 'down', 'left', 'right') and
         values are the adjacent location names in that direction.
    3. Compute all-pairs shortest paths on the location graph using Breadth-First
       Search (BFS) starting from every location. The distances are stored in
       `self.dist[l1][l2]`, representing the minimum number of `move` actions
       (or steps on the graph) between `l1` and `l2` ignoring dynamic obstacles.
    4. Compute `self.push_positions`: A mapping from a tuple `(box_location, next_box_location)`
       to the required robot location for pushing. If a box is at `l_box` and
       needs to be pushed to `l_next` (where `l_next` is adjacent to `l_box`
       in direction `dir`), the robot must be at `l_r_push` such that `l_r_push`
       is adjacent to `l_box` in the *same* direction `dir`. This mapping is
       precomputed by iterating through all possible robot positions and push
       directions to find the corresponding box start and end locations.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is a goal state by calling `self.task.goal_reached(state)`.
       If it returns `True`, the heuristic value is 0, and we return immediately.
    2. Parse the current state (a frozenset of fact strings) to identify the
       robot's current location (`robot_loc`) and the current location of each
       box (`box_locs`), stored as a dictionary mapping box names to location names.
    3. Initialize the total heuristic value `h = 0`.
    4. Initialize `min_robot_dist_to_any_push_pos = math.inf`. This variable
       will keep track of the minimum distance from the current robot location
       to any valid pushing position required for any box that is not yet at its goal.
    5. Iterate through each box that has a goal location.
    6. For the current box, retrieve its current location (`current_loc`) from `box_locs`.
    7. If the box is found in the state and its `current_loc` is different from
       its `goal_loc`:
       a. Calculate the shortest path distance from the box's `current_loc` to
          its `goal_loc` using the precomputed distances `self.dist[current_loc][goal_loc]`.
          This distance represents the minimum number of push actions needed for
          this box if the path were clear and the robot always in position.
       b. Add this `box_dist` to the total heuristic `h`. If the distance is
          `math.inf` (meaning the goal is unreachable on the static graph),
          return `math.inf` as the state is likely unsolvable.
       c. Find potential next locations (`l_next`) for the box that are one step
          closer to the goal. Iterate through all locations `l_next` adjacent to
          `current_loc` (using `self.adj_list`). If the precomputed distance
          `self.dist[l_next][goal_loc]` is less than `box_dist`, then `l_next`
          is on a shortest path from `current_loc` to `goal_loc`.
       d. For each such `l_next`:
          i. Determine the required robot pushing position `l_r_push`. This is
             the location the robot must be at to push the box from `current_loc`
             to `l_next`. This position is looked up in `self.push_positions[(current_loc, l_next)]`.
          ii. If a valid `l_r_push` exists (it might not if the location graph
              doesn't support pushing in that direction, e.g., a dead end),
              calculate the shortest path distance from the current `robot_loc`
              to `l_r_push` using the precomputed distances `self.dist[robot_loc][l_r_push]`.
          iii. Update `min_robot_dist_to_any_push_pos` with the minimum of its
               current value and the calculated `robot_dist`. If `robot_dist`
               is `math.inf`, return `math.inf` as the state is likely unsolvable.
    8. After iterating through all boxes, if `min_robot_dist_to_any_push_pos` is
       not `math.inf` (meaning there was at least one box needing movement and
       a reachable pushing position was found), add `min_robot_dist_to_any_push_pos`
       to the total heuristic `h`. This component estimates the robot's effort
       to get into a position to start making progress on *any* box.
    9. Return the final calculated heuristic value `h`.
    """

    def __init__(self, task):
        self.task = task
        self.box_goals = {}
        self.locations = set()
        self.adj_list = collections.defaultdict(list)
        self.adj_map_dir = collections.defaultdict(dict)
        self.dist = collections.defaultdict(dict)
        self.push_positions = {}

        # 1. Parse goal facts
        for fact in task.goals:
            pred_args = self._parse_fact(fact)
            if pred_args[0] == 'at':
                # Goal fact is (at box_name loc_name)
                self.box_goals[pred_args[1]] = pred_args[2]

        # 2. Parse static adjacent facts and build graph structures
        for fact in task.static:
            pred_args = self._parse_fact(fact)
            if pred_args[0] == 'adjacent':
                # Fact is (adjacent loc1 loc2 direction)
                loc1, loc2, direction = pred_args[1], pred_args[2], pred_args[3]
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.adj_list[loc1].append(loc2)
                self.adj_map_dir[loc1][direction] = loc2

        # 3. Compute all-pairs shortest paths using BFS
        for start_node in self.locations:
            self._bfs(start_node)

        # 4. Compute push positions
        # Iterate through all possible robot positions (l_r_push)
        # For each direction (dir) the robot can move from l_r_push
        #   Let l_box be the location adjacent to l_r_push in direction dir
        #   If l_box exists AND l_box is adjacent to l_next in the SAME direction dir
        #   Then robot at l_r_push can push box from l_box to l_next
        for l_r_push in self.locations:
            for dir in ['up', 'down', 'left', 'right']: # Iterate through all possible push directions
                if dir in self.adj_map_dir[l_r_push]:
                    l_box = self.adj_map_dir[l_r_push][dir] # Location of the box
                    if l_box in self.adj_map_dir and dir in self.adj_map_dir[l_box]:
                        l_next = self.adj_map_dir[l_box][dir] # Target location for the box
                        # Robot at l_r_push can push box from l_box to l_next
                        self.push_positions[(l_box, l_next)] = l_r_push


    def _parse_fact(self, fact_string):
        """Parses a PDDL fact string into a tuple (predicate, arg1, arg2, ...)."""
        # Remove outer parentheses and split by space
        # Handles strings like '(at box1 loc_4_4)' -> ['at', 'box1', 'loc_4_4']
        parts = fact_string[1:-1].split()
        return tuple(parts)

    def _bfs(self, start_node):
        """Performs BFS from start_node to compute distances to all reachable nodes."""
        q = collections.deque([start_node])
        self.dist[start_node][start_node] = 0
        visited = {start_node}

        while q:
            curr_node = q.popleft()
            curr_dist = self.dist[start_node][curr_node]

            # Use adj_list for simple graph traversal
            for neighbor in self.adj_list[curr_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    self.dist[start_node][neighbor] = curr_dist + 1
                    q.append(neighbor)

    def sokobanHeuristic(self, state):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        # Check if the goal is reached
        if self.task.goal_reached(state):
            return 0

        robot_loc = None
        box_locs = {} # {box: location}

        # Parse current state to find robot and box locations
        for fact in state:
            pred_args = self._parse_fact(fact)
            pred = pred_args[0]
            if pred == 'at-robot':
                robot_loc = pred_args[1]
            elif pred == 'at':
                # Fact is (at box_name loc_name)
                box, loc = pred_args[1], pred_args[2]
                box_locs[box] = loc
            # Ignore 'clear' facts for this heuristic calculation

        # Initialize heuristic components
        h = 0
        min_robot_dist_to_any_push_pos = math.inf

        # Ensure robot location was found (should always be present in a valid state)
        if robot_loc is None:
             # This indicates an unexpected state structure, return a high value
             return math.inf

        # Iterate through each box that has a goal location
        for box, goal_loc in self.box_goals.items():
            current_loc = box_locs.get(box) # Get current location of the box

            # If the box exists in the current state and is not at its goal location
            if current_loc and current_loc != goal_loc:
                # Calculate box distance (minimum pushes)
                # Use precomputed distance. Handle case where goal is unreachable on the static graph.
                box_dist = self.dist[current_loc].get(goal_loc, math.inf)
                if box_dist == math.inf:
                    # Box cannot reach goal on the static graph - likely unsolvable
                    return math.inf
                h += box_dist

                # Find minimum robot distance to a push position for this box
                # We need to push from current_loc towards goal_loc
                # Find adjacent locations l_next that are one step closer to the goal
                # Iterate through all locations adjacent to current_loc
                for l_next in self.adj_list[current_loc]:
                    # Check if l_next is closer to the goal using precomputed distances
                    dist_l_next_to_goal = self.dist[l_next].get(goal_loc, math.inf)
                    # Check if l_next is indeed closer and reachable from current_loc
                    # (dist[current_loc][l_next] should be 1, which is true for adjacent nodes)
                    if dist_l_next_to_goal < box_dist:
                        # l_next is a valid next step towards the goal
                        # Find the required robot pushing position for this specific push (current_loc -> l_next)
                        l_r_push = self.push_positions.get((current_loc, l_next))

                        # If a valid push position exists for this move
                        if l_r_push:
                            # Calculate robot distance from its current location to the required push position
                            # Use precomputed distance on the static graph. Handle unreachable.
                            robot_dist = self.dist[robot_loc].get(l_r_push, math.inf)
                            if robot_dist == math.inf:
                                # Robot cannot reach the required push position on the static graph - likely unsolvable
                                return math.inf
                            # Update the minimum robot distance needed across all potential first pushes for all boxes
                            min_robot_dist_to_any_push_pos = min(min_robot_dist_to_any_push_pos, robot_dist)

        # Add the minimum robot distance required to get into *any* useful pushing position
        # for any box that needs moving. This component is added only once.
        if min_robot_dist_to_any_push_pos != math.inf:
             h += min_robot_dist_to_any_push_pos
        # Note: If no boxes need moving, min_robot_dist_to_any_push_pos remains math.inf,
        # and this component is not added, correctly resulting in h=0 if the goal is reached.

        return h

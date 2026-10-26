# Use the actual Heuristic base class provided by the planner environment
from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and multiple spaces
    return fact.strip()[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args is empty
    if not args:
        return True # Matches any fact if no pattern is given
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components:
    1. The minimum number of pushes required for each box to reach its goal location (sum of shortest path distances).
    2. The minimum number of moves required for the robot to reach a position from which it can make the first push towards the goal for any box that needs moving.

    # Assumptions
    - The grid structure is defined by `adjacent` predicates.
    - Shortest path distances on this grid represent minimum moves/pushes in an empty grid.
    - The cost of a 'move' action is 1.
    - The cost of a 'push' action is 1.
    - Deadlocks are not explicitly detected or penalized beyond reachability towards the goal. A state where a box cannot move closer to its goal might receive an infinite heuristic value.
    - The heuristic ignores the `clear` predicate, assuming paths are traversable by the robot and pushable by the box based solely on adjacency.

    # Heuristic Initialization
    - Parse all unique locations mentioned in the initial state and static facts.
    - Build a graph representation of the grid based on `adjacent` predicates. Store adjacency lists and direction mappings.
    - Pre-compute all-pairs shortest path distances between locations using BFS on the constructed graph.
    - Store the goal location for each box.
    - Store the mapping from direction strings to their opposite direction strings.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Extract Relevant Information:
       - Identify the current location of the robot.
       - Identify the current location of every box.

    2. Check for Goal State:
       - Determine which boxes are not currently at their designated goal locations.
       - If all boxes are at their goals, the current state is a goal state, and the heuristic value is 0.

    3. Calculate Box-Goal Distance Component:
       - Initialize a running total for the heuristic cost.
       - For each box that is not at its goal location:
         - Find the shortest path distance from the box's current location to its goal location using the pre-computed distances.
         - Add this distance to the total cost. This represents the minimum number of push actions required for this box in an unblocked environment. If a goal is unreachable from a box's current location, return infinity.

    4. Calculate Robot Positioning Component:
       - The robot must be in a specific location adjacent to a box (on the side opposite the push direction) to push it. To make progress, the robot needs to get into position to push *some* box towards its goal.
       - Identify the set of potential "required robot locations": For each box that needs moving, find locations adjacent to the box's current position that are strictly closer to the box's goal location (using pre-computed distances). For each such "next step" location, determine the direction of the required push (from box current location to next step location). The required robot location is the location adjacent to the box's current position in the *opposite* direction of this push.
       - Find the minimum shortest path distance from the robot's current location to *any* location in the set of required robot locations. This is the estimated robot movement cost to get into a useful position.
       - If the set of required robot locations is empty (meaning no box can move closer to its goal) or if the robot cannot reach any of the required robot locations, this state is likely problematic (deadlock or unreachable). In such cases, return infinity.

    5. Sum Components:
       - The final heuristic value is the sum of the total box-goal distance (step 3) and the minimum robot positioning distance (step 4).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the grid graph and pre-computing distances.
        """
        # Call the base class constructor if needed
        super().__init__(task)

        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Collect all unique locations
        all_locations_set = set()
        # Locations from initial state
        for fact in initial_state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at-robot':
                all_locations_set.add(parts[1])
            elif parts and parts[0] == 'at':
                 all_locations_set.add(parts[2])
        # Locations from static facts (adjacent)
        for fact in static_facts:
             parts = get_parts(fact)
             if parts and parts[0] == 'adjacent':
                 all_locations_set.add(parts[1])
                 all_locations_set.add(parts[2])

        self.all_locations = sorted(list(all_locations_set)) # Sort for consistent ID mapping
        self.loc_to_id = {loc: i for i, loc in enumerate(self.all_locations)}
        self.id_to_loc = {i: loc for i, loc in enumerate(self.all_locations)}
        num_locations = len(self.all_locations)

        # 2. Build graph representations
        self.adj_list = {i: [] for i in range(num_locations)} # id -> [(neighbor_id, dir)]
        self.adj_dirs = {} # (id1, id2) -> dir

        self.opposite_dir = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'adjacent':
                l1_str, l2_str, dir_str = parts[1], parts[2], parts[3]
                u = self.loc_to_id[l1_str]
                v = self.loc_to_id[l2_str]
                self.adj_list[u].append((v, dir_str))
                self.adj_dirs[(u, v)] = dir_str

        # 3. Pre-compute all-pairs shortest paths using BFS
        self.distances = {} # start_id -> {end_id -> distance}
        for start_id in range(num_locations):
            self.distances[start_id] = {}
            q = deque([(start_id, 0)])
            visited = {start_id}
            while q:
                curr_id, d = q.popleft()
                self.distances[start_id][curr_id] = d

                # Iterate through neighbors using adj_list
                for neighbor_id, _ in self.adj_list[curr_id]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        q.append((neighbor_id, d + 1))

        # 4. Store goal locations for boxes
        self.goal_locations = {} # box_name -> goal_loc_id
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'at':
                box_name, loc_str = parts[1], parts[2]
                # Ensure goal location is in our graph (should be if problem is well-formed)
                if loc_str in self.loc_to_id:
                    self.goal_locations[box_name] = self.loc_to_id[loc_str]
                else:
                    # This goal location is not in the known graph.
                    # Any state requiring this goal is unsolvable via the defined graph.
                    # We can mark this box's goal as unreachable.
                    self.goal_locations[box_name] = None # Use None to indicate unreachable goal


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify current locations
        robot_loc_str = None
        box_locations_str = {} # box_name -> loc_str

        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at-robot':
                robot_loc_str = parts[1]
            elif parts and parts[0] == 'at':
                box_locations_str[parts[1]] = parts[2]

        # Ensure robot location is known and mapped
        if robot_loc_str is None or robot_loc_str not in self.loc_to_id:
             # This state is malformed or unreachable in a way the heuristic can't handle
             return float('inf')
        robot_id = self.loc_to_id[robot_loc_str]

        box_locations_id = {}
        for b, loc_str in box_locations_str.items():
             if loc_str in self.loc_to_id:
                  box_locations_id[b] = self.loc_to_id[loc_str]
             else:
                  # Box is in an unknown location
                  return float('inf')


        # 2. Identify boxes to move
        boxes_to_move = {}
        for b, bloc_id in box_locations_id.items():
             # Check if box has a defined goal and is not at the goal
             if b in self.goal_locations:
                 gloc_id = self.goal_locations[b]
                 # If goal was marked as unreachable (None), this box can never reach it
                 if gloc_id is None:
                     return float('inf') # Problem is unsolvable if this box needs moving
                 if bloc_id != gloc_id:
                     boxes_to_move[b] = bloc_id
             # else: box doesn't have a goal specified, ignore it for heuristic?
             # Assuming all boxes in initial state have goals.

        # 3. If all boxes are at goals, heuristic is 0
        if not boxes_to_move:
            return 0

        total_cost = 0

        # 4. Calculate total box-goal distance
        for box, bloc_id in boxes_to_move.items():
            gloc_id = self.goal_locations[box]
            # Distance from box current location to goal location
            # Check reachability from box location to goal location
            if bloc_id not in self.distances or gloc_id not in self.distances[bloc_id]:
                 # Goal is unreachable from box location via the grid graph
                 return float('inf')
            total_cost += self.distances[bloc_id][gloc_id]

        # 5. Calculate robot cost
        min_robot_dist = float('inf')
        required_robot_locations_ids = set()

        for box, bloc_id in boxes_to_move.items():
            gloc_id = self.goal_locations[box]

            # Find neighbors of bloc_id that are steps towards gloc_id
            # A neighbor v_id is a step towards gloc_id if dist(v_id, gloc_id) < dist(bloc_id, gloc_id)
            # We need to ensure v_id is reachable from bloc_id and gloc_id is reachable from v_id
            potential_next_steps_ids = [
                v_id for v_id, _ in self.adj_list[bloc_id]
                if v_id in self.distances and gloc_id in self.distances[v_id] and self.distances[v_id][gloc_id] < self.distances[bloc_id][gloc_id]
            ]

            for next_loc_id in potential_next_steps_ids:
                # Find the direction from bloc_id to next_loc_id
                dir_to_next = self.adj_dirs.get((bloc_id, next_loc_id))
                if dir_to_next is None: continue # Should not happen

                # Find the required robot location adjacent to bloc_id in the opposite direction
                opposite_dir = self.opposite_dir.get(dir_to_next)
                if opposite_dir is None: continue # Should not happen

                # Look for a location u_id such that adjacent(bloc_id, u_id, opposite_dir)
                required_rloc_candidates_ids = [
                    u_id for u_id, u_dir in self.adj_list[bloc_id]
                    if u_dir == opposite_dir
                ]

                for rloc_id in required_rloc_candidates_ids:
                    # Add this required location to the set if it's a valid location ID
                    if rloc_id in self.id_to_loc:
                         required_robot_locations_ids.add(rloc_id)


        # Find the minimum distance from the robot to any required push location
        if required_robot_locations_ids:
            # Ensure the robot can actually reach at least one of these required locations
            reachable_required_locations_ids = [
                rloc_id for rloc_id in required_robot_locations_ids
                if robot_id in self.distances and rloc_id in self.distances[robot_id]
            ]

            if reachable_required_locations_ids:
                 min_robot_dist = min(self.distances[robot_id][rloc_id] for rloc_id in reachable_required_locations_ids)
                 robot_cost = min_robot_dist
            else:
                 # Robot cannot reach any required push location for any box that needs moving
                 # This state is likely unsolvable or requires complex path clearing not captured
                 return float('inf')
        else:
             # This means there are boxes not at their goal, but for every such box,
             # there is no adjacent location strictly closer to the goal.
             # This implies the boxes are stuck in local minima w.r.t. distance, likely deadlocked.
             # Return infinity.
             return float('inf')


        total_cost += robot_cost

        return total_cost

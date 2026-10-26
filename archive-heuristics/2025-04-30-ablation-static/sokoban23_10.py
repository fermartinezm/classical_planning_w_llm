from fnmatch import fnmatch
from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def shortest_path_distance_from_source(start, graph, all_locations):
    """Compute shortest path distances from a source using BFS."""
    distances = {loc: float('inf') for loc in all_locations}
    if start not in all_locations:
        return distances # Start location is not in the known locations

    distances[start] = 0
    queue = deque([start])
    visited = {start}

    while queue:
        current_loc = queue.popleft()

        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_loc] + 1
                    queue.append(neighbor)

    return distances

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the minimum
    number of pushes required for each box to reach its goal, plus the minimum
    robot movement cost to get into a position to make the first push for any
    of the boxes that need moving.

    # Assumptions
    - The grid structure is defined by `adjacent` facts.
    - All locations relevant to the problem (initial state, goals, adjacent facts)
      are considered.
    - The heuristic ignores potential conflicts between boxes and assumes boxes
      can be pushed along shortest paths if the robot can reach the required
      push position.
    - Deadlocks (like a box in a corner it cannot be pushed out of) are partially
      handled by returning infinity if a box cannot reach its goal or if the
      robot cannot reach a valid push position for any box.

    # Heuristic Initialization
    - Parse all locations from the initial state, goals, and static facts.
    - Build an undirected graph representing the grid connectivity for distance calculations.
    - Build a directed graph representing adjacency with directions.
    - Build a map from (target_location, push_direction) to the required robot
      location for a push action, based on the push action preconditions.
    - Compute all-pairs shortest path distances on the undirected grid graph.
    - Extract goal locations for each box from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Extract Relevant Information:
       - Identify the current location of the robot.
       - Identify the current location of every box.

    2. Identify Boxes to Move:
       - Create a list of boxes that are not currently at their designated goal locations.
       - If this list is empty, the state is a goal state, and the heuristic is 0.

    3. Calculate Box Movement Cost:
       - Initialize a total cost for box movements to 0.
       - For each box that needs to be moved:
         - Calculate the shortest path distance (minimum number of pushes) from its
           current location to its goal location on the grid graph (ignoring other
           objects). Add this distance to the total box movement cost.
         - If any box cannot reach its goal (distance is infinity), the state is
           likely unsolvable, and the heuristic should be infinity.

    4. Calculate Robot Positioning Cost:
       - Initialize a minimum robot distance to a push position to infinity.
       - For each box that needs to be moved:
         - Determine the location `L_next` that is the first step along a shortest
           path from the box's current location `L_box` towards its goal `L_goal`.
         - Determine the direction `dir` of the push action required to move the
           box from `L_box` to `L_next`. Based on the PDDL `push` action, this
           is the direction `d` such that `(L_next, d)` is in the directed graph
           from `L_box`.
         - Determine the required robot location `L_prev` to perform this push.
           Based on the `push` action precondition `(adjacent ?rloc ?bloc ?dir)`,
           the robot must be at `L_prev` such that `(L_box, dir)` is in the
           directed graph from `L_prev`. Use the pre-computed map to find `L_prev`.
         - Calculate the shortest path distance from the robot's current location
           to this required push location `L_prev`.
         - Update the minimum robot distance found so far with this distance.

    5. Combine Costs:
       - If there are boxes to move (step 2), the total heuristic value is the
         sum of the total box movement cost (step 3) and the minimum robot
         distance to a push position (step 4).
       - If no valid push position was found for any box that needs moving, the
         minimum robot distance remains infinity, resulting in an infinite heuristic.

    6. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by building the grid graph and pre-computing distances."""
        self.goals = task.goals
        static_facts = task.static

        # 1. Parse all locations
        self.locations = set()
        # Get locations from adjacent facts
        for fact in static_facts:
             parts = get_parts(fact)
             if parts and parts[0] == 'adjacent' and len(parts) == 4:
                 self.locations.add(parts[1])
                 self.locations.add(parts[2])
        # Get locations from initial state 'at' and 'at-robot' facts
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] in ('at-robot', 'at') and len(parts) > 1:
                 self.locations.add(parts[-1]) # Last part is the location
        # Get locations from goal 'at' facts
        for goal in task.goals:
             parts = get_parts(goal)
             if parts and parts[0] == 'at' and len(parts) == 3:
                 self.locations.add(parts[-1]) # Last part is the location

        # Initialize graph dictionaries with all locations
        self.graph = {loc: set() for loc in self.locations}
        self.directed_graph = {loc: set() for loc in self.locations}
        # Map (target_loc, push_direction) -> required_robot_loc
        self.push_precondition_robot_pos = {}

        # 2. Build graphs and push position map
        # Interpretation: (adjacent l1 l2 dir) means l1 -> l2 with direction dir
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'adjacent' and len(parts) == 4:
                l1, l2, direction = parts[1], parts[2], parts[3]
                # Ensure locations are valid
                if l1 in self.locations and l2 in self.locations:
                    self.graph[l1].add(l2)
                    self.graph[l2].add(l1) # Undirected

                    self.directed_graph[l1].add((l2, direction))

                    # If l1 -> l2 is direction 'dir', then to push from l2 in direction 'dir',
                    # the robot must be at l1.
                    # Push action: adjacent(?rloc, ?bloc, ?dir) and adjacent(?bloc, ?floc, ?dir)
                    # If box moves from ?bloc to ?floc in direction 'dir_move', then ( ?floc, dir_move ) is in directed_graph[?bloc].
                    # The push direction ?dir in the action is 'dir_move'.
                    # Robot needs to be at ?rloc such that adjacent(?rloc, ?bloc, dir_move).
                    # This means ( ?bloc, dir_move ) is in directed_graph[?rloc].
                    # So, if (target, dir) is in directed_graph[source], then source is the required robot pos
                    # to push from target in direction dir.
                    # Map key: (target_loc, push_direction), Value: required_robot_loc
                    self.push_precondition_robot_pos[(l2, direction)] = l1


        # 3. Compute all-pairs shortest paths on the undirected graph
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = shortest_path_distance_from_source(start_loc, self.graph, self.locations)

        # 4. Extract goal locations for boxes
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'at' and len(parts) == 3: # (at box loc)
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # 1. Identify current locations
        robot_location = None
        box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at-robot' and len(parts) == 2:
                robot_location = parts[1]
            elif parts and parts[0] == 'at' and len(parts) == 3:
                box, location = parts[1], parts[2]
                box_locations[box] = location

        # Ensure robot location is found (should always be the case in valid states)
        if robot_location is None:
             return float('inf') # Should not happen in valid states

        # 2. Identify boxes not at goal
        boxes_to_move = []
        for box, goal_loc in self.goal_locations.items():
            # Check if box exists in the current state and is not at its goal
            if box in box_locations and box_locations[box] != goal_loc:
                 boxes_to_move.append(box)
            elif box not in box_locations:
                 # Goal requires a box that is not in the state (e.g., disappeared?)
                 # This state is likely invalid or unsolvable.
                 return float('inf')


        # 3. If all boxes are at goals, heuristic is 0
        if not boxes_to_move:
            return 0

        total_box_dist = 0
        min_robot_dist_to_push = float('inf')

        # 4. Calculate costs for each box
        for box in boxes_to_move:
            l_box = box_locations[box]
            l_goal = self.goal_locations[box]

            # Get box-to-goal distance (minimum pushes)
            # Use .get() with {} and float('inf') for safety if location not in distances map
            dist_box_goal = self.distances.get(l_box, {}).get(l_goal, float('inf'))

            # If any box cannot reach its goal, the state is likely unsolvable
            if dist_box_goal == float('inf'):
                 return float('inf')

            total_box_dist += dist_box_goal

            # Find required robot push position for the first step towards goal
            l_prev_for_this_box = None
            push_direction = None # This is the PDDL direction for the push action

            # Find a neighbor L_next that is one step closer to the goal
            # Iterate through neighbors of L_box in the undirected graph
            if l_box in self.graph:
                for l_next in self.graph[l_box]:
                    # Check if this neighbor is indeed one step closer to the goal
                    if self.distances.get(l_next, {}).get(l_goal, float('inf')) == dist_box_goal - 1:
                        # Found a valid next step towards goal (L_next)
                        # Find the PDDL direction for the push from L_box to L_next
                        # This is the direction 'd' such that (L_next, d) is in directed_graph[L_box]
                        if l_box in self.directed_graph:
                            for neighbor, direction in self.directed_graph[l_box]:
                                if neighbor == l_next:
                                    push_direction = direction
                                    break
                        # Found L_next and push_direction, break from neighbor loop
                        if push_direction:
                            break

            # Find L_prev using the pre-computed map: (target_loc, push_direction) -> required_robot_loc
            # Here, target_loc is L_box, push_direction is the direction found above
            if push_direction:
                 l_prev_for_this_box = self.push_precondition_robot_pos.get((l_box, push_direction))

            # Calculate robot distance to this required push position
            if l_prev_for_this_box: # Check if a valid push position exists for this specific push
                 dist_robot_to_prev = self.distances.get(robot_location, {}).get(l_prev_for_this_box, float('inf'))
                 min_robot_dist_to_push = min(min_robot_dist_to_push, dist_robot_to_prev)
            # else: If no valid push position found for this box's first step,
            # this box doesn't contribute to reducing min_robot_dist_to_push.
            # min_robot_dist_to_push remains inf if no box has a reachable push position.


        # 5. Total heuristic
        # The robot cost is added only if there are boxes to move and at least one
        # has a reachable push position towards its goal.
        # If min_robot_dist_to_push is still infinity, it means either no boxes
        # need moving (handled by returning 0 earlier), or there are boxes
        # but no valid push position was found or is reachable for any of them.
        # In the latter case, the state is likely unsolvable, and adding infinity
        # results in an infinite heuristic.
        return total_box_dist + min_robot_dist_to_push

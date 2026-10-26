# from heuristics.heuristic_base import Heuristic
# Assuming Heuristic base class is available in the environment
# If not, a mock definition would be needed for local testing:
# class Heuristic:
#     def __init__(self, task): pass
#     def __call__(self, node): raise NotImplementedError

from collections import defaultdict, deque
import math # For infinity

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and remove leading/trailing parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Handle cases where fact might not be a string or properly formatted
         # For this problem, we expect facts as strings like '(predicate arg1 arg2)'
         # If it's not a string, or not a fact string, return empty list or handle error
         if isinstance(fact, str):
             # Attempt basic split if it looks like a fact but maybe malformed
             return fact.strip().split()
         else:
             # Cannot parse non-string
             return []

    return fact[1:-1].split()


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing two components:
    1. The minimum number of pushes required for each misplaced box to reach its goal location,
       calculated as the shortest path distance between the box's current location and its goal location
       on the static adjacency graph.
    2. The minimum number of robot movements required for the robot to reach a position
       from which it can make a useful push (a push that moves any misplaced box
       one step closer along a shortest path to its goal).

    # Assumptions
    - The grid structure and connectivity are defined solely by the `adjacent` predicates.
    - The heuristic ignores dynamic obstacles (other boxes, the robot's current position blocking a path)
      and the `clear` predicate. It assumes any location on a shortest path is traversable by the robot
      or pushable by a box if the required robot push position is reachable.
    - Instances are solvable, meaning all relevant locations are connected in the adjacency graph
      and goal states are reachable in principle.
    - The cost of a `move` action is 1, and the cost of a `push` action is 1.

    # Heuristic Initialization
    - Parses the goal conditions to identify the target location for each box.
    - Builds an undirected graph of locations based on the static `adjacent` facts.
    - Computes all-pairs shortest path distances between locations using BFS on this graph.
    - Precomputes, for every possible box movement `(from_loc, to_loc)`, the required robot
      position `robot_pos` such that the robot can push the box from `from_loc` to `to_loc`.
      This mapping `(from_loc, to_loc) -> robot_pos` is stored in `self.push_positions`.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the robot's current location.
    2. Identify the current location of each box.
    3. Identify all boxes that are not at their goal location (misplaced boxes).
    4. If there are no misplaced boxes, the state is a goal state, and the heuristic is 0.
    5. Calculate the sum of shortest path distances for each misplaced box from its current
       location to its goal location. This estimates the minimum number of push actions required
       for the boxes independently. Let this be `total_box_distance`.
    6. Find the minimum distance the robot needs to travel from its current location to
       any location from which it can perform a "useful" push. A useful push is one that moves
       a misplaced box one step closer along a shortest path.
       - Iterate through all misplaced boxes.
       - For each misplaced box, iterate through all possible single-step pushes from its current location
         for which the required robot position is known (using `self.push_positions`).
       - For each such potential push `(box_loc, next_loc)`, check if moving to `next_loc` is
         one step closer to the box's goal (i.e., on a shortest path).
       - If it is a useful push, get the required robot position `r_pos` from `self.push_positions`.
       - Calculate the shortest path distance from the robot's current location (`robot_loc`)
         to `r_pos`.
       - Keep track of the minimum such distance found across all useful pushes for all
         misplaced boxes. Let this be `min_robot_dist_to_push_pos`.
    7. The total heuristic value is `total_box_distance + min_robot_dist_to_push_pos`.
       If no useful push position is reachable by the robot, `min_robot_dist_to_push_pos`
       will be infinity, resulting in an infinite heuristic value, indicating a likely dead end
       or unsolvable subproblem from this state (within the heuristic's relaxation).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the graph,
        computing distances, and precomputing push positions.
        """
        # Ensure task object has goals and static attributes
        if not hasattr(task, 'goals') or not hasattr(task, 'static'):
             # In a real planner, this might raise an error or log a warning.
             # For this context, we assume valid task objects are provided.
             # We'll proceed but some parts might fail if attributes are missing.
             self.goals = frozenset()
             static_facts = frozenset()
        else:
            self.goals = task.goals  # Goal conditions (frozenset of strings).
            static_facts = task.static  # Facts that are not affected by actions (frozenset of strings).


        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                # Assuming any 'at' goal is for a box
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
            # Note: Other goal types like (at-robot ...) are ignored by this heuristic,
            # which focuses solely on box placement. This is a domain-specific choice.

        # Build the undirected graph from adjacent facts for distance calculation.
        self.graph = defaultdict(list)
        # Store directed adjacencies and inverse adjacencies to find push positions
        adj_by_l1_l2 = {} # Maps (l1, l2) -> dir
        adj_by_l2_dir = defaultdict(list) # Maps (l2, dir) -> list of l1 (locations adjacent to l2 in direction dir)

        all_locations = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "adjacent" and len(parts) == 4:
                l1, l2, dir = parts[1], parts[2], parts[3]
                self.graph[l1].append(l2)
                self.graph[l2].append(l1) # Undirected edge for distance
                adj_by_l1_l2[(l1, l2)] = dir
                adj_by_l2_dir[(l2, dir)].append(l1)
                all_locations.add(l1)
                all_locations.add(l2)

        # Compute all-pairs shortest paths using BFS.
        self.distances = {}
        for start_node in all_locations:
            self.distances[(start_node, start_node)] = 0
            q = deque([(start_node, 0)])
            visited = {start_node}

            while q:
                curr, d = q.popleft()

                # Use the undirected graph for BFS
                for neighbor in self.graph.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_node, neighbor)] = d + 1
                        q.append((neighbor, d + 1))

        # Precompute required robot push positions.
        # push_positions[(box_from_loc, box_to_loc)] = robot_pos
        self.push_positions = {}
        # Iterate through all possible single-step box movements defined by adjacency
        # A box moves from bloc to floc if adjacent(bloc, floc, dir)
        # Robot must be at rloc such that adjacent(rloc, bloc, dir)
        for (bloc, floc), dir in adj_by_l1_l2.items():
             # Need rloc such that adjacent(rloc, bloc, dir)
             # This means rloc is adjacent to bloc, and the direction from rloc to bloc is dir.
             # We stored adjacencies as (l1, l2, dir) meaning dir is from l1 to l2.
             # So we need to find l1' such that (l1', bloc, dir) is an adjacent fact.
             # Look up (bloc, dir) in adj_by_l2_dir. The list contains potential l1's.
             potential_rlocs = adj_by_l2_dir.get((bloc, dir), [])
             if potential_rlocs:
                 # Assuming at most one valid rloc for a given (bloc, dir) in grid-like domains
                 rloc = potential_rlocs[0]
                 self.push_positions[(bloc, floc)] = rloc


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        # Ensure node object has state attribute
        if not hasattr(node, 'state'):
             # In a real planner, this might raise an error or log a warning.
             # For this context, we assume valid node objects are provided.
             # Returning infinity is a safe default if state is inaccessible.
             return float('inf')

        state = node.state  # Current world state (frozenset of strings).

        # Find robot location and box locations in the current state.
        robot_loc = None
        box_locations = {} # Maps box name to location

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at-robot" and len(parts) == 2:
                robot_loc = parts[1]
            elif predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Only consider objects that are boxes according to goal locations
                if obj in self.goal_locations:
                     box_locations[obj] = loc

        # Identify misplaced boxes.
        misplaced_boxes = [
            box for box, loc in box_locations.items()
            if self.goal_locations.get(box) != loc
        ]

        # If all boxes are at their goals, the heuristic is 0.
        if not misplaced_boxes:
            return 0

        # Calculate the sum of box-to-goal distances.
        total_box_distance = 0
        for box in misplaced_boxes:
            box_loc = box_locations.get(box) # Use .get for safety
            goal_loc = self.goal_locations.get(box) # Use .get for safety

            if box_loc is None or goal_loc is None:
                 # Should not happen if misplaced_boxes list is built correctly,
                 # but defensive check.
                 return float('inf')

            # Use precomputed distances. Handle unreachable locations.
            dist = self.distances.get((box_loc, goal_loc), float('inf'))
            if dist == float('inf'):
                 # If a box cannot reach its goal, the state is likely unsolvable
                 # or requires complex steps not captured by this heuristic.
                 # Return infinity to prune this branch in greedy search.
                 return float('inf')
            total_box_distance += dist

        # Calculate the minimum robot distance to a useful push position.
        min_robot_dist_to_push_pos = float('inf')

        if robot_loc is None:
             # Robot location unknown, cannot make progress
             return float('inf')

        for box in misplaced_boxes:
            box_loc = box_locations[box]
            goal_loc = self.goal_locations[box]
            box_to_goal_dist = self.distances.get((box_loc, goal_loc), float('inf'))

            # Iterate through all possible single-step pushes *from* box_loc
            # These are the keys in self.push_positions that start with box_loc
            for (push_from_loc, push_to_loc), r_pos in self.push_positions.items():
                if push_from_loc == box_loc:
                    next_loc = push_to_loc # This is the location the box would move to

                    # Check if this move (box_loc -> next_loc) is a step closer to the goal
                    dist_from_next_to_goal = self.distances.get((next_loc, goal_loc), float('inf'))

                    if dist_from_next_to_goal == box_to_goal_dist - 1:
                        # next_loc is on a shortest path to the goal

                        # We already have the required robot position r_pos from self.push_positions

                        # Calculate robot's distance from its current location to this required push position
                        robot_dist_to_r_pos = self.distances.get((robot_loc, r_pos), float('inf'))

                        # Update the minimum distance
                        min_robot_dist_to_push_pos = min(min_robot_dist_to_push_pos, robot_dist_to_r_pos)

        # If robot cannot reach any useful push position, return infinity.
        # This happens if min_robot_dist_to_push_pos is still its initial value.
        if min_robot_dist_to_push_pos == float('inf'):
             # This state might be a dead end or require complex maneuvers
             # not captured by the heuristic (e.g., unblocking).
             # Returning infinity guides the search away from such states.
             return float('inf')

        # The heuristic is the sum of box distances and the robot's initial positioning cost.
        return total_box_distance + min_robot_dist_to_push_pos

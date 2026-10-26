import collections
# Assuming heuristics.heuristic_base is available in the specified path
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Handle unexpected input, although PDDL states are usually well-formed
         return []
    return fact[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach a goal state by summing two components for each box not yet at its goal:
    1. The minimum number of pushes required to move the box from its current location to its goal location (box-to-goal distance).
    2. The minimum number of robot moves required to get the robot into the correct position to perform the *first* push for that box towards its goal.

    # Assumptions
    - The grid structure and connectivity are defined by the `adjacent` predicates in the static facts.
    - Goals specify the target location for specific boxes.
    - The heuristic calculates distances on the static grid graph, ignoring dynamic obstacles (other boxes or the robot itself being in the way). This makes the heuristic non-admissible but potentially faster to compute and more informative than a simple goal count.
    - The grid graph defined by `adjacent` facts is assumed to be connected for all locations relevant to the problem (robot, boxes, goals). If a location is isolated, distances involving it will be infinite, correctly indicating unsolvability or high cost.

    # Heuristic Initialization
    - Extracts goal locations for each box from the task goals.
    - Builds a graph representation of the grid connectivity based on `adjacent` static facts. This includes adjacency lists, direction maps (l1 -> l2 -> dir), and reverse direction maps (l2 -> dir -> l1).
    - Precomputes all-pairs shortest path distances between all locations in the graph using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot and each box from the state facts.
    2. Initialize the total heuristic cost to 0.
    3. For each box that has a specified goal location and is not currently at that goal:
        a. Get the box's current location (`box_l`) and its goal location (`goal_l`).
        b. Check if `box_l` or `goal_l` are valid locations in the precomputed graph distances. If not, the state is likely unsolvable, return infinity.
        c. Calculate the shortest path distance between `box_l` and `goal_l` using the precomputed graph distances. This distance represents the minimum number of `push` actions required for this box if there were no obstacles and the robot was always in position. If the distance is infinity, the box cannot reach the goal, return infinity. Add this distance to the total heuristic.
        d. Determine a location (`next_l`) adjacent to `box_l` that is one step closer to `goal_l` along a shortest path. Iterate through neighbors of `box_l` and check their distance to `goal_l`.
        e. If such a `next_l` is found, determine the direction (`dir_box_to_next`) from `box_l` to `next_l` using the precomputed direction map.
        f. Determine the required robot location (`robot_l_needed`) adjacent to `box_l` such that pushing from `robot_l_needed` moves the box in the direction `dir_box_to_next`. Based on the PDDL `push` action definition `(adjacent ?rloc ?bloc ?dir)`, this means `robot_l_needed` is adjacent to `box_l` in the direction `dir_box_to_next`. Find this location using the reverse direction map.
        g. If `robot_l_needed` is found and is a valid location in the precomputed graph distances, calculate the shortest path distance from the robot's current location (`robot_l`) to `robot_l_needed`. This represents the cost for the robot to get into position for the *first* push of this box. If this distance is infinity, the robot cannot reach the push position, return infinity. Add this distance to the total heuristic.
        h. If no `next_l` is found (despite `box_dist > 0`) or `robot_l_needed` is not found, the box is likely in a simple deadlock or cannot be pushed towards the goal from its current position. Return infinity.
    4. If the loop completes without returning infinity, all boxes are either at their goals or are reachable and pushable. Return the total calculated heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goals, building the grid graph,
        and precomputing distances and directional maps.
        """
        self.goals = task.goals
        static_facts = task.static

        self.locations = set()
        self.adj_graph = collections.defaultdict(list)
        self.dir_map = collections.defaultdict(dict) # l1 -> l2 -> dir (direction from l1 to l2)
        self.rev_dir_map = collections.defaultdict(dict) # l2 -> dir -> l1 (location l1 adjacent to l2 in direction dir)
        self.opposite_dir = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

        # Build graph and direction maps from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'adjacent':
                l1, l2, dir = parts[1], parts[2], parts[3]
                self.locations.add(l1)
                self.locations.add(l2)

                # Add edge l1 -> l2
                self.adj_graph[l1].append(l2)
                self.dir_map[l1][l2] = dir # Direction from l1 to l2 is dir

                # Add reverse edge l2 -> l1 (assuming symmetry)
                opp_dir = self.opposite_dir.get(dir)
                if opp_dir:
                     self.adj_graph[l2].append(l1)
                     self.dir_map[l2][l1] = opp_dir # Direction from l2 to l1 is opp_dir

                # Build reverse direction map: l2 -> dir -> l1
                # If adjacent(l1, l2, dir) is true, it means l1 is adjacent to l2 in direction dir.
                # So, from l2, moving in direction 'dir' gets you to l1.
                # This interpretation aligns with the push action (adjacent ?rloc ?bloc ?dir)
                # where ?rloc is adjacent to ?bloc in direction ?dir.
                self.rev_dir_map[l2][dir] = l1


        # Compute all-pairs shortest paths
        self.distances = {}
        # We only need to compute distances from locations that can be the robot's position
        # or a box's position, and to locations that can be a goal.
        # Locations involved in adjacent facts cover the main grid.
        # Locations in goals might be outside the grid (unsolvable).
        # Locations in initial state (robot/boxes) might be outside the grid.
        # We compute BFS from all locations identified in adjacent facts.
        # Reachability of locations from state/goals is checked in __call__.
        for loc in self.locations: # Only run BFS from locations that are part of the grid graph
            self.distances[loc] = self._bfs(loc)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'at':
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location
                # Add goal location to distances if it wasn't in adjacent facts
                if location not in self.locations:
                     self.locations.add(location)
                     # Since it's not in adjacent facts, it's isolated. BFS from it will yield only 0 to itself.
                     self.distances[location] = {location: 0}


    def _bfs(self, start_location):
        """Helper function to perform BFS from a start location."""
        # Initialize distances for all locations known in the graph
        distances = {loc: float('inf') for loc in self.locations}
        if start_location not in self.locations:
             # This should not happen if called correctly from __init__
             return distances # Should not reach here

        distances[start_location] = 0
        queue = collections.deque([start_location])

        while queue:
            current_loc = queue.popleft()
            # Check if current_loc has neighbors in the graph
            if current_loc in self.adj_graph:
                for neighbor in self.adj_graph[current_loc]:
                    # Ensure neighbor is a valid location in our distance map keys
                    if neighbor in distances and distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_loc] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        robot_l = None
        box_locations = {}
        # clear_locations = set() # Not needed for this heuristic calculation

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts
            if parts[0] == 'at-robot':
                robot_l = parts[1]
            elif parts[0] == 'at' and parts[1] in self.goal_locations: # Only track boxes that have goals
                 box_locations[parts[1]] = parts[2]
            # elif parts[0] == 'clear':
            #      clear_locations.add(parts[1]) # Not used

        total_heuristic = 0

        # If robot location is unknown or not in the graph, heuristic is infinite
        if robot_l is None or robot_l not in self.distances:
             return float('inf')

        for box, goal_l in self.goal_locations.items():
            box_l = box_locations.get(box)

            # If box location is unknown, or box is not in the graph, or goal is not in graph, heuristic is infinite
            # Note: goal_l should always be in self.distances if it was added during init from goals
            if box_l is None or box_l not in self.distances or goal_l not in self.distances:
                 return float('inf')

            # If box is already at goal, skip
            if box_l == goal_l:
                continue

            # Distance from box to goal (minimum pushes)
            # If goal is unreachable from box, distance is infinity
            box_dist = self.distances[box_l][goal_l]
            if box_dist == float('inf'):
                 return float('inf') # Box cannot reach goal

            total_heuristic += box_dist

            # Cost for robot to get into position for the first push
            # Find a neighbor next_l on a shortest path from box_l to goal_l
            next_l = None
            dir_box_to_next = None # Direction from box_l to next_l

            # Find *a* neighbor that is one step closer to the goal
            # Iterate through neighbors of box_l
            # Ensure box_l has neighbors in the graph
            if box_l in self.adj_graph:
                for neighbor in self.adj_graph[box_l]:
                     # Ensure neighbor is a valid location in our distance map keys
                     if neighbor in self.distances and goal_l in self.distances[neighbor]:
                         if self.distances[neighbor][goal_l] == box_dist - 1:
                             next_l = neighbor
                             # Find the direction from box_l to next_l
                             if box_l in self.dir_map and next_l in self.dir_map[box_l]:
                                 dir_box_to_next = self.dir_map[box_l][next_l]
                                 break # Found one shortest path step and its direction

            # If we found a direction to push the box towards the goal
            if dir_box_to_next is not None:
                # Robot needs to be at robot_l_needed such that adjacent(robot_l_needed, box_l, dir_box_to_next) is true
                # This means robot_l_needed is adjacent to box_l in the direction dir_box_to_next
                # We find this using the reverse direction map: rev_dir_map[box_l][dir_box_to_next]
                if box_l in self.rev_dir_map and dir_box_to_next in self.rev_dir_map[box_l]:
                    robot_l_needed = self.rev_dir_map[box_l][dir_box_to_next]

                    # Add distance for robot to reach the required push position
                    # Ensure robot_l_needed is a valid location in our distance map
                    if robot_l_needed in self.distances[robot_l]:
                         robot_cost_to_position = self.distances[robot_l][robot_l_needed]
                         # If robot_l_needed is unreachable from robot_l, cost is infinity
                         if robot_cost_to_position == float('inf'):
                             return float('inf') # Robot cannot reach push position
                         total_heuristic += robot_cost_to_position
                    else:
                         # This case should ideally not happen if robot_l_needed is a valid location
                         # in the graph and the graph is connected. If it happens, it implies
                         # robot_l_needed is not a location the robot can reach. Treat as infinity.
                         return float('inf') # Robot cannot reach push position

                # else: robot_l_needed doesn't exist (e.g., box is against a wall in that direction).
                # If box_dist > 0 but no valid push location exists behind the box for any shortest path step,
                # the box is in a simple deadlock. Return infinity.
                else:
                     # This means there is no location l1 such that adjacent(l1, box_l, dir_box_to_next) is true.
                     # The box cannot be pushed in this direction because there's no space for the robot behind it.
                     # If this is true for all shortest path directions, the box is stuck.
                     # The current logic only checks *one* shortest path direction.
                     # A more robust check would be needed for full deadlock detection, but for a non-admissible
                     # heuristic, returning infinity here is a reasonable approximation for a stuck box.
                     return float('inf') # Box cannot be pushed towards goal

            # else: next_l or dir_box_to_next is None. This happens if box_dist > 0 but no neighbor
            # is one step closer to the goal. This implies the box is in a local minimum or stuck.
            # Return infinity.
            elif box_dist > 0:
                 return float('inf') # Box is stuck or in a local minimum


        # If the loop finishes, all boxes are at their goals or reachable.
        # If total_heuristic is 0, all boxes were at goals initially.
        return total_heuristic

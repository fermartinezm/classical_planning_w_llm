from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to extract the components of a PDDL fact
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, maybe return empty list or raise error
        return []
    return fact[1:-1].split()

# Helper function to check if a PDDL fact matches a given pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function to calculate shortest path distance using BFS
def shortest_path_distance(start_loc, end_loc, graph, traversable_locations):
    """
    Calculates the shortest path distance between two locations using BFS.
    Only locations present in the traversable_locations set can be visited.

    Args:
        start_loc (str): The starting location.
        end_loc (str): The target location.
        graph (dict): Adjacency list representation of the location graph.
                      {location: [neighbor1, neighbor2, ...]}
        traversable_locations (set): Set of locations that can be traversed.

    Returns:
        int or float('inf'): The shortest distance, or infinity if unreachable.
    """
    if start_loc == end_loc:
        return 0

    # If start or end is not traversable, it's unreachable (unless start == end)
    if start_loc not in traversable_locations or end_loc not in traversable_locations:
         return float('inf')

    queue = deque([(start_loc, 0)])
    visited = {start_loc}

    while queue:
        curr_loc, dist = queue.popleft()

        # Neighbors are locations reachable from curr_loc in the graph
        for neighbor_loc in graph.get(curr_loc, []):
            # Can only move to a neighbor if it is traversable and not yet visited
            if neighbor_loc in traversable_locations and neighbor_loc not in visited:
                if neighbor_loc == end_loc:
                    return dist + 1 # Found the shortest path
                visited.add(neighbor_loc)
                queue.append((neighbor_loc, dist + 1))

    return float('inf') # Target location is unreachable

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach a goal state by summing, for each box not at its goal,
    the estimated cost to move the robot to the box's location plus the estimated cost to push the box
    from its current location to its goal location.

    # Assumptions
    - The grid structure is defined by `adjacent` facts.
    - The cost of moving the robot one step is 1.
    - The cost of pushing a box one step is 1 (which also moves the robot one step).
    - The heuristic for a single box is the sum of the shortest path distance for the robot to reach the box's location
      (considering current obstacles) and the shortest path distance for the box to reach its goal
      (ignoring dynamic obstacles like other boxes).
    - The total heuristic is the sum of these individual box costs. This ignores potential
      interactions between boxes (e.g., one box blocking another) and robot path sharing,
      making it non-admissible but potentially informative for greedy search.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Builds an undirected graph representing the grid connectivity based on `adjacent` static facts.
    - Stores the set of all possible locations in the grid.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot and all boxes.
    2. Identify the set of locations that are currently clear (empty).
    3. Initialize the total heuristic cost to 0.
    4. For each box that has a specified goal location and is not currently at that goal:
       a. Get the box's current location (`l_box`) and its goal location (`l_goal`).
       b. Calculate the minimum number of pushes required to move the box from `l_box` to `l_goal`.
          This is estimated as the shortest path distance between `l_box` and `l_goal` on the full grid graph,
          ignoring dynamic obstacles (other boxes or the robot). This distance represents the minimum number of `push` actions.
          Use BFS on the graph where all locations are considered traversable.
       c. Calculate the minimum number of robot moves required for the robot to reach the box's location (`l_box`).
          This is estimated as the shortest path distance between the robot's current location (`l_robot`) and `l_box`.
          This BFS must only traverse through locations that are currently `clear` (or the robot's starting location).
       d. If either the box-to-goal distance or the robot-to-box distance is infinite (unreachable),
          the state is likely unsolvable, and a large heuristic value is returned immediately.
       e. Add the sum of the robot-to-box distance and the box-to-goal distance to the total heuristic cost.
    5. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the location graph.

        Args:
            task (Task): The planning task object.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Goal is (at box location)
                box, location = args
                self.goal_locations[box] = location

        # Build the location graph from adjacent facts.
        # This graph is undirected, representing connectivity.
        self.location_graph = {}
        self.all_locations = set()
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                # Fact is (adjacent loc1 loc2 dir)
                _, loc1, loc2, _ = get_parts(fact)
                self.location_graph.setdefault(loc1, []).append(loc2)
                self.location_graph.setdefault(loc2, []).append(loc1) # Assuming adjacency is symmetric
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Use a large value to represent infinity for unreachable states
        self.large_value = 1000000

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach a goal state.

        Args:
            node (Node): The current state node.

        Returns:
            int: The estimated heuristic cost.
        """
        state = node.state  # Current world state (frozenset of fact strings).

        # Parse the current state to find locations of robot, boxes, and clear spots.
        robot_location = None
        box_locations = {} # {box_name: location_name}
        clear_locations = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at-robot":
                # Fact is (at-robot location)
                if len(parts) > 1:
                    robot_location = parts[1]
            elif predicate == "at":
                # Fact is (at box location)
                if len(parts) > 2:
                    box = parts[1]
                    location = parts[2]
                    box_locations[box] = location
            elif predicate == "clear":
                # Fact is (clear location)
                 if len(parts) > 1:
                    clear_locations.add(parts[1])

        # If robot location is not found, something is wrong with the state representation
        if robot_location is None:
             return self.large_value # Should not happen in valid states

        total_cost = 0

        # Calculate cost for each box that is not yet at its goal
        for box, goal_loc in self.goal_locations.items():
            current_box_loc = box_locations.get(box)

            # If the box doesn't exist in the state (shouldn't happen if goal specifies it)
            # or if it's already at its goal location, its cost contribution is 0.
            if current_box_loc is None or current_box_loc == goal_loc:
                continue

            # 1. Estimate pushes needed for the box: Shortest path from box to goal on the full grid.
            #    We assume the path can be cleared, so no obstacles are considered here.
            dist_box_goal = shortest_path_distance(current_box_loc, goal_loc, self.location_graph, traversable_locations=self.all_locations)

            # If the box cannot reach its goal even on an empty grid, the state is unsolvable.
            if dist_box_goal == float('inf'):
                return self.large_value

            # 2. Estimate robot moves needed to reach the box: Shortest path from robot to box location.
            #    The robot can only move into clear locations. The set of traversable locations
            #    for the robot includes all currently clear locations plus the robot's current location
            #    (as it can move *from* there).
            robot_traversable_locations = set(clear_locations)
            robot_traversable_locations.add(robot_location) # Robot can start from its current spot

            dist_robot_box = shortest_path_distance(robot_location, current_box_loc, self.location_graph, traversable_locations=robot_traversable_locations)

            # If the robot cannot reach the box, the state is likely unsolvable.
            if dist_robot_box == float('inf'):
                 return self.large_value

            # Heuristic contribution for this box:
            # Robot moves to reach the box's vicinity (dist_robot_box actions)
            # Box is pushed to the goal (dist_box_goal pushes, each is 1 action)
            # Total for this box = dist_robot_box + dist_box_goal
            total_cost += dist_robot_box + dist_box_goal

        # The heuristic is 0 if and only if all boxes are at their goal locations,
        # which is checked by the loop condition (we only add cost if box != goal).
        # If total_cost is 0, it means all boxes were already at their goals.
        return total_cost


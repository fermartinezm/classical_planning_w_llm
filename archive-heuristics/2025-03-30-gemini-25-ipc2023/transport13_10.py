# -*- coding: utf-8 -*-
import itertools
from collections import deque
from fnmatch import fnmatch
# Assuming Heuristic base class is available in this path,
# otherwise adjust the import path as needed.
from heuristics.heuristic_base import Heuristic
import math # For math.inf

def get_parts(fact):
    """
    Helper function to parse a PDDL fact string.
    Removes parentheses and splits the string by spaces.
    Example: "(at p1 l1)" -> ["at", "p1", "l1"]

    Args:
        fact (str): The PDDL fact string.

    Returns:
        list[str]: A list of strings representing the parts of the fact.
                   Returns an empty list if the fact is malformed (e.g., not starting/ending with parentheses).
    """
    if fact.startswith("(") and fact.endswith(")"):
        return fact[1:-1].split()
    return []

class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL 'transport' domain.

    # Summary
    This heuristic estimates the remaining cost to reach the goal state by summing
    the estimated costs for moving each package to its target destination. It focuses
    on the number of actions (pickup, drop, drive) directly related to transporting
    packages that are not yet in their goal locations. The heuristic is designed
    for use with greedy best-first search and does not need to be admissible.

    # Assumptions
    - The primary goal is to have packages at specific locations, defined by `(at package location)` facts in the goal description. Other goal types (e.g., vehicle locations) are considered when checking for overall goal satisfaction but do not contribute directly to the heuristic value calculation based on package movement.
    - Roads are bidirectional. The pathfinding logic accounts for this.
    - The cost of moving a vehicle *to* a package's location before pickup is ignored to simplify the calculation and keep it fast.
    - Vehicle capacity constraints and the specific assignment of packages to vehicles are ignored.
    - The heuristic assumes the shortest path (in terms of the number of 'drive' actions) will be taken between locations for package transport.
    - All actions (drive, pick-up, drop) have a uniform cost of 1.

    # Heuristic Initialization
    - Extracts all goal conditions from the task definition.
    - Identifies all packages that have a specific `(at package location)` goal condition.
    - Stores the target location for each of these packages.
    - Parses static facts (`road ?l1 ?l2`) to build a graph representation (adjacency list) of the locations and their connections.
    - Identifies all unique locations mentioned in the domain (from roads, initial state 'at', and goal state 'at').
    - Computes all-pairs shortest path distances between all known locations using Breadth-First Search (BFS). The distance represents the minimum number of 'drive' actions. These distances are stored in a lookup table for efficient retrieval during heuristic evaluation.

    # Step-By-Step Thinking for Computing Heuristic Value
    For a given state (represented by a set of facts):
    1. **Goal Check:** Check if the current state satisfies all goal conditions defined in the task. If yes, the heuristic value is 0.
    2. **Initialization:** Initialize the total estimated cost H = 0.
    3. **State Parsing:** Parse the current state to determine:
       - The current position of each package `p` that has a goal location `goal_loc(p)`. A package can be either `(at p current_loc)` or `(in p vehicle)`. Store this in `package_current_pos`.
       - The current location `(at vehicle vehicle_loc)` of all vehicles. Store this in `vehicle_location`.
    4. **Package Cost Calculation:** Iterate through each package `p` with a defined goal location `goal_loc(p)`:
       a. **Check if Package Goal Met:** If the fact `(at p goal_loc(p))` is already present in the state, this package's contribution to the heuristic is 0. Continue to the next package.
       b. **Find Package Position:** Get the current position `current_pos` of package `p` from `package_current_pos`. If the package position is unknown (should not happen in valid states), return infinity as the state is likely invalid or unreachable.
       c. **Calculate Cost based on Position:**
          i. **If `p` is at a location (`current_pos` is a location name):**
             - Let `current_loc = current_pos`.
             - Find the shortest path distance `dist = shortest_path_dist(current_loc, goal_loc(p))` using the precomputed distances.
             - If `dist` is infinity, the goal is unreachable for this package. Return infinity for the entire state.
             - The estimated cost for `p` is: 1 (for pick-up) + `dist` (for driving) + 1 (for drop).
          ii. **If `p` is in a vehicle (`current_pos` is a vehicle name):**
             - Let `vehicle = current_pos`.
             - Find the vehicle's current location `vehicle_loc` from `vehicle_location`. If the vehicle's location is unknown, return infinity.
             - If `vehicle_loc == goal_loc(p)`: The estimated cost for `p` is 1 (for drop).
             - If `vehicle_loc != goal_loc(p)`:
               - Find the shortest path distance `dist = shortest_path_dist(vehicle_loc, goal_loc(p))`.
               - If `dist` is infinity, the goal is unreachable. Return infinity.
               - The estimated cost for `p` is: `dist` (for driving) + 1 (for drop).
       d. **Accumulate Cost:** Add the calculated cost for package `p` to the total heuristic cost H.
    5. **Final Value:**
       - If H is calculated as 0, but the state is not a goal state (checked in step 1), return 1. This ensures that only true goal states have a heuristic value of 0, preventing premature termination of the search if non-package goals exist or if there's an edge case.
       - If H is infinity (due to unreachable goals), return infinity.
       - Otherwise, return the calculated total cost H as an integer.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing task information.
        - Extracts goal locations for packages.
        - Builds the road network graph.
        - Computes all-pairs shortest path distances between locations.

        Args:
            task (Task): The planning task object containing goals, initial state,
                         static facts, etc.
        """
        self.task = task
        self.goals = task.goals
        static_facts = task.static

        # 1. Identify packages and their goal locations from task goals
        self.packages = set()
        self.locations = set()
        self.goal_locations = {} # {package_name: location_name}

        for goal in self.goals:
            parts = get_parts(goal)
            # Assume goals relevant to this heuristic are (at package location)
            if len(parts) == 3 and parts[0] == 'at':
                # Assume the first argument is the package, second is location.
                # Type checking would be more robust if type info were available.
                package, location = parts[1], parts[2]
                self.packages.add(package)
                self.locations.add(location)
                self.goal_locations[package] = location

        # 2. Build road network adjacency list and gather all locations
        self.adj = {}
        found_locations_from_roads = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == 'road':
                loc1, loc2 = parts[1], parts[2]
                found_locations_from_roads.add(loc1)
                found_locations_from_roads.add(loc2)
                # Add edge loc1 -> loc2
                self.adj.setdefault(loc1, []).append(loc2)
                # Add edge loc2 -> loc1 for bidirectionality
                self.adj.setdefault(loc2, []).append(loc1)

        self.locations.update(found_locations_from_roads)

        # 3. Ensure all locations mentioned in init/goal 'at' predicates are included
        all_facts_for_locations = task.initial_state.union(task.goals)
        for fact in all_facts_for_locations:
             parts = get_parts(fact)
             if len(parts) == 3 and parts[0] == 'at':
                 # Assume the third part is always a location
                 loc = parts[2]
                 self.locations.add(loc)
                 # Ensure isolated locations exist in adj map for BFS consistency
                 if loc not in self.adj:
                     self.adj[loc] = [] # Add location with empty neighbor list

        # Ensure adjacency lists have unique neighbors if PDDL listed roads redundantly
        for loc in self.adj:
            self.adj[loc] = list(set(self.adj[loc]))

        # 4. Compute all-pairs shortest paths (using BFS)
        self.distances = self._compute_all_pairs_shortest_paths(list(self.locations), self.adj)


    def _compute_all_pairs_shortest_paths(self, locations, adj):
        """
        Computes shortest path distances (number of drive actions) between all
        pairs of locations using BFS.

        Args:
            locations (list[str]): A list of all known location names.
            adj (dict[str, list[str]]): The adjacency list representation of the road network.

        Returns:
            dict[str, dict[str, float]]: A nested dictionary where distances[start_loc][end_loc]
                                         gives the shortest distance (integer or math.inf).
        """
        distances = {loc: {other_loc: math.inf for other_loc in locations} for loc in locations}

        for start_node in locations:
            # Check if start_node is actually in the graph (might be isolated)
            if start_node not in adj and start_node in locations:
                 distances[start_node][start_node] = 0
                 continue # No paths from here other than to itself

            distances[start_node][start_node] = 0
            queue = deque([start_node])
            # visited_bfs stores nodes visited *during the current BFS run* starting from start_node
            visited_bfs = {start_node}

            while queue:
                current_node = queue.popleft()
                current_dist = distances[start_node][current_node]

                # Iterate through neighbors using adj.get() for safety
                for neighbor in adj.get(current_node, []):
                    if neighbor not in visited_bfs:
                        visited_bfs.add(neighbor)
                        distances[start_node][neighbor] = current_dist + 1
                        queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        Estimates the total number of actions required to move packages to their goals.

        Args:
            node: A node object containing the state (node.state).
                  The state is expected to be a frozenset or set of PDDL fact strings.

        Returns:
            float: The estimated cost (number of actions) to reach the goal.
                   Returns 0 for goal states, math.inf for unreachable states.
        """
        state = node.state

        # Check for goal achievement first using the task's method
        is_goal = self.task.goal_reached(state)
        if is_goal:
            return 0

        total_cost = 0.0
        package_current_pos = {} # {package: location_name or vehicle_name}
        vehicle_location = {} # {vehicle_or_package_at_loc: location_name}

        # Parse the state to find current locations of packages and vehicles
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at' and len(args) == 2:
                obj, loc = args[0], args[1]
                # Store location for all objects initially
                vehicle_location[obj] = loc
                # If the object is a package we care about, update its position status
                if obj in self.packages:
                    package_current_pos[obj] = loc # Mark as 'at location'
            elif predicate == 'in' and len(args) == 2:
                package, vehicle = args[0], args[1]
                # If it's a package we care about, mark it as 'in vehicle'
                if package in self.packages:
                    package_current_pos[package] = vehicle # Mark as 'in vehicle'

        # Calculate estimated cost for each package not at its goal
        for package, goal_loc in self.goal_locations.items():
            # Check if this specific package goal fact is already true
            goal_fact = f"(at {package} {goal_loc})"
            if goal_fact in state:
                continue # This package is already at its destination

            current_pos = package_current_pos.get(package)

            if current_pos is None:
                # Package state is unknown (e.g., not 'at' or 'in').
                # This indicates an invalid or unexpected state. Assign infinite cost.
                # print(f"Warning: Position of package {package} not found in state.")
                return math.inf

            cost_p = 0.0
            # Check if current_pos is a known location name
            if current_pos in self.locations:
                # Package is at location `current_pos`
                current_loc = current_pos
                # Calculate distance to goal location
                # Use .get() defensively for potentially unknown locations in distances dict
                dist = self.distances.get(current_loc, {}).get(goal_loc, math.inf)

                if dist == math.inf:
                    # Goal location is unreachable from the package's current location
                    return math.inf
                # Cost: pickup(1) + drive(dist) + drop(1)
                cost_p = 1.0 + dist + 1.0
            else:
                # Package is in a vehicle `current_pos` (which is the vehicle name)
                vehicle = current_pos
                # Find the location of the vehicle
                current_loc = vehicle_location.get(vehicle)

                if current_loc is None:
                    # Vehicle carrying the package has an unknown location. Error.
                    # print(f"Warning: Location of vehicle {vehicle} carrying {package} not found.")
                    return math.inf

                if current_loc == goal_loc:
                    # Vehicle is already at the goal location. Needs drop only.
                    cost_p = 1.0
                else:
                    # Vehicle needs to drive to the goal location, then drop.
                    dist = self.distances.get(current_loc, {}).get(goal_loc, math.inf)

                    if dist == math.inf:
                        # Goal location is unreachable from the vehicle's current location
                        return math.inf
                    # Cost: drive(dist) + drop(1)
                    cost_p = dist + 1.0

            total_cost += cost_p

        # Final checks and return value
        if total_cost == 0 and not is_goal:
            # If heuristic calculated 0 but it's not a goal state
            # (e.g., goal involves non-package facts, or unreachable goals were missed)
            # return 1 to ensure non-zero cost for non-goal states.
             return 1.0
        elif total_cost == math.inf:
             # If any package goal was found unreachable
             return math.inf

        # Return the calculated cost, ensuring it's an integer if not infinity
        # The costs (1, dist) are integers, so the sum should be convertible.
        return int(total_cost)


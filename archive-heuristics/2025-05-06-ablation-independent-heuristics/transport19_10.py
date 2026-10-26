import collections
import re
import heapq

from heuristics.heuristic_base import Heuristic
from task import Task # Assuming Task class is available in the environment

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the transport domain.

    Summary:
        Estimates the cost to reach the goal by summing the minimum required
        actions for each package that is not yet at its goal location.
        For a package currently at location L_curr and needing to reach L_goal:
        - If the package is at L_curr: Estimated cost is 1 (pick-up) + shortest_distance(L_curr, L_goal) (drive) + 1 (drop).
        - If the package is inside a vehicle at L_curr: Estimated cost is shortest_distance(L_curr, L_goal) (drive) + 1 (drop).
        The shortest distance is computed based on the 'road' predicates.
        This heuristic ignores vehicle capacity and availability constraints,
        and the possibility of transporting multiple packages in one trip.

    Assumptions:
        - The 'road' predicates define a graph. Reachability is determined by this graph.
        - All goal facts relevant to this heuristic are of the form '(at <package> <location>)'.
        - All packages considered in the heuristic calculation have a goal location specified in the task goals.
        - Vehicle locations are always specified by '(at <vehicle> <location>)' facts in any reachable state.

    Heuristic Initialization:
        1. Parses the static 'road' facts to build an adjacency list representation of the road network graph.
        2. Identifies all unique locations mentioned in 'road' facts.
        3. Computes all-pairs shortest path distances between all locations using Breadth-First Search (BFS), as drive actions have uniform cost 1. Stores these distances in a dictionary.
        4. Parses the task goals to create a mapping from each package object to its goal location object. Only 'at' goals for packages are considered.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize the total heuristic value `h` to 0.
        2. Create dictionaries to quickly look up the current location of each locatable (vehicles and packages) and which package is in which vehicle from the current state facts. This involves iterating through the state facts once.
        3. Iterate through each package and its goal location stored during initialization (from task goals).
        4. For the current package P and its goal location L_goal:
            a. Check if P is already at L_goal in the current state. If the fact '(at P L_goal)' exists in the state, this package is done; continue to the next package.
            b. If P is not at L_goal, determine its current status using the lookup dictionaries created in step 2: Is it at a location or inside a vehicle?
            c. If P is inside a vehicle V (P is a key in the package-in-vehicle dictionary):
                i. Find the current location L_curr of vehicle V using the locatable-location dictionary.
                ii. Look up the precomputed shortest distance D between L_curr and L_goal.
                iii. If the distance D is infinity (L_goal is unreachable from L_curr), the state is likely a dead end; return float('inf').
                iv. Add D + 1 (estimated cost: 1 drive action per road segment + 1 drop action) to the total heuristic value `h`.
            d. If P is at a location L_curr (P is a key in the locatable-location dictionary and not in the package-in-vehicle dictionary):
                i. Look up the precomputed shortest distance D between L_curr and L_goal.
                ii. If the distance D is infinity, return float('inf').
                iii. Add D + 2 (estimated cost: 1 pick-up action + 1 drive action per road segment + 1 drop action) to the total heuristic value `h`.
            e. If the package's status (at location or in vehicle) cannot be determined from the state facts (which should not happen in valid states), return float('inf') as a fallback for an unhandled state.
        5. Return the total heuristic value `h`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.package_goals = self._parse_package_goals(task.goals)
        self.locations, self.distances = self._precompute_distances(task.static)

    def _parse_fact(self, fact_str):
        """Helper to parse a fact string into predicate and arguments."""
        # Remove parentheses and split by space
        parts = fact_str.strip('()').split()
        if not parts: # Handle empty string case
            return None, []
        predicate = parts[0]
        args = parts[1:]
        return predicate, args

    def _parse_package_goals(self, goals):
        """Extracts package goal locations from the goal facts."""
        package_goals = {}
        for goal_fact_str in goals:
            predicate, args = self._parse_fact(goal_fact_str)
            # Assuming goal facts for packages are '(at package location)'
            # and the first argument of such an 'at' fact is the package.
            # This heuristic specifically targets package delivery goals.
            if predicate == 'at' and len(args) == 2:
                 # Further check if the first arg is likely a package based on task objects?
                 # For simplicity and based on domain examples, assume 'at' goals
                 # are for packages.
                package_name = args[0]
                location_name = args[1]
                package_goals[package_name] = location_name
        return package_goals

    def _precompute_distances(self, static_facts):
        """Builds road graph and computes all-pairs shortest paths using BFS."""
        adj_list = collections.defaultdict(list)
        locations = set()

        # Build adjacency list from road facts
        for fact_str in static_facts:
            predicate, args = self._parse_fact(fact_str)
            if predicate == 'road' and len(args) == 2:
                l1, l2 = args
                adj_list[l1].append(l2)
                adj_list[l2].append(l1) # Assuming roads are bidirectional
                locations.add(l1)
                locations.add(l2)

        locations = list(locations) # Get a list of all unique locations

        # Compute shortest paths using BFS from each location
        distances = {}
        for start_loc in locations:
            q = collections.deque([(start_loc, 0)])
            visited = {start_loc}
            distances[(start_loc, start_loc)] = 0

            while q:
                current_loc, dist = q.popleft()

                for neighbor in adj_list.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[(start_loc, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

            # Mark unreachable locations with infinity
            for other_loc in locations:
                 if (start_loc, other_loc) not in distances:
                     distances[(start_loc, other_loc)] = float('inf')

        return locations, distances

    def get_distance(self, loc1, loc2):
        """Retrieves the precomputed shortest distance between two locations."""
        # Handle cases where loc1 or loc2 might not be in the precomputed locations
        # (e.g., if the initial state has objects at locations not connected by roads,
        # although this is unlikely in well-formed PDDL).
        # Returning infinity is safe as it indicates unreachability.
        return self.distances.get((loc1, loc2), float('inf'))

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.
        """
        state = node.state
        h_value = 0

        # Build quick lookup for locatable positions and package contents
        locatable_location = {}
        package_in_vehicle = {}
        # Convert state to set for faster lookups if needed, though iterating is also fine
        # depending on state size vs number of lookups. Let's use set for goal check.
        state_set = set(state)

        for fact_str in state:
            predicate, args = self._parse_fact(fact_str)
            if predicate == 'at' and len(args) == 2:
                locatable, location = args
                locatable_location[locatable] = location
            elif predicate == 'in' and len(args) == 2:
                package, vehicle = args
                package_in_vehicle[package] = vehicle

        # Calculate heuristic for each package not at its goal
        for package, goal_location in self.package_goals.items():
            goal_fact = f'(at {package} {goal_location})'

            # Check if package is already at its goal location
            if goal_fact in state_set:
                continue # Package is delivered

            # Package is not delivered, find its current location/status
            estimated_cost = float('inf') # Default to infinity

            if package in package_in_vehicle:
                # Package is in a vehicle
                vehicle = package_in_vehicle[package]
                if vehicle in locatable_location:
                    current_location = locatable_location[vehicle]
                    # Cost: drive from vehicle's location to goal + drop
                    dist = self.get_distance(current_location, goal_location)
                    if dist == float('inf'):
                        return float('inf') # Goal unreachable
                    estimated_cost = dist + 1
                # else: vehicle location unknown? Indicates invalid state.
            elif package in locatable_location:
                # Package is at a location
                current_location = locatable_location[package]
                # Cost: pick-up + drive from package's location to goal + drop
                dist = self.get_distance(current_location, goal_location)
                if dist == float('inf'):
                    return float('inf') # Goal unreachable
                estimated_cost = dist + 2
            # else: package location/status unknown? Indicates invalid state.

            # If we found a status and calculated a finite cost, add it
            if estimated_cost != float('inf'):
                 h_value += estimated_cost
            else:
                 # If a package's status wasn't found or led to an infinite cost,
                 # the state is a dead end or invalid.
                 return float('inf')

        # The heuristic is 0 if and only if all packages in package_goals are at their goal.
        # If the task goal includes other conditions, this heuristic being 0
        # does not guarantee a goal state. However, for typical transport problems
        # where goals are solely package deliveries, this holds.
        # The prompt requires h=0 only for goal states. If task.goal_reached(state_set)
        # is faster than re-checking all package goals, we could use that.
        # But re-checking package goals is already done by the loop resulting in h_value=0.
        # Let's rely on the sum being 0 when all packages are delivered.
        # If h_value is 0, it means all packages in self.package_goals are at their goal.
        # If the Task goal includes ONLY these 'at' facts, then h=0 iff goal.
        # If the Task goal includes other facts, h=0 implies a partial goal achievement.
        # For efficiency and simplicity, we return the calculated sum.

        return h_value

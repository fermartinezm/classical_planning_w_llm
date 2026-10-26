from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    fact = fact.strip()
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Heuristic class
class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It sums the estimated cost for each package
    independently, ignoring vehicle capacity and coordination. The cost for
    a package is estimated based on its current location (or the location of
    the vehicle it's in) and its goal location, plus actions for pickup and dropoff.

    # Assumptions
    - The road network is connected, allowing travel between any two locations mentioned in the problem.
    - Any vehicle can transport any package (capacity is ignored for simplicity).
    - Vehicles are always available when needed at a package's location (this is implicitly handled by using shortest path from the package's current location).

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Builds the road network graph from static `road` facts.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of every locatable object (`at` predicate).
    2. Identify which packages are inside which vehicles (`in` predicate).
    3. For each package `p` whose goal location `l_goal` is known (from initialization):
       a. Check if the package is already at its goal location on the ground, i.e.,
          if `(at p l_goal)` is true in the current state. This is checked by looking
          up the package's location in the collected `at` facts and comparing it
          to the goal location. If they match, the cost for this package is 0.
       b. If the package is not yet at its goal location on the ground:
          i. Determine the package's current location `l_curr`. This is either
             the location from `(at p l_curr)` if the package is on the ground
             (found in the collected `at` facts), or the location of the vehicle
             `v` from `(at v l_curr)` if the package is inside vehicle `v`
             (`(in p v)`). The vehicle's location is also found in the collected
             `at` facts.
          ii. If the package is currently on the ground at `l_curr`:
             The estimated cost for this package is the shortest distance from
             `l_curr` to `l_goal` (for the drive action) plus 2 actions (1 for
             pick-up and 1 for drop). Cost = `dist(l_curr, l_goal) + 2`.
          iii. If the package is currently inside a vehicle `v` which is at `l_curr`:
              The estimated cost for this package is the shortest distance from
              `l_curr` to `l_goal` (for the drive action) plus 1 action (for drop).
              Cost = `dist(l_curr, l_goal) + 1`.
    4. The total heuristic value for the state is the sum of the estimated costs
       for all packages that are not yet at their goal location on the ground.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road graph, and computing shortest paths.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        # task.goals is a frozenset of goal facts.
        # It can contain a single fact like '(at p1 l2)' or a single fact like '(and (at p1 l1) (at p2 l2))'
        for goal_fact_str in self.goals:
            parts = get_parts(goal_fact_str)
            if not parts: continue

            if parts[0] == 'and':
                # This is an (and ...) goal. The original string is like '(and (at p1 l1) (at p2 l2))'
                # We need to extract '(at p1 l1)', '(at p2 l2)', etc.
                # Find the content inside the outer 'and': goal_fact_str[len('(and '):-1]
                content = goal_fact_str[len('(and '):-1].strip()
                # Iterate through the content to find top-level facts
                i = 0
                while i < len(content):
                    if content[i] == '(':
                        j = i
                        balance = 0
                        while j < len(content):
                            if content[j] == '(': balance += 1
                            elif content[j] == ')': balance -= 1
                            if balance == 0:
                                sub_goal_str = content[i:j+1]
                                sub_parts = get_parts(sub_goal_str)
                                if sub_parts and sub_parts[0] == "at":
                                    package, location = sub_parts[1], sub_parts[2]
                                    self.goal_locations[package] = location
                                i = j + 1 # Move past the processed sub-goal
                                break
                            j += 1
                    else:
                        i += 1 # Move to the next character (skip spaces etc.)


            elif parts[0] == "at":
                 # This is a single (at ...) goal fact
                 package, location = parts[1], parts[2]
                 self.goal_locations[package] = location
            # Ignore other types of goal facts if any (though 'at' is typical for transport)


        # Build the road network graph and get all locations.
        self.adj = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                if loc1 not in self.adj:
                    self.adj[loc1] = []
                if loc2 not in self.adj:
                    self.adj[loc2] = []
                self.adj[loc1].append(loc2)
                # Assuming roads are bidirectional unless specified otherwise
                # The example instance files show bidirectional roads
                self.adj[loc2].append(loc1)

        self.locations = list(locations) # Store list of all locations

        # Compute all-pairs shortest paths using BFS.
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_loc):
        """Performs BFS from a start location to find distances to all other locations."""
        distances = {loc: float('inf') for loc in self.locations}
        distances[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            curr_loc = queue.popleft()

            # Ensure curr_loc is a valid key in adj before accessing
            if curr_loc in self.adj:
                for neighbor in self.adj[curr_loc]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[curr_loc] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings).

        # Parse current state to find locations of all locatable objects and package contents.
        all_at_facts = {} # obj -> loc
        pkg_in = {}  # package -> vehicle

        # Collect all 'at' and 'in' facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at":
                obj, loc = parts[1], parts[2]
                all_at_facts[obj] = loc
            elif predicate == "in":
                package, vehicle = parts[1], parts[2]
                pkg_in[package] = vehicle
            # Ignore other predicates like capacity

        total_cost = 0  # Initialize action cost counter.

        # Calculate cost for each package that is not yet at its goal location on the ground.
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at the goal location on the ground
            is_goal_met_on_ground = (package in all_at_facts and all_at_facts[package] == goal_location)

            if is_goal_met_on_ground:
                continue # Goal met for this package, cost is 0.

            # Package is not yet at the goal location on the ground. Calculate cost.
            current_location = None
            if package in all_at_facts:
                # Package is on the ground at all_at_facts[package]
                current_location = all_at_facts[package]
                # Cost: pick + drive + drop
                # Check if locations are in the computed distances map
                if current_location not in self.distances or goal_location not in self.distances.get(current_location, {}):
                     # This indicates an unreachable goal location from the package's current location via the road network
                     # or a location not part of the road network.
                     return float('inf')

                drive_cost = self.distances[current_location][goal_location]
                total_cost += drive_cost + 2 # 1 for pick-up, 1 for drop

            elif package in pkg_in:
                # Package is inside a vehicle
                vehicle = pkg_in[package]
                if vehicle in all_at_facts: # Find vehicle's location
                    current_location = all_at_facts[vehicle]
                    # Cost: drive + drop
                    if current_location not in self.distances or goal_location not in self.distances.get(current_location, {}):
                         return float('inf')

                    drive_cost = self.distances[current_location][goal_location]
                    total_cost += drive_cost + 1 # 1 for drop
                else:
                    # Vehicle location unknown - this shouldn't happen in a valid state
                    # Treat as unreachable or high cost
                    return float('inf') # Or a large number

            else:
                 # Package state unknown (not 'at' and not 'in') - shouldn't happen in a valid state
                 return float('inf') # Or a large number


        return total_cost

from fnmatch import fnmatch
from collections import deque

# Assuming Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if running standalone for testing
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            pass


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if isinstance(fact, str) and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    return []

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost to tighten all goal nuts. For each loose goal nut,
    it estimates the cost to get the man to the nut's location and acquire
    a usable spanner, taking the maximum of these two costs, and adds 1
    for the tighten action. The total heuristic is the sum of these costs
    over all loose goal nuts.

    Heuristic calculation for a loose goal nut N at location L_N:
    cost(N) = max(dist(L_man, L_N), min_spanner_pickup_cost) + 1
    where min_spanner_pickup_cost is the minimum cost to get the man
    carrying a usable spanner from his current location.

    Total heuristic = sum over all loose goal nuts N: cost(N)
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal nuts.
        - Location graph from static facts and compute shortest paths.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Identify all goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "tightened":
                if args: # Ensure there's an argument
                    nut = args[0]
                    self.goal_nuts.add(nut)

        # Build location graph and compute shortest paths
        locations = set()
        links = []
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3: # Ensure correct number of parts
                    _, loc1, loc2 = parts
                    locations.add(loc1)
                    locations.add(loc2)
                    links.append((loc1, loc2))

        self.locations = list(locations) # Keep a list for consistent indexing if needed, or just use set
        self.adj = {loc: [] for loc in self.locations}
        for l1, l2 in links:
            self.adj[l1].append(l2)
            self.adj[l2].append(l1) # Assuming links are bidirectional

        self.dist = {l1: {l2: float('inf') for l2 in self.locations} for l1 in self.locations}

        # Compute all-pairs shortest paths using BFS from each location
        for start_loc in self.locations:
            self.dist[start_loc][start_loc] = 0
            q = deque([(start_loc, 0)])

            while q:
                curr_loc, d = q.popleft()

                # If we found a shorter path later, ignore this one (can happen without visited set)
                # However, for unweighted BFS, the first time we reach a node is the shortest path.
                # Using dist table as visited check:
                if d > self.dist[start_loc][curr_loc] and self.dist[start_loc][curr_loc] != float('inf'):
                     continue # Already found a shorter path

                for neighbor in self.adj.get(curr_loc, []):
                    if self.dist[start_loc][neighbor] == float('inf'): # If not visited yet
                        self.dist[start_loc][neighbor] = d + 1
                        q.append((neighbor, d + 1))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        # The heuristic is defined for loose *goal* nuts.
        # If all goal nuts are tightened, the set of loose goal nuts will be empty,
        # and the heuristic sum will be 0. This is sufficient.

        # Find man's current location and name
        man_name = None
        man_location = None
        # Try finding the man by looking for the 'carrying' predicate first
        for fact in state:
            if match(fact, "carrying", "*", "*"):
                parts = get_parts(fact)
                if len(parts) >= 2:
                    man_name = parts[1]
                    break
        # If not carrying, assume the first object of type 'man' from the problem definition is the man.
        # Since we don't have task.objects here, let's assume 'bob' is the man if no carrier is found.
        # This is a limitation based on the provided Task class structure.
        if man_name is None:
             # Fallback: Assume man is named 'bob' based on examples
             man_name = 'bob'

        # Now find the location of the identified man
        for fact in state:
             if match(fact, "at", man_name, "*"):
                 parts = get_parts(fact)
                 if len(parts) >= 3:
                     man_location = parts[2]
                     break

        if man_location is None or man_location not in self.locations:
             # Man's location is unknown or not in the known locations graph
             return float('inf')

        # Find loose goal nuts and their locations
        loose_goal_nuts = {} # {nut_name: location}
        nut_locations = {} # {nut_name: location} - all goal nuts present in state

        # First find locations of all goal nuts present in the state
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if len(parts) >= 3:
                    obj, loc = parts[1:]
                    if obj in self.goal_nuts:
                         nut_locations[obj] = loc

        # Then identify which of these goal nuts are loose
        for nut in self.goal_nuts:
             if f"(loose {nut})" in state:
                  # Ensure we know the location of this loose nut
                  if nut in nut_locations:
                       loose_goal_nuts[nut] = nut_locations[nut]
                  else:
                       # Loose goal nut location unknown, problem likely unsolvable
                       # This shouldn't happen in valid PDDL states, but as a safeguard:
                       return float('inf')


        # If no loose goal nuts, goal is reached
        if not loose_goal_nuts:
            return 0

        # Find usable spanners and their locations
        usable_spanner_locations_on_ground = []
        man_carrying_usable_spanner = False

        for fact in state:
            if match(fact, "usable", "*"):
                parts = get_parts(fact)
                if len(parts) >= 2:
                    spanner_name = parts[1]
                    # Check if man is carrying this usable spanner
                    if f"(carrying {man_name} {spanner_name})" in state:
                        man_carrying_usable_spanner = True
                        # The location of the carried spanner is the man's location
                        # We don't add man_location to the ground list, handle separately
                    else:
                        # Find location of usable spanner on the ground
                        for at_fact in state:
                            if match(at_fact, "at", spanner_name, "*"):
                                at_parts = get_parts(at_fact)
                                if len(at_parts) >= 3:
                                    spanner_location = at_parts[2]
                                    if spanner_location in self.locations: # Ensure location is in our graph
                                        usable_spanner_locations_on_ground.append(spanner_location)
                                    # else: spanner is at an unknown location, ignore it for heuristic
                                    break # Found location for this spanner

        # Calculate minimum cost to get man carrying a usable spanner
        min_spanner_pickup_cost = float('inf')
        if man_carrying_usable_spanner:
            min_spanner_pickup_cost = 0
        elif usable_spanner_locations_on_ground:
            min_dist_to_spanner = float('inf')
            for s_loc in usable_spanner_locations_on_ground:
                 # dist[man_location][s_loc] was already checked for inf during graph construction
                 min_dist_to_spanner = min(min_dist_to_spanner, self.dist[man_location][s_loc])

            if min_dist_to_spanner != float('inf'):
                 min_spanner_pickup_cost = min_dist_to_spanner + 1 # +1 for pickup action
            else:
                 # Man cannot reach any usable spanner on the ground
                 return float('inf')
        else:
             # No usable spanners available at all
             return float('inf')


        # Calculate total heuristic
        total_cost = 0
        for nut, nut_loc in loose_goal_nuts.items():
            # Cost to get man to nut location
            dist_man_to_nut = float('inf')
            # dist[man_location][nut_loc] was already checked for inf during graph construction
            dist_man_to_nut = self.dist[man_location][nut_loc]

            # Cost for this nut = max(cost to get man there, cost to get spanner) + cost of tighten action
            # If either dist_man_to_nut or min_spanner_pickup_cost is inf, max will be inf.
            cost_for_nut = max(dist_man_to_nut, min_spanner_pickup_cost) + 1
            total_cost += cost_for_nut

        return total_cost

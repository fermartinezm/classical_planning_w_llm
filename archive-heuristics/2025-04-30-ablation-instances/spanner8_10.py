# Need to import necessary modules
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math # For float('inf')

# Helper functions from Logistics example
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we have enough parts to match the pattern args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the cost to tighten all loose nuts by summing:
    1. The number of loose nuts (representing tighten actions).
    2. The number of spanner pickups required (number of loose nuts minus 1 if the man is already carrying a usable spanner).
    3. An estimate of the walk cost, calculated as the sum of shortest distances from the man's current location to the location of each loose nut.

    # Assumptions
    - There is only one man object, and its name is 'bob' (based on example instance).
    - The man can carry at most one spanner at a time.
    - Each usable spanner can tighten exactly one nut.
    - Enough usable spanners exist initially to tighten all nuts that are initially loose.
    - All locations (including nut and spanner locations relevant to goals) are reachable from any other location via 'link' facts in solvable instances.

    # Heuristic Initialization
    - Build a graph representing the locations and links between them based on static 'link' facts.
    - Identify the names of all nuts that need to be tightened based on the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set of nuts that are currently loose in the state by checking for '(loose ?n)' facts for all known nuts.
    2. If there are no loose nuts, the heuristic value is 0 (goal state).
    3. Find the man's current location by searching for the fact '(at bob ?l)'.
    4. Determine if the man is currently carrying a usable spanner by searching for '(carrying bob ?s)' and checking if '(usable ?s)' is also true.
    5. Calculate the number of loose nuts (`N_loose`).
    6. The cost includes `N_loose` actions for tightening. Add `N_loose` to the heuristic value.
    7. Calculate the number of spanner pickups needed. This is `N_loose` if the man is not carrying a usable spanner, or `N_loose - 1` if he is (since the carried spanner can be used for the first nut). Use `max(0, ...)` to handle the case where `N_loose` is 0 or 1. Add this number to the heuristic value.
    8. Perform a Breadth-First Search (BFS) starting from the man's current location using the precomputed location graph to find the shortest distance to all other locations.
    9. Identify the current location of each loose nut by searching for '(at ?n ?l)' facts for each loose nut `?n`.
    10. Estimate the walk cost by summing the shortest distances from the man's current location to the location of each loose nut. If any nut location is unreachable, return infinity. Add this sum to the heuristic value.
    11. Return the total calculated heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and identifying all nuts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build the location graph from 'link' facts
        self.graph = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.graph.setdefault(loc1, []).append(loc2)
                self.graph.setdefault(loc2, []).append(loc1) # Links are bidirectional

        # Identify all nuts that need tightening from the goal state
        self.all_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                _, nut_name = get_parts(goal)
                self.all_nuts.add(nut_name)

    def bfs(self, start_location):
        """
        Performs a Breadth-First Search from a start location to find distances
        to all reachable locations in the precomputed graph.

        Args:
            start_location: The starting location for the BFS.

        Returns:
            A dictionary mapping reachable locations to their shortest distance
            from the start_location. Unreachable locations are not included.
            Returns {start_location: 0} if start_location is not in the graph.
        """
        if start_location not in self.graph and self.graph:
             # Start location is not in a non-empty graph
             # Treat as unreachable from anywhere else, distance to itself is 0
             return {start_location: 0}
        elif not self.graph:
             # Graph is empty, only start location exists
             return {start_location: 0}


        distances = {start_location: 0}
        queue = deque([start_location])
        visited = {start_location}

        while queue:
            current_loc = queue.popleft()
            current_dist = distances[current_loc]

            # Check if current_loc has neighbors in the graph
            if current_loc in self.graph:
                for neighbor in self.graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # 1. Identify loose nuts
        loose_nuts = {nut for nut in self.all_nuts if f"(loose {nut})" in state}

        # 2. If no loose nuts, return 0
        if not loose_nuts:
            return 0

        # 3. Find the man's current location
        # Assuming the man object is named 'bob'
        man_name = 'bob'
        man_location = None
        for fact in state:
            if match(fact, "at", man_name, "*"):
                man_location = get_parts(fact)[2]
                break

        if man_location is None:
             # Man must be at a location in a valid state. If not found, something is wrong.
             # Return infinity as this state is likely unsolvable or invalid.
             return math.inf

        # 4. Determine if the man is currently carrying a usable spanner
        man_carrying_usable_spanner = False
        carried_spanner_name = None
        for fact in state:
            if match(fact, "carrying", man_name, "*"):
                carried_spanner_name = get_parts(fact)[2]
                break # Assuming man carries at most one spanner

        if carried_spanner_name:
             if f"(usable {carried_spanner_name})" in state:
                  man_carrying_usable_spanner = True

        # 5. Calculate the number of loose nuts
        n_loose = len(loose_nuts)

        # 6. Cost for tighten actions
        heuristic_value = n_loose

        # 7. Calculate the number of spanner pickups needed
        # Each loose nut needs a spanner. If man carries a usable one, that's one less pickup needed.
        # Assuming man can only carry one at a time.
        n_pickups_needed = max(0, n_loose - (1 if man_carrying_usable_spanner else 0))
        heuristic_value += n_pickups_needed

        # 8. Perform BFS from man's current location
        distances = self.bfs(man_location)

        # 9. Identify locations of loose nuts
        nut_locations = set()
        for nut in loose_nuts:
            nut_loc = None
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_loc = get_parts(fact)[2]
                    break # Assuming each nut is at only one location
            if nut_loc:
                 nut_locations.add(nut_loc)
            else:
                 # A loose nut should always be at a location in a valid state
                 # If not found, this state is likely invalid or unsolvable
                 return math.inf


        # 10. Estimate walk cost
        walk_cost = 0
        for loc in nut_locations:
            dist = distances.get(loc, math.inf)
            if dist == math.inf:
                 # If a nut location is unreachable from the man, the state is unsolvable
                 return math.inf
            walk_cost += dist

        heuristic_value += walk_cost

        # 11. Return the total sum
        return heuristic_value

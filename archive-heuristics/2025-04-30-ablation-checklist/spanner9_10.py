from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math # Import math for infinity

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Basic check to prevent index errors if pattern is longer than fact parts
    if len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    This heuristic estimates the cost to tighten all loose nuts. It assumes the
    man can carry multiple spanners. The estimate is based on the number of
    tighten actions, pickup actions, and estimated walk costs.

    The walk cost is estimated by first calculating the cost to visit the
    necessary number of closest usable spanners on the ground (if the man
    doesn't carry enough), and then calculating the cost to visit all remaining
    loose nut locations, always going to the closest available item in each phase.

    Heuristic value is infinity if the problem is determined to be unsolvable
    from the current state (e.g., not enough usable spanners available in total,
    or a required location is unreachable).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing distances between locations
        and identifying static information like nut locations and object types.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state
        self.all_facts = task.facts # Contains typed objects like (man bob)

        # 1. Build location graph and compute all-pairs shortest distances using BFS
        self.location_graph = {}
        self.locations = set()
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                self.locations.add(l1)
                self.locations.add(l2)
                self.location_graph.setdefault(l1, []).append(l2)
                self.location_graph.setdefault(l2, []).append(l1) # Links are bidirectional

        self.dist = {}
        for start_loc in self.locations:
            self.dist[start_loc] = {}
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            while q:
                current, d = q.popleft()
                self.dist[start_loc][current] = d
                for neighbor in self.location_graph.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, d + 1))

        # 2. Identify object types and key objects (man, nuts, spanners)
        self.man_object_name = None
        self.nut_objects = set()
        self.spanner_objects = set()

        # Object types are typically defined by facts like (type object)
        for fact in self.all_facts:
             parts = get_parts(fact)
             if len(parts) == 2: # Expecting (type object) format
                 obj_type, obj_name = parts
                 if obj_type == 'man':
                     self.man_object_name = obj_name
                 elif obj_type == 'nut':
                     self.nut_objects.add(obj_name)
                 elif obj_type == 'spanner':
                     self.spanner_objects.add(obj_name)

        # 3. Identify nut locations (static)
        self.nut_location = {}
        # Nuts are locatable and their location is fixed throughout the problem.
        # Find their initial locations.
        for fact in self.initial_state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 if obj in self.nut_objects:
                     self.nut_location[obj] = loc
        # Also check static facts just in case (less common for initial positions)
        for fact in self.static_facts:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 if obj in self.nut_objects and obj not in self.nut_location:
                     self.nut_location[obj] = loc


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach a goal state (all loose nuts tightened).
        """
        state = node.state

        # 1. Get current state information
        man_location = None
        usable_spanners_carried = set()
        loose_nuts_remaining = set()
        usable_spanners_on_ground_and_locs = [] # List of (spanner, location) tuples

        # Iterate through the state facts to extract relevant information
        for fact in state:
            if match(fact, "at", self.man_object_name, "*"):
                man_location = get_parts(fact)[2]
            elif match(fact, "carrying", self.man_object_name, "*"):
                spanner = get_parts(fact)[2]
                # Check if the carried spanner is usable in this state
                if f"(usable {spanner})" in state:
                     usable_spanners_carried.add(spanner)
            elif match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                # Ensure the object is actually a nut
                if nut in self.nut_objects:
                    loose_nuts_remaining.add(nut)
            elif match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 # Check if the object is a spanner on the ground
                 if obj in self.spanner_objects:
                     # Check if this spanner is usable in the current state
                     if f"(usable {obj})" in state:
                         usable_spanners_on_ground_and_locs.append((obj, loc))

        # If man_location is not found, the state is invalid.
        if man_location is None:
             return float('inf')

        # 2. Compute heuristic value based on remaining tasks

        num_loose_nuts = len(loose_nuts_remaining)

        # If no loose nuts remain, the goal is reached.
        if num_loose_nuts == 0:
            return 0

        num_usable_carried = len(usable_spanners_carried)
        num_usable_on_ground = len(usable_spanners_on_ground_and_locs)
        total_usable_available = num_usable_carried + num_usable_on_ground

        # If there are fewer usable spanners available than nuts to tighten,
        # the problem is unsolvable from this state.
        if num_loose_nuts > total_usable_available:
            return float('inf')

        # Base cost: one tighten action per loose nut.
        h = num_loose_nuts

        # Additional cost: pickup actions needed.
        # The man needs one spanner per nut. If he is already carrying usable ones,
        # he needs to pick up fewer from the ground.
        num_pickups_needed = max(0, num_loose_nuts - num_usable_carried)
        h += num_pickups_needed # Add cost for pickup actions

        # Additional cost: walk actions.
        # Estimate walk cost by considering two phases:
        # Phase 1: Walk to pick up needed spanners (if any).
        # Phase 2: Walk to visit all nut locations.

        walk_cost = 0
        current_loc_for_walk = man_location

        # Phase 1: Collect spanners
        if num_pickups_needed > 0:
            # We need to pick up num_pickups_needed spanners from the ground.
            # Greedily select the closest usable spanners on the ground.
            spanners_to_get_info = sorted(
                usable_spanners_on_ground_and_locs,
                key=lambda item: self.dist.get(current_loc_for_walk, {}).get(item[1], float('inf'))
            )[:min(num_pickups_needed, num_usable_on_ground)] # Take min in case not enough on ground

            # Calculate walk cost to visit these selected spanners sequentially
            # (visiting the closest one first from the current location, then the next closest from there, etc.)
            temp_loc = current_loc_for_walk
            for s, s_loc in spanners_to_get_info:
                 # Check if the spanner location is reachable from the current temporary location
                 if temp_loc not in self.dist or s_loc not in self.dist[temp_loc]:
                     return float('inf') # Unreachable spanner location

                 walk_cost += self.dist[temp_loc][s_loc]
                 temp_loc = s_loc

            current_loc_for_walk = temp_loc # Man is now at the last spanner pickup location

        # Phase 2: Visit nuts
        nuts_to_tighten_locs = [self.nut_location[n] for n in loose_nuts_remaining]
        # Sort nut locations by distance from the location after getting spanners (or initial location)
        nuts_to_tighten_locs.sort(key=lambda loc: self.dist.get(current_loc_for_walk, {}).get(loc, float('inf')))

        # Calculate walk cost to visit these nut locations sequentially
        # (visiting the closest one first from the current location, then the next closest from there, etc.)
        temp_loc = current_loc_for_walk
        for n_loc in nuts_to_tighten_locs:
             # Check if the nut location is reachable from the current temporary location
             if temp_loc not in self.dist or n_loc not in self.dist[temp_loc]:
                 return float('inf') # Unreachable nut location

             walk_cost += self.dist[temp_loc][n_loc]
             temp_loc = n_loc

        h += walk_cost

        return h


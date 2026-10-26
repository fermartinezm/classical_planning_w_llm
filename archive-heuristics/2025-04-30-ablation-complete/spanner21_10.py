# from heuristics.heuristic_base import Heuristic # Assuming this exists

# Define a dummy Heuristic base class if not provided externally
# In a real scenario, this would be imported.
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError

from fnmatch import fnmatch
from collections import deque
from math import inf

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
    # Check if the pattern is longer than the fact parts. If so, it cannot match.
    if len(args) > len(parts):
        return False
    # Use zip to compare parts and args up to the length of the pattern.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It uses a greedy approach: for each remaining loose goal nut, it estimates the cost
    to acquire a usable spanner (if not already carrying one) and travel to the nut's
    location to tighten it, always choosing the closest available resource (spanner or nut).

    # Assumptions
    - The man can only carry one spanner at a time (implied by the domain structure and action preconditions).
    - Tightening a nut consumes the usability of the spanner used.
    - The problem is solvable (enough usable spanners exist initially).
    - Locations form an undirected graph connected by 'link' predicates.

    # Heuristic Initialization
    - Identify all locations, the man, spanners, and nuts from the task facts.
    - Build the location graph based on 'link' static facts.
    - Compute all-pairs shortest path distances between locations using BFS.
    - Identify the set of nuts that need to be tightened (goal nuts).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic estimates the cost by simulating a greedy plan to tighten the remaining loose goal nuts:
    1. Identify the man's current location.
    2. Identify the set of goal nuts that are currently loose.
    3. Identify the set of usable spanners (both carried and on the ground) available in the current state.
    4. Initialize the total heuristic cost to 0.
    5. Initialize a conceptual count of usable spanners the man is carrying based on the current state.
    6. Initialize a conceptual set of usable spanners available on the ground based on the current state.
    7. While there are still loose goal nuts remaining:
        a. Check if the conceptual count of carried usable spanners is greater than 0.
        b. If yes (man conceptually has a usable spanner):
            i. Decrement the conceptual carried count.
            ii. Find the closest remaining loose goal nut based on the man's current location.
            iii. Add the walk distance to this nut's location to the total cost.
            iv. Add 1 for the 'tighten_nut' action.
            v. Update the man's current location to the nut's location.
            vi. Remove the nut from the set of remaining nuts.
        c. If no (man does not conceptually have a usable spanner):
            i. Check if there are any usable spanners left in the conceptual ground pool.
            ii. If not, the state is likely unsolvable; return infinity.
            iii. Find the closest usable spanner in the conceptual ground pool based on the man's current location.
            iv. Add the walk distance to this spanner's location to the total cost.
            v. Add 1 for the 'pickup_spanner' action.
            vi. Update the man's current location to the spanner's location.
            vii. Remove the picked-up spanner from the conceptual ground pool.
            viii. Find the closest remaining loose goal nut based on the man's current location (which is now the spanner location).
            ix. Add the walk distance to this nut's location to the total cost.
            x. Add 1 for the 'tighten_nut' action.
            xi. Update the man's current location to the nut's location.
            xii. Remove the nut from the set of remaining nuts.
    8. Return the total estimated cost.

    The distance calculation uses precomputed shortest paths on the location graph.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting domain objects, building the location
        graph, computing distances, and identifying goal nuts.
        """
        self.goals = task.goals
        self.static = task.static
        self.facts = task.facts # Use task.facts to identify all objects

        self.locations = set()
        self.man = None
        self.spanners = set()
        self.nuts = set()

        # Infer objects and types from all possible ground facts
        # This is a robust way to get all objects and their types based on predicates they appear in.
        objects_by_type = {
            'man': set(),
            'spanner': set(),
            'nut': set(),
            'location': set()
        }

        for fact in self.facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            args = parts[1:]

            if predicate == "at":
                obj, loc = args
                objects_by_type['location'].add(loc)
                # We can't definitively type obj just from 'at', need other predicates
            elif predicate == "carrying":
                 m, s = args
                 objects_by_type['man'].add(m)
                 objects_by_type['spanner'].add(s)
            elif predicate == "usable":
                 s = args[0]
                 objects_by_type['spanner'].add(s)
            elif predicate == "loose" or predicate == "tightened":
                 n = args[0]
                 objects_by_type['nut'].add(n)
            elif predicate == "link":
                 l1, l2 = args
                 objects_by_type['location'].add(l1)
                 objects_by_type['location'].add(l2)

        # Assign inferred objects to self attributes
        self.man = next(iter(objects_by_type['man']), None) # Assuming one man
        self.spanners = objects_by_type['spanner']
        self.nuts = objects_by_type['nut']
        self.locations = objects_by_type['location']


        # Build location graph
        self.graph = {loc: set() for loc in self.locations}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                if l1 in self.graph and l2 in self.graph: # Ensure locations are known
                    self.graph[l1].add(l2)
                    self.graph[l2].add(l1) # Links are bidirectional

        # Compute all-pairs shortest paths
        self.distances = {}
        for start_node in self.locations:
            self.distances[start_node] = self._bfs(start_node)

        # Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal)[1])

    def _bfs(self, start_node):
        """Perform BFS to find shortest distances from start_node to all other locations."""
        distances = {node: inf for node in self.locations}
        if start_node not in self.locations:
             return distances # Cannot start BFS from unknown location

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node in self.graph: # Ensure node exists in graph keys
                for neighbor in self.graph.get(current_node, set()): # Use .get for safety
                    if distances[neighbor] == inf:
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Get the precomputed shortest distance between two locations."""
        if loc1 not in self.distances or loc2 not in self.locations:
             return inf # Unknown location

        return self.distances[loc1].get(loc2, inf) # Use .get for safety


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract current state information
        l_man = None
        spanner_locations = {}
        nut_locations = {}
        spanners_carried_now = set()
        usable_spanners_now = set()
        loose_nuts_now = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            args = parts[1:]

            if predicate == "at":
                obj, loc = args
                if obj == self.man:
                    l_man = loc
                elif obj in self.spanners:
                    spanner_locations[obj] = loc
                elif obj in self.nuts:
                    nut_locations[obj] = loc
            elif predicate == "carrying":
                m, s = args
                if m == self.man and s in self.spanners:
                    spanners_carried_now.add(s)
            elif predicate == "usable":
                s = args[0]
                if s in self.spanners:
                    usable_spanners_now.add(s)
            elif predicate == "loose":
                n = args[0]
                if n in self.nuts:
                    loose_nuts_now.add(n)

        # If man's location is unknown, heuristic is infinity (cannot move)
        if l_man is None:
             return inf

        # Identify goal nuts that are still loose
        nuts_remaining = self.goal_nuts.intersection(loose_nuts_now)

        # If all goal nuts are tightened, heuristic is 0
        if not nuts_remaining:
            return 0

        # Identify usable spanners available in the current state
        usable_spanners_pool = usable_spanners_now.copy()

        # Separate usable spanners currently carried vs on ground
        usable_carried_now = usable_spanners_pool.intersection(spanners_carried_now)
        usable_on_ground_now = usable_spanners_pool.difference(spanners_carried_now)

        h = 0
        current_loc = l_man
        num_usable_carried_concept = len(usable_carried_now) # Conceptually track usable spanners carried for the heuristic plan

        # Greedy simulation
        while nuts_remaining:
            if num_usable_carried_concept > 0:
                # Use a conceptually carried usable spanner
                num_usable_carried_concept -= 1 # One less usable spanner available for future nuts via carrying

                # Find the closest remaining loose goal nut
                closest_nut = None
                min_dist_to_nut = inf
                loc_n_closest = None

                for nut in nuts_remaining:
                    loc_n = nut_locations.get(nut)
                    if loc_n is None: continue # Skip if nut location unknown

                    dist = self.get_distance(current_loc, loc_n)
                    if dist == inf: continue # Skip if unreachable

                    if dist < min_dist_to_nut:
                        min_dist_to_nut = dist
                        closest_nut = nut
                        loc_n_closest = loc_n

                if closest_nut is None:
                     # Cannot reach any remaining nut from current location
                     return inf

                h += min_dist_to_nut # Walk to nut
                h += 1 # Tighten nut

                # Update state for heuristic calculation
                current_loc = loc_n_closest
                nuts_remaining.remove(closest_nut)

            else:
                # Not carrying a usable spanner (conceptually). Need to pick one up from the ground pool.
                if not usable_on_ground_now:
                    # No usable spanners left anywhere (carried count is 0, ground pool is empty)
                    return inf # Indicate unsolvability

                # Find the closest usable spanner on the ground
                closest_spanner = None
                min_dist_to_spanner = inf
                loc_s_closest = None

                # Iterate through usable spanners that are currently on the ground
                for spanner in usable_on_ground_now:
                     loc_s = spanner_locations.get(spanner)
                     if loc_s is None: continue # Spanner must be on the ground to pick up

                     dist = self.get_distance(current_loc, loc_s)
                     if dist == inf: continue # Skip if unreachable

                     if dist < min_dist_to_spanner:
                         min_dist_to_spanner = dist
                         closest_spanner = spanner
                         loc_s_closest = loc_s

                if closest_spanner is None:
                     # Cannot reach any usable spanner on the ground
                     return inf

                h += min_dist_to_spanner # Walk to spanner
                h += 1 # Pickup spanner

                # This spanner is now conceptually carried and will be used for the next nut.
                # It is removed from the ground pool.
                usable_on_ground_now.remove(closest_spanner)

                # Now, from the spanner location, go to the closest remaining nut.
                current_loc = loc_s_closest

                closest_nut = None
                min_dist_to_nut = inf
                loc_n_closest = None
                for nut in nuts_remaining:
                    loc_n = nut_locations.get(nut)
                    if loc_n is None: continue # Skip if nut location unknown

                    dist = self.get_distance(current_loc, loc_n)
                    if dist == inf: continue # Skip if unreachable

                    if dist < min_dist_to_nut:
                        min_dist_to_nut = dist
                        closest_nut = nut
                        loc_n_closest = loc_n

                if closest_nut is None:
                     # Cannot reach any remaining nut from spanner location
                     return inf

                h += min_dist_to_nut # Walk
                h += 1 # Tighten

                current_loc = loc_n_closest
                nuts_remaining.remove(closest_nut)
                # The spanner used is now unusable. It was already removed from the ground pool.

        return h

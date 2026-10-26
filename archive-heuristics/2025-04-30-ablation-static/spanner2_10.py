from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Utility functions to parse PDDL facts
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
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all required nuts.
    It sums the number of necessary 'tighten_nut' actions, the number of necessary
    'pickup_spanner' actions, and an estimate of the travel cost. The travel cost
    is estimated as the minimum distance from the man's current location to any
    location where a task needs to be performed (either tightening a nut or picking
    up a required spanner).

    # Assumptions
    - Each 'tighten_nut' action consumes one usable spanner.
    - The man can carry multiple spanners simultaneously (based on PDDL domain definition).
    - Nuts are static objects at fixed locations determined in the initial state.
    - Locations are connected by 'link' predicates forming an undirected graph.
    - All locations mentioned in the problem are part of the connected graph.

    # Heuristic Initialization
    - Identify all locations from initial state 'at' facts and static 'link' facts.
    - Build an undirected graph based on 'link' predicates.
    - Compute all-pairs shortest paths between locations using BFS and store them in `self.distances`. Unreachable locations will have infinite distance.
    - Identify the man object, all spanner objects, and all nut objects by inspecting predicates in the initial state and goals.
    - Store the static locations of all nuts based on the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the set of loose nuts that are required to be tightened according to the goal conditions (`GoalNuts`). Let `k` be the number of such nuts.
    2. If `k` is 0, it means all required nuts are tightened, so the goal is reached. The heuristic value is 0.
    3. Find the man's current location (`L_M`) from the state facts. If the man's location cannot be found, return a large value indicating a potentially invalid or unsolvable state.
    4. Count the number of usable spanners the man is currently carrying (`c`). This is done by checking for `(carrying man_name spanner_name)` and `(usable spanner_name)` facts in the current state.
    5. Count the number of usable spanners currently located on the ground at various locations (`a`). This is done by checking for `(at spanner_name location_name)` and `(usable spanner_name)` facts in the current state.
    6. Calculate the total number of usable spanners available in the current state (`total_usable = c + a`).
    7. If the number of nuts to tighten (`k`) is greater than the total number of usable spanners available (`total_usable`), the problem is unsolvable from this state. Return a large value (e.g., 1000000).
    8. Calculate the number of additional usable spanners the man needs to pick up from locations (`needed_pickups = max(0, k - c)`). These are spanners he doesn't currently carry but will need for the remaining nuts.
    9. Initialize the heuristic value `h`. This value is the sum of the minimum number of 'tighten_nut' actions (`k`) and the minimum number of 'pickup_spanner' actions (`needed_pickups`). So, `h = k + needed_pickups`.
    10. Identify the locations of the nuts in `GoalNuts` (`GoalNutLocs`) using the pre-stored static nut locations.
    11. Identify the locations of the `needed_pickups` usable spanners that are currently at locations (`UsableAtLoc`). Sort these spanners by their distance from the man's current location (`L_M`). The locations of the first `needed_pickups` spanners in this sorted list form the set `SpannerPickupLocs`.
    12. Combine `GoalNutLocs` and `SpannerPickupLocs` into a set of locations the man needs to visit to make progress towards the goal (`RequiredVisitLocs`).
    13. If `RequiredVisitLocs` is not empty, estimate the travel cost. The heuristic uses a simple estimate: the minimum distance from the man's current location (`L_M`) to any location in `RequiredVisitLocs`. Add this minimum distance to `h`. If `L_M` or any required location is not found in the precomputed distances (indicating a potential graph issue or unreachable location), return a large value.
    14. Return the final calculated heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static information and computing distances."""
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # Identify objects and locations
        self.man_name = None
        self.spanners = set()
        self.nuts = set()
        all_locations_set = set()

        # A robust way to find object types involves parsing the :objects section,
        # but that info isn't directly available in the Task object.
        # We infer types based on predicates they appear in (common domain structure).
        # Collect all potential objects first
        all_objects = set()
        for fact in self.initial_state:
             parts = get_parts(fact)
             all_objects.update(parts[1:]) # Add all arguments as potential objects
        for fact in self.goals:
             parts = get_parts(fact)
             all_objects.update(parts[1:]) # Add all arguments as potential objects
        for fact in self.static_facts:
             parts = get_parts(fact)
             all_objects.update(parts[1:]) # Add all arguments as potential objects

        # Infer types and collect locations
        initial_at_facts = {f for f in self.initial_state if match(f, "at", "*", "*")}
        initial_carrying_facts = {f for f in self.initial_state if match(f, "carrying", "*", "*")}
        initial_usable_facts = {f for f in self.initial_state if match(f, "usable", "*")}
        initial_loose_facts = {f for f in self.initial_state if match(f, "loose", "*")}
        goal_tightened_facts = {f for f in self.goals if match(f, "tightened", "*")}
        link_facts = {f for f in self.static_facts if match(f, "link", "*", "*")}


        for obj in all_objects:
            is_locatable = any(match(f, "at", obj, "*") for f in initial_at_facts)
            is_linked = any(match(f, "link", obj, "*") or match(f, "link", "*", obj) for f in link_facts)

            if is_locatable:
                 if any(match(f, "carrying", obj, "*") for f in initial_carrying_facts):
                     self.man_name = obj
                 elif any(match(f, "usable", obj) for f in initial_usable_facts) or any(match(f, "carrying", "*", obj) for f in initial_carrying_facts):
                     self.spanners.add(obj)
                 elif any(match(f, "loose", obj) for f in initial_loose_facts) or any(match(f, "tightened", obj) for f in goal_tightened_facts):
                     self.nuts.add(obj)
                 else: # If locatable but not man, spanner, or nut, assume it's a location
                     all_locations_set.add(obj)
            elif is_linked: # If not locatable but linked, it must be a location
                 all_locations_set.add(obj)

        # Add locations from initial 'at' facts explicitly
        for fact in initial_at_facts:
             all_locations_set.add(get_parts(fact)[2])


        self.locations = list(all_locations_set)

        # Build location graph from link facts
        self.graph = {loc: [] for loc in self.locations}
        for fact in link_facts:
            l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
            if l1 in self.graph and l2 in self.graph: # Ensure locations are known
                self.graph[l1].append(l2)
                self.graph[l2].append(l1) # Links are typically bidirectional

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = {}
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            self.distances[start_loc][start_loc] = 0

            while q:
                curr_loc, dist = q.popleft()

                for neighbor in self.graph.get(curr_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[start_loc][neighbor] = dist + 1
                        q.append((neighbor, dist + 1))

        # Store static nut locations from initial state
        self.nut_locations = {}
        for fact in initial_at_facts:
            obj, loc = get_parts(fact)[1], get_parts(fact)[2]
            if obj in self.nuts:
                self.nut_locations[obj] = loc


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify loose nuts that are goal conditions
        goal_nuts_names = {get_parts(g)[1] for g in self.goals if match(g, "tightened", "*")}
        loose_nuts_in_state = {n for n in goal_nuts_names if f"(loose {n})" in state}
        k = len(loose_nuts_in_state)

        # 2. If k is 0, the goal is reached
        if k == 0:
            return 0

        # 3. Find the man's current location
        man_location = None
        for fact in state:
             if match(fact, "at", self.man_name, "*"):
                 man_location = get_parts(fact)[2]
                 break
        if man_location is None or man_location not in self.locations:
             # Man is not located anywhere or at an unknown location
             return 1000000 # Problematic state

        # 4. Count usable spanners carried by man
        carried_usable_spanners = {s for s in self.spanners if f"(carrying {self.man_name} {s})" in state and f"(usable {s})" in state}
        c = len(carried_usable_spanners)

        # 5. Count usable spanners at locations
        usable_at_loc = {}
        for fact in state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                 if obj in self.spanners and f"(usable {obj})" in state:
                     usable_at_loc[obj] = loc
        a = len(usable_at_loc)

        # 6. Calculate total usable spanners
        total_usable = c + a

        # 7. Check for unsolvability (not enough usable spanners in total)
        if k > total_usable:
            return 1000000 # Unsolvable

        # 8. Calculate needed pickups
        needed_pickups = max(0, k - c)

        # 9. Initialize heuristic value
        h = k + needed_pickups # Cost for tighten and pickup actions

        # 10. Identify nut locations needing tightening
        goal_nut_locations = {self.nut_locations[n] for n in loose_nuts_in_state if n in self.nut_locations}
        # If a nut needing tightening doesn't have a known location (shouldn't happen in valid problems), treat as unsolvable
        if len(goal_nut_locations) != k:
             return 1000000

        # 11. Identify spanner pickup locations
        spanner_pickup_locations = set()
        if needed_pickups > 0:
            # Get usable spanners at locations as a list of (spanner, location)
            usable_at_loc_list = list(usable_at_loc.items())
            # Sort by distance from man's current location, handling unreachable locations
            usable_at_loc_list.sort(key=lambda item: self.distances[man_location].get(item[1], float('inf')))
            # Take the locations of the needed_pickups closest ones
            for i in range(min(needed_pickups, len(usable_at_loc_list))):
                 spanner_pickup_locations.add(usable_at_loc_list[i][1])

        # 12. Combine required visit locations
        required_visit_locations = goal_nut_locations.union(spanner_pickup_locations)

        # 13. Estimate travel cost
        if required_visit_locations:
            min_dist_to_required = float('inf')
            # Ensure man_location is a valid key before accessing distances (checked earlier)
            for loc in required_visit_locations:
                # Ensure target location is reachable from man_location
                if loc in self.distances[man_location]:
                     min_dist_to_required = min(min_dist_to_required, self.distances[man_location][loc])

            # If any required location is unreachable, min_dist_to_required remains inf
            if min_dist_to_required != float('inf'):
                h += min_dist_to_required
            else:
                 # Required locations are unreachable from man_location
                 return 1000000

        # 14. Return the total heuristic value
        return h


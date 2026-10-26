from fnmatch import fnmatch
from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

# Helper functions for parsing PDDL facts
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Domain-specific helper functions
def get_man_object(initial_state):
    """
    Find the man object name from the initial state facts.
    Infers the man object by looking for an object involved in an 'at' fact
    that is not identified as a nut or spanner based on other initial state facts.
    Assumes there is exactly one man object.
    """
    # Collect names of objects known to be nuts or spanners from initial state facts
    known_nuts_spanners = set()
    for fact in initial_state:
        parts = get_parts(fact)
        if parts[0] in ["loose", "tightened", "usable"]:
            known_nuts_spanners.add(parts[1])
        elif parts[0] == "carrying":
             known_nuts_spanners.add(parts[1]) # The carrier is the man
             known_nuts_spanners.add(parts[2]) # The carried is a spanner

    # Look for an object at a location that is not a known nut or spanner
    for fact in initial_state:
        if match(fact, "at", "*", "*"):
            obj = get_parts(fact)[1]
            if obj not in known_nuts_spanners:
                 return obj # This is likely the man

    # Fallback: If no such object found, look for the carrier in a 'carrying' fact
    for fact in initial_state:
         if match(fact, "carrying", "*", "*"):
             return get_parts(fact)[1]

    # Should not happen in valid problems
    return None


def get_man_location(state, man_obj):
    """Find the man's current location in the state."""
    for fact in state:
        if match(fact, "at", man_obj, "*"):
            return get_parts(fact)[2]
    return None # Man's location not found (should not happen in valid states)


def get_loose_nuts_info(state, goal_facts, nut_initial_locations):
    """
    Find currently loose nuts from goal facts and state, and get their locations.
    Nut locations are static and precomputed.
    """
    loose_nuts = set()
    tightened_nuts = set()

    # Find all nuts mentioned in goals
    goal_nuts = {get_parts(g)[1] for g in goal_facts if match(g, "tightened", "*")}

    # Check state for tightened nuts
    for fact in state:
        if match(fact, "tightened", "*"):
            tightened_nuts.add(get_parts(fact)[1])

    # Loose nuts are goal nuts that are not yet tightened
    loose_nuts = goal_nuts - tightened_nuts

    # Return info (nut name, location) only for currently loose nuts
    return [(nut, nut_initial_locations.get(nut)) for nut in loose_nuts if nut in nut_initial_locations]


def check_carrying_usable_spanner(state, man_obj):
    """Check if the man is carrying a usable spanner in the current state."""
    carried_spanner = None
    for fact in state:
        if match(fact, "carrying", man_obj, "*"):
            carried_spanner = get_parts(fact)[2]
            break
    if carried_spanner:
        # Check if the carried spanner is usable in the current state
        for fact in state:
            if match(fact, "usable", carried_spanner):
                return True
    return False


def get_available_spanner_locations(state):
    """Find locations of usable spanners that are on the ground (not carried)."""
    available_locs = set()
    usable_spanners = set()

    # Find usable spanners
    for fact in state:
        if match(fact, "usable", "*"):
            usable_spanners.add(get_parts(fact)[1])

    # Find locations of usable spanners that are at a location (implies not carried)
    for fact in state:
        if match(fact, "at", "*", "*"):
            obj, loc = get_parts(fact)[1:3]
            if obj in usable_spanners:
                 available_locs.add(loc)

    return available_locs


def build_location_graph(static_facts, initial_state_facts, goal_facts):
    """Build adjacency list graph from link facts and collect all relevant locations."""
    graph = defaultdict(set)
    locations = set()

    # Add locations from link facts
    for fact in static_facts:
        if match(fact, "link", "*", "*"):
            l1, l2 = get_parts(fact)[1:3]
            graph[l1].add(l2)
            graph[l2].add(l1) # Links are bidirectional
            locations.add(l1)
            locations.add(l2)

    # Add locations from at facts in initial state
    for fact in initial_state_facts:
        if match(fact, "at", "*", "*"):
            locations.add(get_parts(fact)[2])

    # Add locations from at facts in goals
    for fact in goal_facts:
        if match(fact, "at", "*", "*"):
            locations.add(get_parts(fact)[2])

    return graph, list(locations)


def compute_distances(graph, locations):
    """Compute all-pairs shortest paths using BFS."""
    INF = 1000 # Represents unreachable locations with a large value
    distance = {l: {l: 0 for l in locations} for l in locations} # Distance to self is 0

    for start_node in locations:
        q = deque([(start_node, 0)])
        visited = {start_node}

        if start_node in graph: # Only explore from nodes that are part of the linked graph
            while q:
                current_node, dist = q.popleft()
                distance[start_node][current_node] = dist # Update distance

                for neighbor in graph.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        # distance[start_node][neighbor] = dist + 1 # Set distance
                        q.append((neighbor, dist + 1))

    # Fill in unreachable distances with INF
    for l1 in locations:
        for l2 in locations:
            if l2 not in distance.get(l1, {}):
                 distance[l1][l2] = INF

    return distance


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost based on the number of loose nuts, the number of
    spanner pickups required, and the estimated travel cost to the first
    necessary location.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing distances and extracting
        static information like nut locations and the man object.
        """
        self.goals = task.goals
        self.initial_state = task.initial_state
        self.static_facts = task.static

        # 1. Precompute distances between all relevant locations
        self.location_graph, self.all_locations = build_location_graph(
            self.static_facts, self.initial_state, self.goals
        )
        self.distance = compute_distances(self.location_graph, self.all_locations)
        self.INF = 1000 # Value representing unreachable locations

        # 2. Extract static information
        self.man_obj = get_man_object(self.initial_state)

        # Store initial locations of all nuts (they are static)
        self.nut_initial_locations = {}
        # Find all nuts mentioned in initial state (loose or tightened)
        all_nuts_in_init = {get_parts(f)[1] for f in self.initial_state if match(f, "loose", "*") or match(f, "tightened", "*")}
        # Find their locations in the initial state
        for fact in self.initial_state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:3]
                 if obj in all_nuts_in_init:
                     self.nut_initial_locations[obj] = loc

        # Get the names of the goal nuts
        self.goal_nuts = {get_parts(g)[1] for g in self.goals if match(g, "tightened", "*")}


    def get_distance(self, loc1, loc2):
        """Safely get distance between two locations, returning INF if unreachable or location is unknown."""
        if loc1 not in self.distance or loc2 not in self.distance.get(loc1, {}):
             return self.INF
        return self.distance[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify loose nuts and their locations
        loose_nuts_info = get_loose_nuts_info(state, self.goals, self.nut_initial_locations)
        num_loose = len(loose_nuts_info)

        # If no loose nuts, goal is reached
        if num_loose == 0:
            return 0

        # 2. Calculate base cost (tighten and pickup actions)
        h = num_loose # Each loose nut needs a tighten action (cost 1)

        # Check if man is carrying a usable spanner
        man_carrying_usable = check_carrying_usable_spanner(state, self.man_obj)

        # Number of pickup actions needed: one for each nut, minus the one if already carrying
        # Assumes man can only carry one spanner and it becomes unusable after one tighten action.
        num_pickups = max(0, num_loose - (1 if man_carrying_usable else 0))
        h += num_pickups # Each pickup is one action (cost 1)

        # 3. Estimate walk cost
        walk_cost = 0
        man_loc = get_man_location(state, self.man_obj)

        # If man's location is unknown or unreachable, return INF
        if man_loc is None or man_loc not in self.all_locations:
             return self.INF

        nut_locations = {loc for nut, loc in loose_nuts_info if loc is not None}
        available_spanner_locs = get_available_spanner_locations(state)

        # Cost of the first significant move:
        # If carrying usable spanner, the first move is towards a nut.
        # If not carrying usable spanner, the first move is towards a spanner (if needed).
        if man_carrying_usable:
            # Go to the closest loose nut location
            if nut_locations:
                closest_nut_loc = min(nut_locations, key=lambda ln: self.get_distance(man_loc, ln))
                dist_to_closest_nut = self.get_distance(man_loc, closest_nut_loc)
                if dist_to_closest_nut == self.INF: return self.INF # Unreachable nut
                walk_cost += dist_to_closest_nut

        elif num_loose > 0: # Not carrying, needs spanner for the first nut
            # Go to the closest available usable spanner location
            if available_spanner_locs:
                closest_spanner_loc = min(available_spanner_locs, key=lambda ls: self.get_distance(man_loc, ls))
                dist_to_closest_spanner = self.get_distance(man_loc, closest_spanner_loc)
                if dist_to_closest_spanner == self.INF: return self.INF # Unreachable spanner
                walk_cost += dist_to_closest_spanner
            else:
                # Needs spanner, not carrying, none available. Unsolvable from here.
                return self.INF # Indicate unsolvability

        # Add the estimated walk cost to the heuristic
        h += walk_cost

        # Ensure heuristic is non-negative (should be guaranteed by logic, but defensive)
        return max(0, h)


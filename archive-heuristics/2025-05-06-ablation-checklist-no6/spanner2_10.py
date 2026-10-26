from fnmatch import fnmatch
from collections import deque
# Assume Heuristic base class is available, e.g.:
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         raise NotImplementedError

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts with patterns
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Assuming Heuristic base class is imported or defined elsewhere
# from heuristics.heuristic_base import Heuristic

class spannerHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost based on the number of loose nuts that are goals,
    the number of spanners that need to be picked up, and the travel cost
    to reach the necessary locations.

    Heuristic Components:
    1. Number of 'tighten_nut' actions needed (equals number of loose goal nuts).
    2. Number of 'pickup_spanner' actions needed (equals number of loose goal nuts minus carried usable spanners, minimum 0).
    3. Travel cost: Estimated as the distance to the first required location
       (closest loose nut or needed usable spanner on the ground) plus a unit cost (2 actions)
       for each subsequent required location.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and goal information.
        Precomputes shortest path distances between all locations.
        Infers man, spanners, and nuts from initial state and goals due to
        limited Task object structure. This inference is brittle.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state # Needed to infer objects

        # Extract all locations and links to build the graph
        self.locations = set()
        self.links = {} # Adjacency list {loc: [neighbor1, neighbor2, ...]}

        # Extract static information: links
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == "link":
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.links.setdefault(loc1, []).append(loc2)
                self.links.setdefault(loc2, []).append(loc1) # Links are bidirectional

        # Infer man, spanners, and nuts from initial state and goals
        initial_at_objects = {get_parts(f)[1] for f in self.initial_state if match(f, "at", "*", "*")}
        initial_usable_objects = {get_parts(f)[1] for f in self.initial_state if match(f, "usable", "*")}
        initial_loose_objects = {get_parts(f)[1] for f in self.initial_state if match(f, "loose", "*")}

        # Assume all initially usable objects are spanners
        self.all_spanners = initial_usable_objects
        # Assume all initially loose objects are nuts
        self.all_nuts = initial_loose_objects

        # The man is the object in initial_at_objects that is not a spanner and not a nut.
        potential_men = initial_at_objects - self.all_spanners - self.all_nuts
        if len(potential_men) == 1:
            self.man = list(potential_men)[0]
        else:
            # Fallback: Look for 'carrying' in initial state
            initial_carrying_facts = [f for f in self.initial_state if match(f, "carrying", "*", "*")]
            if initial_carrying_facts:
                self.man = get_parts(initial_carrying_facts[0])[1]
            else:
                # Fallback: Assume 'bob' - This is brittle and depends on instance names.
                # A robust heuristic needs proper object type information from the task.
                self.man = 'bob' # May fail on instances without 'bob' or carrying initially.


        # Extract goal nuts: nuts that must be tightened
        self.goal_nuts = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "tightened":
                nut = args[0]
                self.goal_nuts.add(nut)

        # Precompute distances using BFS
        self.distances = {}
        for start_loc in self.locations:
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            self.distances[(start_loc, start_loc)] = 0

            while q:
                current_loc, dist = q.popleft()

                if current_loc in self.links:
                    for neighbor in self.links[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[(start_loc, neighbor)] = dist + 1
                            q.append((neighbor, dist + 1))

    def get_distance(self, loc1, loc2):
        """Helper to get precomputed distance, returns infinity if no path."""
        return self.distances.get((loc1, loc2), float('inf'))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
             return 0

        # Get current man location and object locations
        man_loc = None
        obj_locations = {}
        carried_spanner = set()
        usable_spanners_in_state = set()
        loose_nuts_in_state = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                obj_locations[obj] = loc
                if obj == self.man:
                    man_loc = loc
            elif parts[0] == "carrying":
                m, s = parts[1], parts[2]
                if m == self.man:
                    carried_spanner.add(s)
            elif parts[0] == "usable":
                spanner = parts[1]
                usable_spanners_in_state.add(spanner)
            elif parts[0] == "loose":
                nut = parts[1]
                loose_nuts_in_state.add(nut)

        # Identify loose nuts that are goals
        loose_goal_nuts = {n for n in self.goal_nuts if f'(loose {n})' in state}

        num_nuts_to_tighten = len(loose_goal_nuts)

        # Count usable spanners available (carried or on the ground)
        # A spanner is usable if the fact (usable s) is in the state.
        # We only care about usable spanners for tightening.
        usable_spanners_available = usable_spanners_in_state

        num_carried_usable = len(carried_spanner.intersection(usable_spanners_available))
        # Usable spanners on the ground are those usable spanners not carried by the man.
        usable_spanners_on_ground = usable_spanners_available - carried_spanner

        # If we need more spanners than are usable in total, it's unsolvable
        if len(usable_spanners_available) < num_nuts_to_tighten:
             return float('inf') # Problem is likely unsolvable

        # Heuristic calculation
        h = 0

        # 1. Cost for tightening actions: Each loose goal nut needs one tighten action.
        h += num_nuts_to_tighten

        # 2. Cost for pickup actions: Need to pick up spanners if not enough are carried.
        num_pickups_needed = max(0, num_nuts_to_tighten - num_carried_usable)
        h += num_pickups_needed

        # 3. Travel cost:
        # The man needs to visit the location of each loose goal nut.
        # The man needs to visit the location of a usable spanner for each pickup needed.

        required_locations = set()

        # Add locations of loose goal nuts
        for nut in loose_goal_nuts:
            if nut in obj_locations:
                 required_locations.add(obj_locations[nut])
            # else: nut location unknown - should not happen in valid states

        # Add locations of spanners that need to be picked up.
        # We need to pick up `num_pickups_needed` usable spanners from the ground.
        # Find the locations of usable spanners on the ground.
        usable_spanner_locs_on_ground = {
            obj_locations[s] for s in usable_spanners_on_ground
            if s in obj_locations # Spanner must be at a location
        }

        # If pickups are needed, the man must travel to some of these locations.
        # Add the locations of the `num_pickups_needed` closest usable spanners on the ground
        # to the set of required locations.
        # Sort potential pickup locations by distance from man_loc
        sorted_spanner_locs_on_ground = sorted(list(usable_spanner_locs_on_ground),
                                               key=lambda loc: self.get_distance(man_loc, loc))

        pickup_target_locs = set(sorted_spanner_locs_on_ground[:num_pickups_needed])
        required_locations.update(pickup_target_locs)

        # Calculate travel cost based on required locations
        travel_cost = 0
        num_required_locs = len(required_locations)

        if num_required_locs > 0:
            if man_loc not in required_locations:
                # Cost to reach the first required location (actual distance)
                min_dist_to_required = float('inf')
                if required_locations:
                     min_dist_to_required = min(self.get_distance(man_loc, loc) for loc in required_locations)
                travel_cost += min_dist_to_required
                # Cost for subsequent travels: 2 actions per remaining location (e.g., move there, move away)
                travel_cost += max(0, num_required_locs - 1) * 2
            else:
                # Man is already at one required location.
                # Cost to reach the closest *other* required location (actual distance)
                remaining_required_locs = {loc for loc in required_locations if loc != man_loc}
                if remaining_required_locs:
                     min_dist_to_remaining = float('inf')
                     if remaining_required_locs:
                          min_dist_to_remaining = min(self.get_distance(man_loc, loc) for loc in remaining_required_locs)
                     travel_cost += min_dist_to_remaining
                     # Cost for subsequent travels: 2 actions per remaining location
                     travel_cost += max(0, len(remaining_required_locs) - 1) * 2

        h += travel_cost

        return h

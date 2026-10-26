from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to check if a PDDL fact matches a given pattern.
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It counts the number of loose nuts (representing the minimum number of tighten actions),
    adds the estimated travel cost for the man to reach the closest loose nut, and
    adds the estimated cost to acquire a usable spanner if the man is not already
    carrying one when the first nut needs tightening.

    # Assumptions:
    - Each loose nut requires one tighten action with a usable spanner.
    - A spanner becomes unusable after one tighten action.
    - The man can only carry one spanner at a time.
    - The heuristic estimates the cost to address the *first* remaining task (tightening the closest nut),
      plus the total number of remaining tasks. This is a non-admissible estimate.
    - The cost to acquire a spanner (if needed for the first nut) is estimated as
      travel to the closest available usable spanner plus the pickup action cost.
    - There is exactly one man object in the domain.

    # Heuristic Initialization
    - Identify all locations and links from the static facts.
    - Build a graph representing the locations and their links.
    - Compute all-pairs shortest path distances between locations using BFS.
    - Identify the man's name by examining initial state facts (assuming one man).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location. If the man's location is unknown, the state is likely invalid; return infinity.
    2. Identify all loose nuts and their locations. If a loose nut's location is unknown, the state is likely invalid; return infinity.
    3. Identify all usable spanners that are currently at a location.
    4. Check if the man is currently carrying a usable spanner.
    5. Count the total number of usable spanners (at locations + carried).
    6. If the number of loose nuts is greater than the total number of usable spanners,
       the problem is likely unsolvable; return a large value (infinity).
    7. If there are no loose nuts, the goal is reached; return 0.
    8. Initialize the heuristic cost with the number of loose nuts (minimum tighten actions).
    9. Find the loose nut location closest to the man's current location. Add this distance to the heuristic.
    10. If the man is NOT currently carrying a usable spanner:
       a. Find the usable spanner at a location that is closest to the man's current location.
       b. If no usable spanners are available at locations, return a large value (infinity), as the man needs one to pick up for the first nut.
       c. Add the shortest path distance from the man's current location to the closest usable spanner's location to the heuristic.
       d. Add 1 to the heuristic for the 'pickup_spanner' action.
    11. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing distances between locations
        and identifying the man's name.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state # Access initial state to find man name

        # Build the graph of locations
        self.locations = set()
        self.graph = {} # Adjacency list: location -> set of connected locations

        # Extract locations and links from static facts
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.graph.setdefault(loc1, set()).add(loc2)
                self.graph.setdefault(loc2, set()).add(loc1) # Links are bidirectional

        # Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_loc in self.locations:
            self.dist[start_loc] = {}
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            while q:
                current_loc, d = q.popleft()
                self.dist[start_loc][current_loc] = d
                if current_loc in self.graph:
                    for neighbor in self.graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append((neighbor, d + 1))

        # Set a large value for unreachable locations
        self.infinity = len(self.locations) * 1000 if self.locations else 1000 # Use a larger value, handle empty locations

        # Identify the man's name from the initial state
        self.man_name = None
        nuts_in_init = {get_parts(f)[1] for f in self.initial_state if match(f, "loose", "*") or match(f, "tightened", "*")}
        spanners_in_init = {get_parts(f)[1] for f in self.initial_state if match(f, "usable", "*") or match(f, "carrying", "*", "*")}
        locatables_at_loc_init = {get_parts(f)[1] for f in self.initial_state if match(f, "at", "*", "*")}
        men_candidates_init = locatables_at_loc_init - nuts_in_init - spanners_in_init
        if len(men_candidates_init) == 1:
            self.man_name = list(men_candidates_init)[0]
        elif len(men_candidates_init) > 1:
             # Should not happen in standard spanner domain, but handle defensively
             self.man_name = sorted(list(men_candidates_init))[0] # Pick first alphabetically
        # If len is 0, problem definition is likely malformed or assumptions are wrong.
        # In a real system, might raise an error. For competition, might return infinity.
        # Assuming valid spanner problems have exactly one man locatable at init.


    def get_distance(self, loc1, loc2):
        """Helper to get shortest distance, returning infinity if unreachable."""
        if loc1 not in self.dist or loc2 not in self.dist[loc1]:
             return self.infinity
        return self.dist[loc1][loc2]

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify man's current location
        man_loc = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                man_loc = get_parts(fact)[2]
                break
        if man_loc is None:
             # Man is not at any location? Should not happen in valid states.
             # Return infinity as state is likely invalid or unsolvable.
             return self.infinity


        # 2. Identify loose nuts and their locations
        loose_nuts = {} # nut_name -> location
        for fact in state:
            if match(fact, "loose", "*"):
                nut_name = get_parts(fact)[1]
                # Find location of this loose nut
                nut_loc = None
                for f_at in state:
                    if match(f_at, "at", nut_name, "*"):
                        nut_loc = get_parts(f_at)[2]
                        break
                if nut_loc:
                    loose_nuts[nut_name] = nut_loc
                # If a loose nut is not at a location, problem is likely unsolvable.
                # The domain implies nuts are static objects at locations.
                elif f"(tightened {nut_name})" not in state: # Only loose nuts matter
                     return self.infinity # Loose nut not at a location

        # 3. Identify usable spanners that are currently at a location.
        usable_spanners_at_locs = {} # spanner_name -> location
        usable_spanners_carried_count = 0
        total_usable_spanners_count = 0

        for fact in state:
             if match(fact, "usable", "*"):
                 spanner_name = get_parts(fact)[1]
                 total_usable_spanners_count += 1
                 # Check if carried
                 if match(fact, "carrying", self.man_name, spanner_name):
                      usable_spanners_carried_count += 1
                 else:
                      # Find location if not carried
                      spanner_loc = None
                      for f_at in state:
                           if match(f_at, "at", spanner_name, "*"):
                                spanner_loc = get_parts(f_at)[2]
                                break
                      if spanner_loc:
                           usable_spanners_at_locs[spanner_name] = spanner_loc
                      # If a usable spanner is not carried and not at a location, it's effectively lost for planning.
                      # usable_spanners_at_locs correctly only includes those at locations.


        num_loose_nuts = len(loose_nuts)

        # 7. Goal state check
        if num_loose_nuts == 0:
            return 0

        # 6. Check solvability based on spanners
        # We need at least one usable spanner per loose nut.
        if num_loose_nuts > total_usable_spanners_count:
             return self.infinity # Not enough usable spanners in total

        # 8. Initialize heuristic cost with minimum tighten actions
        h = num_loose_nuts

        # Find the loose nut location closest to the man
        closest_nut_loc = None
        min_dist_to_nut = self.infinity
        # num_loose_nuts > 0 is checked above
        for nut_loc in loose_nuts.values():
            dist = self.get_distance(man_loc, nut_loc)
            if dist < min_dist_to_nut:
                min_dist_to_nut = dist
                closest_nut_loc = nut_loc

        # 9. Add travel cost to the closest nut
        # closest_nut_loc should not be None if num_loose_nuts > 0
        h += min_dist_to_nut

        # 10. Add cost to get a spanner if needed for the first nut
        man_is_carrying_usable = usable_spanners_carried_count > 0

        if not man_is_carrying_usable:
             # Man needs to pick up a spanner. Find the closest usable spanner at a location.
             closest_spanner_loc_to_man = None
             min_dist_spanner_to_man = self.infinity

             if not usable_spanners_at_locs:
                  # No usable spanners available at locations to pick up.
                  # If man needs one (num_loose_nuts > 0 and not man_is_carrying_usable),
                  # and none are available to pick up, it's impossible unless he was already carrying one (which is false here).
                  # This state is unsolvable.
                  return self.infinity

             for s_loc in usable_spanners_at_locs.values():
                  dist = self.get_distance(man_loc, s_loc)
                  if dist < min_dist_spanner_to_man:
                      min_dist_spanner_to_man = dist
                      closest_spanner_loc_to_man = s_loc

             # Add travel cost to the closest spanner + pickup action
             # closest_spanner_loc_to_man should not be None if usable_spanners_at_locs is not empty
             h += min_dist_spanner_to_man + 1

        # 11. Return the total heuristic cost.
        return h

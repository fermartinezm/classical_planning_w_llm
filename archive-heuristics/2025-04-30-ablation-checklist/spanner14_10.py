import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    # Ensure the number of parts matches the number of args, unless args has wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the actions needed (tighten, pickup) and the travel required
    to reach the necessary locations (spanners and nuts).

    # Assumptions:
    - The man can only carry one spanner at a time.
    - Each tighten action consumes one usable spanner.
    - Nuts are static.
    - The graph of locations defined by 'link' predicates is traversable.

    # Heuristic Initialization
    - Computes all-pairs shortest paths between locations using BFS based on 'link' facts.
    - Identifies all nut objects and their fixed locations from the initial state/goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all loose nuts in the current state. If none are loose, the heuristic is 0.
    2. Identify the man's current location.
    3. Identify all usable spanners (carried or on the ground) and their locations.
    4. Count the number of loose nuts (`N_loose`). This is a lower bound on `tighten_nut` actions.
    5. Count the number of usable spanners the man is currently carrying (`N_carried_usable`).
    6. Calculate the number of additional spanners that need to be picked up from the ground: `N_pickups_needed = max(0, N_loose - N_carried_usable)`. This is a lower bound on `pickup_spanner` actions.
    7. The heuristic starts with the sum of these minimum action counts: `h = N_loose + N_pickups_needed`.
    8. Estimate the walk cost: The man needs to travel to pick up `N_pickups_needed` spanners and visit the location of each loose nut to tighten it.
    9. Identify the locations of usable spanners currently on the ground.
    10. Identify the unique locations of all loose nuts.
    11. The man must visit a set of locations: the locations of the `N_pickups_needed` nearest usable ground spanners (to pick them up) and the locations of all loose nuts (to tighten them).
    12. Calculate the distance from the man's current location to *each* location in this required set using the precomputed BFS distances. Sum these distances. This sum is added to the heuristic. This overestimates travel but encourages moving towards necessary items/locations.
    13. If any required location is unreachable (infinite distance), the heuristic returns infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and computing
        all-pairs shortest paths between locations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # Build the location graph from 'link' facts
        self.locations = set()
        self.graph = collections.defaultdict(set)
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.graph[loc1].add(loc2)
                self.graph[loc2].add(loc1) # Links are bidirectional

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # Identify all nut objects and their locations (nuts are static)
        # We can find nut locations from the initial state or goals.
        # Assuming nuts are always 'at' a location in the initial state.
        self.nut_locations = {}
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 # Check if the object is a nut type (heuristic doesn't have type info directly,
                 # but we can infer from goal facts or object names if needed.
                 # For simplicity, let's assume objects starting with 'nut' are nuts).
                 # A more robust way would parse types from the domain, but this is faster.
                 if obj.startswith("nut"):
                     self.nut_locations[obj] = loc

        # Also get nut locations from goals, just in case (though typically nuts are static)
        for goal in self.goals:
             if match(goal, "tightened", "*"):
                 nut = get_parts(goal)[1]
                 # We need the location of this nut. It must be in the initial state.
                 # If not found in initial state, this heuristic might fail or need refinement.
                 # Assuming all goal nuts are present and located in the initial state.
                 pass # Location already captured from initial state

    def _bfs(self, start_node):
        """Perform BFS to find shortest distances from start_node to all other nodes."""
        distances = {node: float('inf') for node in self.locations}
        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node not in self.graph:
                 continue # Handle locations with no links

            for neighbor in self.graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify loose nuts and their locations
        loose_nuts = set()
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                loose_nuts.add(nut)

        # If no nuts are loose, the goal is reached.
        if not loose_nuts:
            return 0

        # 2. Identify man's location
        man_loc = None
        carried_spanners = set()
        usable_spanners_on_ground_locs = {} # Map location to list of usable spanners there

        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                # Assuming there's only one man and his name is not 'spanner' or 'nut'
                if not obj.startswith("spanner") and not obj.startswith("nut") and obj in self.locations: # Simple check for man object
                     man_loc = loc
            elif match(fact, "carrying", "*", "*"):
                 _, m, s = get_parts(fact)
                 # Assuming the man object name is 'm' and spanner is 's'
                 carried_spanners.add(s)

        # Find usable spanners on the ground
        usable_spanners = set()
        for fact in state:
             if match(fact, "usable", "*"):
                  spanner = get_parts(fact)[1]
                  usable_spanners.add(spanner)

        for spanner in usable_spanners:
             if spanner not in carried_spanners:
                  # Find location of this ground spanner
                  spanner_loc = None
                  for fact in state:
                       if match(fact, "at", spanner, "*"):
                            spanner_loc = get_parts(fact)[2]
                            break
                  if spanner_loc:
                       if spanner_loc not in usable_spanners_on_ground_locs:
                            usable_spanners_on_ground_locs[spanner_loc] = []
                       usable_spanners_on_ground_locs[spanner_loc].append(spanner)


        # Check if man_loc was found (should always be the case in valid states)
        if man_loc is None:
             # This indicates an invalid state representation or parsing issue
             return float('inf') # Should not happen in a well-formed problem

        # 3. Calculate minimum actions (tighten + pickup)
        N_loose = len(loose_nuts)
        N_carried_usable = len(carried_spanners.intersection(usable_spanners))
        N_pickups_needed = max(0, N_loose - N_carried_usable)

        # Check solvability based on available usable spanners
        total_usable_spanners = N_carried_usable + sum(len(spanners) for spanners in usable_spanners_on_ground_locs.values())
        if N_loose > total_usable_spanners:
             return float('inf') # Not enough usable spanners in the state

        # Heuristic starts with non-walk actions
        h = N_loose + N_pickups_needed

        # 4. Estimate walk cost
        locations_to_visit = set()

        # Add locations of loose nuts
        for nut in loose_nuts:
            if nut in self.nut_locations:
                locations_to_visit.add(self.nut_locations[nut])
            # else: nut location not found, problem with setup? Assume nuts are located.

        # Add locations of spanners to pick up
        if N_pickups_needed > 0:
            # Find usable ground spanners and sort their locations by distance from man_loc
            ground_spanner_locs_with_dist = []
            for loc, spanners in usable_spanners_on_ground_locs.items():
                 if loc in self.distances[man_loc]:
                      dist = self.distances[man_loc][loc]
                      if dist != float('inf'):
                           # Add each spanner at this location as a potential pickup
                           for spanner in spanners:
                                ground_spanner_locs_with_dist.append((dist, loc, spanner))

            # Sort by distance and take the locations of the N_pickups_needed nearest spanners
            ground_spanner_locs_with_dist.sort()
            picked_spanner_locations = set()
            spanners_accounted_for = set()

            for dist, loc, spanner in ground_spanner_locs_with_dist:
                 if len(picked_spanner_locations) < N_pickups_needed:
                      # Ensure we count distinct spanners if multiple are at one location
                      if spanner not in spanners_accounted_for:
                           picked_spanner_locations.add(loc)
                           spanners_accounted_for.add(spanner)
                 else:
                      break # Found enough spanner locations

            locations_to_visit.update(picked_spanner_locations)


        # Add walk cost: sum of distances from man_loc to each required location
        for loc in locations_to_visit:
            if man_loc in self.distances and loc in self.distances[man_loc]:
                dist = self.distances[man_loc][loc]
                if dist == float('inf'):
                    return float('inf') # Required location is unreachable
                h += dist
            else:
                 # Should not happen if graph includes all locations, but safety check
                 return float('inf')


        return h


from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math

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


class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost based on the number of nuts to tighten, spanners to pick up,
    and estimated travel cost.

    Heuristic Components:
    1. Number of loose goal nuts (estimate for tighten_nut actions).
    2. Number of spanners the man needs to pick up (total goal nuts minus
       usable spanners currently carried).
    3. Estimated travel cost to visit all necessary locations (locations of
       loose goal nuts and locations of needed usable spanners on the ground).
       This is estimated as the sum of shortest path distances from the man's
       current location to each unique relevant location.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing distances between locations
        and identifying goal nuts and the man object.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Identify the man object. We assume there is one man.
        # Prioritize finding the man via a 'carrying' fact in the initial state.
        self.man_obj = None
        for fact in task.initial_state:
             if match(fact, "carrying", "*", "*"):
                  self.man_obj = get_parts(fact)[1]
                  break
        # Fallback: find the first object in an 'at' fact that doesn't look like a spanner or nut.
        if self.man_obj is None:
             for fact in task.initial_state:
                 if match(fact, "at", "*", "*"):
                     parts = get_parts(fact)[1:] # [obj, loc]
                     obj = parts[0]
                     # Simple check based on common naming conventions
                     if not obj.startswith("spanner") and not obj.startswith("nut"):
                          self.man_obj = obj
                          break
        # Last resort: assume 'bob' based on example instances. This is fragile.
        if self.man_obj is None:
             self.man_obj = 'bob'


        # Collect all locations mentioned in the initial state and static facts.
        all_locations = set()
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                  all_locations.add(get_parts(fact)[2])
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                all_locations.add(get_parts(fact)[1])
                all_locations.add(get_parts(fact)[2])


        # Build location graph from link facts. Assume links are bidirectional.
        graph = {loc: [] for loc in all_locations} # Initialize graph with all known locations
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                # Only add links if both locations are in our collected set
                if l1 in graph and l2 in graph:
                    graph[l1].append(l2)
                    graph[l2].append(l1) # Links are bidirectional

        # Compute all-pairs shortest paths using BFS.
        self.distances = {}
        for start_loc in all_locations:
            self.distances[(start_loc, start_loc)] = 0
            queue = deque([start_loc])
            visited = {start_loc}
            while queue:
                curr = queue.popleft()
                # Ensure location exists in graph (might be an isolated location)
                if curr in graph:
                    for neighbor in graph[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[(start_loc, neighbor)] = self.distances[(start_loc, curr)] + 1
                            queue.append(neighbor)

        # Identify goal nuts from the task goals.
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions from the
        current state to reach a goal state.
        """
        state = node.state

        # 1. Find man's current location.
        man_location = None
        for fact in state:
            if match(fact, "at", self.man_obj, "*"):
                man_location = get_parts(fact)[2]
                break
        # If man's location is not found, the state is likely invalid or terminal.
        if man_location is None:
             return math.inf

        # 2. Count usable spanners currently carried by the man.
        carried_usable_count = 0
        # Find all spanners the man is carrying.
        carried_spanners = set()
        for fact in state:
             if match(fact, "carrying", self.man_obj, "*"):
                  spanner = get_parts(fact)[2]
                  carried_spanners.add(spanner)

        # Check which of the carried spanners are usable in the current state.
        for spanner in carried_spanners:
             if "(usable {})".format(spanner) in state:
                  carried_usable_count += 1

        # 3. Find usable spanners on the ground and their locations.
        usable_spanners_on_ground_locations = {} # {spanner_obj: location}
        for fact in state:
             if match(fact, "at", "*", "*"):
                  parts = get_parts(fact)
                  obj, loc = parts[1], parts[2]
                  # Check if the object is a spanner and is usable in this state.
                  if obj.startswith("spanner") and "(usable {})".format(obj) in state:
                       usable_spanners_on_ground_locations[obj] = loc

        # 4. Find loose goal nuts in the current state and their locations.
        loose_goal_nut_locations = {} # {nut_obj: location}
        for nut in self.goal_nuts:
             # Check if the nut is loose in the current state.
             if "(loose {})".format(nut) in state:
                  # Find the location of the loose nut.
                  for fact in state:
                       if match(fact, "at", nut, "*"):
                            loose_goal_nut_locations[nut] = get_parts(fact)[2]
                            break # Assuming a nut is only at one location

        # If all goal nuts are already tightened (i.e., no loose goal nuts), heuristic is 0.
        num_loose_goal_nuts = len(loose_goal_nut_locations)
        if num_loose_goal_nuts == 0:
             return 0

        # 5. Calculate heuristic components.

        # h_tighten: Estimate for tighten_nut actions. Each loose goal nut needs one.
        h_tighten = num_loose_goal_nuts

        # h_pickup: Estimate for pickup_spanner actions.
        # Man needs a total of num_loose_goal_nuts usable spanners.
        # He starts with carried_usable_count.
        # The number of additional pickups needed is the difference, minimum 0.
        pickups_needed = max(0, num_loose_goal_nuts - carried_usable_count)
        h_pickup = pickups_needed

        # h_travel: Estimate for walk actions.
        # The man must visit the location of each loose goal nut.
        target_nut_locations = set(loose_goal_nut_locations.values())
        relevant_locations_to_visit = set(target_nut_locations)

        # If pickups are needed, the man must also visit locations of usable spanners on the ground.
        # We assume he will go to the 'pickups_needed' closest usable spanners.
        if pickups_needed > 0:
             available_spanner_locs_list = list(usable_spanners_on_ground_locations.values())
             # If pickups are needed but no usable spanners are on the ground, the state is likely unsolvable.
             if not available_spanner_locs_list:
                  return math.inf

             # Sort available spanner locations by distance from the man's current location.
             sorted_spanner_locs = sorted(available_spanner_locs_list,
                                          key=lambda loc: self.distances.get((man_location, loc), math.inf))

             # Add the locations of the 'pickups_needed' closest spanners to the set of relevant locations to visit.
             # Ensure we don't try to add more locations than there are available usable spanners on the ground.
             num_spanner_locs_to_consider = min(pickups_needed, len(sorted_spanner_locs))
             relevant_locations_to_visit.update(sorted_spanner_locs[:num_spanner_locs_to_consider])

        # Calculate travel cost: Sum of shortest path distances from the man's
        # current location to each unique relevant location. This is a non-admissible
        # estimate that overcounts shared travel but captures the cost of reaching
        # all necessary areas.
        h_travel = 0
        for loc in relevant_locations_to_visit:
             dist = self.distances.get((man_location, loc), math.inf)
             # If any relevant location is unreachable, the state is likely unsolvable.
             if dist == math.inf:
                  return math.inf
             h_travel += dist

        # Total heuristic is the sum of the estimated costs for each component.
        total_heuristic = h_tighten + h_pickup + h_travel

        return total_heuristic


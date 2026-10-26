from collections import deque
from heuristics.heuristic_base import Heuristic
import math # For infinity

def parse_fact(fact_string):
    """Parses a PDDL fact string into a tuple (predicate, [arg1, arg2, ...])."""
    # Remove outer parentheses and split by whitespace
    parts = fact_string[1:-1].split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    Estimates the cost to reach the goal state (all goal nuts tightened)
    by summing the estimated costs for tightening each loose goal nut.
    The cost for tightening a single nut is estimated as the cost to get
    a usable spanner to the man, then get the man to the nut's location,
    plus the pickup and tighten action costs. It greedily selects the
    cheapest spanner-nut pair sequence at each step, prioritizing using
    a spanner the man is already carrying.

    Assumptions:
    - The goal is solely to tighten a specific set of nuts.
    - The man can carry at most one spanner at a time.
    - Spanners become unusable after one use for tightening.
    - No new usable spanners appear during the plan.
    - Object types (man, spanner, nut, location) can be inferred from
      predicate usage or naming conventions (e.g., 'bob' is man,
      'spannerX' is spanner, 'nutX' is nut, others are locations).
    - All locations mentioned in the state and goal are part of the static link graph.

    Heuristic Initialization:
    In the constructor, the heuristic precomputes the shortest path
    distances between all pairs of locations based on the static 'link'
    facts provided in the task definition. This involves building a graph
    of locations and running BFS from each location.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state to identify:
       - The man's current location.
       - The locations of all objects (spanners, nuts).
       - Which spanners are currently usable.
       - Which nuts are currently loose.
       - If the man is carrying a spanner.
    2. Identify the set of loose nuts that are part of the goal.
    3. Identify the set of usable spanners that are available (either at a
       location or being carried by the man).
    4. Check for solvability: If the number of loose goal nuts exceeds the
       total number of available usable spanners, the problem is considered
       unsolvable in this domain, and the heuristic returns infinity.
    5. Initialize the heuristic value `h` to 0. Set the man's current
       location for the heuristic calculation to his actual location in the state.
       Create working sets of remaining loose goal nuts and available usable
       spanners (those at locations). Track if the man currently holds a
       usable spanner.
    6. If the man is currently carrying a usable spanner and there are loose
       goal nuts remaining:
       - Find the loose goal nut that is closest to the man's current location.
       - Add the cost to walk to this nut's location plus the tighten action cost (1)
         to `h`.
       - Update the man's current location for the heuristic calculation to the
         location of the nut just processed.
       - Remove the processed nut from the set of remaining nuts.
       - Note that the carried spanner is now used up.
    7. While there are still loose goal nuts remaining:
       - Find the pair of a remaining loose goal nut `n` and a remaining
         usable spanner `s` (at a location) that minimizes the cost of the
         sequence: walk from the man's current location to `Loc(s)`, pickup `s` (cost 1),
         walk from `Loc(s)` to `Loc(n)`, and tighten `n` (cost 1). The cost for
         a pair (n, s) is `Dist(current_man_location, Loc(s)) + 1 + Dist(Loc(s), Loc(n)) + 1`.
       - If no such pair can be found (e.g., no usable spanners left), return infinity.
       - Add the minimum cost found to `h`.
       - Update the man's current location for the heuristic calculation to the
         location of the nut just processed.
       - Remove the processed nut from the set of remaining nuts.
       - Remove the used spanner from the set of remaining usable spanners at locations.
    8. Return the final value of `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.static = task.static

        # Precompute location graph and shortest paths
        self.location_graph = {}
        self.locations = set()
        self.dist = {} # dist[l1][l2] = shortest path from l1 to l2

        for fact_string in self.static:
            pred, args = parse_fact(fact_string)
            if pred == 'link':
                l1, l2 = args
                self.locations.add(l1)
                self.locations.add(l2)
                self.location_graph.setdefault(l1, set()).add(l2)
                self.location_graph.setdefault(l2, set()).add(l1) # Links are bidirectional

        # Compute all-pairs shortest paths using BFS
        for start_loc in list(self.locations): # Iterate over a copy as locations might be added later if state has new ones
            self.dist[start_loc] = {loc: math.inf for loc in self.locations}
            self.dist[start_loc][start_loc] = 0
            queue = deque([start_loc])

            while queue:
                curr = queue.popleft()
                current_dist = self.dist[start_loc][curr]

                for neighbor in self.location_graph.get(curr, set()):
                    if neighbor not in self.locations:
                         # Add new location found in graph but not in initial locations
                         self.locations.add(neighbor)
                         # Re-initialize distances for the new location
                         for loc in self.locations:
                             self.dist.setdefault(loc, {})[neighbor] = math.inf
                             self.dist.setdefault(neighbor, {})[loc] = math.inf
                         self.dist[neighbor][neighbor] = 0 # Distance to self is 0

                    if self.dist[start_loc][neighbor] == math.inf:
                        self.dist[start_loc][neighbor] = current_dist + 1
                        queue.append(neighbor)

    def __call__(self, node):
        state = node.state

        # 1. Parse current state
        man_location = None
        object_locations = {} # obj_name -> location_name
        usable_spanners_in_state = set() # spanner_name
        loose_nuts_in_state = set() # nut_name
        man_carrying_spanner_name = None # spanner_name
        man_name = None # Assume there's only one man

        for fact_string in state:
            pred, args = parse_fact(fact_string)
            if pred == 'at':
                obj, loc = args
                object_locations[obj] = loc
                # Infer man's name and location - rely on 'bob' from examples
                if obj == 'bob':
                    man_name = obj
                    man_location = loc

            elif pred == 'carrying':
                 # Assuming args are [man_name, spanner_name]
                 man_name_carrier, spanner_name_carried = args
                 man_carrying_spanner_name = spanner_name_carried
                 man_name = man_name_carrier # Confirm man's name

            elif pred == 'usable':
                spanner_name_usable = args[0]
                usable_spanners_in_state.add(spanner_name_usable)

            elif pred == 'loose':
                nut_name_loose = args[0]
                loose_nuts_in_state.add(nut_name_loose)

        # Fallback for man_name if not found via 'at bob' or 'carrying'
        if man_name is None:
             # This is a weak point, assuming the first object at a location is the man if not identified otherwise
             for obj, loc in object_locations.items():
                 # Crude type check based on name patterns from examples
                 is_spanner = 'spanner' in obj.lower()
                 is_nut = 'nut' in obj.lower()
                 if not is_spanner and not is_nut:
                      man_name = obj
                      man_location = loc
                      break

        # Ensure man_location is identified and is a known location
        if man_location is None or man_location not in self.locations:
             # This state is likely invalid or represents a dead end
             return math.inf

        # 2. Identify loose goal nuts
        loose_goal_nuts = {n for n in loose_nuts_in_state if f'(tightened {n})' in self.goals}

        # 3. Identify available usable spanners
        man_has_usable_spanner = (man_carrying_spanner_name is not None and man_carrying_spanner_name in usable_spanners_in_state)
        available_usable_spanners_at_loc = set() # spanners at locations that are usable and not carried
        for s in usable_spanners_in_state:
            if s != man_carrying_spanner_name:
                if s in object_locations:
                    available_usable_spanners_at_loc.add(s)

        # 4. Check for solvability
        num_nuts_to_tighten = len(loose_goal_nuts)
        num_usable_spanners = len(available_usable_spanners_at_loc) + (1 if man_has_usable_spanner else 0)

        if num_nuts_to_tighten > num_usable_spanners:
            return math.inf

        # If already goal state, return 0
        if num_nuts_to_tighten == 0:
             return 0

        # 5. Initialize heuristic calculation
        h = 0
        current_man_location = man_location
        remaining_nuts = set(loose_goal_nuts)
        remaining_spanners_at_loc = set(available_usable_spanners_at_loc)
        man_currently_has_usable_spanner = man_has_usable_spanner

        # 6. Use carried spanner if available and usable
        if man_currently_has_usable_spanner and remaining_nuts:
            # Find closest nut to use the carried spanner on
            closest_nut = None
            min_dist = math.inf

            # Ensure current_man_location has precomputed distances
            if current_man_location in self.dist:
                for nut in remaining_nuts:
                    nut_loc = object_locations.get(nut)
                    # Ensure nut_loc is valid and reachable from current_man_location
                    if nut_loc and nut_loc in self.locations and current_man_location in self.dist and nut_loc in self.dist[current_man_location]:
                        dist = self.dist[current_man_location][nut_loc]
                        if dist < min_dist:
                            min_dist = dist
                            closest_nut = nut

            if closest_nut: # Should find one if remaining_nuts is not empty and locations are valid/reachable
                h += min_dist + 1 # walk + tighten
                current_man_location = object_locations[closest_nut]
                remaining_nuts.remove(closest_nut)
                man_currently_has_usable_spanner = False # Spanner is used up
            # else: If no reachable nut found, the problem might be unsolvable despite counts.
            # The loop in step 7 will handle returning infinity if no path is found.


        # 7. Greedily pick up spanner and tighten nut for remaining nuts
        while remaining_nuts:
            best_nut = None
            best_spanner = None
            min_cost = math.inf

            # Ensure current_man_location has precomputed distances
            if current_man_location not in self.dist:
                 return math.inf # Should not happen if initial man_location was valid

            for nut in remaining_nuts:
                nut_loc = object_locations.get(nut)
                if not nut_loc or nut_loc not in self.locations: continue # Skip if nut location is unknown or invalid

                for spanner in remaining_spanners_at_loc:
                    spanner_loc = object_locations.get(spanner)
                    if not spanner_loc or spanner_loc not in self.locations: continue # Skip if spanner location is unknown or invalid

                    # Cost = walk to spanner + pickup + walk to nut + tighten
                    # Ensure all required distances are precomputed and finite
                    if (current_man_location in self.dist and spanner_loc in self.dist[current_man_location] and
                        spanner_loc in self.dist and nut_loc in self.dist[spanner_loc]):

                        dist_to_spanner = self.dist[current_man_location][spanner_loc]
                        dist_spanner_to_nut = self.dist[spanner_loc][nut_loc]

                        # If any distance is infinity, this path is impossible
                        if dist_to_spanner == math.inf or dist_spanner_to_nut == math.inf:
                            continue # This spanner-nut pair is not reachable

                        cost = dist_to_spanner + 1 + dist_spanner_to_nut + 1 # walk + pickup + walk + tighten

                        if cost < min_cost:
                            min_cost = cost
                            best_nut = nut
                            best_spanner = spanner

            if best_nut is None: # No reachable spanner-nut pair found for remaining nuts
                return math.inf # Problem is unsolvable from this state

            h += min_cost
            current_man_location = object_locations[best_nut]
            remaining_nuts.remove(best_nut)
            remaining_spanners_at_loc.remove(best_spanner)

        # 8. Return the final value of h
        return h

import re
import heapq
import logging
from collections import deque, defaultdict

# Assuming Heuristic, Operator, Task are available from the planner's core
from heuristics.heuristic_base import Heuristic
# from task import Operator, Task # Uncomment if not imported via heuristic_base

# Helper function to parse a PDDL fact string
def parse_fact(fact_string):
    """Removes surrounding brackets and splits by spaces."""
    # Handle potential leading/trailing whitespace and ensure it's a string
    fact_string = str(fact_string).strip()
    if not fact_string.startswith('(') or not fact_string.endswith(')'):
        # Not a valid fact string format we expect, return empty
        # logging.warning(f"Unexpected fact format: {fact_string}") # Avoid excessive logging
        return None, []
    # Remove surrounding brackets and split by spaces
    parts = fact_string[1:-1].split()
    if not parts:
        return None, [] # Empty fact?
    predicate = parts[0]
    objects = parts[1:]
    return predicate, objects

class spannerHeuristic(Heuristic):
    """
    Summary:
    A domain-dependent heuristic for the spanner domain. It estimates the cost
    to reach the goal state (all goal nuts tightened) by summing the estimated
    minimum costs for tightening each currently loose goal nut. The cost for
    tightening a single nut is estimated based on the travel required for the
    man and a usable spanner to reach the nut's location, plus the pickup and
    tighten actions. Since each nut requires a unique usable spanner, the
    heuristic approximates the minimum total cost by finding the cheapest
    pairing of loose nuts to available usable spanners using a greedy approach
    on pre-calculated individual nut-spanner costs.

    Assumptions:
    - The PDDL domain follows the structure provided, specifically regarding
      predicates ('at', 'carrying', 'usable', 'link', 'tightened', 'loose')
      and object types implicitly defined by predicate usage.
    - There is exactly one man object in the domain, identifiable as the first
      argument of the 'carrying' predicate or as the sole 'locatable' object
      in initial 'at' facts that is not a spanner or nut.
    - Nut locations are static and can be determined from the initial state.
    - The location graph defined by 'link' facts is connected for all relevant
      locations (man's initial location, spanner locations, nut locations).
      If not connected, the heuristic correctly returns infinity.

    Heuristic Initialization:
    In the constructor, the heuristic performs the following steps once:
    1.  Identifies the man object, spanner objects, nut objects, and location
        objects by examining all facts (initial state, goals, static, operator
        pre/add/del effects) and inferring types based on predicate usage
        ('carrying', 'usable', 'tightened', 'loose', 'at', 'link').
    2.  Identifies the set of goal nuts from the task's goal facts.
    3.  Determines the static location of each goal nut from the initial state.
    4.  Builds a graph of locations based on the static 'link' facts.
    5.  Computes all-pairs shortest path distances between all identified
        locations using Breadth-First Search (BFS). These distances represent
        the minimum number of 'walk' actions required to travel between locations.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state `s`:
    1.  Identify the man's current location in state `s`.
    2.  Identify all usable spanners in state `s` and their current locations
        (either carried by the man or at a specific location).
    3.  Identify all goal nuts that are currently 'loose' in state `s`. If there
        are no loose goal nuts, the state is a goal state, and the heuristic is 0.
    4.  If the number of loose goal nuts exceeds the number of available usable
        spanners, the goal is unreachable in this relaxation, return infinity.
    5.  For each loose goal nut `n` at its static location `l_n`, and for each
        available usable spanner `s` at its current location `l_s` (where `l_s`
        is the man's location if `s` is carried, or the spanner's location if
        `s` is at a location), calculate the estimated cost to use spanner `s`
        to tighten nut `n`:
        - If spanner `s` is carried by the man: Cost = `dist(man_location, l_n)` (walk) + 1 (tighten).
        - If spanner `s` is at location `l_s`: Cost = `dist(man_location, l_s)` (walk to spanner) + 1 (pickup) + `dist(l_s, l_n)` (walk to nut) + 1 (tighten).
        Store these costs as a list of (cost, nut, spanner) tuples.
    6.  Sort the list of (cost, nut, spanner) tuples in ascending order of cost.
    7.  Greedily select assignments from the sorted list. Iterate through the
        sorted tuples. For each tuple (cost, nut, spanner), if the nut has not
        already been assigned a spanner and the spanner has not already been
        assigned to a nut, select this assignment. Add the cost to the total
        heuristic value, and mark the nut and spanner as used.
    8.  Continue selecting assignments until all loose goal nuts have been assigned
        a unique usable spanner.
    9.  The total sum of the costs of the selected assignments is the heuristic value.
        If it's impossible to assign a unique usable spanner to each loose nut
        (which should be caught by step 4, but this is a safeguard), return infinity.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task

        # --- Identify Objects and Locations ---
        self.man_name = None
        self.spanner_names = set()
        self.nut_names = set()
        self.locations = set()

        all_fact_strings = set(self.task.initial_state) | self.task.goals | self.task.static
        for op in self.task.operators:
             all_fact_strings |= op.preconditions | op.add_effects | op.del_effects

        potential_men = set()
        potential_spanners = set()
        potential_nuts = set()
        potential_locations = set()

        for fact_string in all_fact_strings:
             predicate, objects = parse_fact(fact_string)
             if predicate is None: continue # Skip invalid facts

             if predicate == 'at' and len(objects) == 2:
                 obj, loc = objects
                 potential_locations.add(loc)
             elif predicate == 'carrying' and len(objects) == 2:
                 man_obj, spanner_obj = objects
                 potential_men.add(man_obj)
                 potential_spanners.add(spanner_obj)
             elif predicate == 'usable' and len(objects) == 1:
                 spanner_obj = objects[0]
                 potential_spanners.add(spanner_obj)
             elif predicate == 'tightened' and len(objects) == 1:
                 nut_obj = objects[0]
                 potential_nuts.add(nut_obj)
             elif predicate == 'loose' and len(objects) == 1:
                 nut_obj = objects[0]
                 potential_nuts.add(nut_obj)
             elif predicate == 'link' and len(objects) == 2:
                 l1, l2 = objects
                 potential_locations.add(l1)
                 potential_locations.add(l2)

        # Infer types based on predicate usage
        men_in_carrying = {parse_fact(f)[1][0] for f in all_fact_strings if parse_fact(f)[0] == 'carrying' and len(parse_fact(f)[1]) == 2}
        if len(men_in_carrying) == 1:
            self.man_name = list(men_in_carrying)[0]
        elif len(men_in_carrying) > 1:
            logging.warning(f"Multiple potential man objects found based on 'carrying': {men_in_carrying}. Using the first one.")
            self.man_name = list(men_in_carrying)[0]
        else:
             # Fallback: Try to find an object in initial state 'at' fact that is not a spanner or nut
             initial_at_objects = {parse_fact(f)[1][0] for f in self.task.initial_state if parse_fact(f)[0] == 'at' and len(parse_fact(f)[1]) == 2}
             potential_men_fallback = initial_at_objects - potential_spanners - potential_nuts - potential_locations
             if len(potential_men_fallback) == 1:
                  self.man_name = list(potential_men_fallback)[0]
             else:
                  logging.error("Could not uniquely identify the man object. Heuristic may be incorrect.")
                  # Assign a placeholder, likely leading to infinite heuristic if man isn't found
                  self.man_name = 'unknown_man'


        self.spanner_names = potential_spanners
        self.nut_names = potential_nuts
        self.locations = potential_locations

        # --- Identify Goal Nuts and Their Static Locations ---
        self.goal_nuts = set()
        self.nut_locations = {} # nut_name -> location_name
        for goal_fact_string in self.task.goals:
            predicate, objects = parse_fact(goal_fact_string)
            if predicate == 'tightened' and len(objects) == 1:
                nut_name = objects[0]
                self.goal_nuts.add(nut_name)
                # Find the location of this nut from the initial state
                found_loc = False
                for fact_string in self.task.initial_state:
                     pred, objs = parse_fact(fact_string)
                     if pred == 'at' and len(objs) == 2 and objs[0] == nut_name:
                         self.nut_locations[nut_name] = objs[1]
                         found_loc = True
                         break
                if not found_loc:
                    logging.error(f"Location for goal nut {nut_name} not found in initial state.")
                    self.nut_locations[nut_name] = None # Indicate unknown location


        # --- Build Location Graph and Compute Shortest Paths ---
        self.links = defaultdict(set)
        for fact_string in self.task.static:
            predicate, objects = parse_fact(fact_string)
            if predicate == 'link' and len(objects) == 2:
                l1, l2 = objects
                self.links[l1].add(l2)
                self.links[l2].add(l1)
                # self.locations is already populated

        # Ensure all locations mentioned in nut_locations are included
        self.locations.update(self.nut_locations.values())

        # Compute all-pairs shortest paths
        self.dist = {}
        for start_loc in self.locations:
            self.dist[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_location):
        """Computes shortest path distances from start_location to all others."""
        distances = {loc: float('inf') for loc in self.locations}
        if start_location not in self.locations:
             # This location wasn't identified in __init__, likely an issue.
             logging.warning(f"BFS started from unknown location: {start_location}")
             return distances # Cannot compute from here

        distances[start_location] = 0
        queue = deque([start_location])

        while queue:
            current_loc = queue.popleft()

            if current_loc in self.links: # Check if current_loc has any links
                for neighbor in self.links[current_loc]:
                    if neighbor in distances and distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_loc] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """
        Computes the spanner heuristic value for the given state.
        """
        state = node.state

        man_location = None
        carried_spanners = set()
        usable_spanners_at_loc = {} # spanner_name -> location_name
        all_usable_spanners_in_state = set() # Usable spanners currently in the state
        loose_goal_nuts = set()

        # Parse current state to find dynamic information
        for fact_string in state:
            predicate, objects = parse_fact(fact_string)
            if predicate is None: continue # Skip invalid facts

            if predicate == 'at' and len(objects) == 2:
                obj_name, loc_name = objects
                if obj_name == self.man_name:
                    man_location = loc_name
                elif obj_name in self.spanner_names:
                     # Location of spanners will be processed after finding usable ones
                     pass
            elif predicate == 'carrying' and len(objects) == 2:
                carrier, item = objects
                if carrier == self.man_name and item in self.spanner_names:
                    carried_spanners.add(item)
            elif predicate == 'usable' and len(objects) == 1:
                spanner_name = objects[0]
                if spanner_name in self.spanner_names:
                     all_usable_spanners_in_state.add(spanner_name)
            elif predicate == 'loose' and len(objects) == 1:
                nut_name = objects[0]
                # Check if this loose nut is a goal nut
                if nut_name in self.goal_nuts:
                    loose_goal_nuts.add(nut_name)
            # 'tightened' facts are implicitly handled by checking 'loose' for goal nuts

        # If man_location wasn't found (e.g., initial state didn't have it?), return inf
        if man_location is None:
             logging.error("Man's location not found in state.")
             return float('inf')

        # Now that we have all usable spanners, find their locations if not carried
        for spanner_name in all_usable_spanners_in_state:
            if spanner_name not in carried_spanners:
                # Find where this usable spanner is located in the current state
                found_loc = False
                for fact_string in state:
                    pred, objs = parse_fact(fact_string)
                    if pred == 'at' and len(objs) == 2 and objs[0] == spanner_name:
                        usable_spanners_at_loc[spanner_name] = objs[1]
                        found_loc = True
                        break
                if not found_loc:
                     # Usable spanner is not carried and not at a location? It's lost or deleted.
                     # It won't be included in the usable spanners pool for cost calculation
                     pass

        # Pool of usable spanners available for assignment
        available_usable_spanners = carried_spanners.intersection(all_usable_spanners_in_state) | set(usable_spanners_at_loc.keys())

        # --- Heuristic Calculation ---
        if not loose_goal_nuts:
            return 0 # Goal reached

        num_loose_goal_nuts = len(loose_goal_nuts)
        num_available_usable_spanners = len(available_usable_spanners)

        if num_available_usable_spanners < num_loose_goal_nuts:
            # Not enough usable spanners for all loose goal nuts
            return float('inf')

        # Calculate costs for all possible (loose nut, usable spanner) assignments
        potential_assignments = []
        for nut_name in loose_goal_nuts:
            nut_location = self.nut_locations.get(nut_name)
            if nut_location is None:
                # Location of goal nut unknown, problem parsing error?
                logging.error(f"Location for goal nut {nut_name} unknown.")
                return float('inf')

            # Check if nut_location is in our known locations and reachable from man_location
            if nut_location not in self.dist or man_location not in self.dist[nut_location] or self.dist[man_location][nut_location] == float('inf'):
                 logging.warning(f"Goal nut {nut_name} at {nut_location} is unreachable from man's location {man_location}.")
                 # If the nut location itself is unreachable, this nut cannot be tightened
                 return float('inf')


            for spanner_name in available_usable_spanners:
                spanner_loc = None
                is_carried = spanner_name in carried_spanners

                if is_carried:
                    spanner_loc = man_location # Spanner is with the man
                else:
                    spanner_loc = usable_spanners_at_loc.get(spanner_name)

                if spanner_loc is None:
                     # Should not happen if spanner is in available_usable_spanners
                     continue

                # Check if spanner_loc is in our known locations and reachable from/to man/nut locations
                if spanner_loc not in self.dist or man_location not in self.dist[spanner_loc] or self.dist[man_location][spanner_loc] == float('inf') or nut_location not in self.dist[spanner_loc] or self.dist[spanner_loc][nut_location] == float('inf'):
                     logging.warning(f"Usable spanner {spanner_name} at {spanner_loc} is unreachable from man {man_location} or nut {nut_name} at {nut_location}.")
                     continue # Cannot use this spanner for this nut

                cost_ns = float('inf')
                if is_carried:
                    # Cost: walk from man's current location to nut location + tighten
                    cost_ns = self.dist[man_location][nut_location] + 1
                else:
                    # Cost: walk from man to spanner + pickup + walk from spanner location to nut + tighten
                    cost_ns = self.dist[man_location][spanner_loc] + 1 + self.dist[spanner_loc][nut_location] + 1

                if cost_ns < float('inf'):
                    potential_assignments.append((cost_ns, nut_name, spanner_name))

        # Sort assignments by cost
        potential_assignments.sort()

        # Greedily select assignments (approximation of min-cost matching)
        total_heuristic = 0
        used_nuts = set()
        used_spanners = set()

        for cost, nut, spanner in potential_assignments:
            if nut not in used_nuts and spanner not in used_spanners:
                total_heuristic += cost
                used_nuts.add(nut)
                used_spanners.add(spanner)
                if len(used_nuts) == num_loose_goal_nuts:
                    break # All loose nuts assigned

        # If we couldn't assign a spanner to every loose nut, it's unreachable
        if len(used_nuts) < num_loose_goal_nuts:
             # This case should ideally be covered by the initial check on num_available_usable_spanners,
             # but handles cases where spanners/nuts are in disconnected components or no valid paths exist.
             return float('inf')

        return total_heuristic

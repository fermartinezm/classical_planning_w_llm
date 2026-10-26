# Assuming Heuristic base class is available as heuristics.heuristic_base.Heuristic
from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import deque
import math # For infinity

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
    # Ensure the number of parts matches the number of args, unless args has wildcards at the end
    if len(parts) != len(args) and '*' not in args:
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(start_node, graph):
    """Performs BFS to find shortest distances from start_node."""
    distances = {start_node: 0}
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()
        distance = distances[current_node]

        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distance + 1
                    queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(graph):
    """Computes shortest paths between all pairs of nodes in the graph."""
    all_distances = {}
    # Collect all unique nodes from keys and values in the graph
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    for start_node in all_nodes:
        all_distances[start_node] = bfs(start_node, graph)
    return all_distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost to tighten all loose nuts by greedily processing
    nuts based on distance, accounting for man movement and spanner acquisition.
    Assumes one man. Assumes nuts and spanners don't move from initial locations.
    Assumes man can carry multiple spanners.
    Infers object types (man, nut, spanner) based on predicate usage in
    initial state and goals.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing location distances
        and identifying static object properties.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # Build location graph from static link facts
        self.location_graph = {}
        all_locations = set()
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, []).append(loc2)
                self.location_graph.setdefault(loc2, []).append(loc1)
                all_locations.add(loc1)
                all_locations.add(loc2)

        # Add locations mentioned in the initial state 'at' facts to ensure they are in the graph nodes
        initial_locations_in_at = set()
        for fact in self.initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)[1:]
                 initial_locations_in_at.add(loc)

        for loc in initial_locations_in_at:
             self.location_graph.setdefault(loc, []) # Add node even if no links

        # Compute all-pairs shortest paths
        self.distances = compute_all_pairs_shortest_paths(self.location_graph)

        # Identify objects and infer types based on predicate usage
        potential_nuts = set()
        potential_spanners = set()
        potential_men = set()
        objects_with_at = set()

        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                objects_with_at.add(parts[1])
            elif parts[0] == 'carrying':
                 potential_men.add(parts[1])
                 potential_spanners.add(parts[2])
            elif parts[0] == 'usable':
                 potential_spanners.add(parts[1])
            elif parts[0] == 'loose':
                 potential_nuts.add(parts[1])

        for goal in self.goals:
             if match(goal, "tightened", "*"):
                  potential_nuts.add(get_parts(goal)[1])

        # Assign types based on collected potentials. Assume one man.
        self.all_nuts = potential_nuts
        self.all_spanners = potential_spanners

        # The man is the object that can carry spanners. Assume there's only one.
        if potential_men:
             self.man = list(potential_men)[0]
        else:
             # Fallback: If no object is in 'carrying', assume the man is the
             # only object in 'at' that is not a nut or spanner.
             man_candidates_from_at = objects_with_at - self.all_nuts - self.all_spanners
             if man_candidates_from_at:
                  self.man = list(man_candidates_from_at)[0]
             else:
                  # This case should ideally not happen in a valid spanner instance
                  # where a man exists.
                  self.man = None


        # Store initial locations for confirmed nuts and spanners
        self.initial_nut_locations = {}
        self.initial_spanner_locations = {}
        for fact in self.initial_state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 if obj in self.all_nuts:
                     self.initial_nut_locations[obj] = loc
                 elif obj in self.all_spanners:
                     self.initial_spanner_locations[obj] = loc


        # Store goal nuts (nuts that need to be tightened)
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify current state elements
        current_L_M = None
        current_carried_spanners = set()
        current_usable_spanners = set()
        current_loose_nuts = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] == self.man:
                current_L_M = parts[2]
            elif parts[0] == 'carrying' and parts[1] == self.man:
                current_carried_spanners.add(parts[2])
            elif parts[0] == 'usable':
                current_usable_spanners.add(parts[1])
            elif parts[0] == 'loose':
                current_loose_nuts.add(parts[1])

        # Identify usable spanners currently carried by the man
        current_carried_usable_count = len(current_carried_spanners.intersection(current_usable_spanners))

        # Identify usable spanners currently at locations (not carried)
        current_at_loc_usable_spanners_grouped = {} # {location: [spanner1, spanner2, ...], ...}
        for fact in state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 if obj in self.all_spanners and obj in current_usable_spanners and obj not in current_carried_spanners:
                     current_at_loc_usable_spanners_grouped.setdefault(loc, []).append(obj)

        # Filter loose nuts to only include those that are goal nuts
        loose_goal_nuts = list(current_loose_nuts.intersection(self.goal_nuts))

        # If all goal nuts are already tightened, cost is 0
        if not loose_goal_nuts:
            return 0

        # Check if enough usable spanners exist in total
        total_usable_spanners_at_loc = sum(len(sp_list) for sp_list in current_at_loc_usable_spanners_grouped.values())
        total_usable_spanners_available = current_carried_usable_count + total_usable_spanners_at_loc
        if len(loose_goal_nuts) > total_usable_spanners_available:
             # Not enough spanners to tighten all required nuts
             return 1000000 # Effectively infinity

        # Check if man's current location is in the distance graph
        if current_L_M not in self.distances:
             # Man is in an isolated location. Check if any nut or spanner location is reachable.
             all_relevant_locations = set(self.initial_nut_locations[n] for n in loose_goal_nuts if n in self.initial_nut_locations) | set(current_at_loc_usable_spanners_grouped.keys())
             if any(loc in self.distances.get(current_L_M, {}) for loc in all_relevant_locations):
                  # Some relevant locations are reachable, proceed. Unreachable ones will cause KeyError later.
                  pass
             else:
                  # No relevant locations are reachable from the man's isolated start.
                  return 1000000


        # 2. Estimate cost using greedy approach
        total_cost = 0
        current_man_location = current_L_M
        loose_nuts_with_locations = [(n, self.initial_nut_locations[n]) for n in loose_goal_nuts if n in self.initial_nut_locations]

        # Sort loose nuts by distance from the man's *initial* location in this state
        # This sorting is static for the duration of this heuristic call.
        try:
            loose_nuts_with_locations.sort(key=lambda item: self.distances.get(current_L_M, {}).get(item[1], math.inf))
        except Exception:
             # Should not happen if initial location check passed, but as safeguard
             return 1000000


        # Iterate through sorted loose nuts
        for i, (nut, nut_location) in enumerate(loose_nuts_with_locations):
            # Cost to move man to nut_location
            try:
                move_cost = self.distances[current_man_location][nut_location]
            except KeyError:
                 # Nut location is unreachable from current man location
                 return 1000000

            total_cost += move_cost
            current_man_location = nut_location # Man is now at the nut location

            # Cost to get a usable spanner
            if current_carried_usable_count == 0:
                # Man needs to pick up a spanner
                nuts_remaining = len(loose_nuts_with_locations) - i

                # Find the nearest location with usable spanners
                nearest_spanner_group_loc = None
                min_dist_to_spanner_group = float('inf')

                # Filter locations with spanners to only include those reachable from current_man_location
                reachable_spanner_group_locations = [
                    loc for loc in current_at_loc_usable_spanners_grouped.keys()
                    if loc in self.distances.get(current_man_location, {})
                ]

                if not reachable_spanner_group_locations:
                     # No reachable usable spanners at locations
                     return 1000000 # Unreachable

                for spanner_location in reachable_spanner_group_locations:
                    try:
                        dist_to_spanner_group = self.distances[current_man_location][spanner_location]
                        if dist_to_spanner_group < min_dist_to_spanner_group:
                            min_dist_to_spanner_group = dist_to_spanner_group
                            nearest_spanner_group_loc = spanner_location
                    except KeyError:
                         # Should not happen due to reachable_spanner_group_locations filtering
                         pass # Skip this location

                if nearest_spanner_group_loc is None:
                     # Should not happen if reachable_spanner_group_locations was not empty
                     return 1000000 # No reachable spanner group found

                # Number of spanners available at the nearest location
                num_spanners_at_group = len(current_at_loc_usable_spanners_grouped[nearest_spanner_group_loc])

                # Number of spanners to pick up in this trip
                spanners_to_pickup = min(nuts_remaining, num_spanners_at_group)

                # Cost to go get the spanners and return to the nut location
                # Man is currently at nut_location.
                walk_to_spanner_cost = self.distances[current_man_location][nearest_spanner_group_loc]
                pickup_cost = spanners_to_pickup # Each pickup action costs 1
                walk_back_to_nut_cost = self.distances[nearest_spanner_group_loc][nut_location] # Man is at spanner_location after pickup

                spanner_acquisition_cost = walk_to_spanner_cost + pickup_cost + walk_back_to_nut_cost
                total_cost += spanner_acquisition_cost

                # Update man's location after picking up spanners and returning to nut
                current_man_location = nut_location

                # Update carried spanners count
                current_carried_usable_count += spanners_to_pickup

                # Remove picked up spanners from the available list
                picked_up_spanners = current_at_loc_usable_spanners_grouped[nearest_spanner_group_loc][:spanners_to_pickup]
                current_at_loc_usable_spanners_grouped[nearest_spanner_group_loc] = current_at_loc_usable_spanners_grouped[nearest_spanner_group_loc][spanners_to_pickup:]
                if not current_at_loc_usable_spanners_grouped[nearest_spanner_group_loc]:
                     del current_at_loc_usable_spanners_grouped[nearest_spanner_group_loc]


            # Use one spanner for the current nut
            if current_carried_usable_count > 0:
                 current_carried_usable_count -= 1
            else:
                 # This case should not be reached if spanners_to_pickup logic is correct
                 # and total spanners check passed.
                 return 1000000 # Logic error

            # Cost for tighten_nut action
            total_cost += 1

        return total_cost


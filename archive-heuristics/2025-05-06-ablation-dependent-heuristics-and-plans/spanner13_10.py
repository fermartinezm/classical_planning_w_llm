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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose
    nuts specified in the goal. It uses a greedy approach: the man prioritizes
    getting a usable spanner if he doesn't have one, and then going to the
    closest remaining loose goal nut to tighten it.

    # Assumptions
    - There is exactly one man object.
    - Nuts do not move from their initial locations.
    - Spanners become unusable after tightening one nut.
    - The man can only carry one spanner at a time (implied by domain structure,
      no mention of multiple grippers or carrying multiple items).
    - There are enough usable spanners in the initial state to tighten all goal nuts.
    - The location graph is connected, allowing movement between any two locations
      relevant to the problem (man's initial location, nut locations, spanner locations).

    # Heuristic Initialization
    - Build a graph of locations based on `link` predicates found in static facts.
    - Compute shortest path distances between all pairs of locations using BFS.
    - Identify the man object by looking for objects that are not nuts, spanners, or locations.
    - Identify goal nuts from the task goals.
    - Store the static location for each nut based on the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic estimates the remaining cost as follows:
    1. Identify the man's current location from the state.
    2. Identify usable spanners currently in the state (those with the `(usable S)` predicate).
    3. Identify which of these usable spanners the man is currently carrying.
    4. Identify usable spanners currently on the ground and their locations.
    5. Identify loose nuts in the state that are also goal nuts (these are the remaining subproblems).
    6. If there are no remaining loose goal nuts, the heuristic is 0.
    7. If the number of remaining loose goal nuts exceeds the total number of currently usable spanners (carried + on ground), the problem is unsolvable from this state, return infinity.
    8. Initialize total estimated cost to 0.
    9. Create mutable copies of the remaining loose goal nuts and usable spanners on the ground for simulation. Get the current count of carried usable spanners.
    10. While there are still loose goal nuts remaining:
        a. If the man is currently carrying a usable spanner (simulation count > 0):
            i. Find the loose goal nut closest to the man's current location (simulation location).
            ii. Add the shortest distance to this nut's location plus 1 (for the `tighten_nut` action) to the total cost.
            iii. Update the man's current location in the simulation to the nut's location.
            iv. Decrement the count of carried usable spanners in the simulation.
            v. Remove the nut from the set of remaining loose goal nuts.
        b. If the man is not carrying a usable spanner (simulation count == 0):
            i. If there are no usable spanners left on the ground (simulation set is empty), return infinity (cannot solve remaining nuts).
            ii. Find the usable spanner on the ground (in simulation) closest to the man's current location (simulation location).
            iii. Add the shortest distance to this spanner's location plus 1 (for the `pickup_spanner` action) to the total cost.
            iv. Update the man's current location in the simulation to the spanner's location.
            v. Remove the spanner from the set of remaining usable spanners on the ground in the simulation.
            vi. Set the count of carried usable spanners in the simulation to 1.
            vii. Now that the man has a spanner, find the loose goal nut closest to the man's *new* current location (the spanner's location).
            viii. Add the shortest distance to this nut's location plus 1 (for the `tighten_nut` action) to the total cost.
            ix. Update the man's current location in the simulation to the nut's location.
            x. Decrement the count of carried usable spanners in the simulation.
            xi. Remove the nut from the set of remaining loose goal nuts.
    11. Return the total estimated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and computing
        shortest path distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # 1. Build location graph and compute distances
        self.locations = set()
        links = []
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                self.locations.add(l1)
                self.locations.add(l2)
                # Assuming links are bidirectional
                links.append((l1, l2))
                links.append((l2, l1))

        self.dist = {}
        for start_loc in self.locations:
            self.dist[start_loc] = {}
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            while q:
                current_loc, d = q.popleft()
                self.dist[start_loc][current_loc] = d
                for l1, l2 in links:
                    if l1 == current_loc and l2 not in visited:
                        visited.add(l2)
                        q.append((l2, d + 1))

        # Fill in unreachable locations with infinity distance
        for l1 in self.locations:
            for l2 in self.locations:
                if l2 not in self.dist[l1]:
                    self.dist[l1][l2] = math.inf


        # 2. Identify goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # 3. Store static nut locations
        self.nut_location = {}
        # Nuts are locatable objects. Find them in the initial state.
        # Assuming nuts don't move, their initial location is their static location.
        # Identify all potential nut objects first
        all_nuts = {get_parts(fact)[1] for fact in initial_state if match(fact, "loose", "*")} | \
                   {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 if obj in all_nuts:
                     self.nut_location[obj] = loc

        # 4. Identify the man object
        self.man_obj = None
        # Identify all potential spanner objects
        all_spanners = {get_parts(fact)[1] for fact in initial_state if match(fact, "usable", "*")} | \
                      {get_parts(fact)[1] for fact in initial_state if match(fact, "carrying", "*", "*")} | \
                      {get_parts(fact)[1] for fact in initial_state if match(fact, "at", "*", "*") and get_parts(fact)[1].startswith('spanner')} # Heuristic: starts with spanner

        # The man is the object that is 'at' a location but is not a known nut, spanner, or location itself.
        for fact in initial_state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj not in all_nuts and obj not in all_spanners and obj not in self.locations:
                    self.man_obj = obj
                    break

        if self.man_obj is None:
             # Fallback/Error case: Could not identify the man.
             # This should ideally not happen in valid problem instances.
             # For robustness, we could try other heuristics or raise an error.
             # For this exercise, we assume the man is found.
             print("Warning: Could not identify the man object in spanner heuristic init.")


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Get man's current location
        current_man_location = None
        for fact in state:
            if match(fact, "at", self.man_obj, "*"):
                current_man_location = get_parts(fact)[2]
                break
        if current_man_location is None:
             # Man is not at any location? Invalid state.
             return math.inf # Should not happen in valid states

        # 2. Get carried spanners and usable spanners in state
        carried_spanners = {get_parts(fact)[2] for fact in state if match(fact, "carrying", self.man_obj, "*")}
        usable_spanners_in_state = {get_parts(fact)[1] for fact in state if match(fact, "usable", "*")}

        # 3. Get usable spanners on ground and their locations
        usable_spanners_on_ground_locations = {}
        for spanner in usable_spanners_in_state:
            if spanner not in carried_spanners:
                # Find its location
                for fact in state:
                    if match(fact, "at", spanner, "*"):
                        usable_spanners_on_ground_locations[spanner] = get_parts(fact)[2]
                        break

        # 4. Get loose nuts in state and identify loose goal nuts
        loose_nuts_in_state = {get_parts(fact)[1] for fact in state if match(fact, "loose", "*")}
        loose_goal_nuts = {nut for nut in loose_nuts_in_state if (f"(tightened {nut})") in self.goals}

        # 5. If no remaining loose goal nuts, return 0
        if not loose_goal_nuts:
            return 0

        # Count usable spanners carried
        current_carried_usable_spanners = len({s for s in carried_spanners if s in usable_spanners_in_state})

        # 6. Check if enough usable spanners exist for remaining nuts
        total_usable_spanners_available = current_carried_usable_spanners + len(usable_spanners_on_ground_locations)
        if len(loose_goal_nuts) > total_usable_spanners_available:
             # Cannot tighten all remaining nuts, problem is unsolvable from here
             return math.inf

        # 7. Initialize cost
        cost = 0

        # Create mutable copies for the greedy simulation
        remaining_loose_goal_nuts = set(loose_goal_nuts)
        remaining_usable_spanners_on_ground = dict(usable_spanners_on_ground_locations)
        current_man_location_sim = current_man_location
        current_carried_usable_spanners_sim = current_carried_usable_spanners

        # 8. Greedy simulation
        while remaining_loose_goal_nuts:
            if current_carried_usable_spanners_sim > 0:
                # Man has a usable spanner, go tighten the closest remaining loose goal nut
                closest_nut = None
                min_dist = math.inf
                for nut in remaining_loose_goal_nuts:
                    nut_loc = self.nut_location.get(nut) # Use precomputed static location
                    if nut_loc is None:
                         # Location of a goal nut not found in init? Invalid state/task.
                         return math.inf

                    # Check reachability
                    if current_man_location_sim not in self.dist or nut_loc not in self.dist[current_man_location_sim]:
                         return math.inf # Unreachable location

                    dist = self.dist[current_man_location_sim][nut_loc]
                    if dist < min_dist:
                        min_dist = dist
                        closest_nut = nut

                if closest_nut is None:
                     # Should not happen if remaining_loose_goal_nuts is not empty and locations are reachable
                     return math.inf

                nut_loc = self.nut_location[closest_nut]
                cost += self.dist[current_man_location_sim][nut_loc] + 1  # walk + tighten
                current_man_location_sim = nut_loc
                current_carried_usable_spanners_sim -= 1
                remaining_loose_goal_nuts.remove(closest_nut)

            else: # current_carried_usable_spanners_sim == 0
                # Man needs a usable spanner first
                if not remaining_usable_spanners_on_ground:
                     # No usable spanners left on the ground
                     return math.inf # Cannot solve remaining nuts

                # Find closest usable spanner S on ground
                closest_spanner = None
                min_dist_spanner = math.inf
                for spanner, spanner_loc in remaining_usable_spanners_on_ground.items():
                    # Check reachability
                    if current_man_location_sim not in self.dist or spanner_loc not in self.dist[current_man_location_sim]:
                         return math.inf # Unreachable location

                    dist = self.dist[current_man_location_sim][spanner_loc]
                    if dist < min_dist_spanner:
                        min_dist_spanner = dist
                        closest_spanner = spanner

                if closest_spanner is None:
                     # Should not happen if remaining_usable_spanners_on_ground is not empty and locations are reachable
                     return math.inf

                spanner_loc = remaining_usable_spanners_on_ground[closest_spanner]
                cost += self.dist[current_man_location_sim][spanner_loc] + 1 # walk + pickup
                current_man_location_sim = spanner_loc
                del remaining_usable_spanners_on_ground[closest_spanner]
                current_carried_usable_spanners_sim = 1 # Now carrying one usable spanner

                # Now go tighten the closest remaining loose goal nut from the spanner location
                closest_nut = None
                min_dist_nut = math.inf
                for nut in remaining_loose_goal_nuts:
                    nut_loc = self.nut_location.get(nut) # Use precomputed static location
                    if nut_loc is None:
                         # Location of a goal nut not found in init? Invalid state/task.
                         return math.inf

                    # Check reachability
                    if current_man_location_sim not in self.dist or nut_loc not in self.dist[current_man_location_sim]:
                         return math.inf # Unreachable location

                    dist = self.dist[current_man_location_sim][nut_loc]
                    if dist < min_dist_nut:
                        min_dist_nut = dist
                        closest_nut = nut

                if closest_nut is None:
                     # Should not happen if remaining_loose_goal_nuts is not empty and locations are reachable
                     return math.inf

                nut_loc = self.nut_location[closest_nut]
                cost += self.dist[current_man_location_sim][nut_loc] + 1 # walk + tighten
                current_man_location_sim = nut_loc
                current_carried_usable_spanners_sim -= 1
                remaining_loose_goal_nuts.remove(closest_nut)

        # 9. Return total estimated cost
        return cost

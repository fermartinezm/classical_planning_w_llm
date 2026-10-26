from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts specified in the goal.
    It considers the need to walk to the nut's location, pick up a usable spanner, and finally tighten the nut.
    The heuristic uses a simplified shortest path calculation based on the 'link' predicates.

    # Assumptions:
    - The goal is always to tighten a set of nuts.
    - The cost of each action (walk, pickup_spanner, tighten_nut) is 1.
    - We prioritize using a usable spanner if available and walking to the nut's location.

    # Heuristic Initialization
    - Extracts the goal nuts that need to be tightened.
    - Pre-processes the static 'link' facts to enable shortest path calculations between locations.

    # Step-By-Step Thinking for Computing Heuristic
    For each nut that is specified as 'tightened' in the goal and is currently 'loose' in the state:
    1. Initialize the estimated cost for this nut to 0.
    2. Check if the man is at the nut's location. If not, calculate the shortest path (in terms of number of 'link' actions)
       from the man's current location to the nut's location using Breadth-First Search (BFS) on the location graph derived from 'link' predicates.
       Add the length of this shortest path to the cost. If already at the nut's location, the walking cost is 0.
    3. Check if the man is carrying a usable spanner. If not, increment the cost by 1 (for 'pickup_spanner' action).
    4. Increment the cost by 1 for the 'tighten_nut' action itself.
    5. Sum up the costs calculated for each goal nut that is currently loose.
    This sum represents the total estimated cost to reach the goal state.
    """

    def __init__(self, task):
        """
        Initialize the spanner heuristic.
        Extract goal nuts and pre-process static link information for pathfinding.
        """
        self.goals = task.goals
        self.static_facts = task.static

        self.goal_nuts = set()
        for goal_fact in self.goals:
            if match(goal_fact, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal_fact)[1])

        self.links = collections.defaultdict(list)
        for static_fact in self.static_facts:
            if match(static_fact, "link", "*", "*"):
                l1, l2 = get_parts(static_fact)[1], get_parts(static_fact)[2]
                self.links[l1].append(l2)
                self.links[l2].append(l1) # links are bidirectional in this heuristic

    def __call__(self, node):
        """
        Calculate the heuristic value for the given state.
        Estimate the cost to tighten all goal nuts that are currently loose.
        """
        state = node.state
        heuristic_value = 0

        current_man_location = None
        carried_spanner = None
        usable_spanner_available = False
        nut_locations = {}
        spanner_locations = {}

        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if parts[1] == 'bob':
                    current_man_location = parts[2]
                elif parts[1] in self.goal_nuts:
                    nut_locations[parts[1]] = parts[2]
                elif parts[1].startswith('spanner'):
                    spanner_locations[parts[1]] = parts[2]
            elif match(fact, "carrying", "bob", "*"):
                carried_spanner = get_parts(fact)[2]
            elif match(fact, "usable", "*"):
                usable_spanner_available = True # At least one spanner is usable (not necessarily carried)

        carried_usable_spanner = False
        if carried_spanner and "(usable {})".format(carried_spanner) in state:
            carried_usable_spanner = True

        for nut in self.goal_nuts:
            if "(loose {})".format(nut) in state:
                nut_location = nut_locations.get(nut)
                if nut_location is None: # Should not happen given problem definition, but for robustness
                    continue

                nut_cost = 1 # For tighten_nut action

                # Walking cost
                if current_man_location != nut_location:
                    shortest_path_len = self.shortest_path_length(current_man_location, nut_location)
                    if shortest_path_len is not None:
                        nut_cost += shortest_path_len
                    else:
                        nut_cost += 1 # If no path found, assume 1 walk action (might be suboptimal but avoids infinite cost)


                # Spanner cost
                if not carried_usable_spanner:
                    nut_cost += 1 # For pickup_spanner action

                heuristic_value += nut_cost

        return heuristic_value

    def shortest_path_length(self, start_location, goal_location):
        """
        Calculate the shortest path length between two locations using BFS.
        Returns the path length or None if no path exists.
        """
        if start_location == goal_location:
            return 0

        queue = collections.deque([(start_location, 0)]) # (location, distance)
        visited = {start_location}

        while queue:
            current_location, distance = queue.popleft()

            for neighbor in self.links[current_location]:
                if neighbor not in visited:
                    if neighbor == goal_location:
                        return distance + 1
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        return None # No path found

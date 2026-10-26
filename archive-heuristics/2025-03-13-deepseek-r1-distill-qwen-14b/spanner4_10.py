from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import sys

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at man1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.

    # Assumptions:
    - The man can carry multiple spanners but can only use one at a time.
    - The man must be at the same location as a loose nut to tighten it.
    - If the man is not carrying a spanner, he must pick one up before tightening a nut.

    # Heuristic Initialization
    - Extract static facts to build a graph of locations and their connections.
    - Store the goal locations for each nut.

    # Step-by-Step Thinking for Computing Heuristic
    1. Check if all nuts are already tightened. If yes, return 0.
    2. For each loose nut, determine the man's current location and whether he is carrying a spanner.
    3. Calculate the shortest path from the man's current location to the nut's location.
    4. If the man is not carrying a spanner, add the actions needed to pick one up.
    5. Sum the actions needed for all nuts, considering whether the man needs to move and whether he needs to pick up a spanner.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Static facts (location links) to build a graph.
        - Goal conditions (tightened nuts).
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts (location links)

        # Build a graph of locations based on static links
        self.location_graph = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = get_parts(fact)[1], get_parts(fact)[2]
                if loc1 not in self.location_graph:
                    self.location_graph[loc1] = []
                self.location_graph[loc1].append(loc2)
                if loc2 not in self.location_graph:
                    self.location_graph[loc2] = []
                self.location_graph[loc2].append(loc1)

        # Precompute all possible distances between locations using BFS
        self.distances = {}
        for loc in self.location_graph:
            self.distances[loc] = {}
            queue = deque()
            queue.append((loc, 0))
            visited = {loc}
            while queue:
                current, dist = queue.popleft()
                for neighbor in self.location_graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[loc][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state

        # Check if all nuts are already tightened
        all_tight = True
        nuts = []
        for fact in state:
            if match(fact, "loose", "*"):
                nuts.append(fact)
            if match(fact, "tightened", "*"):
                pass
            else:
                all_tight = False
        if all_tight:
            return 0

        # Extract current state information
        man_location = None
        carrying_spanners = []
        loose_nuts = []
        for fact in state:
            if match(fact, "at", "bob", "*"):
                man_location = get_parts(fact)[2]
            elif match(fact, "carrying", "bob", "*"):
                carrying_spanners.append(get_parts(fact)[2])
            elif match(fact, "loose", "*"):
                loose_nuts.append(get_parts(fact)[1])

        # If man's location is not found, assume it's the initial location (shed)
        if man_location is None:
            man_location = "shed"

        # For each loose nut, calculate the required actions
        total_actions = 0
        has_spanner = len(carrying_spanners) > 0

        for nut in loose_nuts:
            nut_location = None
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_location = get_parts(fact)[2]
                    break
            if nut_location is None:
                continue  # Nut is not present in the state (should not happen)

            # Calculate the shortest path from man's location to nut's location
            if man_location == nut_location:
                distance = 0
            else:
                if nut_location not in self.distances.get(man_location, {}):
                    # If no path exists, assume it's unreachable (should not happen in spanner domain)
                    continue
                distance = self.distances[man_location][nut_location]

            # Add the distance (walking actions)
            total_actions += distance

            # If not carrying a spanner, need to pick one up
            if not has_spanner:
                # Find the nearest spanner location
                nearest_spanner = None
                min_spanner_distance = float('inf')
                for spanner_fact in state:
                    if match(spanner_fact, "at", "*", "*") and match(spanner_fact, "*", "spanner", "*"):
                        spanner_loc = get_parts(spanner_fact)[2]
                        if spanner_loc != man_location:
                            if spanner_loc in self.distances[man_location]:
                                spanner_distance = self.distances[man_location][spanner_loc]
                                if spanner_distance < min_spanner_distance:
                                    min_spanner_distance = spanner_distance
                                    nearest_spanner = spanner_loc
                if nearest_spanner is None:
                    # No spanner available (should not happen in spanner domain)
                    continue
                else:
                    # Move to spanner
                    total_actions += min_spanner_distance
                    # Pick up the spanner
                    total_actions += 1
                    has_spanner = True

            # Tighten the nut
            total_actions += 1

        return total_actions

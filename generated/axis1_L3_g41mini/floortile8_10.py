from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(tightened nut1 locatable11)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, pattern) for part, pattern in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (spanner) domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal conditions,
    primarily the predicates (tightened-2 ?locatable ?location) from the current state.
    It considers the current tightened locations of nuts and the target "tightened-2" relocated 
    facts, along with the loose-2 nut locations and reachable links. 

    # Assumptions
    - The goal involves relocating locatable objects with tightened nuts to specific final locations (tightened-2 predicates).
    - Each nut has exactly one tightened locatable and one loose-2 location.
    - Movement of tightened links (locatable objects that link and carry tightened nuts) along loose chains is needed.
    - Actions can move tightened nuts along locatable chains (loose / carrying / usable predicates).
    - There is exactly one "at-2" location for the agent at a time.
    - The domain predicates and actions combine movement and tightening of nuts via spanners.

    # Heuristic Initialization
    - Extract the goal goals: all (tightened-2 locatable location) facts that must hold at the end.
    - From static facts, extract
      - Chains of locatables connected by (loose ?v1 ?v2), (carrying ?v1 ?v2), and (usable ?v1 ?v2).
      - Which locatables are currently (link ?locatable).
      - The set of loosened nuts and their current loose-2 locations.
      - The agent's current location(s) via (at-2 ?location).
    - Precompute adjacency info on locatables from static facts for quick path distance estimates.

    # Step-By-Step Thinking for Computing Heuristic
    1. Return heuristic = 0 if all goal facts (tightened-2 locatable location) are true in the current state.
    2. For each goal (tightened-2 locatable location_goal), find current (tightened-2 locatable location_current) or
       locate the locatable object currently tightened to that nut or locatable.
    3. Compute minimal estimated steps to move the tightened locatable from its current position to the goal location.
       This path estimation uses adjacency derived from static facts loose, carrying, usable edges, 
       modeling the chain paths the spanner can move along.
    4. Count also the cost related to tightening nuts (like tightening loose nuts to new locations).
       Detect how many nuts need to be tightened at the goal and add estimated costs accordingly.
    5. Add a penalty cost for moving the agent itself if it is not at the needed locations (use current at-2 location).
    6. Sum the estimated per-goal costs into overall heuristic estimate.
    
    This heuristic is informed and domain-dependent but not guaranteed admissible, 
    trading admissibility for better guidance in greedy best-first search.

    The heuristic tries to capture both the "distance" of relocating tightened spanners 
    and the necessary nut tightening operations left.

    The calculation deliberately avoids expensive search computations and instead uses
    shortest path approximations on the locatable graph plus simple counts of nuts needing tightening.
    """

    def __init__(self, task):
        self.goals = task.goals  # goal facts (e.g., tightened-2 locatable location)
        self.static = task.static  # static facts for adjacency and structure

        # Parse and store goal tightened-2 targets: {locatable: location}
        self.goal_tightened_2 = {}
        for fact in self.goals:
            if match(fact, "tightened-2", "*", "*"):
                parts = get_parts(fact)
                locatable, location = parts[1], parts[2]
                self.goal_tightened_2[locatable] = location

        # Build adjacency maps for locatable movement graph from static facts:
        # Connections are undirected edges derived from loose, carrying, usable predicates.
        self.adjacency = {}  # locatable -> set of locatables
        def add_edge(a, b):
            self.adjacency.setdefault(a, set()).add(b)
            self.adjacency.setdefault(b, set()).add(a)

        for fact in self.static:
            # For undirected edges on locatable objects
            if match(fact, "loose", "*", "*") or match(fact, "carrying", "*", "*") or match(fact, "usable", "*", "*"):
                parts = get_parts(fact)
                a, b = parts[1], parts[2]
                add_edge(a, b)

        # Extract all locatables linked by (link ?locatable) predicate - currently usable by spanners
        self.current_links = {
            get_parts(fact)[1] for fact in self.static if match(fact, "link", "*")
        }

        # Extract the set of loose nuts and their locations for potential tightening
        self.loose2_nuts = {}  # nut -> location
        for fact in self.static:
            if match(fact, "loose-2", "*", "*"):
                parts = get_parts(fact)
                nut, location = parts[1], parts[2]
                self.loose2_nuts[nut] = location

        # Extract agent location(s) via (at-2 ?location) facts, expected 1 (or handle multiple as set)
        self.agent_locations = {
            get_parts(fact)[1] for fact in self.static if match(fact, "at-2", "*")
        }

    def __call__(self, node):
        state = node.state

        # 1) Return 0 if all goal facts are satisfied
        if self.goals <= state:
            return 0

        # Helper to find all facts in the current state matching a pattern
        def facts_matching(*pattern):
            return {fact for fact in state if match(fact, *pattern)}

        # Create lookup for tightened-2 facts in current state: locatable -> location
        current_tightened_2 = {}
        for fact in facts_matching("tightened-2", "*", "*"):
            parts = get_parts(fact)
            locatable, location = parts[1], parts[2]
            current_tightened_2[locatable] = location

        # Create lookup for tightened facts (tightened ?nut ?locatable) in current state: nut -> locatable
        tightened = {}
        for fact in facts_matching("tightened", "*", "*"):
            parts = get_parts(fact)
            nut, locatable = parts[1], parts[2]
            tightened[nut] = locatable

        # Create lookup for loose-2 facts (loose-2 ?nut ?location) in current state: nut -> location
        loose_2 = {}
        for fact in facts_matching("loose-2", "*", "*"):
            parts = get_parts(fact)
            nut, location = parts[1], parts[2]
            loose_2[nut] = location

        # Extract all locatables currently (link ?locatable) in the current state
        current_links = {get_parts(f)[1] for f in facts_matching("link", "*")}

        # Extract agent location(s) (at-2 ?location) in current state
        agent_locations = {get_parts(f)[1] for f in facts_matching("at-2", "*")}

        # Helper function: shortest path distance on locatable adjacency graph
        # Uses BFS, with a short cutoff for performance and returns large default if unreachable
        def shortest_dist(start, goal, cutoff=50):
            if start == goal:
                return 0
            visited = {start}
            frontier = [(start, 0)]
            while frontier:
                current, dist = frontier.pop(0)
                if dist >= cutoff:
                    return cutoff + 1  # over cutoff
                for neighbor in self.adjacency.get(current, ()):
                    if neighbor == goal:
                        return dist + 1
                    if neighbor not in visited:
                        visited.add(neighbor)
                        frontier.append((neighbor, dist + 1))
            return cutoff + 1  # no path found within cutoff

        heuristic_value = 0

        # Cost factors (approximate units per action)
        move_cost = 1       # Moving tightened locatable along one edge
        tighten_cost = 1    # Cost to tighten a loose nut to new location (approximate)
        agent_move_cost = 1 # Cost for moving the agent (at-2) between locations

        # Estimate cost for each goal tightened-2 fact
        for goal_locatable, goal_location in self.goal_tightened_2.items():
            # Where is this locatable currently tightened-2?
            current_location = current_tightened_2.get(goal_locatable, None)

            # If goal locatable is not tightened-2 yet, find locatable where this locatable's nut is tightened
            # Note: The domain relates tightened nuts to locatables; some locatables may need movement
            # We can also try to find the nut related to this locatable for tightening estimation.
            # The goal facts attach locatables to locations tightened-2, so we check current tightened-2 positions first

            if current_location is None:
                # If not tightened-2 yet, maybe it's linked to a nut that needs tightening-2 at goal location
                # Heuristic: assume full cost of tightening operation and movement needed.
                # Just add tightening cost here as unknown current location.
                heuristic_value += tighten_cost * 2  # heuristic penalty for missing tightening

                # Also add some penalty for agent movement (conservatively 1)
                heuristic_value += agent_move_cost
                continue

            # Compute movement cost from current location to goal location on locatable chain graph.
            dist = shortest_dist(current_location, goal_location)

            # If unreachable, assign a large cost to motivate search away from this assumption
            if dist > 50:
                dist = 50

            heuristic_value += dist * move_cost

        # Count how many nuts remain loose-2 in current state that must be tightened eventually
        # Each such nut corresponds roughly to one tighten action (plus potential movements)
        loose_nuts_to_tighten = 0
        for nut, loose_loc in loose_2.items():
            # If this nut is still loose-2 in a location different from the goal tightened-2 location:
            # Actually, check where this nut is tightened currently, if not at goal then need tightening
            current_tightened_locatable = tightened.get(nut, None)
            target_location = None
            # The goal location for tightened-2 is actually per locatable,
            # but here to estimate we try to find if the nut's tightened locatable is at a goal location
            # We check if any goal tightened-2 fact relates to that locatable
            if current_tightened_locatable is not None:
                target_location = self.goal_tightened_2.get(current_tightened_locatable, None)

            if target_location is None or loose_loc != target_location:
                loose_nuts_to_tighten += 1

        heuristic_value += loose_nuts_to_tighten * tighten_cost

        # Estimate agent movement cost: if agent is not at any goal location of tightened-2 locatables,
        # add minimal distance cost from agent location to closest goal location to encourage approach.
        if agent_locations and self.goal_tightened_2:
            min_agent_move_dist = min(
                sum(
                    shortest_dist(agent_loc, goal_loc)
                    for goal_loc in self.goal_tightened_2.values()
                )
                for agent_loc in agent_locations
            )
            # Normalize by number of goals to approximate average distance
            avg_agent_move_dist = min_agent_move_dist / max(len(self.goal_tightened_2), 1)
            heuristic_value += avg_agent_move_dist * agent_move_cost

        # Ensure heuristic is at least 1 if non-zero (to avoid zero costs when not goal)
        if heuristic_value > 0:
            heuristic_value = max(1, heuristic_value)

        return heuristic_value

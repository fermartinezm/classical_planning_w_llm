from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(tightened nut1 locatable1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the 'spanner' (secret_domain) domain.

    Summary:
    This heuristic estimates the number of actions needed to achieve the goal conditions 
    in the secret_domain, which concerns moving "tightened" locatable objects to target locations
    and updating the "tightened-2" predicate accordingly. The heuristic focuses on the main
    goal predicate `(tightened-2 locatable location)`, estimating how many moves of tightened
    objects must be done to reach these goal configurations.

    Assumptions:
    - The goal requires certain locatables associated with tightened nuts to be at certain locations
      (expressed by `(tightened-2 locatable location)` predicates).
    - Static knowledge about the linkage of locatables (`link`), the "loose" relations between locatables,
      and the usability relations is available.
    - Tightened nuts move between locatables via spanners moving on links; moving requires walking/spanning
      across loosed/tightened connections.
    - The heuristic does not have to be admissible but should provide a finite estimate guiding the search.
    - The domain likely requires moving 'link' to the locatable where we want to "tighten-2" the locatable to the goal location.

    Heuristic Initialization:
    - Extract all goal `(tightened-2 locatable location)` facts as the targets to achieve.
    - Extract static information:
      * The set of `link` locatables (where the spanner can be).
      * The `loose` relations between locatables (edges).
      * The `at` relations between locatables (adjacency).
      * The `usable` relation, indicating possible valid spanner moves.
      * The `carrying` relations describing carrying relationships between locatables.
    - Organize the static facts into adjacency mappings and sets for quick lookup.

    Step-By-Step Thinking for Computing Heuristic:
    1. If the current state satisfies all the goal `(tightened-2 locatable location)` facts, heuristic = 0.
    2. For each goal `(tightened-2 locatable loc_goal)`:
       - Check if this fact holds in the current state; if yes, no cost.
       - Otherwise, estimate the minimal number of spanner moves to bring the spanner-link to the target location,
         so that the "tighten" action can be applied.
       - The spanner moves on `link` locatables; thus, we find the minimal "link" path distance between current link-positions 
         associated to the locatable and the target location.
       - Count 1 action per spanner move (walk, pickup_spanner, etc.), plus 1 action to apply `tighten_nut`.
    3. Sum these minimal costs for all goal locatables not yet satisfied.
    4. For each nut, consider the distance between current tightened locatable and the goal tightened-2 location.
       Possibly approximate cost by the shortest path length over `loose` edges.
    5. Heuristic = total estimated spanner moves + tighten actions needed.
    6. This heuristic is not admissible but will guide greedy best-first search effectively.

    Implementation details:
    - Use BFS to find shortest paths in linkage graphs.
    - Only consider reachable `link` locatables for spanner moves.
    - Estimate cost as number of spanner moves + tightening action per missing goal fact.

    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goals and static connectivity data."""
        self.goals = set(fact for fact in task.goals if fact.startswith("(tightened-2 "))
        self.static = task.static

        # Extract all locatables that have a 'link' predicate (positions where spanner can be)
        self.links = {get_parts(fact)[1] for fact in self.static if match(fact, "link", "*")}

        # Build adjacency for 'loose' relation between locatables (undirected)
        self.loose_adjacency = {}
        for fact in self.static:
            if match(fact, "loose", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.loose_adjacency.setdefault(l1, set()).add(l2)
                self.loose_adjacency.setdefault(l2, set()).add(l1)

        # Build adjacency for 'at' relation between locatables (directed)
        # Probably represents a graph or path that can be walked on
        self.at_adjacency = {}
        for fact in self.static:
            if match(fact, "at", "*", "*"):
                _, from_loc, to_loc = get_parts(fact)
                self.at_adjacency.setdefault(from_loc, set()).add(to_loc)

        # Usable edges, usable locatable pairs (directed)
        self.usable = {}
        for fact in self.static:
            if match(fact, "usable", "*", "*"):
                _, u_from, u_to = get_parts(fact)
                self.usable.setdefault(u_from, set()).add(u_to)

        # Carrying relation between locatables (directed)
        self.carrying = {}
        for fact in self.static:
            if match(fact, "carrying", "*", "*"):
                _, c_from, c_to = get_parts(fact)
                self.carrying.setdefault(c_from, set()).add(c_to)

    def __call__(self, node):
        """
        Estimate the number of actions required to reach the goal from the current state.

        We approximate by summing for each goal tightened-2 fact not achieved:
        - One tighten_nut action.
        - Plus estimated shortest path of spanner moves on 'link' locatables 
          from current location associated with that locatable to the goal location.

        The state representation includes facts of interest, especially:
        - (tightened-2 locatable location): locatable moved to location
        - (tightened nut locatable): nut tightened at locatable
        - (loose-2 nut location): nut loose at location
        - (link locatable): spanner is at locatable locations

        To avoid complexity, heuristic focuses on number of missing tightened-2 facts
        and the spanner moves needed to achieve them, via BFS on link locatables.

        Returns:
            int: heuristic estimate of remaining actions, 0 if goal state.
        """
        state = node.state

        # Quick check: if all goal (tightened-2 locatable location) hold, heuristic = 0
        if self.goals <= state:
            return 0

        # Extract current tightened-2 facts (locatable-position to location)
        current_tightened_2 = {
            (get_parts(fact)[1], get_parts(fact)[2])
            for fact in state if match(fact, "tightened-2", "*", "*")
        }

        # Map each goal: locatable -> required location
        goals_map = {}
        for fact in self.goals:
            _, locatable, loc_goal = get_parts(fact)
            goals_map[locatable] = loc_goal

        # Extract current spanner positions (link locatables where spanner is)
        current_spanner_positions = {get_parts(fact)[1] for fact in state if match(fact, "link", "*")}
        # If no link position, approximate from tightened facts of nuts if possible
        # Extract tightened (nut locatable) facts: nut location
        tightened_nuts = {}
        for fact in state:
            if match(fact, "tightened", "*", "*"):
                _, nut, locatable = get_parts(fact)
                tightened_nuts[nut] = locatable

        # Candidate spanner positions if none links available: locatables of tightened nuts
        if not current_spanner_positions and tightened_nuts:
            current_spanner_positions = set(tightened_nuts.values())

        # BFS helpers - shortest path lengths between link locatables
        def bfs_shortest_distance(start_set, target):
            """Return shortest distance from any start in start_set to target locatable using 'link' adjacency."""

            # Build adjacency graph for links (undirected)
            graph = {}
            for link_loc in self.links:
                # Find neighbors linked by 'at', 'loose', 'carrying', or 'usable' on locatables with link predicate
                neighbors = set()

                if link_loc in self.at_adjacency:
                    neighbors |= self.at_adjacency[link_loc]
                # Also consider loose adjacency if locatables are also links
                if link_loc in self.loose_adjacency:
                    neighbors |= (loc for loc in self.loose_adjacency[link_loc] if loc in self.links)
                if link_loc in self.usable:
                    neighbors |= (loc for loc in self.usable[link_loc] if loc in self.links)
                if link_loc in self.carrying:
                    neighbors |= (loc for loc in self.carrying[link_loc] if loc in self.links)

                graph[link_loc] = set(neighbors)

            from collections import deque

            visited = set()
            queue = deque()
            for s in start_set:
                if s == target:
                    return 0
                queue.append((s, 0))
                visited.add(s)

            while queue:
                current, dist = queue.popleft()
                if current == target:
                    return dist
                for nb in graph.get(current, []):
                    if nb not in visited:
                        visited.add(nb)
                        queue.append((nb, dist + 1))
            # If unreachable, assign a large cost
            return 20

        heuristic = 0

        # For each goal (tightened-2 locatable location) not satisfied yet,
        # estimate cost: 
        # - 1 action to tighten (tighten_nut)
        # - plus minimal moves for a spanner to reach the locatable to apply this action, 
        #   approximated as shortest distance between any current spanner link position and the target locatable (treated as link)
        for locatable, goal_loc in goals_map.items():
            if (locatable, goal_loc) in current_tightened_2:
                continue  # Goal achieved for this locatable

            # Locate the spanner start positions - approximate by current spanner links, or maybe locatable's tightened location

            # The goal locatable needs the spanner link there to perform tighten_nut-2.
            # We approximate the distance for spanner move by shortest path on 'link' locatables.
            # If goal_loc also corresponds to a locatable with link predicate, use it; else approximate by locatable itself.

            # Check if goal loc is a link locatable; if not, approximate by a closest link
            # For heuristic purposes, assume goal_loc itself is link locatable (or else just count 1 for tighten)

            # Compute spanner move distance
            spanner_move_distance = bfs_shortest_distance(current_spanner_positions, locatable)
            # Add 1 action at least for tighten_nut (or tighten_nut-2)
            actions_for_goal = spanner_move_distance + 1

            heuristic += actions_for_goal

        # Put a floor to heuristic to be at least 1 if goals are still missing
        return max(heuristic, 1)

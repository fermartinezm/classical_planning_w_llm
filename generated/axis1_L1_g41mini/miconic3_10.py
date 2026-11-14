from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by stripping parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Check if a PDDL fact matches a pattern with possible wildcards '*'.
    """
    parts = get_parts(fact)
    return all(fnmatch(p, pat) for p, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain (ud3uf11z).

    Summary:
    The heuristic estimates the number of actions needed to achieve the goal of having
    all relevant nxbd1hc3 objects in the (a7sb56om ?) state predicate. It leverages
    the domain structure, where objects of type nxbd1hc3 must be "activated" via a two-step
    transition through predicates and locations of type t4rdxh32, while moving the 
    "omggbn9z" fact along a connected graph defined by acovodrf relations.

    Assumptions:
    - The goal involves having (a7sb56om ?p) true for certain nxbd1hc3 objects.
    - Each nxbd1hc3 object transitions from (kut2ksnk ?p ?f) with omggbn9z at ?f,
      then diwlmgg9 action sets (f2xo7hb6 ?p).
    - Then (klw11kjf ?p ?f') with omggbn9z at ?f' triggers y7nik9q2 action, producing (a7sb56om ?p).
    - The predicate omggbn9z indicates an active location, moved using qs8pdrqj and i0veo5kv actions
      along the acovodrf graph.
    - Switching between predicates kut2ksnk and klw11kjf for each object requires omggbn9z to be at their respective t4rdxh32 locations.
    - Achieving (a7sb56om ?p) requires first converting (kut2ksnk ?p ?f) via diwlmgg9 at ?f
      (omggbn9z must be there), then using y7nik9q2 at ?f'.
    - Movement of omggbn9z among locations is along the acovodrf graph and costs one action per edge.
    - Objects independently require these sequences; actions interleave due to shared omggbn9z location movement.

    Heuristic Initialization:
    - Store the goal objects requiring (a7sb56om ?p).
    - Store static graphs: 
      - The acovodrf graph of location connectivity (undirected).
      - Mappings kut2ksnk: nxbd1hc3 → set of locations.
      - Mappings klw11kjf: nxbd1hc3 → set of locations.
    - Initialize lookup for which nxbd1hc3 objects are already in states (a7sb56om ?p) (goal achieved),
      or (f2xo7hb6 ?p) (middle activated), or still needing activation.

    Step-By-Step Thinking for Computing Heuristic:
    1. Extract from the current state:
       - The current location(s) of the unique omggbn9z marker.
       - For each nxbd1hc3 object:
         - Whether (a7sb56om ?p) holds (goal done).
         - Whether (f2xo7hb6 ?p) holds (middle activated).
    2. For each object that still needs (a7sb56om ?p):
       a) If already (a7sb56om ?p), cost = 0 for that object.
       b) Else if (f2xo7hb6 ?p), then:
          - Need to move omggbn9z to one of its klw11kjf locations,
          - Perform y7nik9q2 to achieve (a7sb56om ?p).
          - Estimate cost = shortest distance from omggbn9z current location to any klw11kjf location + 1 action.
       c) Else (neither a7sb56om nor f2xo7hb6):
          - Need to move omggbn9z to one of its kut2ksnk locations,
          - Perform diwlmgg9 (+1 action),
          - Move omggbn9z to one of its klw11kjf locations,
          - Perform y7nik9q2 (+1 action).
          - Estimate cost = 
              shortest distance from omggbn9z current location to any kut2ksnk location
              + 1 (diwlmgg9)
              + shortest distance from that kut2ksnk location to any klw11kjf location
              + 1 (y7nik9q2)
    3. Since omggbn9z is a single marker, full serial planning would require sequencing movements.
       However, to keep heuristic efficient, sum costs for all objects assuming independent movement
       starting from the current omggbn9z location (overestimating cost but guiding search well).
    4. Return the sum over all objects' estimated costs as heuristic value.

    This heuristic is not necessarily admissible but is designed for efficiency and effective guidance.
    """

    def __init__(self, task):
        # Save goal facts (we focus on (a7sb56om ?p) goals).
        self.goals = task.goals

        # Store static info: 
        # Build graph of acovodrf as adjacency map: undirected edges for traversal.
        self.adj = {}
        # Maps: nxbd1hc3 -> set of locations for kut2ksnk and klw11kjf
        self.kut2ksnk_map = {}
        self.klw11kjf_map = {}

        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "acovodrf" and len(parts) == 3:
                f1, f2 = parts[1], parts[2]
                self.adj.setdefault(f1, set()).add(f2)
                self.adj.setdefault(f2, set()).add(f1)
            elif parts[0] == "kut2ksnk" and len(parts) == 3:
                p, f = parts[1], parts[2]
                self.kut2ksnk_map.setdefault(p, set()).add(f)
            elif parts[0] == "klw11kjf" and len(parts) == 3:
                p, f = parts[1], parts[2]
                self.klw11kjf_map.setdefault(p, set()).add(f)

        # Precompute goal objects requiring (a7sb56om ?p).
        self.goal_objects = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "a7sb56om" and len(parts) == 2:
                self.goal_objects.add(parts[1])

    def shortest_dist(self, start, targets):
        """
        Compute shortest distance from start to any of the targets in the graph self.adj using BFS.
        If no target reachable, return a large number.
        """
        if not targets:
            return float('inf')
        if start in targets:
            return 0
        visited = {start}
        frontier = [(start, 0)]

        while frontier:
            current, dist = frontier.pop(0)
            for neigh in self.adj.get(current, []):
                if neigh not in visited:
                    if neigh in targets:
                        return dist + 1
                    visited.add(neigh)
                    frontier.append((neigh, dist + 1))
        return float('inf')

    def __call__(self, node):
        state = node.state

        # Quick check for goal state: heuristic = 0
        if self.goals <= state:
            return 0

        # Find current omggbn9z location(s) - there should be exactly one.
        omggbn9z_locs = {get_parts(fact)[1] for fact in state if fact.startswith("(omggbn9z")}
        if not omggbn9z_locs:
            # Defensive fallback: no omggbn9z active means hard to proceed,
            # assign large heuristic value.
            return 1000000
        assert len(omggbn9z_locs) == 1, "Expected exactly one omggbn9z location"
        omggbn9z_loc = next(iter(omggbn9z_locs))

        total_cost = 0

        # For each goal object, estimate cost.
        for p in self.goal_objects:
            # Check if p already satisfies goal (a7sb56om)
            if f"(a7sb56om {p})" in state:
                continue  # cost zero for this object

            # If not goal, check if middle activated (f2xo7hb6)
            has_f2xo7hb6 = f"(f2xo7hb6 {p})" in state

            kut2ksnk_locs = self.kut2ksnk_map.get(p, set())
            klw11kjf_locs = self.klw11kjf_map.get(p, set())

            if has_f2xo7hb6:
                # Cost = shortest distance from omggbn9z_loc to klw11kjf location + 1 (y7nik9q2)
                dist = self.shortest_dist(omggbn9z_loc, klw11kjf_locs)
                if dist == float('inf'):
                    # No reachable klw11kjf location -> large cost
                    dist = 1000000
                cost = dist + 1
            else:
                # Need two moves + actions:
                # move1: omggbn9z_loc -> kut2ksnk location + 1 (diwlmgg9)
                # move2: kut2ksnk location -> klw11kjf location + 1 (y7nik9q2)
                # Since we don't know which kut2ksnk location is best,
                # pick minimal total cost over kut2ksnk locations.

                best_cost = float('inf')

                # For each candidate kut2ksnk location:
                for k_loc in kut2ksnk_locs:
                    dist1 = self.shortest_dist(omggbn9z_loc, {k_loc})
                    if dist1 == float('inf'):
                        continue
                    dist2 = self.shortest_dist(k_loc, klw11kjf_locs)
                    if dist2 == float('inf'):
                        continue
                    total = dist1 + 1 + dist2 + 1  # two moves plus two actions
                    if total < best_cost:
                        best_cost = total

                cost = best_cost
                if cost == float('inf'):
                    # No path found, assign large cost.
                    cost = 1000000

            total_cost += cost

        return total_cost

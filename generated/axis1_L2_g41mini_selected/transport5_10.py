from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The full fact string, e.g. "(predicate arg1 arg2)".
    - `args`: Pattern parts, where '*' matches anything.
    - Returns True if matches pattern, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary:
    This heuristic estimates the number of actions required to reach a goal state in the secret_domain
    by analyzing the relationships between key predicates involving the types hodp29q0 (entities with
    some state), val3ypio/g83opnlj (agents or holders), r86varlt (values or states), and hmhptaxe (positions
    or attributes). It focuses primarily on satisfying the goal facts of the form (jft9ynda agent position).

    Assumptions:
    - The goal facts are of the form (jft9ynda ?gpvob ?pk9o1) specifying desired ownership/position relation.
    - Achieving the goal requires arranging (jft9ynda) facts correctly.
    - Operators sw2fg9ql, k12p0czo, and vyn92k55 allow transferring states or ownership between objects.
    - Static facts define a kind of connectivity or order between hmhptaxe objects via (z6o4oitz).
    - The heuristic does not guarantee admissibility but aims to estimate needed action count efficiently.
    - The heuristic assumes static facts never change and can be used to estimate minimal steps between elements of type hmhptaxe.

    Heuristic Initialization:
    - Extract goal facts of type (jft9ynda ?gpvob ?pk9o1) and store desired goal pairs.
    - Extract static connectivity facts (z6o4oitz ?xbsk8 ?r3kn5) and build a directed graph between hmhptaxe nodes.
    - From the directed graph, precompute shortest paths lengths between hmhptaxe nodes using BFS for fast heuristic computation.
    - Track for each agent (g83opnlj) their current position via facts (jft9ynda ?gpvob ?pk9o1).

    Step-By-Step Thinking for Computing Heuristic:
    1. If the state already satisfies the goal (all goal facts are in state), heuristic = 0.
    2. For each goal fact (jft9ynda g p), check if fact in state:
       - If yes, no cost for this fact.
       - If no, estimate cost to achieve it as:
         a) Find the current position(s) of agent g via (jft9ynda g ?pos) in state.
         b) For each current position, get shortest distance along connectivity graph to goal position p.
         c) Take the minimal distance found, as moving agent's position to p requires at least that many k12p0czo actions.
    3. All goal facts differ by agent and target position; sum all their estimated costs for total heuristic value.
    4. The heuristic also considers needed changes in (yarbdfgc) facts (that relate hodp29q0 to r86varlt) which affect preconditions,
       by counting how many goal predicates referencing (jft9ynda) lack matching (yarbdfgc) facts. Each discrepancy is estimated to cost at least one sw2fg9ql or vyn92k55 action.
    5. Add a small penalty per missing yarbdfgc fact to the sum from step 3 to reflect these required adjustments.
    6. Finally, account for ownership or pairing facts (uypgj7u3) if relevant by counting missing pairs towards achieving sw2fg9ql or vyn92k55 action applicability.
    7. Return the summed heuristic value as an estimate of minimal actions needed to reach the goal from the current state.

    This heuristic avoids full planning or search for efficiency and leverages static connectivity and predicate presence to guide greedy best-first search.
    """

    def __init__(self, task):
        # Save goal facts of interest (jft9ynda ?gpvob ?pk9o1)
        self.goal_jft9ynda = set(
            fact for fact in task.goals if fact.startswith("(jft9ynda ")
        )
        # Store static connectivity for hmhptaxe elements from (z6o4oitz ?xbsk8 ?r3kn5)
        # Graph is adjacency: key -> set of reachable successors
        self.z6o4oitz_graph = {}
        for fact in task.static:
            if match(fact, "z6o4oitz", "*", "*"):
                parts = get_parts(fact)
                src, dst = parts[1], parts[2]
                self.z6o4oitz_graph.setdefault(src, set()).add(dst)

        # Precompute all shortest path lengths between hmhptaxe nodes reachable in graph
        # We store a dict of dicts: distances[src][dst] = minimal edges count or None if unreachable
        self.distances = {}
        for src in self.z6o4oitz_graph:
            self.distances[src] = self._bfs_distances(src)

    def _bfs_distances(self, start):
        """Compute shortest distances from start to all reachable nodes in z6o4oitz graph."""
        distances = {start: 0}
        queue = [start]
        while queue:
            current = queue.pop(0)
            d = distances[current]
            for succ in self.z6o4oitz_graph.get(current, []):
                if succ not in distances:
                    distances[succ] = d + 1
                    queue.append(succ)
        return distances

    def __call__(self, node):
        state = node.state

        # If goal is reached, heuristic is zero.
        if self.goal_jft9ynda <= state:
            return 0

        # Helper: get current positions of an agent (g83opnlj) via (jft9ynda ?gpvob ?pk9o1)
        def agent_positions(agent):
            positions = set()
            for fact in state:
                if match(fact, "jft9ynda", agent, "*"):
                    parts = get_parts(fact)
                    positions.add(parts[2])
            return positions

        # Helper: Extract all (yarbdfgc ?pk9o1 ?fr3fe) facts
        current_yarbdfgc_facts = set(
            fact for fact in state if fact.startswith("(yarbdfgc ")
        )

        # For each goal (jft9ynda g p), estimate cost to achieve (if missing)
        total_cost = 0

        for goal_fact in self.goal_jft9ynda:
            if goal_fact in state:
                continue
            parts = get_parts(goal_fact)
            predicate, agent, goal_pos = parts[0], parts[1], parts[2]
            # Find agent positions now
            current_positions = agent_positions(agent)
            if not current_positions:
                # Agent not located => must at least perform one action to establish jft9ynda 
                # Use heuristic cost 1 as minimal
                min_dist = 1
            else:
                # For each position, find distance to goal_pos in z6o4oitz graph
                dists = []
                for pos in current_positions:
                    dist_map = self.distances.get(pos, {})
                    dist = dist_map.get(goal_pos)
                    if dist is not None:
                        dists.append(dist)
                if dists:
                    min_dist = min(dists)
                else:
                    # No path found in graph from any current pos to goal_pos
                    # Assume a default cost (e.g. 2) for a needed indirect repositioning via actions
                    min_dist = 2
            # Each move along z6o4oitz corresponds to one k12p0czo action cost,
            # plus possibly other actions (sw2fg9ql or vyn92k55),
            # so multiply by a small factor to reflect additional costs.
            total_cost += min_dist + 1  # +1 accounts for related transfer action costs

        # Consider missing yarbdfgc facts that may block achieving goal_jft9ynda:
        # For each goal (jft9ynda agent position), check if corresponding (yarbdfgc position ?val) exists for some val.
        # Count how many such yarbdfgc facts are missing or need changing and estimate additional cost.

        # Extract goal (yarbdfgc) facts from state and goals
        # The domain does not specify direct goal yarbdfgc facts, but we estimate missing based on goals.
        # We consider yarbdfgc needed for each goal position: since operators update yarbdfgc,
        # we count mismatch as cost of at least one action.

        # Gather goal positions from jft9ynda goals
        goal_positions = {get_parts(fact)[2] for fact in self.goal_jft9ynda}

        # Check for each position if some yarbdfgc with that position exists in state, else count missing
        missing_yarbdfgc = 0
        for pos in goal_positions:
            # Find any yarbdfgc with that position in current state
            has_yarbdfgc = any(
                match(f, "yarbdfgc", pos, "*") for f in current_yarbdfgc_facts
            )
            if not has_yarbdfgc:
                missing_yarbdfgc += 1

        # We add a cost penalty proportional to missing yarbdfgc facts to heuristic
        total_cost += missing_yarbdfgc

        # The predicate uypgj7u3 relates g83opnlj and hodp29q0; it appears in preconditions.
        # Missing such pairs likely require one action each (sw2fg9ql or vyn92k55).
        # Check for all pairs used in any precondition of operators or implied by goals.

        # For efficiency, check number of missing uypgj7u3 pairs that appear linked to agents and hodp29q0 objects from state.

        # Extract all uypgj7u3 facts in state
        current_uypgj7u3 = set(f for f in state if f.startswith("(uypgj7u3 "))

        # To estimate, for each agent in goals check if it has a paired hodp29q0 via uypgj7u3, else add a small penalty
        # Extract agents from goals (agents are g83opnlj), and known hodp29q0 from state (yarbdfgc keys)
        agents = {get_parts(fact)[1] for fact in self.goal_jft9ynda}
        hodp29q0_objs = set(
            get_parts(fact)[1] for fact in state if fact.startswith("(yarbdfgc ")
        )

        missing_pairs = 0
        # For each agent and hodp29q0 object check if pair exists, if not, increase penalty moderately
        for agent in agents:
            has_pair = False
            for obj in hodp29q0_objs:
                pair_fact = f"(uypgj7u3 {agent} {obj})"
                if pair_fact in current_uypgj7u3:
                    has_pair = True
                    break
            if not has_pair:
                missing_pairs += 1

        # Add missing pairs penalty (less than main cost, just a heuristic nudging)
        total_cost += missing_pairs * 0.5

        # Return final estimated cost, rounded or as float remains acceptable for heuristic guidance
        return total_cost

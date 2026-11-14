from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the tokens inside a PDDL fact string, e.g. "(on b1 b2)" -> ["on","b1","b2"]."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the classical Blocksworld (STRIPS) domain.

    The heuristic estimates how many block moves (unstack/pickup + stack/putdown)
    are still required to achieve the on/on-table portion of the goal. It combines
    two intuitive signals:
      - For each block that is required to be in a particular support chain in the
        goal, how many positions in that goal chain are not yet realized for that block
        (i.e., how many links in the desired chain starting at the block are missing).
      - How many blocks are stacked above those blocks in the current state (these
        blocks have to be moved away to access the block).

    The heuristic returns 0 exactly on goal states (w.r.t. the goal facts given).
    It always returns a finite integer for solvable states and is efficiently computable.

    Assumptions
    -----------
    - We only consider goal facts of type (on X Y) and (on-table X). Other goal predicates
      (e.g., (clear X)) are ignored for positioning computations (they are implicitly
      handled because correct stacking yields correct clears).
    - The planner's internal state represents facts as strings like "(on b1 b2)".
    - If a block is (holding X) in the current state we treat its parent as "hand".
    - If a block is not mentioned as being on anything, it is treated as having no parent.
    - The heuristic is allowed to be inadmissible (it aims to reduce node expansions).

    Heuristic Initialization
    ------------------------
    - Extract goal parent relations from the task goals:
        * For every (on B P) in the goals we record goal_parent[B] = P.
        * For every (on-table B) in the goals we record goal_parent[B] = "table".
    - For each block that appears as a subject in the goal (i.e., blocks whose position
      is constrained by the goal) we precompute its goal chain: a list [B, parent(B), parent(parent(B)), ...]
      following goal_parent links until "table" or a parent not present in the goal parent map.
      These goal chains allow quick comparison with the current state's downward chain.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state (node.state):
    1. Parse the state's "(on ...)", "(on-table ...)" and "(holding ...)" facts and build:
         - current_parent[block] = parent (a block name, "table", or "hand")
         - children mapping for quick access to the block directly on top of a given block.
    2. For each block B that was constrained by the goal (i.e., appears in goal_parent):
         a) Build the current downward chain for B by following current_parent:
               current_chain = [B, current_parent(B), current_parent(parent), ...]
            stopping once the parent is "table", "hand", unknown, or not a block.
         b) Compare the current_chain to the previously computed goal_chain for B,
            counting how many consecutive entries match from the start
            (i.e., how many links of the desired chain are already in place).
            If matched = len(goal_chain) then B is correctly placed (w.r.t. its goal chain).
            Otherwise B contributes (len(goal_chain) - matched) to the heuristic because those
            many links are missing and at least that many placements are required.
         c) Count how many blocks are stacked above B in the current state (num_above):
            these must be moved away to access B; add num_above to the heuristic.
    3. Sum the contributions across all goal-constrained blocks:
          heuristic = sum_for_B( (len(goal_chain_B) - matched_B) + num_above_B )
    4. Return the summed integer value.

    Notes on behavior
    -----------------
    - The heuristic may double-count some moves (e.g., a block moved away to clear one block
      may also be moved for another), which makes it non-admissible but typically more
      informative and effective for greedy best-first search.
    - The heuristic is zero exactly when all goal-positioned blocks are in their goal chains
      and there are no extraneous blocks above them (thus matching the goal on/on-table facts).
    """

    def __init__(self, task):
        """Extract and preprocess goal information and static facts."""
        # Save the raw goal facts (a set of strings)
        self.goals = task.goals
        # static facts may exist, but Blocksworld has none; keep for completeness
        self.static = task.static

        # Build goal_parent mapping: block -> parent ('table' for on-table)
        self.goal_parent = {}
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "on" and len(parts) >= 3:
                child, parent = parts[1], parts[2]
                self.goal_parent[child] = parent
            elif parts[0] == "on-table" and len(parts) >= 2:
                child = parts[1]
                self.goal_parent[child] = "table"

        # Precompute goal chains for all blocks that have a goal parent.
        # Each chain is a list [block, parent, parent2, ...] following goal_parent
        self.goal_chains = {}
        for block in self.goal_parent.keys():
            chain = [block]
            parent = self.goal_parent.get(block)
            # follow parent links as long as parent is a goal-constrained block
            while parent and parent != "table" and parent in self.goal_parent:
                chain.append(parent)
                parent = self.goal_parent.get(parent)
            # if parent is table include it in chain for completeness
            if parent == "table":
                chain.append("table")
            self.goal_chains[block] = chain

    def __call__(self, node):
        """Compute heuristic value for the given search node (state)."""
        state = node.state  # frozenset of fact strings

        # Build current parent mapping and children mapping
        current_parent = {}  # block -> parent ("table", "hand", or another block)
        children = {}  # parent -> direct block on top (in well-formed states a single block)

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "on" and len(parts) >= 3:
                child, parent = parts[1], parts[2]
                current_parent[child] = parent
                # register child as top-of-stack on parent; if multiple, keep one (domain well-formed)
                children[parent] = child
            elif parts[0] == "on-table" and len(parts) >= 2:
                child = parts[1]
                current_parent[child] = "table"
            elif parts[0] == "holding" and len(parts) >= 2:
                child = parts[1]
                current_parent[child] = "hand"

        # Helper: build current downward chain for a block
        def build_current_chain(block):
            chain = [block]
            parent = current_parent.get(block)
            # Follow until "table", "hand", or unknown parent or parent that is not a block name
            while parent and parent != "table" and parent != "hand" and parent in current_parent:
                chain.append(parent)
                parent = current_parent.get(parent)
            # If parent is table or hand, include that sentinel
            if parent == "table":
                chain.append("table")
            elif parent == "hand":
                chain.append("hand")
            return chain

        # Helper: count how many blocks are stacked above `block` in the current state
        def count_above(block):
            cnt = 0
            # children mapping points parent -> child (one directly on top)
            cur = children.get(block)
            # follow the single chain upwards counting elements
            while cur:
                cnt += 1
                cur = children.get(cur)
            return cnt

        total = 0

        # Only consider blocks that have a goal parent (blocks that the goal constrains)
        for block, goal_chain in self.goal_chains.items():
            current_chain = build_current_chain(block)

            # compute matched prefix length (how many entries from block downward match)
            matched = 0
            for a, b in zip(current_chain, goal_chain):
                if a == b:
                    matched += 1
                else:
                    break

            # number of missing links in the goal chain for this block
            missing_links = len(goal_chain) - matched

            # number of blocks above this block that need to be moved to access it
            above = count_above(block)

            total += missing_links + above

        # If the goal has no positioning constraints, heuristic is 0
        return int(total)
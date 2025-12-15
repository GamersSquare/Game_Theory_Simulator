import numpy as np
from itertools import combinations, product
from tabulate import tabulate
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple, Dict, Optional, Set

class NormalFormGame:
    # Normal Form Game    
    def __init__(self, payoff_A, payoff_B, strategy_labels=None, game_name="Unnamed Game"):
        self.A = np.array(payoff_A, dtype=float)
        self.B = np.array(payoff_B, dtype=float)
        assert self.A.shape == self.B.shape, 'Payoff matrices must have same shape (m,n)'
        self.m, self.n = self.A.shape
        self.game_name = game_name
        
        if strategy_labels is None:
            self.labels = ([f'A{i}' for i in range(self.m)], [f'B{j}' for j in range(self.n)])
        else:
            self.labels = strategy_labels

    def pure_nash_equilibria(self) -> List[Tuple[int, int]]:
        # Find all pure strategy Nash equilibria
        equilibria = []
        for i in range(self.m):
            for j in range(self.n):
                pi = self.A[:, j]
                pj = self.B[i, :]
                if self.A[i,j] >= pi.max() - 1e-9 and self.B[i,j] >= pj.max() - 1e-9:
                    equilibria.append((i,j))
        return equilibria

    def expected_payoffs(self, sigma0, sigma1):
        # Calculate expected payoffs for mixed strategies
        sigma0 = np.array(sigma0, dtype=float)
        sigma1 = np.array(sigma1, dtype=float)
        u0 = float(sigma0 @ self.A @ sigma1)
        u1 = float(sigma0 @ self.B @ sigma1)
        return u0, u1

    def support_enumeration_mixed_nash(self):
        # Find mixed strategy Nash equilibria using support enumeration
        equilibria = []
        supports0 = [set(s) for k in range(1, self.m+1) for s in combinations(range(self.m), k)]
        supports1 = [set(s) for k in range(1, self.n+1) for s in combinations(range(self.n), k)]
        
        for S0 in supports0:
            for S1 in supports1:
                sigma0, sigma1 = self._solve_support_pair(S0, S1)
                if sigma0 is None or sigma1 is None:
                    continue
                    
                payoffs0 = (np.array(sigma1) @ self.A.T)
                payoffs1 = (np.array(sigma0) @ self.B)
                
                if not all(abs(payoffs0[i] - payoffs0[list(S0)[0]]) <= 1e-5 for i in S0):
                    continue
                if not all(abs(payoffs1[j] - payoffs1[list(S1)[0]]) <= 1e-5 for j in S1):
                    continue
                if any(payoffs0[k] > payoffs0[list(S0)[0]] + 1e-6 for k in set(range(self.m)) - S0):
                    continue
                if any(payoffs1[k] > payoffs1[list(S1)[0]] + 1e-6 for k in set(range(self.n)) - S1):
                    continue
                    
                def close_to_existing(s, list_of_s):
                    for t in list_of_s:
                        if np.allclose(s, t, atol=1e-6):
                            return True
                    return False
                    
                if close_to_existing(sigma0, [e[0] for e in equilibria]) and close_to_existing(sigma1, [e[1] for e in equilibria]):
                    continue
                    
                equilibria.append((np.array(sigma0), np.array(sigma1)))
        return equilibria

    def _solve_support_pair(self, S0, S1):
        # Helper to solve linear equations for mixed strategies
        S0 = sorted(list(S0))
        S1 = sorted(list(S1))
        k0, k1 = len(S0), len(S1)
        
        if k0 == 0 or k1 == 0:
            return None, None
            
        try:
            # Solve for Player 0
            M0 = []
            b0 = []
            for j in S1[:-1]:
                row = [(self.B[i, j] - self.B[i, S1[-1]]) for i in S0]
                M0.append(row)
                b0.append(0.0)
            M0.append([1.0]*k0)
            b0.append(1.0)
            
            M0 = np.array(M0, dtype=float)
            b0 = np.array(b0, dtype=float)
            sol0, *_ = np.linalg.lstsq(M0, b0, rcond=None)
            
            sigma0 = np.zeros(self.m)
            for idx, i in enumerate(S0):
                sigma0[i] = sol0[idx]
            
            # Solve for Player 1
            M1 = []
            b1 = []
            for i in S0[:-1]:
                row = [(self.A[i, j] - self.A[S0[-1], j]) for j in S1]
                M1.append(row)
                b1.append(0.0)
            M1.append([1.0]*k1)
            b1.append(1.0)
            
            M1 = np.array(M1, dtype=float)
            b1 = np.array(b1, dtype=float)
            sol1, *_ = np.linalg.lstsq(M1, b1, rcond=None)
            
            sigma1 = np.zeros(self.n)
            for idx, j in enumerate(S1):
                sigma1[j] = sol1[idx]
            
            if (sigma0 < -1e-7).any() or (sigma1 < -1e-7).any():
                return None, None
                
            sigma0 = np.clip(sigma0, 0.0, None)
            sigma1 = np.clip(sigma1, 0.0, None)
            
            if sigma0.sum() <= 0 or sigma1.sum() <= 0:
                return None, None
                
            sigma0 = sigma0 / sigma0.sum()
            sigma1 = sigma1 / sigma1.sum()
            
            return sigma0, sigma1
        except Exception:
            return None, None

    def strictly_dominated_pure(self, player: int) -> List[int]:
        # Find strictly dominated pure strategies
        dominated = []
        if player == 0:
            for i in range(self.m):
                for i2 in range(self.m):
                    if i == i2: continue
                    if all(self.A[i2, j] > self.A[i, j] + 1e-9 for j in range(self.n)):
                        dominated.append(i)
                        break
        else:
            for j in range(self.n):
                for j2 in range(self.n):
                    if j == j2: continue
                    if all(self.B[i, j2] > self.B[i, j] + 1e-9 for i in range(self.m)):
                        dominated.append(j)
                        break
        return dominated

    def weakly_dominated_pure(self, player: int) -> List[int]:
        # Find weakly dominated pure strategies
        dominated = []
        if player == 0:
            for i in range(self.m):
                for i2 in range(self.m):
                    if i == i2: continue
                    if (all(self.A[i2, j] >= self.A[i, j] - 1e-9 for j in range(self.n)) and
                        any(self.A[i2, j] > self.A[i, j] + 1e-9 for j in range(self.n))):
                        dominated.append(i)
                        break
        else:
            for j in range(self.n):
                for j2 in range(self.n):
                    if j == j2: continue
                    if (all(self.B[i, j2] >= self.B[i, j] - 1e-9 for i in range(self.m)) and
                        any(self.B[i, j2] > self.B[i, j] + 1e-9 for i in range(self.m))):
                        dominated.append(j)
                        break
        return dominated

    def iterative_elimination_strictly_dominated(self):
        # Iteratively eliminate strictly dominated strategies
        remaining_rows = list(range(self.m))
        remaining_cols = list(range(self.n))
        changed = True
        
        while changed:
            changed = False
            for i in remaining_rows.copy():
                for i2 in remaining_rows:
                    if i==i2: continue
                    if all(self.A[i2, j] > self.A[i, j] + 1e-9 for j in remaining_cols):
                        remaining_rows.remove(i)
                        changed = True
                        break
                if changed: break
                
            if changed: continue
            
            for j in remaining_cols.copy():
                for j2 in remaining_cols:
                    if j==j2: continue
                    if all(self.B[i, j2] > self.B[i, j] + 1e-9 for i in remaining_rows):
                        remaining_cols.remove(j)
                        changed = True
                        break
                        
        A_red = self.A[np.ix_(remaining_rows, remaining_cols)]
        B_red = self.B[np.ix_(remaining_rows, remaining_cols)]
        return (remaining_rows, remaining_cols), NormalFormGame(A_red, B_red)

    def best_response(self, player: int, opponent_strategy):
        # Find best response strategies for a player
        opponent_strategy = np.array(opponent_strategy, dtype=float)
        if player == 0:
            expected = opponent_strategy @ self.A.T
            max_val = expected.max()
            return [i for i in range(self.m) if expected[i] >= max_val - 1e-9]
        else:
            expected = self.B @ opponent_strategy
            max_val = expected.max()
            return [j for j in range(self.n) if expected[j] >= max_val - 1e-9]

    def rationalizable_strategies(self):
        # Find rationalizable strategies 
        (rows, cols), _ = self.iterative_elimination_strictly_dominated()
        return rows, cols
    
    def to_extensive_form(self):
        # Convert Normal Form to Extensive Form

        tree = {'root': {'player': 0, 'actions': {}}}
        
        # Player 0 moves first (Root)
        for i, strat0 in enumerate(self.labels[0]):
            node_p0 = f"P0_{strat0}"
            tree['root']['actions'][strat0] = node_p0
            
            # Create node for Player 1
            tree[node_p0] = {'player': 1, 'actions': {}}
            
            # Player 1 moves second
            for j, strat1 in enumerate(self.labels[1]):
                node_terminal = f"End_{strat0}_{strat1}"
                tree[node_p0]['actions'][strat1] = node_terminal
                
                # Terminal node with payoffs
                tree[node_terminal] = {'payoff': (self.A[i, j], self.B[i, j])}
        
        return ExtensiveFormGame(tree, ['Player 0', 'Player 1'], f"{self.game_name} (Extensive)")

    def pretty_print(self):
        # Print the game in a formatted table
        print(f"\n{'='*60}")
        print(f"Game: {self.game_name}")
        print(f"{'='*60}")
        headers = ['Player 0 \\ Player 1'] + self.labels[1]
        table = []
        for i in range(self.m):
            row = [self.labels[0][i]]
            for j in range(self.n):
                row.append(f'({self.A[i,j]:.2f}, {self.B[i,j]:.2f})')
            table.append(row)
        print(tabulate(table, headers=headers, tablefmt='grid'))

    def comprehensive_analysis(self):
        # Perform and display comprehensive game analysis (Nash, Dominance, etc
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE ANALYSIS: {self.game_name}")
        print(f"{'='*60}\n")
        
        # 1. Pure Nash Equilibria
        print("1. PURE STRATEGY NASH EQUILIBRIA:")
        pure_ne = self.pure_nash_equilibria()
        if pure_ne:
            for i, j in pure_ne:
                print(f"   ({self.labels[0][i]}, {self.labels[1][j]}) with payoffs ({self.A[i,j]:.2f}, {self.B[i,j]:.2f})")
        else:
            print("   No pure strategy Nash equilibria found.")
        
        # 2. Mixed Nash Equilibria
        print("\n2. MIXED STRATEGY NASH EQUILIBRIA:")
        mixed_ne = self.support_enumeration_mixed_nash()
        if mixed_ne:
            for idx, (sig0, sig1) in enumerate(mixed_ne, 1):
                print(f"   Equilibrium {idx}:")
                print(f"      Player 0: {dict(zip(self.labels[0], np.round(sig0, 4)))}")
                print(f"      Player 1: {dict(zip(self.labels[1], np.round(sig1, 4)))}")
                u0, u1 = self.expected_payoffs(sig0, sig1)
                print(f"      Expected payoffs: ({u0:.4f}, {u1:.4f})")
        else:
            print("   No additional mixed strategy Nash equilibria found (beyond pure).")
        
        # 3. Dominated Strategies
        print("\n3. DOMINATED STRATEGIES:")
        dom0_strict = self.strictly_dominated_pure(0)
        dom1_strict = self.strictly_dominated_pure(1)
        dom0_weak = self.weakly_dominated_pure(0)
        dom1_weak = self.weakly_dominated_pure(1)
        
        if dom0_strict:
            print(f"   Player 0 strictly dominated: {[self.labels[0][i] for i in dom0_strict]}")
        if dom1_strict:
            print(f"   Player 1 strictly dominated: {[self.labels[1][j] for j in dom1_strict]}")
        if dom0_weak:
            print(f"   Player 0 weakly dominated: {[self.labels[0][i] for i in dom0_weak]}")
        if dom1_weak:
            print(f"   Player 1 weakly dominated: {[self.labels[1][j] for j in dom1_weak]}")
        if not any([dom0_strict, dom1_strict, dom0_weak, dom1_weak]):
            print("   No dominated strategies found.")
        
        # 4. Rationalizable Strategies
        print("\n4. RATIONALIZABLE STRATEGIES:")
        rat0, rat1 = self.rationalizable_strategies()
        print(f"   Player 0: {[self.labels[0][i] for i in rat0]}")
        print(f"   Player 1: {[self.labels[1][j] for j in rat1]}")
        
        print(f"\n{'='*60}\n")

    def plot_payoff_heatmap(self, filename='payoff_heatmaps.png'):
        # Generate heatmap visualization of payoff matrices

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        cax1 = ax1.matshow(self.A, cmap='RdYlGn', alpha=0.7)
        ax1.set_title(f'Player 0 Payoff Matrix\n{self.game_name}', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(self.n))
        ax1.set_yticks(range(self.m))
        ax1.set_xticklabels(self.labels[1])
        ax1.set_yticklabels(self.labels[0])
        ax1.set_xlabel('Player 1 Strategy')
        ax1.set_ylabel('Player 0 Strategy')
        
        # Add values to cells
        for i in range(self.m):
            for j in range(self.n):
                ax1.text(j, i, f'{self.A[i,j]:.2f}', ha='center', va='center', fontweight='bold')
        
        fig.colorbar(cax1, ax=ax1)
        
        cax2 = ax2.matshow(self.B, cmap='RdYlGn', alpha=0.7)
        ax2.set_title(f'Player 1 Payoff Matrix\n{self.game_name}', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(self.n))
        ax2.set_yticks(range(self.m))
        ax2.set_xticklabels(self.labels[1])
        ax2.set_yticklabels(self.labels[0])
        ax2.set_xlabel('Player 1 Strategy')
        ax2.set_ylabel('Player 0 Strategy')
        
        # Add values to cells
        for i in range(self.m):
            for j in range(self.n):
                ax2.text(j, i, f'{self.B[i,j]:.2f}', ha='center', va='center', fontweight='bold')
        
        fig.colorbar(cax2, ax=ax2)
        
        plt.tight_layout()
        plt.savefig(f"Output/{filename}", dpi=300, bbox_inches='tight')
        print(f"✓ Payoff heatmaps saved as {filename}")
        plt.close()


class ExtensiveFormGame:
    # Extensive Form Game with Backward Induction and Normal Form Conversion
    
    def __init__(self, tree: Dict, players: List[str], game_name: str = "Unnamed Extensive Game"):
        self.tree = tree
        self.players = players
        self.game_name = game_name
        self._validate_tree()

    def _validate_tree(self):
        # Validate the game tree structure
        if 'root' not in self.tree:
            raise ValueError("Tree must have a 'root' node")
        
        visited = set()
        def visit(node):
            if node in visited:
                raise ValueError(f"Cycle detected at node {node}")
            visited.add(node)
            if node not in self.tree:
                raise ValueError(f"Node {node} referenced but not defined")
            if 'payoff' not in self.tree[node] and 'actions' not in self.tree[node]:
                raise ValueError(f"Node {node} must have either 'payoff' or 'actions'")
            if 'actions' in self.tree[node]:
                for action, next_node in self.tree[node]['actions'].items():
                    visit(next_node)
        
        visit('root')

    def backward_induction(self, node='root'):
        # Perform backward induction to find subgame perfect Nash equilibrium

        if 'payoff' in self.tree[node]:
            return self.tree[node]['payoff'], {}
        
        player = self.tree[node]['player']
        actions = self.tree[node]['actions']
        
        # Recursively solve subgames
        action_values = {}
        strategies = {}
        
        for action, next_node in actions.items():
            value, substrat = self.backward_induction(next_node)
            action_values[action] = value
            strategies[action] = substrat
        
        # Find best action for current player
        if player == 0:
            best_action = max(action_values.keys(), key=lambda a: action_values[a][0])
            best_value = action_values[best_action]
        else:
            best_action = max(action_values.keys(), key=lambda a: action_values[a][1])
            best_value = action_values[best_action]
        
        # Build strategy profile
        equilibrium_strategy = {node: best_action}
        equilibrium_strategy.update(strategies[best_action])
        
        return best_value, equilibrium_strategy

    def find_spne(self):
        # Find Subgame Perfect Nash Equilibrium
        value, strategy = self.backward_induction()
        return strategy, value

    def to_normal_form(self) -> NormalFormGame:
        # Convert the Extensive Form to a Normal Form
          
        # 1. Identify all decision nodes for each player
        nodes_by_player = {0: [], 1: []}
        
        def find_decision_nodes(node_id):
            node_data = self.tree[node_id]
            if 'payoff' in node_data:
                return
            
            player = node_data['player']
            nodes_by_player[player].append(node_id)
            
            for _, next_node_id in node_data['actions'].items():
                find_decision_nodes(next_node_id)
                
        find_decision_nodes('root')
        
        # 2. Generate all pure strategies
        # A strategy is a combination of actions: one action for each decision node the player controls
        strategies = {0: [], 1: []}
        
        for p in [0, 1]:
            nodes = nodes_by_player[p]
            if not nodes:
                # If a player has no moves, they have 1 dummy strategy
                strategies[p] = [("NoMove",)] 
            else:
                lists_of_actions = []
                for node in nodes:
                    lists_of_actions.append(list(self.tree[node]['actions'].keys()))
                strategies[p] = list(product(*lists_of_actions))

        # 3. Create Payoff Matrices
        m = len(strategies[0])
        n = len(strategies[1])
        A = np.zeros((m, n))
        B = np.zeros((m, n))
        
        # Helper to traverse tree given a profile
        def get_payoff(p0_strat_idx, p1_strat_idx):
            p0_moves = dict(zip(nodes_by_player[0], strategies[0][p0_strat_idx]))
            p1_moves = dict(zip(nodes_by_player[1], strategies[1][p1_strat_idx]))
            all_moves = {**p0_moves, **p1_moves}
            
            current_node = 'root'
            while 'payoff' not in self.tree[current_node]:
                node_data = self.tree[current_node]
                # If player has no moves (dummy), this logic naturally skips them if they don't own the node
                # But here we just look up the node owner
                action = all_moves[current_node]
                current_node = node_data['actions'][action]
            
            return self.tree[current_node]['payoff']

        # Fill matrices
        for i in range(m):
            for j in range(n):
                payoff = get_payoff(i, j)
                A[i, j] = payoff[0]
                B[i, j] = payoff[1]

        # 4. Generate Labels
        # Strategy name format: "Action1_Action2" representing actions at subsequent nodes
        labels0 = ["-".join(s) for s in strategies[0]]
        labels1 = ["-".join(s) for s in strategies[1]]

        return NormalFormGame(A, B, (labels0, labels1), f"{self.game_name} (Normal Form)")

    def pretty_print(self):
        # Print the extensive form game structure
        
        print(f"\n{'='*60}")
        print(f"Extensive Form Game: {self.game_name}")
        print(f"{'='*60}")
        print("Game Tree Structure:")
        
        def print_node(node, indent=0):
            prefix = "  " * indent
            if 'payoff' in self.tree[node]:
                print(f"{prefix}└─ {node}: Terminal Node, Payoff {self.tree[node]['payoff']}")
            else:
                player = self.tree[node]['player']
                print(f"{prefix}├─ {node}: {self.players[player]}'s turn")
                actions = self.tree[node]['actions']
                for i, (action, next_node) in enumerate(actions.items()):
                    is_last = (i == len(actions) - 1)
                    connector = "└─" if is_last else "├─"
                    print(f"{prefix}│  {connector} Action '{action}':")
                    print_node(next_node, indent + 2)
        
        print_node('root')

    def comprehensive_analysis(self):
        # Perform comprehensive analysis of the extensive form game

        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE ANALYSIS: {self.game_name}")
        print(f"{'='*60}\n")
        
        print("1. GAME TREE STRUCTURE:")
        self.pretty_print()
        
        print("\n2. SUBGAME PERFECT NASH EQUILIBRIUM (Backward Induction):")
        strategy, value = self.find_spne()
        print(f"   Equilibrium Payoff: {value}")
        print(f"   Equilibrium Strategy Profile:")
        for node, action in strategy.items():
            if node in self.tree and 'player' in self.tree[node]:
                player = self.tree[node]['player']
                print(f"      At node '{node}': {self.players[player]} plays '{action}'")
        
        print(f"\n{'='*60}\n")

    def plot_tree(self, filename='game_tree.png'):
        # Generate visual representation of the game tree
        
        G = nx.DiGraph()
        pos = {}
        node_labels = {}
        edge_labels = {}
        node_colors = []
        
        def add_nodes(node, x, y, level, width=4):
            pos[node] = (x, y)
            
            if 'payoff' in self.tree[node]:
                node_labels[node] = f"{node}\n{self.tree[node]['payoff']}"
                node_colors.append('lightcoral')
            else:
                player = self.tree[node]['player']
                node_labels[node] = f"{node}\n{self.players[player]}"
                node_colors.append('lightblue' if player == 0 else 'lightgreen')
            
            if 'actions' in self.tree[node]:
                actions = list(self.tree[node]['actions'].items())
                num_actions = len(actions)
                new_width = width / (level + 1)
                
                for i, (act, next_node) in enumerate(actions):
                    offset = (i - (num_actions - 1) / 2) * new_width
                    G.add_edge(node, next_node)
                    edge_labels[(node, next_node)] = act
                    add_nodes(next_node, x + offset, y - 1.5, level + 1, new_width)
        
        add_nodes('root', 0, 0, 1)
        
        plt.figure(figsize=(16, 10))
        nx.draw(G, pos, with_labels=True, labels=node_labels, 
                node_color=node_colors, node_size=3500, 
                font_size=9, font_weight='bold', arrows=True, 
                edge_color='gray', width=2, arrowsize=20)
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10, 
                                     font_weight='bold', font_color='red')
        
        plt.title(f'Extensive Form Game Tree: {self.game_name}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"Output/{filename}", dpi=300, bbox_inches='tight')
        print(f"✓ Game tree saved as {filename}")
        plt.close()
from game_theory_core import NormalFormGame, ExtensiveFormGame
import numpy as np
import os
import sys

class GameTheorySimulator:
    # Main interactive simulator class
    
    def __init__(self):
        self.current_game = None
        self.game_library = self._initialize_game_library()
        
    def _initialize_game_library(self):
        # Initialize library of predefined games

        return {
            '1': ('Prisoner\'s Dilemma', self.make_prisoners_dilemma),
            '2': ('Battle of the Sexes', self.make_battle_of_sexes),
            '3': ('Matching Pennies', self.make_matching_pennies),
            '4': ('Hawk-Dove Game', self.make_hawk_dove),
            '5': ('Prisoner\'s Dilemma (Extensive)', self.make_prisoners_dilemma_extensive),
            '6': ('Sequential Entry Game (Extensive)', self.make_entry_game),
        }
    
    # Normal Form Game Definitions

    def make_prisoners_dilemma(self):
        # Prisoner's Dilemma
        A = np.array([[-1, -4], [0, -3]])
        B = np.array([[-1, 0], [-4, -3]])
        labels = (['Cooperate', 'Defect'], ['Cooperate', 'Defect'])
        return NormalFormGame(A, B, labels, "Prisoner's Dilemma")
    
    def make_battle_of_sexes(self):
        # Battle of the Sexes coordination game

        A = np.array([[2, 0], [0, 1]])
        B = np.array([[1, 0], [0, 2]])
        labels = (['Opera', 'Football'], ['Opera', 'Football'])
        return NormalFormGame(A, B, labels, "Battle of the Sexes")
    
    def make_matching_pennies(self):
        # Zero-sum Matching Pennies game

        A = np.array([[1, -1], [-1, 1]])
        B = -A
        labels = (['Heads', 'Tails'], ['Heads', 'Tails'])
        return NormalFormGame(A, B, labels, "Matching Pennies")
    
    def make_hawk_dove(self):
        # Hawk-Dove game

        A = np.array([[-2, 3], [0, 1]])
        B = np.array([[-2, 0], [3, 1]])
        labels = (['Hawk', 'Dove'], ['Hawk', 'Dove'])
        return NormalFormGame(A, B, labels, "Hawk-Dove Game")
    
    # Extensive Form Game Definitions

    def make_prisoners_dilemma_extensive(self):
        # Prisoner's Dilemma in extensive form

        tree = {
            'root': {'player': 0, 'actions': {'C': 'p0_C', 'D': 'p0_D'}},
            'p0_C': {'player': 1, 'actions': {'C': 'CC', 'D': 'CD'}},
            'p0_D': {'player': 1, 'actions': {'C': 'DC', 'D': 'DD'}},
            'CC': {'payoff': (-1, -1)},
            'CD': {'payoff': (-4, 0)},
            'DC': {'payoff': (0, -4)},
            'DD': {'payoff': (-3, -3)},
        }
        return ExtensiveFormGame(tree, ['Player 0', 'Player 1'], "Prisoner's Dilemma (Extensive)")
    
    def make_entry_game(self):
        # Sequential market entry game

        tree = {
            'root': {'player': 0, 'actions': {'Enter': 'entrant_enter', 'Stay_Out': 'stay_out'}},
            'entrant_enter': {'player': 1, 'actions': {'Fight': 'fight', 'Accommodate': 'accommodate'}},
            'stay_out': {'payoff': (0, 2)},
            'fight': {'payoff': (-1, -1)},
            'accommodate': {'payoff': (1, 1)},
        }
        return ExtensiveFormGame(tree, ['Entrant', 'Incumbent'], "Sequential Entry Game")
    
    def create_custom_normal_form(self):
        # Interactive custom game creation

        print("\n" + "="*60)
        print("CREATE CUSTOM NORMAL FORM GAME")
        print("="*60)
        
        try:
            # Get game dimensions
            m = int(input("\nNumber of strategies for Player 0: "))
            n = int(input("Number of strategies for Player 1: "))
            
            if m < 1 or n < 1 or m > 10 or n > 10:
                print("❌ Invalid dimensions. Must be between 1 and 10.")
                return None
            
            # Get strategy labels
            print("\nEnter strategy names for Player 0:")
            labels_0 = [input(f"  Strategy {i+1}: ").strip() or f"A{i}" for i in range(m)]
            
            print("\nEnter strategy names for Player 1:")
            labels_1 = [input(f"  Strategy {i+1}: ").strip() or f"B{i}" for i in range(n)]
            
            # Get payoffs
            print("\nEnter payoffs (Player0_payoff, Player1_payoff) for each strategy profile:")
            A = np.zeros((m, n))
            B = np.zeros((m, n))
            
            for i in range(m):
                for j in range(n):
                    while True:
                        try:
                            payoff_str = input(f"  ({labels_0[i]}, {labels_1[j]}): ")
                            payoffs = [float(x.strip()) for x in payoff_str.split(',')]
                            if len(payoffs) != 2:
                                raise ValueError
                            A[i, j], B[i, j] = payoffs
                            break
                        except:
                            print("    ❌ Invalid format. Enter as: payoff0, payoff1")
            
            game_name = input("\nEnter a name for this game: ").strip() or "Custom Game"
            
            return NormalFormGame(A, B, (labels_0, labels_1), game_name)
            
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Game creation cancelled.")
            return None
    
    def analyze_game_interactive(self, game):
        # Interactive game analysis menu

        while True:
            print("\n" + "="*60)
            print(f"ANALYZING: {game.game_name}")
            print("="*60)
            print("\nAnalysis Options:")
            print("  1. Show Payoff Matrix")
            print("  2. Find Pure Nash Equilibria")
            print("  3. Find Mixed Nash Equilibria")
            print("  4. Check Dominated Strategies")
            print("  5. Find Rationalizable Strategies")
            print("  6. Calculate Best Responses")
            print("  7. Comprehensive Analysis (All Above)")
            print("  8. Simulate Strategy Profile")
            print("  9. Generate Visualizations")
            print("  10. Convert to Extensive Form & Analyze")
            print("  0. Back")
            
            choice = input("\nEnter your choice: ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                game.pretty_print()
            elif choice == '2':
                self._show_pure_nash(game)
            elif choice == '3':
                self._show_mixed_nash(game)
            elif choice == '4':
                self._show_dominated_strategies(game)
            elif choice == '5':
                self._show_rationalizable(game)
            elif choice == '6':
                self._show_best_responses(game)
            elif choice == '7':
                game.comprehensive_analysis()
            elif choice == '8':
                self._simulate_strategy_profile(game)
            elif choice == '9':
                self._generate_visualizations(game)
            elif choice == '10':
                ef_game = game.to_extensive_form()
                print(f"\n✓ Converted to Extensive Form: {ef_game.game_name}")
                self.analyze_extensive_game(ef_game)
            else:
                print("❌ Invalid choice. Please try again.")
    
    def _show_pure_nash(self, game):
        # Display pure strategy Nash equilibria

        print("\n" + "-"*60)
        print("PURE STRATEGY NASH EQUILIBRIA")
        print("-"*60)
        pure_ne = game.pure_nash_equilibria()
        if pure_ne:
            for i, j in pure_ne:
                print(f"  • ({game.labels[0][i]}, {game.labels[1][j]})")
                print(f"    Payoffs: ({game.A[i,j]:.2f}, {game.B[i,j]:.2f})")
        else:
            print("  No pure strategy Nash equilibria found.")
    
    def _show_mixed_nash(self, game):
        # Display mixed strategy Nash equilibria

        print("\n" + "-"*60)
        print("MIXED STRATEGY NASH EQUILIBRIA")
        print("-"*60)
        mixed_ne = game.support_enumeration_mixed_nash()
        if mixed_ne:
            for idx, (sig0, sig1) in enumerate(mixed_ne, 1):
                print(f"\n  Equilibrium {idx}:")
                print(f"    Player 0: {dict(zip(game.labels[0], np.round(sig0, 4)))}")
                print(f"    Player 1: {dict(zip(game.labels[1], np.round(sig1, 4)))}")
                u0, u1 = game.expected_payoffs(sig0, sig1)
                print(f"    Expected payoffs: ({u0:.4f}, {u1:.4f})")
        else:
            print("  No additional mixed Nash equilibria (beyond pure).")
    
    def _show_dominated_strategies(self, game):
        # Display dominated strategies

        print("\n" + "-"*60)
        print("DOMINATED STRATEGIES")
        print("-"*60)
        
        dom0_strict = game.strictly_dominated_pure(0)
        dom1_strict = game.strictly_dominated_pure(1)
        dom0_weak = game.weakly_dominated_pure(0)
        dom1_weak = game.weakly_dominated_pure(1)
        
        print("\nStrictly Dominated:")
        if dom0_strict:
            print(f"  Player 0: {[game.labels[0][i] for i in dom0_strict]}")
        if dom1_strict:
            print(f"  Player 1: {[game.labels[1][j] for j in dom1_strict]}")
        if not dom0_strict and not dom1_strict:
            print("  None")
        
        print("\nWeakly Dominated:")
        if dom0_weak:
            print(f"  Player 0: {[game.labels[0][i] for i in dom0_weak]}")
        if dom1_weak:
            print(f"  Player 1: {[game.labels[1][j] for j in dom1_weak]}")
        if not dom0_weak and not dom1_weak:
            print("  None")
    
    def _show_rationalizable(self, game):
        # Display rationalizable strategies

        print("\n" + "-"*60)
        print("RATIONALIZABLE STRATEGIES")
        print("-"*60)
        rat0, rat1 = game.rationalizable_strategies()
        print(f"  Player 0: {[game.labels[0][i] for i in rat0]}")
        print(f"  Player 1: {[game.labels[1][j] for j in rat1]}")
    
    def _show_best_responses(self, game):
        # Display best responses for pure strategies

        print("\n" + "-"*60)
        print("BEST RESPONSES TO PURE STRATEGIES")
        print("-"*60)
        
        print("\nPlayer 0's Best Responses:")
        for j in range(game.n):
            opp_strat = [1 if k == j else 0 for k in range(game.n)]
            br = game.best_response(0, opp_strat)
            print(f"  vs {game.labels[1][j]}: {[game.labels[0][i] for i in br]}")
        
        print("\nPlayer 1's Best Responses:")
        for i in range(game.m):
            opp_strat = [1 if k == i else 0 for k in range(game.m)]
            br = game.best_response(1, opp_strat)
            print(f"  vs {game.labels[0][i]}: {[game.labels[1][j] for j in br]}")
    
    def _simulate_strategy_profile(self, game):
        # Simulate outcomes for user-specified strategies

        print("\n" + "-"*60)
        print("SIMULATE STRATEGY PROFILE")
        print("-"*60)
        
        try:
            print("\nEnter mixed strategy for Player 0")
            print(f"Strategies: {game.labels[0]}")
            print("Format: comma-separated probabilities (must sum to 1)")
            
            sig0_str = input("Player 0 strategy: ")
            sig0 = [float(x.strip()) for x in sig0_str.split(',')]
            
            if len(sig0) != game.m or abs(sum(sig0) - 1.0) > 1e-6:
                print("❌ Invalid strategy. Must have correct length and sum to 1.")
                return
            
            print("\nEnter mixed strategy for Player 1")
            print(f"Strategies: {game.labels[1]}")
            sig1_str = input("Player 1 strategy: ")
            sig1 = [float(x.strip()) for x in sig1_str.split(',')]
            
            if len(sig1) != game.n or abs(sum(sig1) - 1.0) > 1e-6:
                print("❌ Invalid strategy. Must have correct length and sum to 1.")
                return
            
            u0, u1 = game.expected_payoffs(sig0, sig1)
            print(f"\n✓ Expected Payoffs: Player 0 = {u0:.4f}, Player 1 = {u1:.4f}")
            
            # Check if it's a Nash equilibrium
            br0 = game.best_response(0, sig1)
            br1 = game.best_response(1, sig0)
            
            is_nash = True
            for i in range(game.m):
                if sig0[i] > 1e-6 and i not in br0:
                    is_nash = False
                    break
            if is_nash:
                for j in range(game.n):
                    if sig1[j] > 1e-6 and j not in br1:
                        is_nash = False
                        break
            
            if is_nash:
                print("✓ This is a Nash equilibrium!")
            else:
                print("✗ This is NOT a Nash equilibrium.")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def _generate_visualizations(self, game):
        # Generate visual outputs for the game

        print("\n" + "-"*60)
        print("GENERATE VISUALIZATIONS")
        print("-"*60)
        
        try:
            base_name = game.game_name.replace(" ", "_").replace("'", "")
            game.plot_payoff_heatmap(f'{base_name}_heatmap.png')
            print("✓ All visualizations generated successfully!")
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
    
    def analyze_extensive_game(self, game):
        # Analyze extensive form game

        while True:
            print("\n" + "="*60)
            print(f"ANALYZING: {game.game_name}")
            print("="*60)
            print("\nAnalysis Options:")
            print("  1. Show Game Tree Structure")
            print("  2. Find Subgame Perfect Nash Equilibrium")
            print("  3. Comprehensive Analysis")
            print("  4. Generate Tree Visualization")
            print("  5. Convert to Normal Form & Analyze")
            print("  0. Back")
            
            choice = input("\nEnter your choice: ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                game.pretty_print()
            elif choice == '2':
                strategy, value = game.find_spne()
                print("\nSubgame Perfect Nash Equilibrium:")
                print(f"  Equilibrium Payoff: {value}")
                print(f"  Strategy Profile:")
                for node, action in strategy.items():
                    if node in game.tree and 'player' in game.tree[node]:
                        player = game.tree[node]['player']
                        print(f"    At '{node}': {game.players[player]} plays '{action}'")
            elif choice == '3':
                game.comprehensive_analysis()
            elif choice == '4':
                try:
                    base_name = game.game_name.replace(" ", "_").replace("'", "")
                    game.plot_tree(f'{base_name}_tree.png')
                except Exception as e:
                    print(f"❌ Error: {e}")
            elif choice == '5':
                nf_game = game.to_normal_form()
                print(f"\n✓ Converted to Normal Form: {nf_game.game_name}")
                self.analyze_game_interactive(nf_game)
            else:
                print("❌ Invalid choice.")
    
    def run(self):
        # Main program loop

        print("\n" + "="*60)
        print("  GAME THEORY SIMULATOR - Strategic Decision Making")
        print("="*60)
        
        while True:
            print("\n" + "="*60)
            print("MAIN MENU")
            print("="*60)
            print("\nPredefined Games (Normal Form):")
            for key in ['1', '2', '3', '4']:
                name, _ = self.game_library[key]
                print(f"  {key}. {name}")
            
            print("\nPredefined Games (Extensive Form):")
            for key in ['5', '6']:
                name, _ = self.game_library[key]
                print(f"  {key}. {name}")
            
            print("\nCustom Games:")
            print("  C. Create Custom Normal Form Game")
            
            print("\nOther Options:")
            print("  H. Help & Game Theory Concepts")
            print("  Q. Quit")
            
            choice = input("\nEnter your choice: ").strip().upper()
            
            if choice == 'Q':
                print("\n✓ Thank you for using the Game Theory Simulator!")
                break
            elif choice == 'H':
                self.show_help()
            elif choice == 'C':
                custom_game = self.create_custom_normal_form()
                if custom_game:
                    self.analyze_game_interactive(custom_game)
            elif choice in self.game_library:
                name, constructor = self.game_library[choice]
                game = constructor()
                if isinstance(game, NormalFormGame):
                    self.analyze_game_interactive(game)
                else:
                    self.analyze_extensive_game(game)
            else:
                print("❌ Invalid choice. Please try again.")
    
    def show_help(self):
        # Display game theory concepts

        print("\n" + "="*60)
        print("GAME THEORY CONCEPTS")
        print("="*60)
        
        concepts = {
            "Nash Equilibrium": "A strategy profile where no player can improve "
                               "their payoff by unilaterally changing strategy.",
            
            "Dominant Strategy": "A strategy that is best regardless of what "
                                "other players do.",
            
            "Dominated Strategy": "A strategy that is always worse than another "
                                 "available strategy.",
            
            "Rationalizable Strategy": "A strategy that survives iterative elimination "
                                      "of strictly dominated strategies.",
            
            "Best Response": "The optimal strategy given the opponent's strategy.",
            
            "Mixed Strategy": "A probability distribution over pure strategies.",
            
            "Subgame Perfect Equilibrium": "A Nash equilibrium that represents optimal "
                                          "play in every subgame (extensive form).",
            
            "Backward Induction": "Solving extensive form games by reasoning backwards "
                                 "from terminal nodes."
        }
        
        for concept, description in concepts.items():
            print(f"\n{concept}:")
            print(f"  {description}")
        
        print("\n" + "="*60)


if __name__ == '__main__':
    simulator = GameTheorySimulator()
    simulator.run()
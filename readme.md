# Game Theory Simulator - Strategic Decision Making

## 📋 Project Overview

This is a comprehensive Python-based Game Theory simulator designed for exploring strategic decision-making scenarios. The system implements fundamental game theory concepts and provides an interactive interface for analyzing both Normal Form and Extensive Form games.

## 🎯 Project Objectives

The simulator implements all major Game Theory topics:

- ✅ **Multiple Game Types**: Support for predefined games (Prisoner's Dilemma, Battle of the Sexes, etc.) and custom games
- ✅ **Game Representations**: Both Extensive Form (game trees) and Normal Form (payoff matrices)
- ✅ **Strategy Analysis**: Pure strategies, mixed strategies, and expected payoffs
- ✅ **Equilibrium Concepts**: Nash Equilibria, Subgame Perfect Nash Equilibria
- ✅ **Strategic Dominance**: Dominated strategies, best responses, rationalizable strategies
- ✅ **Visualizations**: Tables, heatmaps, game trees, and charts
- ✅ **Interactive Interface**: User-friendly CLI for exploration and learning

## 🚀 Features

### Core Game Theory Concepts

1. **Nash Equilibrium Analysis**
   - Pure strategy Nash equilibria
   - Mixed strategy Nash equilibria (support enumeration algorithm)
   - Expected payoff calculations

2. **Strategic Dominance**
   - Strictly dominated strategies
   - Weakly dominated strategies
   - Iterative elimination of dominated strategies

3. **Best Response Analysis**
   - Best response correspondences
   - Best response dynamics visualization

4. **Rationalizability**
   - Identification of rationalizable strategy sets
   - Iterative elimination procedures

5. **Welfare Analysis**
   - Pareto efficiency analysis
   - Social welfare optimization
   - Welfare comparisons

6. **Extensive Form Games**
   - Game tree representation
   - Backward induction algorithm
   - Subgame Perfect Nash Equilibrium

### Predefined Games

**Normal Form Games:**
1. **Prisoner's Dilemma** - Classic cooperation dilemma
2. **Battle of the Sexes** - Coordination with conflicting preferences
3. **Matching Pennies** - Zero-sum competitive game
4. **Hawk-Dove Game** - Conflict escalation model
5. **Coordination Game** - Pure coordination scenario
6. **Stag Hunt** - Risk-dominant vs payoff-dominant equilibria
7. **Rock-Paper-Scissors** - Cyclic zero-sum game

**Extensive Form Games:**
8. **Prisoner's Dilemma (Sequential)** - Sequential move version
9. **Sequential Entry Game** - Market entry with strategic commitment

## 📦 Installation

### Requirements

```bash
numpy>=1.21.0
matplotlib>=3.4.0
networkx>=2.6.0
tabulate>=0.8.9
```

### Setup

```bash
# Install required packages
pip install numpy matplotlib networkx tabulate

# Download the project files
# game_theory_core.py - Core game theory classes and algorithms
# interactive_interface.py - Interactive CLI interface
```

## 🎮 Usage

### Running the Simulator

```bash
python interactive_interface.py
```

### Interactive Menu Options

The simulator provides an intuitive menu system:

1. **Select a Predefined Game**: Choose from 9 classic game theory scenarios
2. **Create Custom Game**: Design your own game with custom strategies and payoffs
3. **Analyze Game**: Comprehensive analysis including:
   - Payoff matrix visualization
   - Nash equilibria (pure and mixed)
   - Dominated strategies
   - Rationalizable strategies
   - Best response analysis
   - Pareto efficiency
   - Strategy simulation
4. **Generate Visualizations**: Create heatmaps and game trees
5. **Access Help**: Learn game theory concepts

### Example: Analyzing Prisoner's Dilemma

```
1. Select option "1" for Prisoner's Dilemma
2. Choose "8" for Comprehensive Analysis
3. Review:
   - Pure Nash Equilibrium: (Defect, Defect)
   - Dominated strategies: Cooperate is dominated for both players
   - Pareto efficiency: (Cooperate, Cooperate) is Pareto efficient but not Nash
   - Social optimum vs Nash equilibrium comparison
```

### Example: Creating a Custom Game

```python
# Via interactive menu:
1. Select "C" for Custom Game
2. Enter number of strategies for each player
3. Name strategies (e.g., "High", "Low")
4. Input payoffs for each strategy profile
5. Analyze the custom game
```

### Example: Programmatic Usage

```python
from game_theory_core import NormalFormGame
import numpy as np

# Create a simple coordination game
A = np.array([[3, 0], [0, 3]])
B = np.array([[3, 0], [0, 3]])
labels = (['Strategy A', 'Strategy B'], ['Strategy A', 'Strategy B'])
game = NormalFormGame(A, B, labels, "Coordination Game")

# Run comprehensive analysis
game.comprehensive_analysis()

# Find specific equilibria
pure_nash = game.pure_nash_equilibria()
mixed_nash = game.support_enumeration_mixed_nash()

# Check for dominated strategies
dominated_p0 = game.strictly_dominated_pure(0)
dominated_p1 = game.strictly_dominated_pure(1)

# Generate visualizations
game.plot_payoff_heatmap('coordination_heatmap.png')
```

## 📊 Analysis Capabilities

### Normal Form Game Analysis

```python
# Available analysis methods
game.pure_nash_equilibria()              # Find pure Nash equilibria
game.support_enumeration_mixed_nash()    # Find mixed Nash equilibria
game.strictly_dominated_pure(player)     # Find strictly dominated strategies
game.weakly_dominated_pure(player)       # Find weakly dominated strategies
game.rationalizable_strategies()         # Find rationalizable strategies
game.best_response(player, strategy)     # Compute best responses
game.pareto_efficient_outcomes()         # Find Pareto efficient outcomes
game.social_optimum()                    # Find social welfare maximizer
game.comprehensive_analysis()            # Full analysis report
```

### Extensive Form Game Analysis

```python
# Available analysis methods
ext_game.backward_induction()     # Solve via backward induction
ext_game.find_spne()              # Find subgame perfect Nash equilibrium
ext_game.comprehensive_analysis() # Full analysis including strategy profile
ext_game.plot_tree(filename)      # Generate game tree visualization
```

## 🎓 Educational Features

### Built-in Concept Explanations

The simulator includes educational content explaining:
- Nash Equilibrium
- Dominant and Dominated Strategies
- Rationalizability
- Best Response
- Pareto Efficiency
- Mixed Strategies
- Subgame Perfect Equilibrium
- Backward Induction

Access via the "H" (Help) option in the main menu.

### Learning by Example

Each predefined game demonstrates different strategic scenarios:
- **Prisoner's Dilemma**: Individual rationality vs collective benefit
- **Battle of the Sexes**: Multiple equilibria and coordination
- **Matching Pennies**: No pure Nash equilibrium, mixed strategies essential
- **Hawk-Dove**: Asymmetric equilibria
- **Stag Hunt**: Risk dominance vs payoff dominance

## 📈 Visualization Outputs

The simulator generates professional visualizations:

1. **Payoff Heatmaps**: Color-coded matrices showing payoff distributions
2. **Game Trees**: Visual representation of extensive form games with:
   - Node labels (player and decision points)
   - Edge labels (actions)
   - Terminal payoffs
   - Color-coded by player
3. **Payoff Tables**: Formatted matrices with grid layout

## 🔬 Advanced Features

### Support Enumeration Algorithm

Implements the support enumeration method for finding mixed strategy Nash equilibria:
1. Enumerate all possible support combinations
2. Solve indifference equations for each support pair
3. Verify equilibrium conditions (best response, non-negative probabilities)
4. Remove duplicate solutions

### Backward Induction

For extensive form games:
1. Recursively traverse from terminal nodes
2. Calculate optimal actions at each decision node
3. Construct subgame perfect Nash equilibrium strategy profile
4. Return equilibrium path and payoffs

### Iterative Elimination

Implements iterative elimination of strictly dominated strategies:
1. Identify dominated strategies for all players
2. Remove dominated strategies
3. Repeat until no more eliminations possible
4. Return reduced game

## 🛠️ Implementation Details

### Class Structure

**NormalFormGame Class**
- Attributes: payoff matrices (A, B), strategy labels, dimensions
- Methods: equilibrium finding, dominance checking, visualization

**ExtensiveFormGame Class**
- Attributes: tree structure, player list
- Methods: backward induction, tree visualization, SPNE computation

### Algorithm Complexity

- **Pure Nash Equilibria**: O(m × n) where m, n are strategy counts
- **Mixed Nash Equilibria**: O(2^(m+n)) in worst case (exponential in support enumeration)
- **Dominated Strategies**: O(m² × n) or O(n² × m) 
- **Backward Induction**: O(|V| + |E|) where V = nodes, E = edges

## 📝 Example Output

```
============================================================
COMPREHENSIVE ANALYSIS: Prisoner's Dilemma
============================================================

1. PURE STRATEGY NASH EQUILIBRIA:
   (Defect, Defect) with payoffs (-3.00, -3.00)

2. MIXED STRATEGY NASH EQUILIBRIA:
   No additional mixed strategy Nash equilibria found (beyond pure).

3. DOMINATED STRATEGIES:
   Player 0 strictly dominated: ['Cooperate']
   Player 1 strictly dominated: ['Cooperate']

4. RATIONALIZABLE STRATEGIES:
   Player 0: ['Defect']
   Player 1: ['Defect']

5. PARETO EFFICIENT OUTCOMES:
   (Cooperate, Cooperate) with payoffs (-1.00, -1.00)
   (Cooperate, Defect) with payoffs (-4.00, 0.00)
   (Defect, Cooperate) with payoffs (0.00, -4.00)

6. SOCIAL OPTIMUM (Maximum Total Welfare):
   (Cooperate, Cooperate) with total welfare -2.00

============================================================
```

## 🤝 Contributing

To extend the simulator:

1. **Add New Games**: Create new game constructors in `interactive_interface.py`
2. **Implement New Algorithms**: Add methods to `NormalFormGame` or `ExtensiveFormGame` classes
3. **Enhance Visualizations**: Extend plotting methods in `game_theory_core.py`
4. **Add Analysis Tools**: Create new analysis methods following existing patterns

## 📚 References

Key game theory concepts implemented:
- Nash, J. (1950). "Equilibrium points in n-person games"
- Osborne & Rubinstein (1994). "A Course in Game Theory"
- Fudenberg & Tirole (1991). "Game Theory"

## 🐛 Troubleshooting

**Common Issues:**

1. **Import Errors**: Ensure all required packages are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Visualization Not Showing**: Check matplotlib backend
   ```python
   import matplotlib
   matplotlib.use('TkAgg')  # or 'Qt5Agg'
   ```

3. **Large Games Running Slowly**: Mixed Nash equilibrium computation is exponential; limit game size to 4×4 or smaller for mixed equilibria

## 📄 License

This project is designed for educational purposes in game theory courses.

## 🎯 Learning Outcomes

After using this simulator, students should be able to:
- Represent strategic situations as games
- Identify Nash equilibria in various game types
- Understand strategic dominance and rationalizability
- Apply backward induction to extensive form games
- Analyze welfare properties of equilibria
- Design and analyze custom game scenarios

---

**Version**: 1.0  
**Last Updated**: December 2025  
**Author**: Game Theory Course Project

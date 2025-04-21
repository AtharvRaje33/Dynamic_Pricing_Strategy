
# Dynamic Pricing System with Reinforcement Learning

## Project Overview
This project implements a **Dynamic Pricing System** using reinforcement learning (RL) algorithms to optimize pricing strategies for products based on the UCI Online Retail dataset. The system simulates a pricing environment for two popular products and evaluates the performance of two RL algorithms: **DQN** (Deep Q-Network) and **A2C** (Advantage Actor-Critic). The goal is to maximize revenue by dynamically adjusting prices based on simulated demand.

### Key Features
- **Dataset**: UCI Online Retail dataset (`OnlineRetail.xlsx`), filtered to focus on the top two products by transaction count (StockCodes: `85123A` and `85099B`).
- **Environment**: Custom `DynamicPricingEnv` built with Gymnasium, simulating a pricing scenario with normalized prices (0.5–1.5) and demand based on historical data.
- **RL Algorithms**: Implements DQN and A2C from Stable-Baselines3 to learn optimal pricing strategies.
- **Evaluation**: Metrics include mean reward, final revenue, average purchase rate, and optimal price trends, visualized with plots and summarized in a table.
- **Visualization**: Generates plots for cumulative revenue, purchase rate, and optimal price over time for each model and product.

## Installation

### Prerequisites
- Python 3.10+
- Conda or virtualenv for environment management
- Required libraries listed in `requirements.txt`

### Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/dynamic-pricing-rl.git
   cd dynamic-pricing-rl
   ```

2. **Create a Virtual Environment**:
   ```bash
   conda create -n rl-dps python=3.10
   conda activate rl-dps
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Dataset**:
   - Download the UCI Online Retail dataset (`OnlineRetail.xlsx`) from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/online+retail) or Kaggle.
   - Place the `OnlineRetail.xlsx` file in the project root directory.

## Usage

### Running the Code
1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook dps.ipynb
   ```

2. **Execute the Notebook**:
   - Run all cells in `dps.ipynb` to:
     - Load and preprocess the dataset.
     - Train DQN and A2C models for the top two products (`85123A` and `85099B`).
     - Save trained models to `Training/Saved Models/`.
     - Generate evaluation plots saved to `Images/`.
     - Display an aggregate evaluation metrics table.

### Output
- **Saved Models**: Trained models are saved as `.zip` files in `Training/Saved Models/` (e.g., `model_DQN_85123A.zip`).
- **Plots**: Visualization of cumulative revenue, purchase rate, and optimal price are saved in `Images/` (e.g., `DQN_85123A_3in1.png`).
- **Metrics Table**: A table summarizing mean reward, final revenue, average purchase rate, and optimal prices is displayed in the notebook.

## Project Structure
```
dynamic-pricing-rl/
├── OnlineRetail.xlsx         # UCI Online Retail dataset
├── dps.ipynb                 # Main Jupyter Notebook with code
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── Training/
│   └── Saved Models/         # Directory for saved RL models
├── Images/                   # Directory for output plots
```

## Dependencies
Key libraries used in the project (see `requirements.txt` for full list):
- `numpy`
- `pandas`
- `gymnasium`
- `stable-baselines3`
- `matplotlib`
- `scikit-learn`

## Results
The system evaluates DQN and A2C for two products:
- **StockCode 85123A** (White Hanging Heart T-Light Holder)
- **StockCode 85099B** (Jumbo Bag Red Retrospot)

### Sample Evaluation Metrics
| StockCode | Model | Mean Reward    | Final Revenue | Avg Purchase Rate | Final Optimal Price | Avg Optimal Price |
|-----------|-------|----------------|---------------|-------------------|---------------------|-------------------|
| 85123A    | DQN   | 1302.14 ± 96.02 | 68001.38      | 0.62              | 1.28                | 1.26              |
| 85123A    | A2C   | 1296.76 ± 96.97 | 58625.64      | 0.70              | 1.17                | 1.10              |
| 85099B    | DQN   | 1811.25 ± 158.75 | 100533.57    | 0.63              | 1.28                | 1.25              |
| 85099B    | A2C   | 1479.59 ± 53.85 | 86765.09      | 0.76              | 0.83                | 0.98              |

### Observations
- **DQN** generally achieves higher final revenue but lower purchase rates, indicating a strategy favoring higher prices.
- **A2C** maintains higher purchase rates with lower prices, leading to slightly lower revenue but more consistent sales.
- Plots show how each model adapts prices over time to balance revenue and demand.

## Limitations
- The demand model is simplified, using average quantity and a linear price-demand relationship.
- Only two products are analyzed to reduce computational complexity.
- The environment assumes stationary demand, which may not reflect real-world dynamics.

## Future Improvements
- Incorporate more sophisticated demand models (e.g., time-series or external factors like seasonality).
- Expand to more products or product categories.
- Experiment with additional RL algorithms (e.g., PPO, SAC).
- Add real-time data integration for live pricing adjustments.

## Acknowledgments
- UCI Machine Learning Repository for the Online Retail dataset.
- Stable-Baselines3 for RL algorithm implementations.
- Gymnasium for the reinforcement learning environment framework.

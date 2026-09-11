# Adaptive Particle Swarm Optimization for Blockchain Systems

This repository contains the source code, data generation scripts, and datasets used in the research paper: "An Adaptive Particle Swarm Optimization Approach for Minimizing Cost and Latency in Blockchain-Based Systems"

The primary objective of this repository is to ensure full transparency and reproducibility of our experimental results, specifically concerning the retrieval of real-world Ethereum data, the calibration of our algorithmic metrics, and the optimization simulations.

##  Repository Structure

The repository is organized into Scripts (Code) and Datasets (Real & Synthetic):

### 1. Data Retrieval and Processing Scripts
These Python scripts demonstrate how empirical data was gathered and formatted to be used within the Particle Swarm Optimization (PSO) framework.

*   [fetch_real_ethereum_data.py](fetch_real_ethereum_data.py): Connects directly to Ethereum public RPC nodes to fetch live, real-time block and transaction data.
*   [process_kaggle_dataset.py](process_kaggle_dataset.py): Cleans the raw historical Kaggle dataset, calculates critical internal metrics (e.g., node propagation, validation delays), and correlates them with real extracted gas prices.
*   [generate_calibrated_simulation.py](generate_calibrated_simulation.py): Generates the heavily calibrated, synthetic datasets used for large-scale stress testing (varying sizes from N=100 to N=5000) by combining real baselines with statistical distributions.
*   [blockchain_dataset_generator.py](blockchain_dataset_generator.py): Fallback utility for generating generic, uncalibrated synthetic baseline datasets for comparative testing.

### 2. Datasets
The datasets provided here represent both the foundation of our empirical research and the calibrated outputs fed into the algorithm.

**Raw / Real-World Datasets:**
*   [kaggle_raw_transactions.csv](kaggle_raw_transactions.csv): The primary historical Ethereum dataset downloaded and used for baseline extraction.
*   [live_raw_transactions.csv](live_raw_transactions.csv): A sample output of real-time RPC data fetched by the scripts.
*   Etherscan API Exports (e.g., [export-AvgGasPrice.csv](export-AvgGasPrice.csv), [export-GasUsed.csv](export-GasUsed.csv),[export-BlockTime](export-BlockTime.csv)): Empirical confirmation of standard gas and latency trends.

**Calibrated / Synthetic Datasets:**
*   [blockchain_real_calibrated_N1000.csv](blockchain_real_calibrated_N1000.csv): Our standard N=1000 dataset deeply calibrated against the real-world Ethereum metrics.
*   [blockchain_simulation_calibrated_N1000.csv](blockchain_simulation_calibrated_N1000.csv): The purely synthetic evaluation set for N=1000.
*   [blockchain_simulation_calibrated_N5000.csv](blockchain_simulation_calibrated_N5000.csv): The large-scale stress test dataset utilized in the paper to prove algorithmic scalability.

*   The *process_kaggle_dataset.py* script takes  *kaggle_raw_transactions.csv* to build calibrated algorithmic metrics in *blockchain_simulation_calibrated_N1000.csv*

## 🚀 How to Reproduce the Data

To regenerate the calibrated metrics from the raw data:

1. Ensure you have Python 3.8+ installed along with `pandas` and `numpy`.
2. Clone this repository to your local machine.
3. Run the processing script against the raw Kaggle data:
   ```bash
   python process_kaggle_dataset.py

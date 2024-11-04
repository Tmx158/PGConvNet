# PGConvNet: A Multi-Scale Feature Embedding Framework for Time Series Imputation

🤗🤗🤗Welcome to the PGConvNet repository! This project presents an efficient solution for time series imputation using a novel convolutional architecture designed specifically for handling missing values in multivariate time series data.

## Overview of PGConvNet

🔎🔎🔎PGConvNet is a state-of-the-art model that leverages a two-stage architecture to effectively capture temporal and inter-variable dependencies. By transforming 1D time series data into a 2D representation, it enhances the model's ability to process complex interactions across multiple variables, ensuring robust imputation performance.
![image](https://github.com/user-attachments/assets/4d7b458e-3581-4d5c-8975-00e172f18cb6)


## Module Descriptions

### 1. MSGBlock (Multi-Scale Grouped Convolutional Block)
The MSGBlock is designed to extract multi-scale temporal features while preserving the interdependencies among variables. It utilizes grouped convolutions to capture both short-term and long-term patterns, making it well-suited for diverse time series datasets.  
![image](https://github.com/user-attachments/assets/e12a3bdf-189d-45fb-83d4-1d1f2c9943bc)


### 2. PGCBlock (Parametric Grouped Convolutional Block)
The PGCBlock dynamically adapts to the random positioning of missing values by employing parametric convolutions. This approach allows for precise extraction of relevant temporal and variable information, effectively replacing traditional attention mechanisms.
![image](https://github.com/user-attachments/assets/df671f60-33ef-473e-b1a6-ccce312878bc)


## Main Results
PGConvNet has demonstrated consistent state-of-the-art performance in time series imputation tasks, significantly outperforming existing models, particularly under high rates of missing data.
![image](https://github.com/user-attachments/assets/8d81e0a3-1a3a-4893-8e1b-6219c793295f)


## Getting Started
To use PGConvNet, follow these simple steps:
   **Install Dependencies**: First, install the necessary packages by running:
   ```bash
   pip install -r requirements.txt
   ```
   **Run the Code**: Following command for your dataset (for example, using ETTh1):
   ```bash
   bash PGConvNet/task_imputation/scripts//ETTh1/ETTh1.sh  
   ```

## Acknowledgments
😘😘😘We would like to extend our gratitude to the following repositories for their invaluable contributions to the code base and datasets:
- [ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis](https://github.com/luodhhh/ModernTCN)
- [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://github.com/haoyu0221/Informer)
- [Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift](https://github.com/ts-kim/RevIN)
- [TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://github.com/thuml/Time-Series-Library/)
Special thanks to [THUML Time-Series-Library](https://github.com/thuml/Time-Series-Library) for providing benchmark testing code.
Join the vibrant community of multivariate time series researchers and practitioners to share insights, resources, and collaborate on exciting projects!

## Explore More
👏👏👏We are thrilled to see the rise of another vibrant community focused on advancing time series imputation methods.  This community is dedicated to fostering collaboration, sharing resources, and driving innovation in tackling the challenges posed by missing data in time series analysis.  Discover more about this community at [TSI-Bench： Benchmarking Time Series Imputation](https://github.com/WenjieDu/Awesome_Imputation).

Happy coding!

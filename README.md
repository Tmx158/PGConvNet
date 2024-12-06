


# PGConvNet: A Multi-Scale Feature Embedding Framework Using Grouped and Parametric Convolutions for Efficient Time Series Imputation

🤗🤗🤗Welcome to the PGConvNet repository! This project presents an efficient solution for time series imputation using a novel convolutional architecture designed specifically for handling missing values in multivariate time series data.
![figure2](https://github.com/user-attachments/assets/a24d45db-605e-40a3-a99d-3cd9bdd040a7)

## Introduction to PGConvNet

🧸🧸🧸Multivariate time series data are crucial in various fields, such as finance, healthcare, and environmental monitoring, where accurate imputation is essential for decision-making. However, handling missing values in such datasets is challenging, especially when missing data points occur randomly across time series.Recent experiments on datasets like Weather, Electricity, and ETTh1 reveal strong correlations between neighboring variables, as shown in the Pearson correlation heatmaps. 

![figure1](https://github.com/user-attachments/assets/094efe38-67d0-4fed-88e2-89afc441949f)

These correlations suggest that nearby variables exhibit higher inter-variable dependencies, which are essential for effective imputation.Moreover, a significant challenge arises from the random distribution of missing values. While traditional Transformer models utilize attention mechanisms to capture relationships across sequences, their performance is hindered by the unpredictable nature of missing data, leading to inefficiencies and reduced accuracy. PGConvNet addresses these issues by leveraging a two-stage convolutional architecture that adapts dynamically to missing values, ensuring accurate and efficient imputation even under high missing rates.



🔎🔎🔎PGConvNet is a state-of-the-art model that leverages a two-stage architecture to effectively capture temporal and inter-variable dependencies. By transforming 1D time series data into a 2D representation, it enhances the model's ability to process complex interactions across multiple variables, ensuring robust imputation performance.

## Component Overview

### 1. MSGBlock (Multi-Scale Grouped Convolutional Block)
The MSGBlock is designed to extract multi-scale temporal features while preserving the interdependencies among variables. It utilizes grouped convolutions to capture both short-term and long-term patterns, making it well-suited for diverse time series datasets.  
![image](https://github.com/user-attachments/assets/e12a3bdf-189d-45fb-83d4-1d1f2c9943bc)


### 2. PGCBlock (Parametric Grouped Convolutional Block)
The PGCBlock dynamically adapts to the random positioning of missing values by employing parametric convolutions. This approach allows for precise extraction of relevant temporal and variable information, effectively replacing traditional attention mechanisms.
![image](https://github.com/user-attachments/assets/df671f60-33ef-473e-b1a6-ccce312878bc)


## Results
PGConvNet has demonstrated consistent state-of-the-art performance in time series imputation tasks, significantly outperforming existing models, particularly under high rates of missing data.

![Result](https://github.com/user-attachments/assets/db35c827-57b2-4ce7-80dc-e66ea2f5992b)



## How to Begin
To use PGConvNet, follow these simple steps:

   **Download the Dataset**: First, download the dataset file `all_datasets.zip` from the Baidu Netdisk link: [https://pan.baidu.com/s/1z_EZfKehfqqZvlum1BtcNA?pwd=etr6](https://pan.baidu.com/s/1z_EZfKehfqqZvlum1BtcNA?pwd=etr6).
   Then, extract it directly into the root directory of PGConvNet.
   **Install Dependencies**: install the necessary packages by running:
   ```bash
   pip install -r requirements.txt
   ```
   **Run the Code**: Following command for your dataset (for example, using ETTh1):
   ```bash
   bash PGConvNet/scripts/ETTh1.sh  
   ```

## Acknowledgments
😘😘😘We would like to extend our gratitude to the following repositories for their invaluable contributions to the code base and datasets:
- [ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis](https://github.com/luodhhh/ModernTCN)
- [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://github.com/haoyu0221/Informer)
- [Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift](https://github.com/ts-kim/RevIN)
- [TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://github.com/thuml/Time-Series-Library/)
Special thanks to [THUML Time-Series-Library](https://github.com/thuml/Time-Series-Library) for providing benchmark testing code.
Join the vibrant community of multivariate time series researchers and practitioners to share insights, resources, and collaborate on exciting projects!

## Explore 
👏👏👏We are thrilled to see the rise of another vibrant community focused on advancing time series imputation methods.  This community is dedicated to fostering collaboration, sharing resources, and driving innovation in tackling the challenges posed by missing data in time series analysis.  Discover more about this community at [TSI-Bench： Benchmarking Time Series Imputation](https://github.com/WenjieDu/Awesome_Imputation).

Happy coding!

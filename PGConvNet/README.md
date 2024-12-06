🧸🧸🧸Multivariate time series data are crucial in various fields, such as finance, healthcare, and environmental monitoring, where accurate imputation is essential for decision-making.
However, handling missing values in such datasets is challenging, especially when missing data points occur randomly across time series.Recent experiments on datasets like Weather, Electricity, and ETTh1 reveal strong correlations between neighboring variables, as shown in the Pearson correlation heatmaps.

These correlations suggest that nearby variables exhibit higher inter-variable dependencies, which are essential for effective imputation.Moreover, a significant challenge arises from the random distribution of missing values.
While traditional Transformer models utilize attention mechanisms to capture relationships across sequences, their performance is hindered by the unpredictable nature of missing data, leading to inefficiencies and reduced accuracy. 
PGConvNet addresses these issues by leveraging a two-stage convolutional architecture that adapts dynamically to missing values, ensuring accurate and efficient imputation even under high missing rates.


🔎🔎🔎PGConvNet is a state-of-the-art model that leverages a two-stage architecture to effectively capture temporal and inter-variable dependencies.
By transforming 1D time series data into a 2D representation, it enhances the model's ability to process complex interactions across multiple variables, ensuring robust imputation performance.


To use PGConvNet, follow these simple steps: Install Dependencies: First, install the necessary packages by running:

pip install -r requirements.txt

Run the Code: Following command for your dataset (for example, using ETTh1):

bash PGConvNet/scripts/ETTh1.sh  


😘😘😘We would like to extend our gratitude to the following repositories for their invaluable contributions to the code base and datasets:

ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis
Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift
TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis Special thanks to THUML Time-Series-Library for providing benchmark testing code. Join the vibrant community of multivariate time series researchers and practitioners to share insights, resources, and collaborate on exciting projects!

Explore
👏👏👏We are thrilled to see the rise of another vibrant community focused on advancing time series imputation methods. This community is dedicated to fostering collaboration, sharing resources, and driving innovation in tackling the challenges posed by missing data in time series analysis. Discover more about this community at TSI-Bench： Benchmarking Time Series Imputation.

Happy coding!
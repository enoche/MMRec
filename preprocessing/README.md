# Preprocessing from raw data 从原始数据处理
- datasets: [Amazon](http://jmcauley.ucsd.edu/data/amazon/links.html)  
-- Rating file in `Files/Small subsets for experimentation`  
-- Meta files in `Per-category files`, [metadata], [image features]

## Step by step
1. Performing 5-core filtering, re-indexing -- `0rating2inter.ipynb`
2. Train/Valid/Test data spliting -- `1spliting.ipynb`
3. 
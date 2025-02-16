# Traffic Accident Prediction Using GNNs with Slope Feature  

## Overview  

This project builds upon the **ML4RoadSafety** repository, which applies **Graph Neural Networks (GNNs)** to predict traffic accidents. While the original model used road network and meteorological features, our contribution introduces **slope** as an additional feature to improve predictive performance.

By incorporating slope data into the edge features, we observed a **60 basis point improvement** in AUROC, demonstrating the significance of elevation changes in accident prediction.

## Original Repository  

[Original ML4RoadSafety Repository](https://github.com/VirtuosoResearch/ML4RoadSafety)  

## Key Enhancements in This Fork  

✅ **Feature Engineering**: Added a new **slope feature** for edges, calculated using elevation data.  
✅ **Improved AUROC**: The addition of slope **increased AUROC from 82.62% to 83.22%**.  
✅ **Data Preprocessing Pipeline**: Developed scripts to retrieve elevation data and compute slope efficiently.  

## Dataset Modifications  

The dataset was sourced from **Harvard Dataverse** and includes Montana traffic data (2016-2020). The **new slope feature** was derived using the **Open-Elevation API** and integrated into the edge features tensor.  

The slope for each edge was computed as:  

$$
\text{slope} = \frac{E_{\text{target}} - E_{\text{source}}}{L}
$$

Where:  
- $\(E_{\text{target}}\) and \(E_{\text{source}}\)$ are the elevations of the target and source nodes, respectively.  
- $\(L\)$ is the road segment length.  

### New Scripts in This Fork  

The following scripts were added to the dataset processing pipeline:  

📂 **slope_inclusion/** (New folder for slope processing)  
- `9_elevation_dataset_modification.py` → Retrieves elevation data using the Open-Elevation API.  
- `8_batches_split.py` → Splits dataset into manageable batches for elevation API calls.  
- `10_nodes_elevation_concat.py` → Integrates elevation data with node features.  
- `11_edge_slope_inclusion.py` → Computes slope values for each edge in the road network.  

## Model & Training  

The **Graph Convolutional Network (GCN)** model used:  
- **2 graph convolutional layers**  
- **256 hidden dimensions**  
- **Learning rate**: 0.001  
- **Optimizer**: Adam  
- **Epochs**: 100  

The training and testing splits remained consistent with the original study:  
- **Training**: 2016–2017  
- **Validation**: 2018  
- **Testing**: 2019–2020  

## Results  

Adding slope improved **AUROC**, particularly on the test set (**+60 basis points**):  

| **Metric**       | **Model - Slope** | **Model - No Slope** | **Improvement (b.p)** |
|----------------|--------------|------------------|------------------|
| **Train AUROC** | 83.31 ± 0.35 | 82.99 ± 0.15 | +32 |
| **Valid AUROC** | 83.03 ± 0.12 | 82.73 ± 0.04 | +30 |
| **Test AUROC** | 83.22 ± 0.30 | 82.62 ± 0.21 | **+60** |
| **Test Precision** | 8.13 ± 2.77 | 5.26 ± 1.25 | **+287** |
| **Test Recall** | 47.14 ± 8.47 | 52.01 ± 1.34 | **-487** |

## Future Work  

- Improve elevation data accuracy (**Google Maps API** for higher resolution).  
- Optimize slope feature encoding (**categorical bins: flat, mild, steep**).  
- Extend analysis to **different states** for generalization.  

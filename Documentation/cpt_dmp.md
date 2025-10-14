# Data Management Plan: Lithostratigraphy Identification from CPTs

**Project Title:** A Sequence Modeling Approach to Automated Lithostratigraphy Identification from Cone Penetration Test Data

**Creator:** Rhine Rift Analytics

**Affiliation:** Project Data Science, UHasselt

**Last Modified:** 2025-10-14

---

## Project Abstract

This project aims to develop a machine learning model to automate the identification of lithostratigraphic units from Cone Penetration Test (CPT) data. The project directly addresses the current challenge of slow and resource-intensive manual interpretation. By utilising a dataset of 242 labeled CPTs, we will employ **sequence modeling techniques** to predict the `lithostrat_id`. The core methodology involves **feature engineering**, including the calculation of derivatives and rolling window statistics to capture layer transitions and soil texture. The final deliverable will be a trained model demonstrated through a **dashboard**, providing VITO with a clear path towards a computer-assisted interpretation tool for geologists.

---

#### **1. Research Data Summary**

*This table outlines all data that will be used or generated during the project lifecycle.*

| Dataset Name | Description | New or Reused | Digital or Physical | Digital Data Type | Digital Data Format | Est. Digital Volume |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Raw Labeled CPT Data** | The initial dataset from VITO containing measurements and labels for 242 CPTs. | Reused | Digital | Observational | .csv | <100MB |
| **Structured Sequence Data** | Raw data cleaned and structured into sequences, with each CPT treated as an independent time-series. The foundational dataset for all further work. | Generate new | Digital | Compiled | .parquet, .csv | <100MB |
| **Engineered Feature Sets** | The core analytical dataset. Includes derivatives (gradients, curvature) and rolling window statistics (mean, std dev) to capture local texture and layer transitions. | Generate new | Digital | Compiled | .parquet | <200MB |
| **Split Datasets (Train/Test)** | The engineered feature sets partitioned for model training and unbiased final evaluation, ensuring sequence integrity is maintained. | Generate new | Digital | Software | .pkl | <200MB |
| **Trained Lithostratigraphy Models** | Serialized model artifacts, including baseline (e.g., RandomForest), sequence (e.g., CRF, LSTM/GRU), and any potential post-processing models. | Generate new | Digital | Software | .pkl | <500MB |
| **Model Predictions & Performance** | The model's predictions on the test set, including class labels and uncertainty scores, alongside detailed per-class performance metrics. | Generate new | Digital | Simulation | .csv, .json | <50MB |
| **MVP Visualization Data** | A curated subset of the test data and model predictions, formatted specifically for the Streamlit/Plotly dashboard. | Generate new | Digital | Compiled | .csv | <10MB |
| **Code & Documentation** | All Python/R scripts, Jupyter notebooks, README files, this DMP, and the final report. Version controlled in Git. | Generate new | Digital | Software & Doc. | .py, .ipynb, .md | <20MB |

---

#### **2. Data Handling & Core Methodology**

*This section details our core technical strategy for data handling and modeling.*

**2.1 Data Handling: DataFrame Approach**
The primary tool for all data loading, cleaning, and manipulation will be the **Pandas DataFrame**. This choice is justified by:
*   **Scale & Performance:** The total dataset size is estimated at ~60,000 observations, which fits comfortably in memory. A DataFrame provides high-performance, in-memory computation, enabling rapid iteration.
*   **Ecosystem Integration:** The entire Python data science stack (scikit-learn, PyTorch, Plotly, Streamlit) is optimized to work directly with DataFrames.
*   **Analytical Power:** Pandas provides a rich and flexible API perfectly suited for the complex sequence-based feature engineering (grouping, rolling windows, derivatives) required for this project.

**2.2 Feature Engineering Strategy**
To capture the domain knowledge that "order and texture matter," we will generate a rich set of features beyond the raw measurements. For each CPT sequence (`sondeernummer`), we will compute:
*   **1st and 2nd Derivatives:** To model the rate of change and identify sharp transitions between layers (e.g., `qc_gradient`).
*   **Sliding Window Statistics:** Using various window sizes (e.g., 50cm, 1m) to calculate local `mean`, `std dev`, `min`, and `max` for key variables. This captures the homogeneity and texture of the soil.

**2.3 Modeling Strategy: Layer-Aware Sequence Modeling**
Our approach explicitly treats this as a **sequence labeling task**:
1.  **Exploratory Analysis:** We will first produce visualizations (e.g., box plots per `lithostrat_id`) to find "evidence of power" in our engineered features and confirm their discriminative ability.
2.  **Baseline Model:** A non-sequential model (e.g., RandomForestClassifier) will be trained to establish a performance baseline and validate feature importance.
3.  **Primary Sequence Model:** The core of the project will be the development of a sequence-aware model. We will explore **Conditional Random Fields (CRFs)** for their interpretable transition logic and **Recurrent Neural Networks (LSTMs/GRUs)** for their ability to learn complex, long-range dependencies in the data.
4.  **Handling Geolocation:** In line with client requirements for a generalizable model, the coordinates (`x`, `y`) and absolute depth (`diepte_mtaw`) will be **excluded** from the primary sequence model's features. They may be used in a separate, optional post-processing step to refine predictions based on spatial proximity to known CPTs.
5.  **Evaluation:** Model performance will be assessed not just on overall accuracy, but with a detailed **per-class classification report** (precision, recall, F1-score) to ensure the model effectively identifies both common and rare/thin layers.

---

#### **3. Documentation and Metadata**

*A detailed codebook will be created for the `Engineered Feature Sets`, which is especially critical for defining the generated features. A `README.md` will document the setup, directory structure, and workflow. All code will be commented and version controlled.*

---

#### **4. Data Storage & Back-up During the Research Project**

*All code will be stored and versioned in a private **Git repository**. All data and documentation will reside on a secure institutional shared drive with automated backups. This ensures a clear separation of code and data, promotes reproducibility, and prevents data loss.*

---

#### **5. Data Preservation After the End of the Research Project**

*At the conclusion of this project, we will deliver a single, organized package to VITO containing the following documents.*

*   *The final `Engineered Feature Sets` and `Structured Sequence Data`.*
*   *The detailed `Codebook` and project `README.md`.*
*   *The final, commented `Data Processing & Modeling Scripts`.*
*   *The final `Trained Lithostratigraphy Models`.*
*   *The `Final Project Report`.*

---
#### **6. Responsibilities**

*   **Who will manage data documentation and metadata?**
    *   The **Data Scientist(s)/Researcher(s)** on the project team will be responsible for creating and maintaining all documentation (README, Codebook, code comments) as an ongoing activity throughout the project lifecycle.

*   **Who will manage data storage and backup?**
    *   The **Project Lead** is responsible for establishing the secure storage and code repository environments. The **Data Scientist(s)** are responsible for the day-to-day use of these systems, including regular commits and pushes of code to the Git repository.

*   **Who will manage data preservation and sharing?**
    *   The **Project Lead**, in consultation with the team, will be responsible for preparing the final archival package and depositing it in the chosen repository (Zenodo) at the conclusion of the project.
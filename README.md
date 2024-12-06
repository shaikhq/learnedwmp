# LearnedWMP: Workload Memory Prediction Using Distribution of Query Templates - Experiment Datasets and Code Assets

This repository has the source code for the following paper (under review):

Quader, Shaikh, Andres Jaramillo, Sumona Mukhopadhyay, Ghadeer Abuoda, Calisto Zuzarte, David Kalmuk, Marin Litoiu, and Manos Papagelis. "LearnedWMP: Workload Memory Prediction Using Distribution of Query Templates."

### Repository Overview

This Git repository contains the source code for the research paper referenced in this README. It also provides instructions for running the code included in this repository. Below is an overview of the repository structure and its contents.

---

### Repository Structure

The repository is organized into two main folders: 

1. **`models`**  
2. **`templates`**

#### **`templates` Folder**
This folder contains code and data for generating learning templates using two distinct approaches:  

- **Approach 1:** Generating templates from query text.  
- **Approach 2:** Generating templates from query plan features.  

These approaches enable flexible and efficient creation of templates for various workloads.

#### **`models` Folder**
The `models` folder is further divided into four subdirectories, each tailored for specific training tasks:  

- **`job_query`:**  
  - Contains data and notebooks training models at the level of individual queries from Join Order Benchmark (JOB) queries.  

- **`job_workload`:**  
  - Includes data and notebooks for training models at the workload level (i.e., a batch of queries) using queries from Join Order Benchmark (JOB) queries.  

- **`tpcds_query`:**  
  - Provides datasets and notebooks for training query-level models for TPC-DS queries.  

- **`tpcds_workload`:**  
  - Contains resources for workload-level model training using TPC-DS queries.  

---

### Instructions

1. Begin by exploring the folders to understand their purpose and content.  
2. Follow the provided notebooks and datasets in each folder to replicate the training and template generation approaches described. The following is the suggested order of running code:
- First, **`templates`** folder 
- Second, **`models`** folder

#### Steps to running the notebooks

To run the Jupyter Notebooks from the [LearnedWMP repository](https://github.com/shaikhq/learnedwmp), follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/shaikhq/learnedwmp.git
   cd learnedwmp
   ```

2. **Set Up a Virtual Environment**
   ```bash
   python -m venv .venv
   ```
***The recommended python level is python 3.12.3.***

3. **Activate the Virtual Environment**
   - **On Windows:**
     ```bash
     .venv\Scripts\activate
     ```
   - **On macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```

4. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   ```

5. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
   This command will open the Jupyter Notebook interface in your default web browser.

6. **Open and Run Notebooks**
   In the Jupyter interface, navigate to the notebook you wish to run and click on it to open. Execute the cells sequentially to run the code.

**Note:**
Ensure that your environment meets all prerequisites specified in the `requirements.txt` file.

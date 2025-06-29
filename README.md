# Generator Machine Learning Emission Modeling
### Created by: Daniel Sitompul 


This model is created to assess the use of machine learning to predict generator ussage within a time period. The data that we have is voltage and frequency (within a 2 minute period) from 2 locations in the healthcare facilities.

In this repository, I use 3 different machine learning model:
1. Logistic Regression
2. Random Forest
3. XG-Boost

More information about the methodology and results can be found in the presentation file:  
*Presentation - Climate Impact Assessment of Generator Usage using Machine Learning: A Case Study from DR Congo.pdf*

---

## Prepare Environment

To prepare the environment, follow these steps:

### 1. Create a Virtual Environment
```bash
git clone https://github.com/Danielstevends/Generator_ML_Emission_Modeling
python -m venv venv
source venv/bin/activate # for mac
venv\Scripts\activate # for windows
```

### 2. Install Dependencies
```bash
# For the modeling
pip install -r requirements.txt

# For the functions
cd ModelingFunctions
pip install -r requirements.txt
```

### 3. Verify installation
```bash
pip freeze
```

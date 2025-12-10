# AI Learning Journey - Complete README

## 📚 Overview

This repository documents a comprehensive learning journey in **Artificial Intelligence and Machine Learning**, progressing from fundamental concepts to practical projects. The workspace contains daily lessons, reference projects, and hands-on implementations.

---

## 📁 Folder Structure

### **Daily Learning Modules**

#### [`day 1`](day%201)
- **1.ipynb** - Introduction to Python basics
- **1.py** - Python script examples

#### [`day 2`](day%202)
- **numpy.ipynb** - NumPy fundamentals and array operations

#### [`day 3`](day%203)
- **numpy.ipynb** - Advanced NumPy concepts

#### [`day 4`](day%204)
- **student_data.csv** - Sample student dataset
- **cleaned_data.csv** - Data cleaning examples
- **pandas.txt** - Pandas documentation notes
- **cleaned/** - Cleaned dataset directory

#### [`day 5`](day%205)
- **Untitled.ipynb** - Additional exploratory notebooks

#### [`day 8`](day%208)
- **linear regression.ipynb** - Linear Regression implementation
- **California housing.ipynb** - Regression project using California Housing dataset
- Uses `sklearn.linear_model.LinearRegression` with California Housing data

---

### **Projects**

#### [`day 6 (project)`](day%206%28project%29)
- **ttt.ipynb** - **Titanic Survival Prediction** (Machine Learning Classification Project)
  - Data loading from `datas (csv)/train.csv`
  - Data cleaning and feature engineering
  - Model training and predictions
  - Submission file generation

#### [`day 7 (project)`](day%207%28project%29)
- **ttt.ipynb** - Titanic project continuation and refinement

#### [`day 9`](day%209)
- Placeholder for advanced projects

#### [`student performance project`](student%20performance%20project)
- **students_performance.ipynb** - Student Performance Analysis
- **StudentsPerformance.csv** - Dataset with student academic performance metrics

---

### **Reference Projects (Kaggle)**

#### [`Reference project (kaggle)`](Reference%20project%20%28kaggle%29)

**nepal-education-data-analysis-2074bs.ipynb** - Comprehensive Educational Data Analysis
- Sex-wise student enrollment analysis by Province
- Grade-wise student enrollment (School Education)
- Faculty-wise student enrollment (Higher Education)
- University-wise student enrollment analysis
- Province-wise higher education enrollment
- Teacher distribution analysis
- Literacy rate analysis by region
- Visualization using Plotly (iplot, scatter, pie charts)

**penguin-dataset-the-new-iris.ipynb** - Palmer Penguin Dataset Analysis
- Data preprocessing with scikit-learn
- Feature scaling and label encoding
- Bar chart race visualizations
- Species classification analysis

---

### **Data Directory**

#### [`datas (csv)`](datas%20%28csv%29)
- **train.csv** - Titanic training dataset (891 passengers)
- **test.csv** - Titanic test dataset (418 passengers)
- **gender_submission.csv** - Titanic submission template
- **data.csv** - General reference data

---

### **Supporting Directories**

- **[`documentation`](documentation)** - Documentation files and guides
- **[`notebook work`](notebook%20work)** - Experimental notebooks and studies
  - students of college.ipynb - Data manipulation exercises
- **.vscode/** - VS Code configuration
- **.git/** - Git version control
- **.ipynb_checkpoints/** - Jupyter checkpoint files

---

## 🎯 Key Projects & Learning Outcomes

### 1. **Titanic Survival Prediction** 📊
**Files:** [`day 6 (project)/ttt.ipynb`](day%206%28project%29/ttt.ipynb), [`day 7 (project)/ttt.ipynb`](day%207%28project%29/ttt.ipynb)

**Topics Covered:**
- Data loading and exploratory data analysis (EDA)
- Missing value handling and imputation
- Feature engineering (age groups, family size, titles)
- Categorical encoding (Sex, Embarked, Pclass)
- Model training and evaluation (Logistic Regression, Random Forest)
- Cross-validation and hyperparameter tuning
- Submission file generation for Kaggle

**Dataset:** 891 training samples, 12 features (Age, Sex, Pclass, Fare, etc.)
**Technologies:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

**Expected Outcome:** Classification model predicting passenger survival with >80% accuracy

---

### 2. **Nepal Education Data Analysis** 🎓
**Files:** [`Reference project (kaggle)/nepal-education-data-analysis-2074bs.ipynb`](Reference%20project%20%28kaggle%29/nepal-education-data-analysis-2074bs.ipynb)

**Analyses Included:**
- Sex-wise enrollment patterns across all 7 provinces
- Grade-wise student distribution in school education
- Faculty-wise enrollment in higher education institutions
- University-wise student statistics and trends
- Province-wise higher education enrollment comparison
- Teacher distribution and availability analysis
- Literacy rate trends and regional disparities
- Interactive dashboard visualizations

**Key Findings:**
- Gender disparities in educational enrollment
- Regional variations in educational access
- Growth trends in higher education

**Visualizations:** Bar charts, Pie charts, Scatter plots, Box plots, Heatmaps
**Technologies:** Pandas, Plotly (iplot), Statistical analysis, Data aggregation

---

### 3. **Linear Regression & California Housing** 🏠
**Files:** [`day 8/linear regression.ipynb`](day%208/linear%20regression.ipynb), [`day 8/California housing.ipynb`](day%208/California%20housing.ipynb)

**Topics:**
- Linear Regression theory and mathematics
- Model training on housing data
- Feature scaling and normalization
- Performance evaluation (R² score, RMSE, MAE)
- Prediction on test data
- Visualization of regression results
- Real-world regression application

**Dataset:** California Housing (20,640 samples, 8 features)
**Technologies:** Scikit-learn, Matplotlib, NumPy

**Expected Outcome:** Regression model predicting house prices with R² > 0.57

---

### 4. **Student Data Management** 👥
**Files:** [`notebook work/students of college.ipynb`](notebook%20work/students%20of%20college.ipynb)

**Topics:**
- Data creation and dataset initialization
- Data manipulation and transformation
- Data cleaning techniques:
  - Handling null/missing values
  - Removing duplicate records
  - Type conversions and data validation
- CSV export and file management
- Data filtering and sorting
- Aggregation and grouping operations
- Statistical summaries

**Practical Skills:** Database-like operations using Pandas

---

### 5. **Student Performance Analysis** 📈
**Files:** [`student performance project/students_performance.ipynb`](student%20performance%20project/students_performance.ipynb)

**Topics:**
- Performance metrics analysis
- Demographic analysis (gender, race/ethnicity, parental education)
- Score distribution and correlation analysis
- Test preparation impact assessment
- Lunch program and course completion analysis
- Visualization of performance patterns
- Statistical hypothesis testing

**Dataset:** StudentsPerformance.csv (1000 students, multiple performance indicators)

---

## 🛠️ Technologies & Libraries Used

### Core Libraries
- **pandas** (v1.3+) - Data manipulation and analysis
- **numpy** (v1.21+) - Numerical computing and array operations
- **matplotlib** (v3.4+) - Static data visualization
- **seaborn** (v0.11+) - Statistical data visualization
- **scikit-learn** (v0.24+) - Machine Learning models and preprocessing
- **plotly** (v5.0+) - Interactive visualizations and dashboards

### Tools & Environments
- **Jupyter Notebook** - Interactive development environment
- **Python 3.7+** - Programming language
- **VS Code** - Code editor and IDE
- **Git** - Version control system

---

## 📊 Datasets Used

| Dataset | Location | Samples | Features | Purpose |
|---------|----------|---------|----------|---------|
| Titanic | `datas (csv)/train.csv` | 891 | 12 | Classification project |
| Student Data | `day 4/student_data.csv` | Varies | Multiple | Data cleaning practice |
| Nepal Education | Kaggle | ~50K | Multiple | Exploratory data analysis |
| California Housing | sklearn.datasets | 20,640 | 8 | Regression modeling |
| Student Performance | `student performance project/StudentsPerformance.csv` | 1,000 | 8 | Statistical analysis |
| Palmer Penguins | Kaggle | 344 | 7 | Classification and clustering |

---

## 🚀 Getting Started

### Prerequisites
```bash
# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn plotly jupyter ipython
```

### Installation Steps
```bash
# Clone the repository
git clone git@github.com:sandeshbhatta495/AI.git
cd AI

# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Running Notebooks

**Option 1: Using Jupyter from VS Code**
- Open any .ipynb file in VS Code
- Click "Run All" or run individual cells

**Option 2: Using Command Line**
```bash
# Navigate to project directory
cd "day 6 (project)"

# Start Jupyter server
jupyter notebook ttt.ipynb
```

### Key Project Locations
- **Data Cleaning:** [`day 4`](day%204) notebooks
- **Linear Regression:** [`day 8/linear regression.ipynb`](day%208/linear%20regression.ipynb)
- **Titanic Project:** [`day 6 (project)/ttt.ipynb`](day%206%28project%29/ttt.ipynb)
- **Education Analysis:** [`Reference project (kaggle)/nepal-education-data-analysis-2074bs.ipynb`](Reference%20project%20%28kaggle%29/nepal-education-data-analysis-2074bs.ipynb)
- **Student Performance:** [`student performance project/students_performance.ipynb`](student%20performance%20project/students_performance.ipynb)

---

## 📈 Learning Progression

**Foundational** → **Intermediate** → **Advanced** → **Applied**

| Phase | Days | Topics | Outcomes |
|-------|------|--------|----------|
| **Foundational** | 1-3 | Python basics, NumPy | Array operations, mathematical computing |
| **Intermediate** | 4-5 | Pandas, data cleaning | Data manipulation, EDA |
| **Applied** | 6-8 | ML models, visualization | Classification, regression, insights |
| **Advanced** | 9+ | Complex projects, optimization | Real-world applications |

---

## 🔑 Key Insights & Best Practices

### Data Science Workflow
1. **Data Loading** - Import and inspect data structure
2. **Exploratory Analysis** - Understand data patterns and distributions
3. **Data Cleaning** - Handle missing values, outliers, inconsistencies
4. **Feature Engineering** - Create meaningful features from raw data
5. **Model Selection** - Choose appropriate algorithms
6. **Model Training** - Fit model to training data
7. **Evaluation** - Assess performance on test data
8. **Visualization** - Communicate findings effectively

### Key Principles
- **Data Cleaning:** 80% of project time; essential for model quality
- **Exploratory Analysis:** Visualizations reveal hidden patterns and anomalies
- **Feature Engineering:** Transforms raw data into model-ready, predictive features
- **Model Evaluation:** Use multiple metrics (accuracy, precision, recall, F1-score)
- **Cross-Validation:** Prevents overfitting and ensures generalization

---

## 📝 Author & Contact

**Sandesh Bhatta** 
- **Year:** 2025
- **Status:** Ongoing Learning & Development
- **Focus:** AI/ML, Data Science, Python Development

This repository documents a comprehensive journey from AI/ML fundamentals to practical implementation with real-world datasets.

**GitHub Repository:** [github.com/sandeshbhatta495/AI](https://github.com/sandeshbhatta495/AI)

**Repository URL:** `git@github.com:sandeshbhatta495/AI.git`

---

## ✅ Comprehensive Checklist

### Core Concepts
- [x] Python fundamentals and syntax
- [x] NumPy arrays and operations
- [x] Pandas data structures and manipulation
- [x] Data cleaning and preprocessing

### Data Analysis
- [x] Exploratory Data Analysis (EDA)
- [x] Data visualization (Matplotlib, Seaborn)
- [x] Interactive visualizations (Plotly)
- [x] Statistical analysis

### Machine Learning
- [x] Classification models
- [x] Regression models
- [x] Feature engineering
- [x] Model evaluation and validation
- [x] Hyperparameter tuning

### Projects
- [x] Titanic Survival Prediction (Classification)
- [x] California Housing (Regression)
- [x] Education Data Analysis (EDA)
- [x] Penguin Dataset (Classification)
- [x] Student Performance Analysis (Statistics)

### Development Practices
- [x] Git version control
- [x] Jupyter Notebook workflows
- [x] Code organization
- [x] Documentation

---

## 📞 Important Notes

### File Organization
- All notebooks use **Jupyter Notebook** format (.ipynb)
- CSV data files stored in **`datas (csv)`** directory
- Reference projects sourced from **Kaggle competitions**
- Work organized chronologically by day for easy tracking of progress

### Git Workflow
```bash
# After making changes:
git add .
git commit -m "Descriptive message"
git push origin main
```

### Common Issues & Solutions
- **Import errors:** Ensure all libraries installed via `pip install -r requirements.txt`
- **File path errors:** Use relative paths or absolute paths from project root
- **Jupyter kernel issues:** Restart kernel and rerun cells
- **Memory issues:** Use data sampling for large datasets

---

## 🎓 Learning Resources

### Official Documentation
- [Pandas Documentation](https://pandas.pydata.org/)
- [NumPy Documentation](https://numpy.org/)
- [Scikit-learn Guide](https://scikit-learn.org/)
- [Matplotlib Guide](https://matplotlib.org/)

### External Resources
- Kaggle Competitions
- GitHub repositories
- Online courses and tutorials
- Research papers and journals

---

## 📋 Project Statistics

- **Total Projects:** 5 major projects
- **Total Datasets:** 6 datasets
- **Total Hours:** 100+ hours of learning
- **Code Files:** 20+ Jupyter notebooks
- **Data Files:** 10+ CSV datasets
- **Lines of Code:** 5,000+ lines

---

**Last Updated:** December 10, 2025 | **Repository Status:** Active Development 🚀
**Next Steps:** Advanced ML models, Deep Learning, Real-world deployment
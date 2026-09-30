# Online Education Platform — Graduation Project

## Project Overview

This project analyzes student purchasing behavior for an online education platform.

The main business problem is to understand which courses generate the most revenue, how student purchasing behavior relates to spending, and how students can be grouped into meaningful segments.

The project combines exploratory data analysis, statistical hypothesis testing, RFM analysis, K-Means segmentation, and an interactive Streamlit dashboard.

---

## Business Question

**How can the online education platform use student purchasing data to understand student behavior and improve revenue?**

---

## Data Description

The dataset contains **400 simulated purchase transactions** covering the period from **2023 to 2024**.

The main variables are:

* `purchase_date` — date of purchase
* `course` — purchased course
* `student_id` — unique student identifier
* `price` — course price

The dataset was generated using Python for educational and portfolio purposes.

---

## Project Files

### `online_education_project.py`

Contains the main analysis, including:

* Business problem definition
* Initial data exploration
* Data cleaning
* Advanced exploratory data analysis
* Hypothesis testing
* RFM analysis
* K-Means student segmentation
* Business findings and recommendations

### `online_education_dashboard.py`

Contains the interactive Streamlit dashboard with:

* Date filters
* Course filters
* Key performance indicators
* Revenue trend
* Course performance
* Student segmentation
* Hypothesis test result
* Key business insights

### `final_report.md`

Contains the final executive summary, business insights, hypothesis-test interpretation, segmentation results, and recommendations.

---

## Key Findings

1. **Data Analysis generated the highest course revenue**, with total revenue of 5,040.
2. **Purchase frequency has a very strong positive relationship with total student spending.**
3. **Students who purchased Data Analysis had higher average total spending** than students who did not purchase it.
4. **RFM and K-Means identified four different student segments** with different purchasing behaviors.

---

## Dashboard

The Streamlit dashboard allows users to interactively explore the data using:

* Purchase date range
* Course selection
* Revenue KPIs
* Revenue trends
* Course performance
* Student segments

The dashboard was tested to confirm that the filters update the displayed analysis.

---

## How to Run the Project

### 1. Open the project folder

Open the project folder in VS Code.

### 2. Open the terminal

In VS Code, select:

**Terminal → New Terminal**

### 3. Run the main analysis

```bash
python online_education_project.py
```

### 4. Run the interactive dashboard

```bash
streamlit run online_education_dashboard.py
```

The Streamlit dashboard will open in the browser.

---

## Challenges and Solutions

### Challenge 1 — Understanding student purchasing behavior

The raw transaction-level data did not directly show which students were more valuable.

**Solution:** Student-level summaries and RFM analysis were created to measure recency, frequency, and monetary value.

### Challenge 2 — Identifying meaningful student groups

Students have different purchasing patterns, making it difficult to treat all students in the same way.

**Solution:** K-Means clustering was applied to the RFM variables to create four student segments.

### Challenge 3 — Making the analysis accessible to business users

Technical analysis alone is not sufficient for decision-making.

**Solution:** An interactive Streamlit dashboard was developed to present KPIs, trends, course performance, segmentation, and business insights in an easy-to-understand format.

---

## If I Had More Time

If more time were available, the project could be extended by:

* Using real customer transaction data instead of simulated data.
* Adding more detailed student demographics.
* Developing predictive models for future purchases.
* Adding customer lifetime value analysis.
* Connecting the dashboard to a live database so that results update automatically.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* SciPy
* Scikit-learn
* Streamlit

---

## Project Purpose

This project demonstrates how data analysis can be used to transform transaction data into actionable business insights and support data-driven decision-making.

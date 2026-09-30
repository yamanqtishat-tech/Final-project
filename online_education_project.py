import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


print("=" * 70)
print("ONLINE EDUCATION PLATFORM - FINAL PROJECT")
print("=" * 70)


# ============================================================
# STEP 1 - GENERATE ONLINE EDUCATION DATA
# ============================================================

np.random.seed(21)

n_purchases = 400

dates = pd.date_range(
    "2023-01-01",
    "2024-12-31",
    freq="D"
)

courses = [
    "Python Basics",
    "Data Analysis",
    "SQL Mastery",
    "Excel Pro",
    "Statistics 101"
]

course_prices = {
    "Python Basics": 40,
    "Data Analysis": 60,
    "SQL Mastery": 50,
    "Excel Pro": 30,
    "Statistics 101": 45
}

purchases = pd.DataFrame({
    "purchase_date": np.random.choice(
        dates,
        n_purchases
    ),
    "course": np.random.choice(
        courses,
        n_purchases
    ),
    "student_id": np.random.randint(
        1,
        121,
        n_purchases
    )
})

purchases["price"] = purchases["course"].map(
    course_prices
)

purchases = purchases.sort_values(
    "purchase_date"
).reset_index(drop=True)


print("\nFirst 10 purchases:")
print(purchases.head(10))

print("\nDataset shape:")
print(purchases.shape)
# ============================================================
# STEP 2 - QUICK EXPLORATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 - QUICK EXPLORATION")
print("=" * 70)


# Basic information
print("\nDataset Shape:")
print(purchases.shape)

print("\nDescriptive Statistics:")
print(purchases.describe())


# Number of unique students
unique_students = purchases["student_id"].nunique()

print("\nNumber of Unique Students:")
print(unique_students)


# Number of courses sold by course
course_sales = purchases["course"].value_counts()

print("\nNumber of Courses Sold by Course:")
print(course_sales)


# ============================================================
# MONTHLY SALES
# ============================================================

purchases["month"] = purchases["purchase_date"].dt.to_period("M")

monthly_sales = purchases.groupby("month")["price"].sum()

print("\nMonthly Revenue:")
print(monthly_sales)


# Convert period to timestamp for plotting
monthly_sales_plot = monthly_sales.copy()
monthly_sales_plot.index = monthly_sales_plot.index.to_timestamp()


# ============================================================
# MONTHLY SALES PLOT
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales_plot.index,
    monthly_sales_plot.values,
    marker="o"
)

plt.title("Monthly Revenue - Online Education Platform")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.grid(True)

plt.tight_layout()
plt.show()
# ============================================================
# STEP 3 - PREDICTIVE ANALYSIS (LINEAR REGRESSION)
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 - LINEAR REGRESSION")
print("=" * 70)


# Create sequential month number
monthly_regression = monthly_sales.reset_index()

monthly_regression["month_number"] = np.arange(
    1,
    len(monthly_regression) + 1
)

X = monthly_regression[["month_number"]]
y = monthly_regression["price"]


# Train Linear Regression model
regression_model = LinearRegression()

regression_model.fit(X, y)


# Predictions for existing months
monthly_regression["predicted_revenue"] = (
    regression_model.predict(X)
)


# Slope and intercept
slope = regression_model.coef_[0]
intercept = regression_model.intercept_

print("\nRegression Slope:")
print(f"{slope:.2f}")

print("\nRegression Intercept:")
print(f"{intercept:.2f}")


# R-squared
r_squared = regression_model.score(X, y)

print("\nR-squared:")
print(f"{r_squared:.4f}")


# Business interpretation
if slope > 0:
    print(
        f"\nBusiness Interpretation: Revenue shows an overall "
        f"upward trend of approximately {slope:.2f} per month."
    )
else:
    print(
        f"\nBusiness Interpretation: Revenue shows an overall "
        f"downward trend of approximately {abs(slope):.2f} per month."
    )


print(
    f"R-squared of {r_squared:.4f} indicates how much of the "
    "variation in monthly revenue is explained by the linear trend."
)


# ============================================================
# FORECAST NEXT 3 MONTHS
# ============================================================

last_month_number = monthly_regression["month_number"].max()

future_month_numbers = np.array([
    last_month_number + 1,
    last_month_number + 2,
    last_month_number + 3
]).reshape(-1, 1)

future_predictions = regression_model.predict(
    future_month_numbers
)

future_dates = pd.date_range(
    start=monthly_regression["month"].iloc[-1].to_timestamp()
    + pd.offsets.MonthBegin(1),
    periods=3,
    freq="MS"
)

forecast = pd.DataFrame({
    "month": future_dates,
    "predicted_revenue": future_predictions
})

print("\nNext 3 Months Revenue Forecast:")
print(forecast)


# ============================================================
# PLOT ACTUAL DATA + TREND + FORECAST
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_regression["month"].dt.to_timestamp(),
    y,
    marker="o",
    label="Actual Revenue"
)

plt.plot(
    monthly_regression["month"].dt.to_timestamp(),
    monthly_regression["predicted_revenue"],
    linestyle="--",
    label="Regression Trend"
)

plt.plot(
    forecast["month"],
    forecast["predicted_revenue"],
    marker="o",
    linestyle="--",
    label="3-Month Forecast"
)

plt.title(
    "Monthly Revenue: Actual, Regression Trend, and Forecast"
)

plt.xlabel("Month")
plt.ylabel("Revenue")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
# ============================================================
# STEP 4 - STUDENT SEGMENTATION (RFM + K-MEANS)
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 - STUDENT SEGMENTATION (RFM + K-MEANS)")
print("=" * 70)


# ============================================================
# RFM CALCULATION
# ============================================================

analysis_date = purchases["purchase_date"].max()

rfm = purchases.groupby("student_id").agg(
    recency=("purchase_date", lambda x:
             (analysis_date - x.max()).days),
    frequency=("course", "count"),
    monetary=("price", "sum")
).reset_index()


print("\nRFM Table - First 10 Students:")
print(rfm.head(10))

print("\nNumber of Students:")
print(len(rfm))


# ============================================================
# STANDARDIZATION
# ============================================================

rfm_values = rfm[
    ["recency", "frequency", "monetary"]
]

scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(
    rfm_values
)


print("\nStandardized RFM - First 5 Rows:")
print(rfm_scaled[:5])


# ============================================================
# ELBOW METHOD
# ============================================================

print("\n" + "=" * 70)
print("ELBOW METHOD")
print("=" * 70)

inertia_values = []

for k in range(2, 8):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(rfm_scaled)

    inertia_values.append(
        kmeans.inertia_
    )

    print(
        f"K = {k}, "
        f"Inertia = {kmeans.inertia_:.2f}"
    )


plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 8),
    inertia_values,
    marker="o"
)

plt.title("Elbow Method - Student Segmentation")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(range(2, 8))
plt.grid(True)

plt.tight_layout()
plt.show()
# ============================================================
# FINAL K-MEANS MODEL
# ============================================================

print("\n" + "=" * 70)
print("FINAL K-MEANS MODEL - K = 4")
print("=" * 70)

kmeans_final = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["segment"] = kmeans_final.fit_predict(rfm_scaled)


# ============================================================
# SEGMENT SUMMARY
# ============================================================

segment_summary = rfm.groupby("segment").agg(
    students=("student_id", "count"),
    avg_recency=("recency", "mean"),
    avg_frequency=("frequency", "mean"),
    avg_monetary=("monetary", "mean"),
    total_revenue=("monetary", "sum")
).round(2)

segment_summary["percentage"] = (
    segment_summary["students"]
    / len(rfm) * 100
).round(2)

print("\nSegment Summary:")
print(segment_summary)


# ============================================================
# SEGMENT INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("SEGMENT INTERPRETATION")
print("=" * 70)

for segment in segment_summary.index:

    row = segment_summary.loc[segment]

    if row["avg_recency"] <= 30 and row["avg_frequency"] >= 5:
        segment_name = "Champions"

    elif row["avg_recency"] >= 100:
        segment_name = "At Risk / Inactive"

    elif row["avg_frequency"] <= 4:
        segment_name = "Occasional Students"

    else:
        segment_name = "Regular Students"

    print(
        f"Segment {segment}: {segment_name} | "
        f"Students: {int(row['students'])} | "
        f"Avg Recency: {row['avg_recency']:.2f} days | "
        f"Avg Frequency: {row['avg_frequency']:.2f} | "
        f"Avg Monetary: {row['avg_monetary']:.2f}"
    )


# ============================================================
# VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    rfm["frequency"],
    rfm["monetary"],
    c=rfm["segment"],
    cmap="viridis",
    alpha=0.7
)

plt.xlabel("Purchase Frequency")
plt.ylabel("Total Spending")
plt.title("Student Segmentation - Frequency vs Spending")
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# HIGHEST REVENUE SEGMENT
# ============================================================

highest_revenue_segment = (
    segment_summary["total_revenue"].idxmax()
)

highest_revenue = (
    segment_summary.loc[
        highest_revenue_segment,
        "total_revenue"
    ]
)

print("\nHighest Revenue Segment:")
print(f"Segment {highest_revenue_segment}")

print(f"Total Revenue: {highest_revenue:.2f}")


print("\nPROJECT STATUS:")
print("Student segmentation completed successfully.")
# ============================================================
# FINAL BUSINESS SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL BUSINESS SUMMARY")
print("=" * 70)

print("\n1. REVENUE TREND")
print(
    f"Monthly revenue has an overall trend of "
    f"{slope:.2f} per month."
)

print(
    f"The regression R-squared is {r_squared:.4f}, "
    f"showing that the linear trend explains very little "
    f"of the variation in monthly revenue."
)

print("\n2. REVENUE FORECAST")

for _, row in forecast.iterrows():
    print(
        f"{row['month'].strftime('%Y-%m')}: "
        f"{row['predicted_revenue']:.2f}"
    )

print("\n3. STUDENT SEGMENTATION")

print(
    f"Total students analyzed: {len(rfm)}"
)

print(
    f"Highest revenue segment: Segment {highest_revenue_segment}"
)

print(
    f"Revenue generated by this segment: "
    f"{highest_revenue:.2f}"
)

print("\n4. BUSINESS RECOMMENDATIONS")

print(
    "Recommendation 1: Focus retention campaigns on "
    "At Risk / Inactive students to encourage new course purchases."
)

print(
    "Recommendation 2: Reward Champions with loyalty benefits, "
    "course bundles, or early access to new courses."
)

print(
    "Recommendation 3: Develop targeted promotions for "
    "Occasional Students to increase purchase frequency."
)

print("\n" + "=" * 70)
print("EXECUTIVE SUMMARY")
print("=" * 70)

print(
    "\nThe analysis combined monthly revenue forecasting with "
    "RFM-based student segmentation to understand the performance "
    "of the online education platform."
)

print(
    f"Monthly revenue showed a very weak overall linear trend "
    f"with an R-squared of {r_squared:.4f}. "
    f"The three-month forecast is approximately "
    f"{forecast['predicted_revenue'].mean():.2f} per month."
)

print(
    f"RFM and K-Means analysis identified {len(rfm)} students "
    f"across four segments. Segment {highest_revenue_segment} "
    f"generated the highest total revenue of "
    f"{highest_revenue:.2f}."
)

print(
    "The segmentation results can support targeted marketing, "
    "retention, and customer engagement strategies."
)

print("\nPROJECT COMPLETED SUCCESSFULLY!")
# ============================================================
# GRADUATION PROJECT - WEEK 8
# ONLINE EDUCATION PLATFORM
# ============================================================

print("\n" + "=" * 70)
print("GRADUATION PROJECT - WEEK 8")
print("ONLINE EDUCATION PLATFORM")
print("=" * 70)

print("\nBUSINESS PROBLEM")
print(
    "The online education platform needs to understand student "
    "purchasing behavior, identify valuable and at-risk students, "
    "and improve revenue through targeted strategies."
)

print("\nMAIN BUSINESS QUESTION")
print(
    "How can the platform use purchasing data to understand "
    "student behavior and improve revenue?"
)

print("\nHYPOTHESIS QUESTION")
print(
    "Do students who purchase the Data Analysis course have a "
    "different average total spending from students who do not?"
)

print("\nPROJECT OBJECTIVES")
print("1. Explore and clean the purchasing data.")
print("2. Identify important patterns in student purchasing behavior.")
print("3. Test whether spending differs between student groups.")
print("4. Segment students using RFM and K-Means.")
print("5. Provide business recommendations based on the analysis.")
# ============================================================
# STEP 2 - INITIAL EXPLORATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 - INITIAL EXPLORATION")
print("=" * 70)

print("\nDataset Shape:")
print(purchases.shape)

print("\nData Types:")
print(purchases.dtypes)

print("\nMissing Values:")
print(purchases.isnull().sum())

print("\nDuplicate Rows:")
print(purchases.duplicated().sum())

print("\nDescriptive Statistics:")
print(purchases.describe())

print("\nUnique Students:")
print(purchases["student_id"].nunique())

print("\nUnique Courses:")
print(purchases["course"].nunique())

print("\nCourse Distribution:")
print(purchases["course"].value_counts())

print("\nPurchase Date Range:")
print(
    purchases["purchase_date"].min(),
    "to",
    purchases["purchase_date"].max()
)

print("\nInitial Exploration Completed.")
# ============================================================
# STEP 3 - DATA CLEANING
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 - DATA CLEANING")
print("=" * 70)

# Check missing values
missing_before = purchases.isnull().sum().sum()

print("\nMissing Values Before Cleaning:")
print(missing_before)

# Check duplicates
duplicates_before = purchases.duplicated().sum()

print("\nDuplicate Rows Before Cleaning:")
print(duplicates_before)

# Check invalid prices
invalid_prices = purchases[
    (purchases["price"] <= 0)
]

print("\nInvalid Price Records:")
print(len(invalid_prices))

# Check invalid student IDs
invalid_students = purchases[
    (purchases["student_id"] <= 0)
]

print("\nInvalid Student IDs:")
print(len(invalid_students))

# Check invalid courses
valid_courses = [
    "Python Basics",
    "Data Analysis",
    "SQL Mastery",
    "Excel Pro",
    "Statistics 101"
]

invalid_courses = purchases[
    ~purchases["course"].isin(valid_courses)
]

print("\nInvalid Course Records:")
print(len(invalid_courses))


# ============================================================
# CLEANING DECISIONS
# ============================================================

print("\nCleaning Decisions:")

if missing_before == 0:
    print("- No missing values were found, so no imputation was required.")

if duplicates_before == 0:
    print("- No duplicate rows were found, so no duplicate records were removed.")

if len(invalid_prices) == 0:
    print("- No invalid price values were found.")

if len(invalid_students) == 0:
    print("- No invalid student IDs were found.")

if len(invalid_courses) == 0:
    print("- All course names match the expected course list.")


# Create a clean copy
clean_purchases = purchases.copy()

print("\nClean Dataset Shape:")
print(clean_purchases.shape)

print("\nDATA CLEANING COMPLETED.")
# ============================================================
# STEP 4 - ADVANCED EDA
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 - ADVANCED EXPLORATORY DATA ANALYSIS")
print("=" * 70)


# ============================================================
# 4.1 PRICE DISTRIBUTION + SKEWNESS
# ============================================================

print("\n4.1 PRICE DISTRIBUTION")

price_skewness = clean_purchases["price"].skew()

print("Price Skewness:", round(price_skewness, 3))

plt.figure(figsize=(8, 5))
plt.hist(clean_purchases["price"], bins=5, edgecolor="black")
plt.title("Distribution of Course Prices")
plt.xlabel("Course Price")
plt.ylabel("Number of Purchases")
plt.tight_layout()
plt.show()

print(
    "Interpretation: Course prices are concentrated around the "
    "available course price levels, with limited variation."
)


# ============================================================
# 4.2 STUDENT-LEVEL SUMMARY
# ============================================================

student_summary = clean_purchases.groupby("student_id").agg(
    purchase_frequency=("course", "count"),
    total_spending=("price", "sum"),
    last_purchase=("purchase_date", "max")
).reset_index()

print("\nStudent-Level Summary:")
print(student_summary.head())

print("\nStudent Summary Statistics:")
print(student_summary[[
    "purchase_frequency",
    "total_spending"
]].describe())


# ============================================================
# 4.3 CORRELATION MATRIX
# ============================================================

correlation_data = student_summary[
    ["purchase_frequency", "total_spending"]
]

correlation_matrix = correlation_data.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(6, 5))
plt.imshow(correlation_matrix, cmap="Blues")
plt.colorbar()

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.title("Correlation Matrix - Student Behavior")

for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(
            j,
            i,
            round(correlation_matrix.iloc[i, j], 2),
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()

print(
    "Interpretation: The correlation matrix shows the relationship "
    "between purchase frequency and total student spending."
)


# ============================================================
# 4.4 GROUPBY ANALYSIS - COURSE PERFORMANCE
# ============================================================

course_analysis = clean_purchases.groupby("course").agg(
    purchases=("course", "count"),
    total_revenue=("price", "sum"),
    average_price=("price", "mean")
).sort_values("total_revenue", ascending=False)

print("\nCourse Performance:")
print(course_analysis)


# ============================================================
# 4.5 PLOT - REVENUE BY COURSE
# ============================================================

plt.figure(figsize=(9, 5))

course_analysis["total_revenue"].plot(
    kind="bar"
)

plt.title("Total Revenue by Course")
plt.xlabel("Course")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(
    "Interpretation: The chart compares total revenue generated "
    "by each course and identifies the strongest revenue contributors."
)


# ============================================================
# 4.6 PLOT - PURCHASE FREQUENCY BY COURSE
# ============================================================

plt.figure(figsize=(9, 5))

course_analysis["purchases"].sort_values(
    ascending=False
).plot(kind="bar")

plt.title("Number of Purchases by Course")
plt.xlabel("Course")
plt.ylabel("Number of Purchases")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(
    "Interpretation: This chart shows which courses have the highest "
    "number of purchases and therefore attract more student demand."
)


# ============================================================
# 4.7 PLOT - STUDENT FREQUENCY VS SPENDING
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    student_summary["purchase_frequency"],
    student_summary["total_spending"]
)

plt.title("Student Purchase Frequency vs Total Spending")
plt.xlabel("Purchase Frequency")
plt.ylabel("Total Spending")

plt.tight_layout()
plt.show()

print(
    "Interpretation: Students with more purchases generally have "
    "higher total spending, indicating a relationship between "
    "purchase frequency and student value."
)


# ============================================================
# 4.8 MONTHLY REVENUE TREND
# ============================================================

monthly_revenue = (
    clean_purchases
    .groupby("month")["price"]
    .sum()
)

plt.figure(figsize=(12, 5))

monthly_revenue.plot()

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print(
    "Interpretation: Monthly revenue fluctuates across the two-year "
    "period, indicating changes in purchasing activity over time."
)


print("\nSTEP 4 - ADVANCED EDA COMPLETED.")
# ============================================================
# STEP 5 - HYPOTHESIS TEST
# ============================================================

from scipy.stats import ttest_ind

print("\n" + "=" * 70)
print("STEP 5 - HYPOTHESIS TEST")
print("=" * 70)

# Identify students who purchased Data Analysis
data_analysis_students = clean_purchases[
    clean_purchases["course"] == "Data Analysis"
]["student_id"].unique()

# Create student groups
group_data_analysis = student_summary[
    student_summary["student_id"].isin(data_analysis_students)
]["total_spending"]

group_other_courses = student_summary[
    ~student_summary["student_id"].isin(data_analysis_students)
]["total_spending"]

# Calculate group means
mean_data_analysis = group_data_analysis.mean()
mean_other_courses = group_other_courses.mean()

print("\nGroup 1 - Data Analysis Students")
print("Number of Students:", len(group_data_analysis))
print("Average Total Spending:", round(mean_data_analysis, 2))

print("\nGroup 2 - Students Without Data Analysis")
print("Number of Students:", len(group_other_courses))
print("Average Total Spending:", round(mean_other_courses, 2))

# Independent T-Test
t_stat, p_value = ttest_ind(
    group_data_analysis,
    group_other_courses,
    equal_var=False
)

print("\nT-Test Results:")
print("T-Statistic:", round(t_stat, 3))
print("P-Value:", round(p_value, 4))

# Significance level
alpha = 0.05

print("\nHypotheses:")
print("H0: There is no significant difference in average total spending.")
print("H1: There is a significant difference in average total spending.")

if p_value < alpha:
    print("\nConclusion:")
    print(
        "Reject H0. There is a statistically significant difference "
        "in average total spending between the two groups."
    )
else:
    print("\nConclusion:")
    print(
        "Fail to reject H0. There is no statistically significant "
        "difference in average total spending between the two groups."
    )

print("\nSTEP 5 - HYPOTHESIS TEST COMPLETED.")
# ============================================================
# STEP 6 - ADDITIONAL ANALYSIS
# RFM + K-MEANS CUSTOMER SEGMENTATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 - ADDITIONAL ANALYSIS")
print("RFM + K-MEANS CUSTOMER SEGMENTATION")
print("=" * 70)


# ============================================================
# 6.1 CREATE RFM DATA
# ============================================================

analysis_date = clean_purchases["purchase_date"].max()

rfm = clean_purchases.groupby("student_id").agg(
    recency=("purchase_date", lambda x: (analysis_date - x.max()).days),
    frequency=("purchase_date", "count"),
    monetary=("price", "sum")
).reset_index()

print("\nRFM Data:")
print(rfm.head())

print("\nRFM Statistics:")
print(rfm[["recency", "frequency", "monetary"]].describe())


# ============================================================
# 6.2 STANDARDIZE RFM VARIABLES
# ============================================================

rfm_features = rfm[
    ["recency", "frequency", "monetary"]
]

scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(rfm_features)


# ============================================================
# 6.3 K-MEANS CLUSTERING
# ============================================================

kmeans = KMeans(
    n_clusters=4,
    random_state=21,
    n_init=10
)

rfm["segment"] = kmeans.fit_predict(rfm_scaled)


# ============================================================
# 6.4 SEGMENT SUMMARY
# ============================================================

segment_summary = rfm.groupby("segment").agg(
    students=("student_id", "count"),
    avg_recency=("recency", "mean"),
    avg_frequency=("frequency", "mean"),
    avg_monetary=("monetary", "mean"),
    total_revenue=("monetary", "sum")
)

segment_summary["percentage"] = (
    segment_summary["students"] /
    len(rfm) * 100
)

print("\nSegment Summary:")
print(segment_summary.round(2))


# ============================================================
# 6.5 SEGMENT VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 5))

segment_summary["total_revenue"].plot(
    kind="bar"
)

plt.title("Revenue by Student Segment")
plt.xlabel("Segment")
plt.ylabel("Total Revenue")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ============================================================
# 6.6 BUSINESS INTERPRETATION
# ============================================================

print("\nBusiness Interpretation:")

highest_revenue_segment = segment_summary[
    "total_revenue"
].idxmax()

highest_revenue = segment_summary.loc[
    highest_revenue_segment,
    "total_revenue"
]

highest_frequency_segment = segment_summary[
    "avg_frequency"
].idxmax()

print(
    f"- Segment {highest_revenue_segment} generates the highest "
    f"total revenue: {highest_revenue:.2f}."
)

print(
    f"- Segment {highest_frequency_segment} has the highest "
    f"average purchase frequency."
)

print(
    "- RFM segmentation helps the platform identify groups of "
    "students with different purchasing behaviors."
)

print(
    "- The segments can support targeted retention, engagement, "
    "and promotional strategies."
)

print("\nSTEP 6 - ADDITIONAL ANALYSIS COMPLETED.")
# ============================================================
# WEEK 8 - FINAL PROGRESS SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("WEEK 8 - FINAL PROGRESS SUMMARY")
print("=" * 70)

print("\nKEY FINDINGS:")

print(
    "1. Data Analysis was the highest-revenue course, generating "
    "5040 in total revenue."
)

print(
    "2. Purchase frequency and total student spending had a very "
    "strong positive correlation (0.966)."
)

print(
    "3. RFM and K-Means identified four student segments, with "
    "Segment 0 generating the highest revenue at 9600 (44.07%)."
)

print("\nWEEK 9 PLAN:")

print(
    "1. Build an interactive dashboard using Streamlit."
)

print(
    "2. Present key KPIs, revenue trends, course performance, "
    "and student segments."
)

print(
    "3. Prepare the final report with findings, business "
    "interpretations, and recommendations."
)

print("\nWEEK 8 RAW ANALYSIS COMPLETED.")
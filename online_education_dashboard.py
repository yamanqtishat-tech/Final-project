import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Online Education Dashboard",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🎓 Online Education Platform Dashboard")

st.markdown(
    "Interactive dashboard for analyzing student purchasing behavior, "
    "course performance, revenue, and student segments."
)


# ============================================================
# CREATE DATA
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


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🔎 Filters")

min_date = purchases["purchase_date"].min().date()
max_date = purchases["purchase_date"].max().date()

selected_dates = st.sidebar.date_input(
    "Purchase Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

selected_courses = st.sidebar.multiselect(
    "Select Course",
    options=courses,
    default=courses
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_data = purchases.copy()

if len(selected_dates) == 2:

    start_date = pd.Timestamp(
        selected_dates[0]
    )

    end_date = pd.Timestamp(
        selected_dates[1]
    )

    filtered_data = filtered_data[
        (filtered_data["purchase_date"] >= start_date)
        &
        (filtered_data["purchase_date"] <= end_date)
    ]

filtered_data = filtered_data[
    filtered_data["course"].isin(
        selected_courses
    )
]


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_revenue = filtered_data["price"].sum()

active_students = filtered_data[
    "student_id"
].nunique()

total_purchases = len(filtered_data)

if active_students > 0:
    average_spending = (
        total_revenue / active_students
    )
else:
    average_spending = 0


# ============================================================
# KPI DISPLAY
# ============================================================

st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Revenue",
    f"${total_revenue:,.0f}"
)

col2.metric(
    "Active Students",
    f"{active_students:,}"
)

col3.metric(
    "Total Purchases",
    f"{total_purchases:,}"
)

col4.metric(
    "Avg Spending / Student",
    f"${average_spending:,.2f}"
)


# ============================================================
# NO DATA WARNING
# ============================================================

if filtered_data.empty:

    st.warning(
        "No data available for the selected filters."
    )

    st.stop()


# ============================================================
# REVENUE TREND
# ============================================================

st.subheader("📈 Revenue Trend")

filtered_data["month"] = (
    filtered_data["purchase_date"]
    .dt.to_period("M")
)

monthly_revenue = (
    filtered_data
    .groupby("month")["price"]
    .sum()
)

monthly_revenue.index = (
    monthly_revenue.index
    .astype(str)
)

fig1, ax1 = plt.subplots(
    figsize=(12, 5)
)

ax1.plot(
    monthly_revenue.index,
    monthly_revenue.values,
    marker="o"
)

ax1.set_title(
    "Monthly Revenue Trend"
)

ax1.set_xlabel(
    "Month"
)

ax1.set_ylabel(
    "Revenue"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

st.pyplot(fig1)


# ============================================================
# COURSE PERFORMANCE
# ============================================================

st.subheader("📚 Course Performance")

course_revenue = (
    filtered_data
    .groupby("course")["price"]
    .sum()
    .sort_values(
        ascending=False
    )
)

fig2, ax2 = plt.subplots(
    figsize=(9, 5)
)

course_revenue.plot(
    kind="bar",
    ax=ax2
)

ax2.set_title(
    "Revenue by Course"
)

ax2.set_xlabel(
    "Course"
)

ax2.set_ylabel(
    "Revenue"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

st.pyplot(fig2)


# ============================================================
# STUDENT SEGMENTATION
# ============================================================

st.subheader("👥 Student Segments")

analysis_date = filtered_data[
    "purchase_date"
].max()

rfm = filtered_data.groupby(
    "student_id"
).agg(
    recency=(
        "purchase_date",
        lambda x: (
            analysis_date - x.max()
        ).days
    ),
    frequency=(
        "purchase_date",
        "count"
    ),
    monetary=(
        "price",
        "sum"
    )
).reset_index()


if len(rfm) >= 4:

    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm[
            [
                "recency",
                "frequency",
                "monetary"
            ]
        ]
    )

    kmeans = KMeans(
        n_clusters=4,
        random_state=21,
        n_init=10
    )

    rfm["segment"] = (
        kmeans.fit_predict(
            rfm_scaled
        )
    )

    segment_revenue = (
        rfm
        .groupby("segment")["monetary"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    fig3, ax3 = plt.subplots(
        figsize=(8, 5)
    )

    segment_revenue.plot(
        kind="bar",
        ax=ax3
    )

    ax3.set_title(
        "Revenue by Student Segment"
    )

    ax3.set_xlabel(
        "Student Segment"
    )

    ax3.set_ylabel(
        "Revenue"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    st.pyplot(fig3)

else:

    st.info(
        "Not enough students for segmentation."
    )


# ============================================================
# HYPOTHESIS TEST RESULT
# ============================================================

st.subheader("📌 Hypothesis Test Result")

st.info(
    "Students who purchased the Data Analysis course had a "
    "higher average total spending than students who did not "
    "purchase the course. The statistical test from Week 8 "
    "showed a significant difference between the two groups."
)


# ============================================================
# KEY BUSINESS INSIGHTS
# ============================================================

st.subheader("💡 Key Business Insights")

st.markdown(
    """
**1. Data Analysis generated the highest course revenue**  
Data Analysis generated 5,040 in revenue during the analyzed period.

**2. Purchase frequency is strongly related to student spending**  
Students with more purchases tend to generate higher total spending.

**3. Student segmentation reveals different purchasing behaviors**  
RFM and K-Means identified four distinct student groups that can
support targeted engagement and retention strategies.
"""
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Online Education Platform | Week 9 Graduation Project"
)
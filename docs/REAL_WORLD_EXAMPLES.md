# Real-World Analysis Examples

These examples demonstrate OpenDsStar's ability to autonomously plan and execute complex data analysis. The agent receives a question and a set of data files — it discovers which files are relevant, plans the analysis, writes and executes code, debugs errors, and interprets the results with domain-aware reasoning.

**Setup:**
- **Data:** 80 CSV files from [DataBench](https://huggingface.co/datasets/cardiffnlp/databench) (Forbes billionaires, Titanic passengers, FIFA players, tornado records, NYC taxi trips, health data, and more)
- **Model:** Claude Opus 4.6
- **Agent:** OpenDsStar (DS-Star)

---

## 1. Cross-Dataset Join: Forbes Billionaires x FIFA Players

**What makes this hard:** The agent must identify that two separate datasets (Forbes billionaires and FIFA players) share a common dimension (country/nationality), filter each independently, then join and aggregate across them — without any hint about which files to use.

> **Q: What is the average age of Forbes billionaires from countries that also have at least 5 FIFA players with an overall rating above 85?**

**OpenDsStar's answer:**

The average age of Forbes billionaires from countries that also have at least 5 FIFA players with an overall rating above 85 is approximately **66.39 years**.

The qualifying countries (those with at least 5 FIFA players rated above 85) are: **Argentina, Brazil, England, France, Germany, and Spain**. A total of 231 Forbes billionaires come from these six countries, and their average age is ~66.4 years.

---

## 2. Trend Analysis: Tornado Severity Over Time

**What makes this hard:** The agent must perform time-series analysis on 67,000+ records, compute decade-level aggregates, run statistical tests, and — critically — provide domain-aware interpretation that goes beyond the raw numbers.

> **Q: Is there a trend in tornado severity over time?**

**OpenDsStar's answer:**

Yes, there is a statistically significant decreasing trend in tornado severity over time, based on an analysis of 67,558 tornado records from 1950 to 2021.

**Key Findings:**
- Linear regression of mean magnitude per year yields a slope of **-0.014 per year**
- The correlation is strong (**r = -0.89**) and highly significant (**p ~ 4.5 x 10<sup>-26</sup>**)

**Decade-by-Decade Average Magnitude:**

| Decade | Mean Magnitude | Tornado Count |
|--------|---------------|---------------|
| 1950s  | 1.31          | 4,793         |
| 1960s  | 1.19          | 6,811         |
| 1970s  | 1.09          | 8,579         |
| 1980s  | 0.84          | 8,195         |
| 1990s  | 0.56          | 12,137        |
| 2000s  | 0.49          | 12,721        |
| 2010s  | 0.61          | 11,629        |
| 2020s  | 0.66          | 2,088         |

**Interpretation & Caveats:**

The data shows that the average recorded tornado magnitude has dropped substantially — from about 1.3 in the 1950s to roughly 0.5-0.7 in recent decades. However, this trend should be interpreted with caution:

- **Detection improvements:** Modern radar (especially Doppler and dual-polarization) and storm-spotter networks detect many more weak (F/EF-0) tornadoes that would have gone unrecorded in earlier decades. The number of recorded tornadoes has more than doubled (from ~4,800 in the 1950s to ~12,700 in the 2000s), largely driven by an increase in weak tornado reports.
- **Scale changes:** The transition from the original Fujita (F) scale to the Enhanced Fujita (EF) scale in 2007 may also affect comparisons.
- **Reporting bias:** In earlier decades, only the more severe and damaging tornadoes tended to be reported, inflating the historical average magnitude.

**Bottom line:** While the data shows a clear statistical decline in average tornado severity, much of this trend is likely driven by improved detection of weak tornadoes rather than a true decrease in the intensity of the strongest tornadoes. The number of violent (F/EF-4 and F/EF-5) tornadoes has remained relatively stable over time.

---

## 3. Survival Analysis: Titanic Women by Class and Family Status

**What makes this hard:** The agent must segment data across three dimensions (gender, class, family status), compute survival rates for each combination, and draw meaningful conclusions about the interaction effects.

> **Q: On the Titanic, what was the survival rate for women traveling alone vs with family by class?**

**OpenDsStar's answer:**

| Class | Travel Status | Survival Rate | Survived / Total |
|-------|--------------|---------------|------------------|
| 1st   | Alone        | 97.06%        | 33 / 34          |
| 1st   | With Family  | 96.67%        | 58 / 60          |
| 2nd   | Alone        | 90.62%        | 29 / 32          |
| 2nd   | With Family  | 93.18%        | 41 / 44          |
| 3rd   | Alone        | 61.67%        | 37 / 60          |
| 3rd   | With Family  | 41.67%        | 35 / 84          |

**Key takeaways:**

- **1st Class:** Survival rates were nearly identical (~97%) regardless of whether women traveled alone or with family.
- **2nd Class:** Rates were also very similar, with women traveling with family having a slightly higher survival rate (93.18%) compared to those alone (90.62%).
- **3rd Class:** This is where the most notable difference appears — women traveling alone had a significantly higher survival rate (61.67%) than those traveling with family (41.67%). This could suggest that women with family in 3rd class may have delayed evacuation while trying to stay with family members, or faced other logistical challenges.

Overall, class was the dominant factor in survival, with 1st and 2nd class women surviving at very high rates regardless of travel status. In 3rd class, traveling alone was actually associated with better survival outcomes.

(Based on 314 female passengers in the dataset: 126 traveling alone and 188 traveling with family.)

---

## 4. Health Analytics: Stroke Risk by Comorbidity Combination

**What makes this hard:** The agent must compute stroke rates across a combinatorial explosion of three binary/categorical variables (2 x 2 x 4 = 16 groups), handle small sample sizes, and identify meaningful patterns in the results.

> **Q: What is the stroke rate for each combination of hypertension, heart disease, and smoking status?**

**OpenDsStar's answer:**

| Hypertension | Heart Disease | Smoking Status   | Stroke Rate | Total | Strokes |
|:------------|:-------------|:----------------|:------------|------:|--------:|
| No          | No           | Unknown          | 2.42%       | 1,449 | 35      |
| No          | No           | formerly smoked  | 6.06%       | 709   | 43      |
| No          | No           | never smoked     | 3.14%       | 1,593 | 50      |
| No          | No           | smokes           | 3.72%       | 672   | 25      |
| No          | Yes          | Unknown          | 7.69%       | 52    | 4       |
| No          | Yes          | formerly smoked  | 10.00%      | 60    | 6       |
| No          | Yes          | never smoked     | 6.67%       | 75    | 5       |
| No          | Yes          | smokes           | 0.00%       | 28    | 0       |
| Yes         | No           | Unknown          | 4.76%       | 63    | 3       |
| Yes         | No           | formerly smoked  | 14.29%      | 112   | 16      |
| Yes         | No           | never smoked     | 8.33%       | 120   | 10      |
| Yes         | No           | smokes           | 10.00%      | 60    | 6       |
| Yes         | Yes          | Unknown          | 0.00%       | 10    | 0       |
| Yes         | Yes          | formerly smoked  | **20.59%**  | 34    | 7       |
| Yes         | Yes          | never smoked     | 14.81%      | 27    | 4       |
| Yes         | Yes          | smokes           | 15.38%      | 26    | 4       |

**Key observations:**

- The highest stroke rate (**20.59%**) is among individuals with both hypertension and heart disease who formerly smoked.
- Having both hypertension and heart disease consistently elevates stroke rates across all smoking categories.
- Among those with hypertension but no heart disease, formerly smoked individuals have the highest stroke rate (14.29%).
- The combination of no hypertension, heart disease present, and current smoker shows a 0% stroke rate, though the sample size is very small (28 individuals).

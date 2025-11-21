# Uplift Modelling on the Hillstrom Email Marketing Dataset

## Predicting Incremental Impact of Marketing Emails Using Uplift Modeling

### 1. Project Overview 🔍 

This project analyzes the impact of two email marketing campaigns (“Men’s Email” and “Women’s Email”) using uplift modeling, a causal machine learning technique that predicts incremental effects rather than just outcomes.

The dataset, provided by Kevin Hillstrom (MineThatData), contains 64,000 customers randomly assigned to:

* Men’s Email (treatment A)

* Women’s Email (treatment B)

* No Email (control)

The **goal** is to answer:

➡️ “Which customers should receive which email to maximize incremental conversions/spend?”

➡️ “Which campaign performs better and for which segments?”

➡️ “How much incremental gain can we expect with an optimal policy?”

###  2. Key Concepts 🎯

#### Uplift Modeling

Traditional ML predicts P(Y = 1 | X).

Uplift modeling predicts incremental change:

Uplift(𝑥)=𝑃(𝑌=1∣𝑇=1,𝑋=𝑥)−𝑃(𝑌=1∣𝑇=0,𝑋=𝑥)

Uplift(x)=P(Y=1∣T=1,X=x)−P(Y=1∣T=0,X=x)

Where:

T=1 → email sent

T=0 → no email

Y → conversion

This allows us to identify:

**Persuadables** → customers likely to convert only because of the email

**Sure Things** → would convert anyway

**Lost Causes** → won’t convert regardless

**Sleeping Dogs** → emailing hurts them

### 3. Methodology 🧮 

We train two S-Learners:

**Model 1: Men’s Email vs Control**

Predicts uplift for sending Men’s Email.

**Model 2: Women’s Email vs Control**

Predicts uplift for sending Women’s Email.

For each customer, we obtain:

* uplift_men

* uplift_women

* together used to choose the best action.

**Models used:**

✔ Logistic Regression

✔ One-hot encoding for categorical features

✔ No leakage 

###  📊 4. Main Results

#### 4.1 Average Treatment Effect (ATE)

| Campaign          | Incremental Visit | Incremental Conversion | Incremental Spend |
| ----------------- | ----------------- | ---------------------- | ----------------- |
| **Men’s Email**   | Positive          | **Higher**             | Positive          |
| **Women’s Email** | Positive          | Lower                  | Mixed             |

➡️ Men’s campaign outperforms Women’s overall.

#### 4.2 Uplift@K (Top K% Customers)

| Metric     | Men’s Email | Women’s Email |
| ---------- | ----------- | ------------- |
| uplift@30% | **0.01056** | 0.00421       |
| uplift@10% | **0.01405** | 0.00593       |

➡️ At every threshold, Men’s Email generates 2–3× more incremental conversions.

#### 4.3 Uplift Curves

Both curves show a smooth gain above the random baseline, expected because:

logistic regression produces smooth uplift distributions

dataset is balanced across treatment arms

But Men’s uplift curve is higher than Women’s.


###  5. Optimal Email Policy

We compute for each customer:

* best_uplift = max(uplift_men, uplift_women)

* best_action = argmax([uplift_men, uplift_women, 0])


Where:

* if uplift_men is highest → send Men’s Email

* if uplift_women is highest → send Women’s Email

* if uplift ≤ 0 → send No Email

**Policy Summary**

| Action        | % of Customers | Interpretation                      |
| ------------- | -------------- | ----------------------------------- |
| Men’s Email   | ~33–40%        | Best performing treatment           |
| Women’s Email | ~10–20%        | Useful for small segments           |
| No Email      | Remainder      | Emailing them reduces effectiveness |


✔ This policy maximizes incremental conversions.

✔ Matches business logic: fewer emails, better ROI.

###  💰 6. ROI Simulation

Using simple assumed values:

* cost_per_email = $0.02

* margin_per_conversion = $10

We simulate expected incremental profit by applying the policy.

Result:

➡️ Policy-based targeting significantly outperforms blast campaigns

➡️ Improved profitability even with same total email volume

###  🔬 7. Segment Insights

**Key findings**:

* High spenders (history > $160) respond strongly to Men’s Email

* Multi-channel customers (Web + Phone) show higher uplift

* Urban customers show better incremental response

* New customers (newbie=1) respond less predictively to either email

These insights can guide marketing strategies beyond uplift modeling.

###  🏁 8. Conclusions

* Uplift modeling reveals which customers truly benefit from marketing.

* Men’s Email campaign is **substantially more effective** than Women’s.

* Optimal targeting strategy **reduces wasted emails** while **maximizing incremental conversions**.

* Even simple models (Logistic Regression) produce actionable uplift insights.

* This methodology is fully extensible to XGBoost, Causal Forest, and Deep Uplift models.

# Credit Memo - APP-00001
_AI-generated draft for credit officer review [MRM-2]._

## Summary
- **Applicant ID:** APP-00001
- **Date:** 2026-10-15
- **Loan Type / Purpose:** Home / Purchase of first home
- **Loan Amount:** INR 29,03,000
- **Loan Term & Interest Rate:** 300 months at 8.56%
- **Annual / Monthly Income:** INR 8,83,000 / INR 73,583
- **Proposed EMI & Existing Debt:** EMI of INR 23,493; existing monthly debt of INR 7,300
- **Credit Score & Risk Grade:** 698 (prime); Risk Grade A (PD 2.7%)
- **Approval Authority:** Branch credit officer [RLP-6]
- **Decision:** APPROVE
- **Model / Version:** Logistic Regression (pd-20261014205833) [RLP-8]

## Risk Drivers
The risk model identified the following key factors influencing the application:
- **Debt-to-income ratio (41.9%):** Increases risk [MRM-1].
- **Loan-to-value ratio (56.4%):** Increases risk [MRM-1].
- **Loan tenure (300 months):** Decreases risk [MRM-1].
- **Years in current employment (11.1):** Decreases risk [MRM-1, KYC-3].
- **Unsecured personal loan (no):** Decreases risk [MRM-1].
- **Home loan (yes):** Decreases risk [MRM-1].

## Policy Assessment
All policy checks passed successfully:
- **RLP-2 (Debt-to-Income Limits):** DTI of 41.9% falls within the 40-50% band. A documented mitigant is present (stable employment of 11.1 years, exceeding the five-year requirement) [RLP-2].
- **RLP-3 (Credit Score Thresholds):** Score of 698 is classified as prime [RLP-3].
- **RLP-4 (Delinquency History):** 0 delinquencies in 12 months and a maximum DPD of 0 in 24 months [RLP-4].
- **RLP-5 (Loan-to-Value for Secured Loans):** LTV of 56.4% satisfies the 80% maximum cap for home loans [RLP-5].
- **KYC-3 (Employment Stability):** Employment duration of 11.1 years exceeds the one-year minimum [KYC-3].
- **AML-1 (Amount vs Income Profile):** Requested amount of INR 29,03,000 is consistent with the applicant's income profile [AML-1].
- **RLP-6 (Risk Grading and Decision Authority):** PD of 2.7% maps to Grade A, which permits branch-level approval [RLP-6].

## Recommendation
APPROVE the home loan application for INR 29,03,000 at 8.56% interest over 300 months. All policy checks have passed, and the appropriate branch credit officer has decision authority [RLP-6, RLP-8].
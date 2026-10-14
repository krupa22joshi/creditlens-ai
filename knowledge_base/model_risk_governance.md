# Model Risk Management and Responsible AI Guidelines (Sample)

> Illustrative governance guidelines for a fictional bank, written for demonstration purposes.

## MRM-1 Model Explainability
Any model used in a credit decision must provide reason codes for each individual decision.
The top factors that increased the applicant's risk must be communicated in plain language
to the credit officer. Feature attribution methods such as SHAP are acceptable.

## MRM-2 Human in the Loop
Model outputs and AI-generated narratives are decision support only. A credit officer must
review and approve every recommendation. AI-generated content must be clearly labelled as such.

## MRM-3 Generative AI Controls
Generative AI used to draft credit documentation must be grounded in approved sources.
Every policy statement in a generated document must cite the policy clause it relies on.
Numerical values in generated text (PD, DTI, income, loan amount) must exactly match the
system of record. Outputs that fail validation must not be shown as final.

## MRM-4 Data Leakage and Feature Governance
Features that contain information not available at the time of application (for example,
post-disbursement repayment status or collection flags) must not be used for training.
Feature lists must be reviewed and approved before deployment.

## MRM-5 Fairness
Protected attributes such as gender, religion, and caste must not be used as model inputs.
Model performance should be monitored across customer segments for disparate impact.

## MRM-6 Monitoring
Model discrimination (ROC-AUC) should be monitored monthly. A drop of more than 5 percentage
points from the validation baseline triggers a model review.

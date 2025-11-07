# Cancer-Detection-Analysis
A simple classification ML that using a logistic regression predicts based on the features the probability for the presence of cancer.

**Product Goal:** To use an ML model as a preliminary cancer detection tool while upholding maximum patient safety .

**Key Decision:** The strategy was to prioritize Recall (97.5%) over general Accuracy (96.0%) to ensure almost no true cancer cases are missed (minimizing catastrophic False Negatives).

**Safety Workflow (HITL)**: A Human-in-the-Loop (HITL) system was designed to integrate the model safely, sending uncertain predictions to mandatory human review within 4 hours.

**Outcome:** A risk-optimized deployment plan that augments (not replaces) medical professionals, minimizing liability and ethical risks.

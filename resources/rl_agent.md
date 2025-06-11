# RL Agent

## Config File

- This is the dynamic file the rl agent will update, e.g:
```yml
thresholds:
    # Low being less than 0.3
    moderate: 0.3
    high: 0.7
explanation:
    # How many shap values to display in the report/give an explanation of
    top_k: 3
```
## Actions

- The possible actions the **RL Agent** can take involve modifying the above config file. These include:
    - **Risk Thresholds:**
        1. `INCREASE_MODERATE_THRESHOLD`
        2. `DECREASE_MODERATE_THRESHOLD`
        3. `INCREASE_HIGH_THRESHOLD`
        4. `DECREASE_HIGH_THRESHOLD`
    - **SHAP/XAI**
        - Note these can easily later be extended to increase/decrease the verbosity of the report
        5. `INCREASE_TOP_K`
        6. `DECREASE TOP_K`
    - **Do Nothing:**
        7. `NOP`

- The step size for the **Risk_Thresholds** should be ~ 0.05, leaving 20 discrete steps.
- The step size for the **SHAP/XAI** will be 1.
- Aggregate actions could also be later added(e.g adjust both the risk threshold and verbosity).

## CSV Patient Data

### Structure

- Each patient will be given their own csv file containing their history.
- The patients history columns will contain the features found in [/resources/data/heart_disease_data.csv](../resources/data/heart_disease_data.csv), plus additional info used by the RL Agent (Q-Learning).
- To satisfy the MDP Criterion, the additional columns must include `(State, Action, Reward, Next_State)`
    - **State**: Prediction probability mapped to 10 discrete levels:
        - **0**: [0, 0.1)
        - **1**: [0.1, 0.2)
        - .
        - .
        - **9**: [0.9, 1.0)
    - **Reward**: +1/-1/0
    - **Action**: INCREASE_MODERATE_THRESHOLD, ... DESCREASE_TOP_K, NOP
- For example, a patient file would have the following additional columns:
    ```
    Id, Date, Prediction, State, Action, Reward, Eval_Date, Eval_Prediction, Next_State, True_Outcome
    ```
- **Note:** State can be removed since it is easily derivable from Prediction via floor(Prediction*10).

### Preprocessing and Synthesizing

- The [dataset](../resources/data/heart_disease_data.csv) will be stretched to account for multiple patients, over varying timeframes.
- Since it is difficult to stretch the data and not introduce biases, noise, and instability, I suggest the following approach:
    1. Firstly compute pairs of "Present" and "Absent" entries in the csv. These values will be used as the anchors of the patient's synthetic history, a patient will drift at some rate, for some period of time, landing at the other anchor. This atleast ostensibly provides realistic inflection points of when someone "gets heart disease".
        - **Improving**: Present->Absent
        - **Worsening**: Absent->Present
        - **Stable**: No Change 
    2. I propose a simple equation for generating this "drift" to avoid the RL algorithm overfitting my beliefs, and to ensure that the RL Agent is indeed learning a policy based on state dependent action outcomes with some semblance of real world stochasticity. For example:
    ```
    P(t+1) = (1-alpha)*P_start(t) + alpha*P_end(t) + gaussian_noise + action_dependent_noise 
    ```
    Where P, P_start, and P_end are prediction probabilities, and alpha is 1/t. An action dependent noise could be a simple in-python check:
    ```python
    if action in (DECREASE_MODERATE_THRESHOLD, DECREASE_HIGH_THRESHOLD):
        p_next = p_base - np.random.uniform(delta_min, delta_max)
    elif action in (INCREASE_MODERATE_THRESHOLD ,DECREASE_HIGH_THRESHOLD):
        p_next = p_base + np.random.uniform(delta_min, delta_max)
    else:
        p_next = p_base
    ``` 

    - The extra complexity associate with the drift equation doesn't immediately feel too substantial to me, it should be testable with a few unit tests. But if it does become too difficult I can always fallback to a simpler increase without the anchors.


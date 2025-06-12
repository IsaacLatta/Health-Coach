# RL Agent System

## Flow

### Tasks

The reinforcement agent component will need to complete various tasks on each patient, these include:

- **Load configuration** (`load_configuration`)
    Load the current configuration for the system's recommendations and shap values.

- **Fetch patient history** (`fetch_patient_data`)  
    Retrieve the patient’s most recent CSV record so we can compare their previous state and action.

- **Validate fetched data** (`validate_patient_data`)  
    Ensure the CSV row has all required fields (`probability`, `state`, `action`) and correct types.

- **Compute current state** (`compute_current_state`)  
    Discretize the new model probability into an integer state (0–9) for the MDP.

- **Compute reward** (`apply_reward`)  
    Calculate the reward (+1/0/–1) by comparing the previous state to the current state.

- **Select next action** (`compute_action`)  
    Use a greedy policy over the Q-table to pick the configuration adjustment.

- **Validate proposed action** (`validate_action`)  
    Check that the chosen action exists and won’t push any threshold or `top_k` out of bounds.

- **Apply configuration update** (`update_configuration`)  
    Mutate the YAML config file according to the chosen action and clamp to valid bounds.

- **Validate configuration file** (`validate_configuration_file`)  
    Confirm the YAML still contains the expected sections/keys after mutation.

- **Update patient record** (`update_patient_data`)  
    Back-fill the last row with reward and next state, then append a fresh row with blank eval fields.

### Tools

The necessary tools needed to complete the above tasks are proposed below. Each set of correlated tasks involve some sort of validating of agent processed data. This pattern could easily be extended later into a *Guardrails Agent(s)*, where after each tasks agents post their output to some sort of _"Validator"_ that gracefully handles any errors redirecting the system's flow.

- **verify_patient_history**
    Check whether a history file already exists for this patient, if not create it and enter the first row

- **read_patient_history**  
    Read the last CSV row for a given patient `id` file; error if file missing or malformed.

- **validate_patient_data**  
    Verify the fetched row includes `probability`, `state`, and `action` fields with correct types.

- **discretize_probability**  
    Map a float from [0,1] into an integer state 0–9.

- **calculate_reward**  
    Return +1/0/–1 by comparing `prev_state` to `current_state`.

- **select_policy_action**  
    Load Q-table, perform greedy action selection on `current_state`.

- **validate_action**  
    Ensure `new_action` is in the allowed set and won’t drive any config value out of valid range.

- **apply_action_to_config**  
    Load & mutate the YAML (thresholds/top_k), clamp values, and save.

- **validate_config_file**  
    Reload YAML and check it still contains `thresholds`, `explanation.top_k`, etc.

- **update_patient_history**  
    Open CSV, back-fill the previous row’s blank columns, then append a new row with blank _next_ columns, populated _current_ columns.

### Agents

Three agents are hypothesized, where each agent's tasks are correlated.

- **Data Manager Agent**  
  Handles all CSV I/O and validation.  
  **Tasks**: `fetch_patient_data`, `validate_patient_data`, `update_patient_data`

- **RL Agent**  
  Computes state, reward, and next action.  
  **Tasks**: `compute_current_state`, `apply_reward`, `compute_action`, `validate_action`

- **Configuration Agent**  
  Validates and applies config changes.  
  **Tasks**: `load_configuration`, `update_configuration`, `validate_configuration_file`


### Integration

The reinforcment centric tasks incorporation can be seen below. In summary, an obvious place for RL is after the original prediction->report pipeline, decoupling the RL from the rest of the system. Thus, leading to a much more modular and resuable component, later the RL system could be refactored into a generic RL pipeline to be used in other applications, where "application specific RL" can be injected, but the process remains constant (e.g load data, compute reward ... update data etc).Only loading the systems current configuration would need to be completed prior to or during the report generation/prediction pipeline.
```mermaid
flowchart TB
  subgraph row1
    direction LR
    LD[Load Configuration|(load_configuration)]
    LD-->RP[Report Pipeline]
    RP --> FPH["Fetch patient history|(fetch_patient_data)"]
    FPH --> VPD["Validate patient data|(validate_patient_data)"]
    VPD --> CCS["Compute current state|(compute_current_state)"]
  end
  subgraph row2
    direction RL
    CCS --> AR["Compute reward|(apply_reward)"]
    AR --> CNA["Select next action|(compute_action)"]
    CNA --> VA["Validate proposed action|(validate_action)"]
  end
  subgraph row3
    direction LR
    VA --> UC["Apply configuration update(update_configuration)"]
    UC --> VCF["Validate configuration file|(validate_configuration_file)"]
    VCF --> UPR["Update patient record|(update_patient_data)"]
  end
  style row1 fill:none,stroke:none
  style row2 fill:none,stroke:none
  style row3 fill:none,stroke:none
```

## Reinforcement Learning

### Configuration File

The reinforcement system will work by updating a YAML config file. To begin the file will only contain 3 modifiable parameters; `threshold.moderate`, `thresholds.high`, and `explanation.top_k`. The thresholds give indicators of when the systems give a _"high risk"_, _"moderate risk"_, or _"low risk"_  indication in the report, prompting the clinician's attention. 

- An example of the dynamic file the RL agent will update:
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


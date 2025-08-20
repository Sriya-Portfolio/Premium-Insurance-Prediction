import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import joblib

from PIL import Image

st.set_page_config(
    page_title="🛡️💰🚑 Premium Insurance Prediction",
    page_icon="Assets/Images/insurance_logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Sidebar

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'abstract'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

with st.sidebar:

    st.title('🛡️💰🚑 Premium Insurance Prediction')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("Abstract", use_container_width=True, on_click=set_page_selection, args=('abstract',)):
        st.session_state.page_selection = 'abstract'

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"


# Data

data = pd.read_csv("Assets/Files/insurance.csv")

# Importing models

rf_regressor = joblib.load('Assets/Files/random_forest_regressor.pkl')

# Pages
if st.session_state.page_selection == "abstract":
    st.subheader("🌀 Abstract")

    st.markdown("## What is Premium Insurance?")

    st.markdown("""
    
    An **insurance premium** is the amount of money an individual or business pays to an insurance company in exchange for coverage under a specific insurance policy.

    """)

    col_insurance_def = st.columns((2,4,2), gap="medium")

    with col_insurance_def[0]:
        st.write(" ")
    with col_insurance_def[1]:
        premium_insurance_def_image = Image.open('Assets/Images/Insurance-Premium-Definition.png')
        st.image(premium_insurance_def_image, caption="Insurance Premium Definition Image")
    with col_insurance_def[2]:
        st.write(" ")

    st.markdown("""

    ---

    ### **1. Why Do We Need to Pay Insurance Premiums?**

    Insurance premiums are the backbone of the insurance system, enabling individuals and businesses to secure protection against financial uncertainties. Paying a premium means transferring the financial risk of unexpected events to the insurer in exchange for regular contributions.

    #### **Key Reasons for Paying Insurance Premiums**

    1. **Risk Management**  
    Life is unpredictable, and events like accidents, illnesses, natural disasters, or theft can lead to significant financial losses. Insurance provides a safety net by managing and sharing this risk:
    - A small, predictable premium prevents massive, unforeseen financial outlays.

    2. **Financial Security and Stability**  
    Insurance ensures you are prepared for emergencies without depleting savings or accumulating debt. For example:
    - Health insurance covers costly medical procedures.
    - Property insurance compensates for damage or theft of valuable assets.

    3. **Protect Loved Ones**  
    Life and health insurance premiums provide financial support to your family in case of an untimely demise or medical emergency:
    - A life insurance payout can help cover debts, daily living expenses, or education costs for dependents.

    4. **Compliance with Legal Requirements**  
    In many cases, paying insurance premiums is mandatory:
    - Auto insurance is legally required in most regions.
    - Business liability insurance may be necessary for operating in certain industries.

    5. **Peace of Mind**  
    Insurance provides emotional relief by ensuring you’re financially prepared for the unexpected:
    - Knowing your medical expenses, car repairs, or family’s future are covered allows you to focus on your life and career without constant worry.

    6. **Investment and Savings Opportunity**  
    Some insurance policies (like whole life insurance or endowment plans) combine protection with savings or investment components:
    - The premium you pay accumulates value over time, which can be used later for specific goals like retirement or education.

    ---

    ### **2. What Is the Need for Premium Insurance?**

    Premium insurance offers enhanced coverage, exclusive benefits, and tailored solutions, making it an essential choice for individuals and organizations with specific needs. It is particularly valuable for those who want comprehensive protection or possess assets requiring higher coverage.

    #### **Why Premium Insurance Stands Out**

    1. **Comprehensive Coverage**  
    Premium insurance policies provide extensive protection beyond the basic plans:
    - **Health Insurance**: Covers advanced treatments, specialized care, and wellness programs.
    - **Auto Insurance**: Includes benefits like rental car reimbursement, roadside assistance, and extended liability coverage.

    2. **High Coverage Limits**  
    Premium policies often have significantly higher coverage limits, ensuring protection for:
    - Expensive medical treatments or surgeries.
    - High-value assets like luxury vehicles, jewelry, or art collections.

    3. **Customizable Options**  
    Premium insurance policies allow you to customize coverage based on your unique needs:
    - Add-ons or riders (e.g., critical illness cover, accidental death benefits).
    - Tailored coverage for business-specific risks or high-risk activities.

    4. **Priority and Exclusive Services**  
    Premium insurance plans often include:
    - Access to private hospitals or doctors.
    - Priority claims processing and dedicated customer service.
    - Wellness programs, preventive care, or global health coverage.

    5. **Protection Against Rare or Niche Risks**  
    Premium insurance is ideal for covering risks not typically included in standard policies:
    - International travel emergencies.
    - Specialized business equipment or operations.

    6. **Security for Long-Term Financial Goals**  
    For policies like life insurance, premium plans often include investment components, helping policyholders meet long-term goals like:
    - Retirement planning.
    - Funding children’s education.

    ---

    ### **3. Who Should Consider Premium Insurance?**

    Premium insurance is not only for the wealthy. It is designed for anyone who values comprehensive protection and is willing to pay more for better benefits and services. Below are the key groups who should consider opting for premium insurance:

    #### **Individuals**

    1. **High-Net-Worth Individuals (HNWIs)**  
    - People with significant assets (e.g., luxury homes, cars, businesses, or collectibles) require premium policies to ensure adequate protection.
    - Premium insurance provides higher limits and broader coverage for these assets.

    2. **Families**  
    - Parents can use premium health or life insurance to secure their family’s future.
    - Policies with maternity benefits, child education riders, or family-wide health coverage are available.

    3. **Frequent Travelers**  
    - Individuals traveling internationally for business or leisure benefit from premium travel insurance, which includes global medical coverage, trip cancellation protection, and emergency evacuation.

    4. **Individuals with Specific Health Needs**  
    - Premium health insurance is invaluable for those with chronic conditions, pre-existing illnesses, or those requiring advanced treatments or therapies.

    5. **People in High-Risk Occupations**  
    - Jobs involving physical danger (e.g., construction workers, pilots, or athletes) may require premium life, disability, or liability coverage.

    6. **Lifestyle-Oriented Individuals**  
    - Those participating in adventure sports or living in areas prone to natural disasters benefit from premium insurance, which covers activities and risks often excluded by standard policies.

    ---

    #### **Businesses**

    1. **Small and Medium Enterprises (SMEs)**  
    - Businesses with specialized needs (e.g., cyber liability or professional indemnity) benefit from premium policies tailored to their operations.

    2. **Large Corporations**  
    - Corporations often require premium group health, property, and liability insurance to protect employees, assets, and operations.

    3. **Entrepreneurs and Startups**  
    - Founders can secure their business and personal assets through premium insurance covering risks like lawsuits, employee injuries, or intellectual property theft.

    ---

    ### **Conclusion**

    An **insurance premium** is not just a cost; it is an investment in security, peace of mind, and financial stability. Premium insurance, though more expensive than standard plans, provides comprehensive coverage, enhanced services, and tailored solutions. It is especially necessary for individuals or businesses with specific needs, valuable assets, or a desire for superior protection.

    Would you like to know more about specific premium insurance types, such as health, life, or property? Or guidance on choosing the right policy?

    """)

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    st.markdown(""" 

    A Streamlit web application that performs **Exploratory Data Analysis (EDA)**, **Data Preprocessing**, and **Supervised Machine Learning** to predict premium insurance from the insurance dataset using **Linear Regression, SVR, and Random Forest Regressor**.

    #### Pages
    1. `Dataset` - Brief description of the Insurance dataset used in this project. 
    2. `EDA` - Exploratory Data Analysis of the Insurance dataset. Includes various visualization.
    3. `Data Cleaning / Pre-processing` - Data cleaning and pre-processing steps such as encoding the species column and splitting the dataset into training and testing sets.
    4. `Machine Learning` - Training the supervised classification model: Linear Regression, SVR, Random Forest Regressor. Includes model evaluation, feature importance, and tree plot.
    5. `Prediction` - Prediction page where users can input values to predict the Brain age using the trained model.
    6. `Conclusion` - Summary of the insights and observations from the EDA and model training.

    """)

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    st.markdown("""
                
    The purposes of this project is to look into different features to observe their relationship, and plot a multiple linear regression based on several features of individual such as age, physical/family condition and location against their existing medical expense to be used for predicting future medical expenses of individuals that help medical insurance to make decision on charging the premium.
                
    `Link:` https://www.kaggle.com/datasets/noordeen/insurance-premium-prediction

    """)

    col_insurance = st.columns((2,4,2), gap="medium")

    with col_insurance[0]:
        st.write(" ")
    with col_insurance[1]:
        insurance_image = Image.open('Assets/Images/InsurancePremium.jpg')
        st.image(insurance_image, caption="Premium Insurance Image")
    with col_insurance[2]:
        st.write(" ")

    # Display the dataset
    st.subheader("Dataset displayed as a Data Frame")
    st.dataframe(data, use_container_width=True, hide_index=True)

    # Descibe Statistics
    st.subheader("Describing Statistics")
    st.dataframe(data.describe(), use_container_width=True)

    st.markdown("""

    The insurance.csv dataset contains 1338 observations (rows) and 7 features (columns). The dataset contains 4 numerical features (age, bmi, children and expenses) and 3 nominal features (sex, smoker and region) that were converted into factors with numerical value designated for each level.

    """)

    info_data = {
        "Column": data.columns,
        "Non-Null Count": data.notnull().sum(),
        "Dtype": data.dtypes
    }
    info_df = pd.DataFrame(info_data)

    # Display in Streamlit
    st.subheader("Data Information")
    st.dataframe(info_df, hide_index=True)

# EDA Page
if st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")

    # Replace infinite values with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    x_axis = ['age', 'bmi', 'children', 'expenses']

    for x in x_axis:
        st.subheader(f'Plots for {x}')
    
        # Set up the figure with increased size for better visibility
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))  # Adjusted size

        # Histogram
        sns.histplot(data[x], ax=axes[0], kde=True, color='skyblue', bins=30, edgecolor='none')
        axes[0].set_title(f'Histogram of {x}', fontsize=20)  # Bigger title font
        axes[0].set_xlabel(f'{x}', fontsize=16)  # Bigger x-axis label font
        axes[0].set_ylabel('Frequency', fontsize=16)  # Bigger y-axis label font

        # Boxplot
        sns.boxplot(data=data[x], ax=axes[1], orient='h', showmeans=True, color='pink')
        axes[1].set_title(f'Boxplot of {x}', fontsize=20)  # Bigger title font
        axes[1].set_xlabel(f'{x}', fontsize=16)  # Bigger x-axis label font
        axes[1].set_ylabel('Value', fontsize=16)  # Bigger y-axis label font
    
        # Adjust layout and display the plot
        plt.tight_layout()
        st.pyplot(fig)

# Data Cleaning Page
if st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    st.dataframe(data.head(), use_container_width=True, hide_index=True)

    st.write("#### Checking for Null values")
    st.dataframe(data.isnull().sum())

    st.write("#### Describing all the columns")
    st.dataframe(data.describe(include = 'all'), use_container_width=True)

    st.write("#### Before Dropping duplicates ")
    st.write("#### Shape of the data: ")
    st.write(data.shape)

    st.write("#### After Dropping duplicates")
    st.write("*Shape of the data:* ")
    data = data.drop_duplicates()
    st.write(data.shape)

    st.dataframe(data.head(), use_container_width=True, hide_index=True)

    st.write("#### Converting All the object data into numerical data manually as the data has less columns which can be easy to convert object data to numerical data")
    
    # Convert 'sex' column
    st.write("##### Converting sex column")
    st.write(data['sex'].unique())
    data['sex'] = data['sex'].map({'female': 0, 'male': 1})
    st.dataframe(data.head(), use_container_width=True, hide_index=True)

    # Convert 'smoker' column
    st.write("##### Converting smoker column")
    st.write(data['smoker'].unique())
    data['smoker'] = data['smoker'].map({'yes': 0, 'no': 1})
    st.dataframe(data.head(), use_container_width=True, hide_index=True)

    # Convert 'region' column
    st.write("##### Converting region column")
    st.write(data['region'].unique())
    data['region'] = data['region'].map({
        'southwest': 1, 'southeast': 2,
        'northwest': 3, 'northeast': 4
    })
    st.dataframe(data.head(), use_container_width=True, hide_index=True)

    # Check column types after conversion
    info_data = {
        "Column": data.columns,
        "Non-Null Count": data.notnull().sum(),
        "Dtype": data.dtypes
    }
    info_df = pd.DataFrame(info_data)

    # Display in Streamlit
    st.subheader("Data Information")
    st.dataframe(info_df, hide_index=True)

    st.subheader("Splitting the dataset for training and testing")

    st.write(data.columns)

    # Split the data into features (X) and target (y)
    X = data.drop('expenses', axis = 1)
    y = data[['expenses']]

    st.code(""" 

    X = data.drop('expenses', axis = 1)
    y = data[['expenses']]

    """)

    st.dataframe(X.head(), use_container_width=True, hide_index=True)
    st.dataframe(y.head(), hide_index=True)

    st.subheader("Train Test Split")

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.code(""" 

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    """)

    st.write("X shape: ", X.shape)
    st.write("X_train shape: ", X_train.shape)
    st.write("X_test shape: ", X_test.shape)

    st.write("#### X_train")
    st.dataframe(X_train, use_container_width=True, hide_index=True)
    
    st.write("#### X_test")
    st.dataframe(X_test, use_container_width=True, hide_index=True)

    st.write("#### y_train")
    st.dataframe(y_train, hide_index=True)

    st.write("#### y_test")
    st.dataframe(y_test, hide_index=True)

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Linear Regression

    st.subheader("Linear Regression")

    st.markdown("""

    ## Linear Regression

    Linear Regression is a fundamental algorithm in machine learning used to model the relationship between a dependent variable (target) and one or more independent variables (predictors). It assumes a linear relationship between them.

    ### How It Works:
    Linear regression tries to fit a straight line through the data points. The line is defined by the equation:
    **y = mx + b**

    Where:
    - `y` is the dependent variable (e.g., stock price),
    - `x` is the independent variable (e.g., volume, open price),
    - `m` is the slope (how much `y` changes with `x`),
    - `b` is the y-intercept.

    The algorithm minimizes the sum of squared differences between the actual and predicted values to find the best-fitting line.

    ### Application in Stock Price Prediction:
    In stock price prediction, linear regression helps forecast the future stock price based on historical data like open, high, low, and volume. The model provides an equation to predict the closing price, assuming a linear relationship between the predictors and the target variable.

    ### Advantages:
    - Simple and easy to implement.
    - Interpretable results.
    - Works well when the relationship between variables is linear.

    ### Limitations:
    - Assumes a linear relationship, which might not always hold.
    - Sensitive to outliers.

    For more details, check the [scikit-learn documentation on Linear Regression](https://scikit-learn.org/stable/modules/linear_model.html#linear-regression).

    """)

    col_lr_fig = st.columns((2, 4, 2), gap='medium')

    with col_lr_fig[0]:
        st.write(' ')

    with col_lr_fig[1]:
        linear_regression_image = Image.open('Assets/Images/linear_regression.png')
        st.image(linear_regression_image, caption='Linear Regression Figure')

    with col_lr_fig[2]:
        st.write(' ')

    st.subheader("Multiple Linear Regression")

    col_mlr_fig = st.columns((2, 4, 2), gap='medium')

    with col_mlr_fig[0]:
        st.write(' ')

    with col_mlr_fig[1]:
        multiple_linear_regression_image = Image.open('Assets/Images/multiple_linear_regression.png')
        st.image(multiple_linear_regression_image, caption='Multiple Linear Regression Figure')

    with col_mlr_fig[2]:
        st.write(' ')

    st.code("""

    lr = LinearRegression()
    lr.fit(X_train, y_train)     
            
    """)

    st.code("""

    # Make predictions on the test set
    y_pred = lr.predict(X_test)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # Print the results rounded to 4 decimal points
    print("Accuracy Score:", round(accuracy, 4))    
    
    """)

    st.write("""
    **Accuracy Score**: 0.8068
    
    That means Accuracy score is 80.68%
    """)

    st.markdown("""
                
    The evaluation of the model includes three important metrics:

    **Accuracy Score**: This metric represents the percentage of correct predictions made by the model. In this case, the accuracy score of 0.8068 means that approximately 80.68% of the predictions were correct.

    """)

    # SVR

    st.markdown("""

    ### Support Vector Regression (SVR)

    Support Vector Regression (SVR) is an extension of the Support Vector Machine (SVM) algorithm used for regression tasks. It is designed to predict continuous variables while being robust to outliers. SVR works by finding a function that approximates the data points within a certain margin of error, while keeping the model as simple as possible.
    """)

    col_svr = st.columns((2,4,2), gap="medium")

    with col_svr[0]:
        st.write(" ")
    with col_svr[1]:
        svr_image = Image.open("Assets/Images/svr.png")
        st.image(svr_image, caption="SVR Image")
    with col_svr[2]:
        st.write(" ")

    st.markdown("""

    #### Key Features of SVR:
    1. **Margin of Tolerance (Epsilon)**: SVR introduces a margin, defined by the parameter **epsilon (ε)**, within which errors are ignored. This means that the model tries to minimize errors only for points that fall outside the epsilon tube.
    2. **Kernel Trick**: Like SVM, SVR uses the kernel trick to map input data into higher dimensions, enabling it to capture non-linear relationships. Common kernels include Radial Basis Function (RBF), polynomial, and linear kernels.
    3. **Support Vectors**: SVR uses a subset of training data points, known as **support vectors**, to define the decision boundary. These points are the most critical in determining the model's output.
    4. **Regularization Parameter (C)**: The parameter **C** controls the trade-off between achieving a low error and keeping the model as simple as possible. A small **C** allows more slack, leading to a simpler model but potentially higher error, while a large **C** forces the model to minimize errors more aggressively.
    5. **Robustness to Outliers**: SVR is less sensitive to outliers compared to traditional regression models like linear regression. This is due to its focus on fitting a model that ignores small errors within the epsilon margin.
    6. **Non-Linear Regression**: SVR can handle complex, non-linear regression tasks by using different kernel functions to map data to higher-dimensional spaces.

    #### Advantages of SVR:
    - **Flexibility**: SVR can model both linear and non-linear relationships by using appropriate kernels.
    - **Outlier Handling**: It performs well in the presence of outliers, making it a good choice for real-world data with noise.
    - **Generalization**: The model has strong generalization capabilities, which helps in avoiding overfitting.

    #### Disadvantages of SVR:
    - **Computationally Expensive**: Training an SVR model can be time-consuming, especially with large datasets, as it requires calculating the support vectors.
    - **Hyperparameter Tuning**: SVR has several parameters (C, epsilon, and kernel type) that need careful tuning to achieve optimal performance.

    #### Applications:
    - Time-series forecasting
    - Stock price prediction
    - Engineering and financial data analysis
    - Any problem where a non-linear relationship exists between features and the target variable.

    For more details, visit the [Support Vector Machines Wiki](https://en.wikipedia.org/wiki/Support_vector_machine).

    """)

    col_svr_ml = st.columns((2,4,2), gap="medium")

    with col_svr_ml[0]:
        st.write(" ")
    with col_svr_ml[1]:
        svr_ml_image = Image.open("Assets/Images/svr_regression.png")
        st.image(svr_ml_image, caption="Support Vector Regression Image")
    with col_svr_ml[2]:
        st.write(" ")

    st.code("""

    svm = SVR() 
    svm.fit(X_train, y_train)    
            
    """)

    st.code("""

    # Make predictions on the test set
    y_pred = svm.predict(X_test)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # Print the results rounded to 4 decimal points
    print("Accuracy Score:", round(accuracy, 4))    
    
    """)

    st.write("""
    **Accuracy Score**: -0.1344
    
    That means Accuracy score is -13.44%
    Which is very bad for our project.

    """)

    st.markdown("""
                
    The evaluation of the model includes three important metrics:

    **Accuracy Score**: This metric represents the percentage of correct predictions made by the model. In this case, the accuracy score of -0.1344 means that approximately -13.44% of the predictions were correct.

    """)

    # Random Forest Regressor

    st.subheader("Random Forest Regressor")

    st.markdown("""

    **Random Forest Regressor** is a machine learning algorithm that is used to predict continuous values by *combining multiple decision trees* which is called `"Forest"` wherein each tree is trained independently on different random subset of data and features.

    This process begins with data **splitting** wherein the algorithm selects various random subsets of both the data points and the features to create diverse decision trees.  

    Each tree is then trained separately to make predictions based on its unique subset. When it's time to make a final prediction each tree in the forest gives its own result and the Random Forest algorithm averages these predictions.

    `Reference:` https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
         
    """)

    # Columns to center the Random Forest Regressor figure image
    col_rfr_fig = st.columns((2, 4, 2), gap='medium')

    with col_rfr_fig[0]:
        st.write(' ')

    with col_rfr_fig[1]:
        decision_tree_parts_image = Image.open('Assets/Images/Random-Forest-Figure.png')
        st.image(decision_tree_parts_image, caption='Random Forest Figure')

    with col_rfr_fig[2]:
        st.write(' ')

    st.code("""

    rfr_regressor = RandomForestRegressor(n_estimators=100, random_state=42, class_weight='balanced')
    rfr_regressor.fit(X_train, y_train)     
            
    """)

    st.code("""

    # Make predictions on the test set
    y_pred = rf.predict(X_test)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # Print the results rounded to 4 decimal points
    print("Accuracy Score:", round(accuracy, 4))    
    
    """)

    st.write("""
    **Accuracy Score**: 0.8816
    
    That means Accuracy score is 88.16%
    Which is consider to be great for our project
    """)

    st.markdown("""
                
    The evaluation of the model includes three important metrics:

    **Accuracy Score**: This metric represents the percentage of correct predictions made by the model. In this case, the accuracy score of 0.8816 means that approximately 88.16% of the predictions were correct.

    """)
    
    st.subheader("Number of Trees")
    st.code("""

    print(f"Number of trees made: {len(rfr_classifier.estimators_)}")
     
    """)

    st.markdown("**Number of trees made:** 100")

    st.subheader("Plotting the Forest")

    forest_image = Image.open('Assets/Images/RFRForest.png')
    st.image(forest_image, caption='Random Forest Regressor - Forest Plot')

    st.markdown("This graph shows **all of the decision trees** made by our **Random Forest Regressor** model which then forms a **Forest**.")

    st.subheader("Forest - Single Tree")

    forest_single_tree_image = Image.open('Assets/Images/RFRTreeOne.png')
    st.image(forest_single_tree_image, caption='Random Forest Regressor - Single Tree')

    st.markdown("This **Tree Plot** shows a single tree from our Random Forest Regressor model.")

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("Prediction")

    st.write("Provide the following details to predict insurance costs:")

    # Create input fields for user data
    age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
    sex = st.radio("Sex", options=["Female", "Male"], index=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=27.5, step=0.1)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
    smoker = st.radio("Smoker", options=["Yes", "No"], index=1)
    region = st.selectbox("Region", options=["Southwest", "Southeast", "Northwest", "Northeast"])

    # Map inputs to model-compatible format
    sex_value = 1 if sex == "Male" else 0
    smoker_value = 0 if smoker == "Yes" else 1
    region_map = {"Southwest": 1, "Southeast": 2, "Northwest": 3, "Northeast": 4}
    region_value = region_map[region]

    # Create a DataFrame for prediction
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex_value],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker_value],
        'region': [region_value]
    })

    # Make prediction when the button is clicked
    if st.button("Predict Insurance Cost"):
        prediction = rf_regressor.predict(input_data)
        st.success(f"Predicted Insurance Cost: ${prediction[0]:,.2f}")

elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    st.markdown("""

    ### Conclusion

    In this project, we have developed a model to predict insurance costs based on several key factors such as age, BMI, number of children, smoking status, and region of residence. The primary goal of this project was to build a reliable predictive model using machine learning, which can provide personalized insurance premium estimates based on an individual's characteristics. By leveraging data science techniques and tools such as Streamlit for the user interface and Random Forest Regressor for the prediction model, we have successfully created an interactive, user-friendly application for insurance cost prediction.

    Through this project, we have not only demonstrated the ability of machine learning to predict insurance premiums accurately, but we have also explored how various factors influence the final cost of insurance. The integration of features like age, smoking habits, and region highlights the importance of personal health and demographic information in determining insurance rates. The model can now assist individuals in understanding how their lifestyle and health conditions may affect their insurance costs, making it easier for them to plan financially for such expenses.

    ---

    ### Importance of Premium Insurance

    Having premium insurance is essential for safeguarding against unexpected financial burdens that may arise due to unforeseen events or health-related issues. Insurance, in essence, provides a safety net that helps individuals mitigate financial risks associated with accidents, illnesses, or other emergencies. Premium insurance goes beyond basic coverage, offering more extensive protection, higher coverage limits, and additional benefits that can significantly reduce out-of-pocket expenses when a claim is made.

    One of the primary reasons for opting for premium insurance is comprehensive healthcare coverage. As medical costs rise globally, especially in countries like India, where healthcare expenses can be prohibitive, having premium health insurance ensures that individuals have access to high-quality care without worrying about the financial implications. Premium insurance plans often cover a wide range of treatments, including surgeries, hospitalization, and specialized care, which basic plans may not fully address.

    Moreover, premium insurance offers peace of mind by providing protection for both short-term and long-term health needs. For instance, life insurance policies with higher premiums often come with added benefits like critical illness coverage, which protects against life-threatening conditions. These policies may also offer endowment plans or wealth-building features, ensuring that policyholders have a financial cushion to rely on in their later years.

    Additionally, the security of premium insurance helps mitigate the stress that can arise from sudden health issues or emergencies. It allows individuals and their families to focus on recovery and well-being instead of struggling with overwhelming medical bills. It also provides protection for families against the financial loss that may result from the premature death of a loved one, ensuring that the family's standard of living is maintained.

    In conclusion, premium insurance is not just about covering immediate healthcare costs but securing a stable future. It offers holistic coverage, ensuring that people can protect their health, finances, and families against unforeseen events. As healthcare expenses and life uncertainties continue to rise, having premium insurance has become more than just a luxury—it's a necessary investment for long-term financial security and peace of mind.

    """)
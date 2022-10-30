import streamlit as st
import numpy as np
import pandas as pd
import math
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(page_title="Vehicle default", page_icon="logo.png", layout='wide', initial_sidebar_state='auto', menu_items=None)
col1, col2, col3 = st.columns(3)
with col2:
   st.image("logo.png", width = 250)


def getData():
    data = pd.read_csv('Data/Train_Dataset.csv')
    return data

def ModelTraining(data):
    #displaying all the rows &columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    originalDataset = getData()

    #Dropping the unwanted data attributes which is not useful to build the model
    processedDataset = originalDataset.copy()

    processedDataset.drop(['ID','Accompany_Client', 'Client_Housing_Type', 'Client_Housing_Type', 'Population_Region_Relative','ID_Days', 'Own_House_Age','Mobile_Tag', 'Homephone_Tag', 'Workphone_Working', 'Client_Occupation','Client_Family_Members', 'Cleint_City_Rating','Application_Process_Day','Application_Process_Hour','Client_Permanent_Match_Tag','Client_Contact_Work_Tag', 'Type_Organization','Score_Source_1','Score_Source_2','Score_Source_3','Social_Circle_Default', 'Phone_Change','Credit_Bureau'], axis=1, inplace=True)

    #checking the columns with null values more than 30% of data and removing those columns form the dataset
    #since high null values might lead the model to work less accurate

    result = 0

    for x in processedDataset:
            result = processedDataset[x].isnull().sum()
            if (result/len(processedDataset.index)) > 0.3 :
                print(processedDataset[x])
                del processedDataset[x]
            result=0

    processedDataset.head()

    toconvert_type_list=['Client_Income','Credit_Amount','Loan_Annuity', 'Age_Days','Employed_Days','Registration_Days']

    numeric_list=['Car_Owned','Bike_Owned','Active_Loan','House_Own','Child_Count', 'Default']

    for x in processedDataset:
          if x in toconvert_type_list:
             processedDataset[x] = pd.to_numeric(processedDataset[x],errors = 'coerce')
             numeric_list.append(x)

    gender_labels = processedDataset['Client_Gender'].unique()
    income_type_labels = processedDataset['Client_Income_Type'].unique()
    education_labels = processedDataset['Client_Education'].unique()
    loan_contract_type_labels = processedDataset['Loan_Contract_Type'].unique()
    marital_status_labels = processedDataset['Client_Marital_Status'].unique()

    # This function will fill the missing values in categorical data with value 'Other'
    def fillMissingCategoricalValues(dataset):
      tempData = dataset
      valueMap = {
        "Client_Income_Type": ['Service','Commercial','Retired' , 'Student' , 'Unemployed'],
        "Client_Education": ['Secondary','Graduation'],
        "Loan_Contract_Type": ['CL', 'RL'],
        "Client_Marital_Status": ['M', 'W', 'S', 'D'],
        "Client_Gender": ['Male', 'Female']
      }

      for column in valueMap.keys():
        tempData[column] = [value if value in valueMap[column] else 'Other' for value in tempData[column]]

      return tempData

    # This function will impute missing numeric values
    from sklearn.impute import SimpleImputer
    import math

    def fillMissingNumericValues(dataset):
      tempData = dataset
      numericColumns = ['Client_Income','Credit_Amount','Loan_Annuity','Age_Days','Employed_Days','Registration_Days']
      wholeNumberColumns = ['House_Own', 'Child_Count', 'Active_Loan', 'Bike_Owned', 'Car_Owned']

      for numericColumn in numericColumns:
        tempData[numericColumn] = pd.to_numeric(tempData[numericColumn],errors = 'coerce')

      imputer = SimpleImputer(strategy='mean', missing_values=np.nan)

      for column in numericColumns:
        data = tempData[[column]]
        imputer = imputer.fit(data)
        tempData[column] = imputer.transform(data)

      for column in wholeNumberColumns:
        imputer = SimpleImputer(strategy='constant',
                            missing_values=np.nan, fill_value=0.0)
        data = tempData[[column]]
        imputer = imputer.fit(data)
        tempData[column] = imputer.transform(data)

      return tempData

    # This function will convert the days to years
    def daysToYears(dataset):
      tempData = dataset
      dayColumns = ['Age_Days','Employed_Days','Registration_Days']

      for column in dayColumns:
        tempData[column] = [math.ceil(days/365) if not math.isnan(days) else 0 for days in tempData[column]]

      return tempData

    processedDataset = fillMissingCategoricalValues(processedDataset)
    processedDataset = fillMissingNumericValues(processedDataset)
    processedDataset = daysToYears(processedDataset)

    for columnName in processedDataset:
      tot = processedDataset[columnName].isnull().sum()
      percentatge = (tot/len(processedDataset.index))
      print(columnName, percentatge)

    # Droping Unwanted rows in the dataset

    #Removing unwanted rows

    #Removing Employed_Days selected rows beacuse the values must be within a range of 0-80 (Collected data must be meaning full & more practical)
    processedDataset.drop(processedDataset[processedDataset['Employed_Days'] > 80].index, inplace=True)
    categoricalColumns = ["Client_Income_Type","Client_Education","Loan_Contract_Type","Client_Marital_Status","Client_Gender"]

    processedDataset['Client_Education'].unique()

    processedDataset.head()

     # This function will convert categorical labels to numbers
    def convertCategoryLabelsToNumber(dataset):
      tempData = dataset.copy()
      valueMap = {
          "Client_Income_Type": {
              "Commercial": 1,
              "Service": 2,
              "Student": 3,
              "Retired": 4,
              "Unemployed": 5,
              "Other": 99
          },
          "Client_Education": {
              "Secondary": 1,
              "Graduation": 2,
              "Other": 99
          },
          "Client_Marital_Status": {
              'M': 1,
              'W': 2,
              'S': 3,
              'D':4,
              'Other': 99
          },
          "Client_Gender": {
              'Male': 1,
              'Female': 2,
              'Other': 99
          },
          "Loan_Contract_Type": {
              'CL': 1,
              'RL': 2,
              'Other': 99
          }

      }
      for column in valueMap.keys():
          tempData[column] = [valueMap[column][value] if valueMap[column][value] else 99 for value in tempData[column]]

      return tempData

    processedDataset = convertCategoryLabelsToNumber(processedDataset)

    for column in categoricalColumns:
      processedDataset.drop(processedDataset[processedDataset[column] == 99].index, inplace=True)


    # Model Building

    finalDataset = processedDataset.copy()
    x = finalDataset.drop('Default', axis=1).values# Input features (attributes)
    y = processedDataset['Default'].values # Target vector

    # re-balance the imbalance dataset
    from imblearn.under_sampling import RandomUnderSampler

    under = RandomUnderSampler()
    x, y = under.fit_resample(x, y)

    ## Random Forest Classifier Model

    from sklearn.ensemble import RandomForestClassifier
    model=RandomForestClassifier(criterion='gini' , max_depth=20 , bootstrap=True , max_features='sqrt' , min_samples_leaf= 2 , min_samples_split=5 , n_estimators=100 )
    model.fit(x,y)

    categoryLabels = {
        "Client_Income_Type": income_type_labels,
        "Client_Education": education_labels,
        "Loan_Contract_Type": loan_contract_type_labels,
        "Client_Marital_Status": marital_status_labels,
        "Client_Gender": gender_labels
}

    return model, processedDataset, categoryLabels

def input_transformer(inputs):
    value_map = {
        "House Owned": {
            "Yes": 1,
            "No": 0
        },
        "Car Owned": {
            "Yes": 1,
            "No": 0
        },
        "Bike Owned": {
            "Yes": 1,
            "No": 0
        },
        "Has Active Loan": {
            "Yes": 1,
            "No": 0
        },
        "Client Income Type": {
          "Commercial": 1,
          "Service": 2,
          "Student": 3,
          "Retired": 4,
          "Unemployed": 5
        },
        "Client Education": {
          "Secondary": 1,
          "Graduation": 2
        },
        "Client Marital Status": {
          'Married': 1,
          'Widow': 2,
          'Single': 3,
          'Divorced':4
        },
        "Client Gender": {
          'Male': 1,
          'Female': 2
        },
        "Loan Contract Type": {
          'Cash Loan': 1,
          'Revolving Loan': 2
        }
    }

    transformed_inputs = []
    for input, value in inputs.items():
       if (value_map[input] != None and value_map[input][value] != None):
        transformed_inputs.append(value_map[input][value])

    return transformed_inputs

mainContainer = st.container()

with mainContainer:


    data = getData()
    model , Encodes , NonEncodes = ModelTraining(data)
    
    tab = st.table()

    tab1 = tab.form(key='my_form')
    tab1.header("Enter Following Details to predict Loan Default status")

    col1 , col2 = tab1.columns(2)

    fName = col1.text_input("Client full name: ")

    active_loan = col1.selectbox("Already has an active loan?" , ("-", "Yes" , "No"))
    education = col1.selectbox("Enter client education: " , ("-", 'Secondary', 'Graduation') , on_change=None)
    employed_days = col1.slider("Enter number of employed years before application: " , min_value = 0 , max_value= 80 , on_change=None)

    income = col1.text_input("Enter client income: "  , value = 0 ,on_change=None)
    income_type = col2.selectbox("Enter income type: " , ("-", 'Commercial','Retired' ,'Service', 'Student' , 'Unemployed') , on_change=None)
    loan_contract_type = col2.selectbox("Enter loan contract type: " , ("-", 'Cash Loan', 'Revolving Loan') , on_change=None)

    loan_amount = col1.text_input("Enter loan amount requested: " , value = 0 , on_change=None)
    loan_annuity = col2.text_input("Enter loan annuity amount: " , value = 0 , on_change=None)

    age = col1.slider("Enter age: " , min_value = 20 , max_value= 60 , on_change=None)

    gender = col1.selectbox("Enter client gender: " , ("-", "Female", "Male" ) , on_change=None)
    child_count = col2.selectbox("Enter child count: " , (0,1,2,3,4,5,6,7,8,9,10) , on_change=None)
    registration = col2.slider("Years since registration: " , min_value = 0 , max_value= 50 , on_change=None)

    marital_status = col1.selectbox("Enter marital status" , ("-", "Divorced", "Single", "Married", "Widow"))
    car_owned = col1.selectbox("Car owner?" , ("-", "Yes" , "No"))
    bike_owned = col1.selectbox("Bike owner?" , ("-", "Yes" , "No"))
    house_owned = col1.selectbox("House owner?" , ("-", "Yes" , "No"))


    Submit = tab1.form_submit_button("Submit")
    if Submit:
        inputs = { "Loan Amount":loan_amount , "Income": income , "Loan Annuity":loan_annuity , "Age": age, "Child Count": child_count, "Employed Days": employed_days, "Years since registration": registration }
        inputs_to_transform = {
                               "House Owned": house_owned,
                               "Car Owned": car_owned,
                               "Bike Owned": bike_owned,
                               "Has Active Loan": active_loan,
                               "Client Income Type": income_type,
                               "Client Education": education,
                               "Client Marital Status": marital_status,
                               "Client Gender": gender,
                               "Loan Contract Type": loan_contract_type
        }


        invalid_inputs = []

        if fName.strip() == "":
            invalid_inputs.append("Client Name")

        if loan_amount.strip() == "0" or loan_amount.strip() == "":
            invalid_inputs.append("Loan Amount")

        for label, value in inputs.items():
            if value == '-' or value == "-" or value == None:
                invalid_inputs.append(label)

        for label, value in inputs_to_transform.items():
            if value == '-' or value == "-" or value == None:
                invalid_inputs.append(label)


        if len(invalid_inputs) > 0:
            invalid_inputs_str = "Following fields are invalid: \n"
            st.error(invalid_inputs_str + ", ".join(invalid_inputs))
        else:
           tab.empty()
           transformed_inputs = input_transformer(inputs_to_transform)
           inputs_array = [list(inputs.values()) + transformed_inputs]
           st.write("Client Name: " + fName)
           st.write("Loan Amount: " + loan_amount)
           # print(inputs)
           prediction  = model.predict(inputs_array)
           if prediction[0] == 0:
               st.success("Please accept the above loan request")
           else:
               st.error("Please reject the above request as client is more prone to default on the loan")








    




  






import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import seaborn as sns
from PIL import Image
import shap
import os

model = joblib.load('Nepal_12_model(severe_damage).pkl')
model1 = joblib.load('Nepal_28_model(Damage_Grade).pkl')
st.title('NEPAL EARTHQUAKE')
tab,tab1,tab2 = st.tabs(['Data Analysis','Nepal District 12(Severe Damage)','Nepal District 28(Damage Grade)'])
with tab:
    uploaded_file = st.file_uploader('Chose a file',type=['csv','db','pdf'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            

            st.success('File was uploaded successfully')
            st.subheader('Preview Of The Dataset')
            st.write(df.head())

            foundation_pivot = pd.pivot_table(
            df,index = 'foundation_type' , values='severe_damage',aggfunc='mean'
            ).sort_values(by='severe_damage',ascending=False)
            st.subheader('Material Durability')
            st.write(foundation_pivot)

            major,minor = df.severe_damage.value_counts(normalize=True)

            fig,ax = plt.subplots()
            plt.title('Chart Of The Severe Damage Of the Materials')
            sns.barplot(data=foundation_pivot,x='severe_damage',y='foundation_type',label='severe_damages')
            plt.axvline(
                major,linestyle='--',color='red',label='Major Damages'
            )
            plt.axvline(
                minor,linestyle='--',color='green',label='Minor Damages'
            )
            plt.legend(loc = 'lower right')
            st.pyplot(fig)
        except Exception as e:
            st.error(f'Error processing file:{e}')
with tab1:
        st.header('Nepal Disrict 12')

        # image = Image.open('my_image.jpg')
        # st.image(image, caption='My Image', use_column_width=True)

        st.subheader('Input Features')

        age_building = st.number_input('Building Age',min_value=1,max_value=100,value=5,key='age_buillding')
        foundation_type = st.selectbox('foundation_type',options=['Other', 'Mud mortar-Stone/Brick', 'Cement-Stone/Brick','Bamboo/Timber', 'RC'],key='foundation_type')
        ground_floor_type = st.selectbox('ground_floor_type',options=['Mud', 'Brick/Stone', 'RC', 'Timber', 'Other'],key='ground_floor_type')
        height_ft_pre_eq = st.number_input('height_ft_pre_eq',min_value=5,max_value=100,value=25,step=10,key='height_ft_pre_eq')
        land_surface_condition = st.selectbox('land_surface_condition',options=['Flat', 'Moderate slope', 'Steep slope'],key='land_surface_condition')
        other_floor_type = st.selectbox('other_floor_type',options=['Not applicable', 'TImber/Bamboo-Mud', 'Timber-Planck','RCC/RB/RBC'],key='other_floor_type')
        plan_configuration = st.selectbox('plan_configuration',options=['Rectangular', 'L-shape', 'Square', 'T-shape', 'Multi-projected','H-shape', 'U-shape', 'Others', 'E-shape','Building with Central Courtyard'],key='plan_configuration')
        plinth_area_sq_ft = st.number_input('plinth_area_sq_ft',min_value=100,max_value=4000,value=2000,step=500,key='plinth_area_sq_ft')
        position = st.selectbox('position',options=['Not attached', 'Attached-1 side', 'Attached-2 side','Attached-3 side'],key='position')
        roof_type = st.selectbox('roof_type',options=['Bamboo/Timber-Light roof', 'Bamboo/Timber-Heavy roof','RCC/RB/RBC'],key='roof_type')
        superstructure = st.selectbox('superstructure',options=['mud_mortar_stone', 'bamboo', 'adobe_mud', 'stone_flag', 'other','cement_mortar_brick', 'mud_mortar_brick', 'timber','cement_mortar_stone', 'rc_non_engineered', 'rc_engineered'],key='superstructure') 
        input_data = {
            'age_building':age_building,
            'foundation_type' : foundation_type,
            'ground_floor_type': ground_floor_type,
            'height_ft_pre_eq' : height_ft_pre_eq,
            'land_surface_condition': land_surface_condition,
            'other_floor_type': other_floor_type,
            'plan_configuration':plan_configuration,
            'plinth_area_sq_ft':plinth_area_sq_ft,
            'position':position,
            'roof_type':roof_type,
            'superstructure':superstructure
        }
        input_df = pd.DataFrame([input_data])
        X_train = pd.DataFrame({
        'age_building':[20,50,100,80],
        'foundation_type' : ['Other', 'Mud mortar-Stone/Brick', 'Cement-Stone/Brick','Cement-Stone/Brick'],
        'ground_floor_type': ['Mud', 'Brick/Stone', 'RC', 'Timber'],
        'height_ft_pre_eq' : [10,6,20,40],
        'land_surface_condition': ['Flat', 'Moderate slope', 'Steep slope','Steep slope'],
        'other_floor_type': ['Not applicable', 'TImber/Bamboo-Mud', 'Timber-Planck','RCC/RB/RBC'],
        'plan_configuration':['Rectangular', 'L-shape', 'Square', 'T-shape'],
        'plinth_area_sq_ft':[1000,450,2000,4000],
        'position':['Not attached', 'Attached-1 side', 'Attached-2 side','NOt attached'],
        'roof_type':['Bamboo/Timber-Light roof', 'Bamboo/Timber-Heavy roof','RCC/RB/RBC','Bamboo/Timber-Light roof'],
        'superstructure':['mud_mortar_stone', 'bamboo', 'adobe_mud', 'stone_flag']
        
        })
        def predict_eq(input_data):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)
            record = input_df.copy()
            record["prediction"] = prediction

            file_exists = os.path.isfile("user_logs.csv")
            record.to_csv("user_logs.csv", mode="a", index=False, header=not file_exists)
            return prediction
        
        if st.button('Predict'):
            prediction = predict_eq(input_data)
            if prediction == 1:
                st.error('The damage was above 3')
            else:
                st.success('Great the damage was below 3')
            
            tree_model = model.named_steps['dt']  
            encoded_input = model.named_steps["oh"].transform(input_df)
            explainer = shap.TreeExplainer(tree_model)
            shap_values = explainer.shap_values(encoded_input)

            shap.initjs()
            st.subheader("SHAP Explanation")

            # Use SHAP waterfall plot for class 1 (if binary classification)
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            st.pyplot(fig)


        
        
with tab2:
    st.header('Nepal Disrict 28')

    st.subheader('Input Features')

    age_building = st.number_input('Building Age',min_value=1,max_value=100,value=5,key='age_building1')
    foundation_type = st.selectbox('foundation_type',options=['Other', 'Mud mortar-Stone/Brick', 'Cement-Stone/Brick','Bamboo/Timber', 'RC'],key='foundation_type1')
    ground_floor_type = st.selectbox('ground_floor_type',options=['Mud', 'Brick/Stone', 'RC', 'Timber', 'Other'],key='ground_floor_type1')
    height_ft_pre_eq = st.number_input('height_ft_pre_eq',min_value=5,max_value=100,value=25,step=10,key='height_ft_pre_eq1')
    land_surface_condition = st.selectbox('land_surface_condition',options=['Flat', 'Moderate slope', 'Steep slope'],key='land_surface_condition1')
    other_floor_type = st.selectbox('other_floor_type',options=['Not applicable', 'TImber/Bamboo-Mud', 'Timber-Planck','RCC/RB/RBC'],key='other_floor_type1')
    plan_configuration = st.selectbox('plan_configuration',options=['Rectangular', 'L-shape', 'Square', 'T-shape', 'Multi-projected','H-shape', 'U-shape', 'Others', 'E-shape','Building with Central Courtyard'],key='plan_configuration1')
    plinth_area_sq_ft = st.number_input('plinth_area_sq_ft',min_value=100,max_value=4000,value=2000,step=500,key='plinth_area_sq_ft1')
    position = st.selectbox('position',options=['Not attached', 'Attached-1 side', 'Attached-2 side','Attached-3 side'],key='position1')
    roof_type = st.selectbox('roof_type',options=['Bamboo/Timber-Light roof', 'Bamboo/Timber-Heavy roof','RCC/RB/RBC'],key='roof_type1')
    superstructure = st.selectbox('superstructure',options=['mud_mortar_stone', 'bamboo', 'adobe_mud', 'stone_flag', 'other','cement_mortar_brick', 'mud_mortar_brick', 'timber','cement_mortar_stone', 'rc_non_engineered', 'rc_engineered'],key='superstructure1') 

    input_data = {
        'age_building':age_building,
        'foundation_type' : foundation_type,
        'ground_floor_type': ground_floor_type,
        'height_ft_pre_eq' : height_ft_pre_eq,
        'land_surface_condition': land_surface_condition,
        'other_floor_type': other_floor_type,
        'plan_configuration':plan_configuration,
        'plinth_area_sq_ft':plinth_area_sq_ft,
        'position':position,
        'roof_type':roof_type,
        'superstructure':superstructure
    }
    input_df = pd.DataFrame([input_data])
    def predict_eq(input_data):
        input_df = pd.DataFrame([input_data])
        prediction = model1.predict(input_df)
        record = input_df.copy()
        record["prediction"] = prediction

        file_exists = os.path.isfile("user_logs1.csv")
        record.to_csv("user_logs.csv", mode="a", index=False, header=not file_exists)
        return prediction

    if st.button('predict'):
          prediction = predict_eq(input_data)
          if prediction == 1:
               st.success('The damage was minimal ')
          elif prediction == 2:
              st.success('The damage was ok')
          elif prediction == 3:
              st.error('The damage was quite significant')
          elif prediction == 4:
              st.error('The damage was bad')
          else:
            st.error('The damage was disaterous') 

          tree_model = model1.named_steps['dt']  
          encoded_input = model1.named_steps["oh"].transform(input_df)
          explainer = shap.TreeExplainer(tree_model)
          shap_values = explainer.shap_values(encoded_input)

          shap.initjs()
          st.subheader("SHAP Explanation")

          # Use SHAP waterfall plot for class 1 (if binary classification)
          fig, ax = plt.subplots()
          shap.plots.waterfall(shap.Explanation(values=shap_values[1][0],base_values=explainer.expected_value[1],data=input_df.iloc[0],feature_names=input_df.columns
          ), max_display=10, show=False)

          st.pyplot(fig)
          plt.clf()




        

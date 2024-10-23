import os
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import json
import pandas as pd
import ast
import time
import openai
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import math
from itertools import groupby
from itertools import product
from zipfile import ZipFile
import io
from pymongo import MongoClient

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key was loaded successfully
if openai_api_key is None:
    raise ValueError("OpenAI API key not found. Please set it in the .env file.")

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=openai_api_key)  # Use the API key from the environment



@st.cache_data(experimental_allow_widgets=True)
def variable_values():
    # try:
        with st.form(key='my_form2'):
            # db_client = MongoClient('mongodb+srv://m220student:Gokulnath@cluster0.qp8h2nr.mongodb.net/')
            # load_dotenv()
            # db_name = "electronics"
            # collection_name = "DOE"
            # collection = db_client[db_name][collection_name]
            # print(collection)
            # # index_name = "langchain_demo"

            # db = db_client.electronics
            # col = db.DOE
            # st.session_state['last_used'] = ''
            dict_values = {}
            mongo_data = {}
            # available_list = [str(id) for id in col.distinct('_id')]
            # available_list = list(filter(lambda x : True if x else False,available_list))
            # st.write(available_list)
            # select_name = st.selectbox('Select preused doe name',available_list,key = 'select_name',index=None,disabled=st.session_state.get('select_name',False))

            # if default:
            # if select_name:
            #     dict_values = col.find_one({ "_id": select_name})
            #     # mongo_data['doe_name'] = doe
            #     try:
            #         dict_values_re = dict_values.get('values')
            #         st.session_state.dict_values_re = dict_values_re
            #     except Exception as e:
            #         st.write(f'No values found for DOE {select_name}')
            #         st.session_state.dict_values_re = {}
            doe_name = st.text_input('DOE NAME',key = 'doe_name',value ='')
            mongo_data['doe_name'] = doe_name
            # doe = st.session_state.get('doe_name','').replace(' ','').lower().strip()
            # st.write(available_list)
            # default = st.form_submit_button(label='Get preused values')

            # st.write(st.session_state.dict_values_re)

            doe_type = st.selectbox('DOE type?',('fractional factorial design', 'full factorial design', 'RMS design'),key = 'doe_type') #st.text_input('DOE TYPE',key = 'doe_type',value = st.session_state.get('doe_type',''))
            num_factors = st.text_input('NUMBER OF FACTORS',key = 'num_factors',value ='4')
            mongo_data['num_factors'] = num_factors
            # time.sleep(30)
            factors_dict = {}
            if not num_factors == 'no_value':
                for i in range(1,int(num_factors)+1):
                    col1, col2 = st.columns([1,1])
                    with col1:
                        st.text_input(f"Factor # {i} (low,high)", key=f'factor{i}',value = st.session_state.get(f'factor{i}',''))
                        mongo_data[f'factor{i}'] = st.session_state.get(f'factor{i}','')
                    with col2:
                        st.text_input(f"values # {i}", key=f'values{i}',value = st.session_state.get(f'values{i}',''))
                        mongo_data[f'values{i}'] = st.session_state.get(f'values{i}','')
                factor_level = st.text_input('FACTOR LEVEL',key = 'factor_level',value ='')
                mongo_data['factor_level'] = st.session_state.get('factor_level','')
                print(type(num_factors))
                print(num_factors)
                resolution = st.text_input('RESOLUTION',key = 'resolution',value = '')
                mongo_data['resolution'] = st.session_state.get('resolution','')
                num_response = st.text_input('NUMBER OF RESPONSE VARIABLES',key='num_response',value = '')
                mongo_data['num_response'] = st.session_state.get('num_response','')
                response_names = st.text_input('RESPONSE VARIABLE NAMES',key = 'response_names',value = '')
                mongo_data['response_names'] = st.session_state.get('response_names','')
                col1, col2 = st.columns([1,1])
                with col1:
                    tmp_button = st.form_submit_button(label='Submit')
                with col2:
                    clear_button = st.form_submit_button("Clear",on_click=clear_cache)
                # if tmp_button:
                #     st.session_state['last_used'] = 'submit'
                # print('factors',factors)
                # print('values',values)
                actual_values = ''
                # for i in range(1,int(num_factors)+1):
                #     actual_values+= f"{st.session_state[f'factor{i}']}: {st.session_state[f'values{i}']},"
                # print(actual_values)
                # prompt = f"Generate a full design of experiment matrix in a table for a {doe_name} example. Factors = {num_factors}, {doe_type}, level {factor_level}, Resolution is {resolution}. Update the table with actual factor values instead of the coded +1 and -1.The actual values are {actual_values} {response_names} with empty values . And return all the 16 runs. Return the response in a json format having each factor and response has key and its respective values in a list and only return the dictionary. Don't return the generated table or any other texts."
                # messages=[{"role": "user", "content": prompt}]
                filled_values = st.session_state[f'factor{int(num_factors)}']
                # if not col.find_one({ "_id": st.session_state.doe_name}):
                #     print('gggggggggggggggggggggggg')
                #     # st.write(dict_values)
                #     col.insert_one({'_id': st.session_state.doe_name,'values' :mongo_data })
                #     print('kkkkkkkkkkkkkkkkkkkkk')
                # else:
                #     col.update_one({'_id':st.session_state.doe_name},{'$set': {'values': mongo_data }})

                if tmp_button and filled_values:
                    for i in range(1,int(num_factors)+1):
                        actual_values+= f"{st.session_state[f'factor{i}']}: {st.session_state[f'values{i}']},"
                    print(actual_values)
                    prompt = f"Generate a full design of experiment matrix in a table for a {doe_name} example. Factors = {num_factors}, {doe_type}, level {factor_level}, Resolution is {resolution}. Update the table with actual factor values instead of the coded +1 and -1.The actual values are {actual_values} each {response_names} with empty values. Return the response in a json format having each factor and response has key and its respective values in a list and only return the dictionary.Return full table.Don't return the generated table, any asumption text or any other texts."
                    messages=[{"role": "user", "content": prompt}]
                    print(doe_name)
                    # st.write(messages)
                    completion = client.chat.completions.create(model="gpt-4-1106-preview",messages=messages,temperature = 0)
                    print('bezbezb')
                    response = completion.choices[0].message.content
                    # response1 = response.split('\n',1)
                    # print('response1',response1)
                    response = response.split('\n',1)[1]
                    print('dict_response',response)
                    response = response.rsplit('\n',1)[0]
                    # st.write(completion.choices[0].message.content)
                    print('final_response',response)
                    # st.write(response)
                    print('type',type(response))
                    dict_response = ast.literal_eval(response)

                    # data = {'Name': ['Tom', 'nick', 'krish', 'jack'], 'Age': [20, 21, 19, 18], 'qbc':["","","",""]}
                    df = pd.DataFrame(dict_response)
                    # print('bsehbeh')
                    # return df
                    # editable_df = st.data_editor(dict_response)
                    # print('bsenbesn')
                    return df,messages
                else:
                    st.stop()
                # pass
        # print('num_factors',num_factors)
        # if num_factors:
        #     print('vgebe')
        #     if isinstance(int(num_factors),int):
        #         break
        
                # pass
    # except Exception as e:
    #     # t = st.form_submit_button(label='Submit')
    #     print(e)
    #     # pass
    #     st.stop()
            # pass
        # for i in range(1,int(num_factors)+1):
        #     st.text_input(f"Factor # {i}", key=f'factor{i}')
        #     st.text_input(f"values # {i}", key=f'values{i}')
        # factor_level = st.text_input('FACTOR LEVEL')
        # print(type(num_factors))
        # print(num_factors)
        # resolution = st.text_input('RESOLUTION')
        # num_response = st.text_input('NUMBER OF RESPONSE VARIABLES')
        # response_names = st.text_input('RESPONSE VARIABLE NAMES')
        # # tmp_button = st.form_submit_button(label='Submit')
        # # if tmp_button:
        # #     st.session_state['last_used'] = 'submit'
        # # print('factors',factors)
        # # print('values',values)
        # actual_values = ''
        # for i in range(1,int(num_factors)+1):
        #     actual_values+= f"{st.session_state[f'factor{i}']}: {st.session_state[f'values{i}']},"
        # print(actual_values)
        # prompt = f"Generate a full design of experiment matrix in a table for a {doe_name} example. Factors = {num_factors}, {doe_type}, level {factor_level}, Resolution is {resolution}. Update the table with actual factor values instead of the coded +1 and -1.The actual values are {actual_values} {response_names} with empty values . And return all the 16 runs. Return the response in a json format having each factor and response has key and its respective values in a list and only return the dictionary. Don't return the generated table or any other texts."
        # messages=[{"role": "user", "content": prompt}]
        # if st.form_submit_button(label='Submit'):
        #     with st.spinner("Executing ...."):
        #         print(doe_name)
        #         completion = client.chat.completions.create(model="gpt-4-1106-preview",messages=messages)
        #         response = completion.choices[0].message.content
        #         # response1 = response.split('\n',1)
        #         # print('response1',response1)
        #         response = response.split('\n',1)[1]
        #         print('dict_response',response)
        #         response = response.rsplit('\n',1)[0]
        #         # st.write(completion.choices[0].message.content)
        #         print('final_response',response)
        #         st.write(response)
        #         dict_response = ast.literal_eval(response)
        #         # data = {'Name': ['Tom', 'nick', 'krish', 'jack'], 'Age': [20, 21, 19, 18], 'qbc':["","","",""]}
        #         # df = pd.DataFrame(data)
        #         # print('bsehbeh')
        #         # return df
        #         # editable_df = st.data_editor(dict_response)
        #         # print('bsenbesn')
        #         return dict_response,messages
        # else:
        #     st.stop()
            # print(favorite_command)
            # if "Taste" in editable_df.columns:
            #     editable_df.drop('Taste',inplace=True,axis = 1)
            # editable_df['Taste'] = favorite_command
            # print('favorite_command',favorite_command)
            # print('editable_df',editable_df)
            # messages.append({"role": "assistant", "content": editable_df.to_json()})
            # with st.form(key='my_form3'):
            #     if st.form_submit_button(label='ANALYSE DOE'):
            #         with st.spinner("Training ongoing"):
            #             favorite_command = editable_df["qbc"].to_list()
            #             print(favorite_command)
            #             print(editable_df.columns)
        # else:
        #     st.stop()
        #     print('bewbewb')




    # st.write(e, "=", st.session_state["data"][e])
# completion = client.chat.completions.create(
#   model="gpt-4-1106-preview",
#   messages=[
#     {"role": "user", "content": "Generate a full design of experiment matrix in a table for a cookie baking example. Factors = 5, Fractional factorial design, level 2, Resolution is 5. Response should be in a table with all the factors values as column name. Update the table with actual factor values instead of the coded +1 and -1.The actual values are Oven Temperature (°F): 325°F, 375°F, Baking Time (Minutes): 10 min, 15 min, Type of Flour: All-Purpose, Whole Wheat, Amount of Sugar (Cups): 0.75 cups, 1.25 cups, Type of Fat: Butter, Margarine and taste as response 1,10 And return all the 16 runs. Return only the generated table and avoid unwanted texts"}
#   ]
# )

# print(completion.choices[0].message.content)
def clear_cache():
    st.cache_data.clear()
    # reasonable_value = False
    st.session_state.reasonable_value = ''
    st.session_state.response_df =''
    variable_values.clear()
    # st.cache_resource.clear()
    st.session_state.analyseDoe = ''
    st.session_state.click = False

def response_clear_cache():
    prediction_profiler.clear()
    st.session_state.response_df =''
    st.session_state.reasonable_value = ''
    st.session_state.analyseDoe = ''

@st.cache_resource(experimental_allow_widgets=True)
def prediction_profiler(editable_df,num_response):
    print('info',editable_df.info())
    print('prediction_profiler main',editable_df)
    resp_columns = editable_df.select_dtypes(include=['object']).columns
    # resp_columns = resp_columns[-int(num_response):]

    # print('object_response',resp_columns)
    # print('num_response',num_response)
    if not st.session_state.reasonable_value:
        for col in resp_columns[-int(num_response):]:
            print(col)
            editable_df[col] = editable_df[col].astype(int)
    # editabledf_row = editable_df.shape[0]

    ols_output = {}
    pred_output = {}
    num_response = int(num_response)
    print(editable_df.info())
    # # editable_df["FlourType"] = editable_df["FlourType"].map({"All-Purpose": 0, "Whole Wheat": 1})
    # # editable_df["FatType"] = editable_df["FatType"].map({"Butter": 0, "Margarine": 1})
    object_features = editable_df.select_dtypes(include=['object']).columns

    resp_variables = editable_df.columns[-num_response:]
    # import math
    # fig, axs = plt.subplots(math.ceil(num_response/2),2,sharex=True)
    # st.write(axs.shape)
    # fig.suptitle('Run test')
    # title = 0
    # axs_columns = axs.shape[1] if len(axs.shape)>1 else 1 
    # st.write(axs_columns)
    # plt.xlim(1, 10)
    # for row in range(axs.shape[0]):
    #     if not axs_columns == 1:
    #         for column in range(axs_columns):
    #             st.write('vebwe',editable_df[resp_variables[title]].to_list())
    #             axs[row,column].plot(range(1,editabledf_row+1), editable_df[resp_variables[title]].to_list(),'-o')
    #             axs[row,column].set_title(resp_columns[title])
    #             title+=1
    #     else:
    #         st.write(resp_variables[title])
    #         axs[row].plot(range(1,editabledf_row+1), editable_df[resp_variables[title]].to_list(),'-o')
    #         axs[row].set_title(resp_variables[title])
    #         title+=1
    # st.pyplot(fig)
    factors = editable_df.columns.to_list()[:-num_response]
    factors_types = editable_df.dtypes[:-num_response]
    st.session_state.factors_types = factors_types
    # factors = list(map(lambda x: f'C({x})' if x in object_features else x,factors))
    # renamed_factors = []
    # for factor in factors:
    #     renamed_factors.append(factor.replace(' ','_'))
    fact_comb = []
    for i in range(1,len(factors)):
        # print(a[i-1:i])
        # print(a[i+1:])
        fact_comb.extend(list(product(factors[i-1:i],factors[i:])))

    # print(l)
    fact_comb = [f'{val[0]}*{val[1]}' for val in fact_comb]
    print('ttttttttttttttttt',type(fact_comb))
    print('fffffffff',type(factors))
    # st.write(fact_comb)
    print('bbbbbbbbbbbbbbb',fact_comb)
    fact_comb = factors + fact_comb
    print('ggggggggggggg',fact_comb)
    # st.write(fact_comb)
    factors_joined = ' + '.join(fact_comb)
    t_values = {}
    print(factors_joined)
    for res in resp_variables:
        ols_eq = f'{res} ~ {factors_joined}'
        print('ols_eq',ols_eq)
        output = ols(ols_eq,data = editable_df).fit()
        pred_output[res] = output
        df = pd.DataFrame(output.params)
        values = df.T.values.tolist()[0]
        ols_output[res] = dict(zip(df.T.columns.to_list(),values))
        df_tvalues = pd.DataFrame(output.tvalues)
        t_valueslist = df_tvalues.T.values.tolist()[0]
        t_values[res] = dict(zip(df_tvalues.T.columns.to_list(),t_valueslist))

    st.session_state.pred_output = pred_output
    st.session_state.ols_output = ols_output
    st.session_state.t_values = t_values
    return ols_output,pred_output

def add_slider(ols_output,pred_output):
    # print('red',st.session_state['oventemp'])
    # print('normal',st.session_state.normal)
    # with st.form(key='my_form5'):
    fig, ax = plt.subplots()
    outputs = []
    response = []
    # for output in ols_output.items():
    #     output_value = 0
    #     print(output)
    #     response.append(output[0])
    #     for key,val in output[1].items():
    #         if key == 'Intercept':
    #             output_value+= val
    #         else:
    #             print(int(val))
    #             output_value+= float(st.session_state[key]) * float(val)
    feature_values = []
    print('st.session_state.features',st.session_state.features)
    print('type',type(st.session_state.features))
    for feature in st.session_state.features:
        feature_values.append(st.session_state[feature])
    print('feature_values',feature_values)
    feature_valuesTypes = []
    feature_valuesTypes_dic = dict(zip(feature_values,st.session_state.factors_types))
    for feature,type1 in feature_valuesTypes_dic.items():
        if type1 == 'object':
            type1 = str
            print(type1)
        elif type1 == 'int64' or type1 == 'int32':
            type1 = int
        elif type1 == 'float64':
            type1 = float
        print(type1)
        feature_valuesTypes.append(type1(feature))
    print('feature_valuesTypes',feature_valuesTypes)
    output_df = pd.DataFrame([feature_valuesTypes],columns=st.session_state.features)
    # print(type(output_df.iloc[0,:]))
    # print(output_df.iloc[0,:])
    print('info',output_df.info())
    print('dataframe_slider',output_df)
    abc = output_df.iloc[0,:]
    # a = [375,10,'All-Purpose',0.75,'Butter']
    # b = ['OvenTemp','BakingTime','FlourType','SugarAmount','FatType']
    # data = pd.DataFrame([a],columns = b)
    # print('data_info',data.info())
    # print('data_type',type(data))
    # data
    for output,ols_object in pred_output.items():
        response.append(output)
        print('ols_object',ols_object)
        print(type(output_df))
        output_value = ols_object.predict(output_df)
        print('output_value',output_value)
        outputs.append(output_value.tolist()[0])
    colors = ['C0','C1']
    print('outputs',outputs)
    ax.bar(response, outputs)
    ax.set_ylim(0, 10)
    # ax.axes([10, 0.3, .5, .5])
    # st.write(outputs)
    return fig
    st.pyplot(fig)
    # st.slider(name, min, max, default)
    # sliders = []
    # ax = plt.axes([0.1, 0.02+pos, 0.8, 0.02], facecolor='lightgoldenrodyellow')
    # slider = Slider(ax, name, min, max)
    # sliders.append(slider)
    # def update(val):
    #     print(slider.__dict__)
    #     print(slider.label)
    #     # print(slider.label[2])
    #     # print(dir(slider.label.text))
    #     print(getattr(slider.label, '_label')) # _text
    #     print(getattr(slider.label, '_text'))
    #     # setattr(self.obj, name, val)
    #     # self.l[0].set_ydata(self.obj.series())
    #     # self.fig.canvas.draw_idle()
    # slider.on_changed(update)
def add_line(ax, xpos, ypos):
    line = plt.Line2D([xpos, xpos], [1, ypos],
                    transform=ax.transAxes, color='gray')
    line.set_clip_on(False)
    ax.add_line(line)

def label_len(my_index,level):
    labels = my_index.get_level_values(level)
    return [(k, sum(1 for i in g)) for k,g in groupby(labels)]

def label_group_bar_table(ax, df):
    ypos = -.1
    scale = 1/df.index.size
    # print('scale',scale)
    # print('df.index.size',df.index.size)
    for level in range(df.index.nlevels)[::-1]:
        pos = 0
        # print(level)
        for label, rpos in label_len(df.index,level):
            lxpos = (pos + 0.5 * rpos)*scale
            # print('lxpos',lxpos)
            ax.text(lxpos, ypos, label, ha='right', transform=ax.transAxes, size = 'xx-small')
            add_line(ax, pos*scale, ypos)
            pos += rpos
        add_line(ax, pos*scale , ypos)
        ypos -= .1

if __name__ == '__main__':
    # st.button(label='Assume reasonable value for the response variables',on_click=add_slider)
    # add_slider()
    try:
        # db_client = MongoClient('mongodb+srv://m220student:Gokulnath@cluster0.qp8h2nr.mongodb.net/')
        # load_dotenv()
        # db_name = "electronics"
        # collection_name = "DOE"
        # collection = db_client[db_name][collection_name]
        # print(collection)
        # # index_name = "langchain_demo"

        # db = db_client.electronics
        # col = db.DOE
        # st.session_state['last_used'] = ''
        # dict_values = {}
        # mongo_data = {}
        # available_list = [str(id) for id in col.distinct('_id')]
        # available_list = list(filter(lambda x : True if x else False,available_list))
        # if 'other' not in available_list:
        #     available_list.append('other')
        # # st.write(available_list)
        # select_name = st.selectbox('Select preused doe name',available_list,key = 'select_name',index=None)
        # st.session_state.click = True
        dict_response,messages = variable_values()

        editable_df = st.data_editor(dict_response)
        print('main',editable_df.info())
        messages.append({"role": "assistant", "content": editable_df.to_json()})
        # st.session_state.response_df = editable_df
        # st.download_button("Download DOE table",st.session_state.get('response_df'),file_name = f'{st.session_state.doe_name}_DOE.txt')
        with st.form(key='my_form3'):
            col1, col2,col3 = st.columns([1,1,1])
            with col1:
                reasonable_button = st.form_submit_button(label='Stimulate values for response variables')
            with col2:
                st.form_submit_button("Clear DOE response values",on_click=response_clear_cache)

            if reasonable_button:
                #   print('besnhsen')
                st.session_state.reasonable_value = 'yes'
                messages.append({"role": "user", "content": "Assume reasonable value for the response variables based on the other features/inputs(values must be in range 0 to 10) and return the response in json format with each feature and response in separate key and its corresponsing values in a list. Don't return any unwanted text and any unicode text."})
                completion = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=messages,temperature = 0)
                response = completion.choices[0].message.content
                # st.write(response)
                print(response)
                print(type(response))
                # response = response.split('\n',1)[1]
                # print('dict_response',response)
                # response = response.rsplit('\n',1)[0]
                # st.write(completion.choices[0].message.content)
                print('final_response',response)
                # st.write(response)
                dict_response = ast.literal_eval(response)
                # data = {'Name': ['Tom', 'nick', 'krish', 'jack'], 'Age': [20, 21, 19, 18], 'qbc':["","","",""]}
                df = pd.DataFrame(dict_response)
                st.dataframe(df)
                st.session_state.response_df = df
                messages.append({"role": "assistant", "content": df.to_json()})
                # print(editable_df.head())
            elif st.session_state.get('reasonable_value',''):
                # st.session_state.response_df = df
                st.dataframe(st.session_state.response_df)
            # else:
            #     st.stop()
        # with st.form(key='my_form4'):
            # col1, col2 = st.columns([1,1])
            # with col1:
            analyse_button = st.form_submit_button(label='ANALYSE DOE')

            if analyse_button:
                print('anaaaaaaaaaaaaaa')
                if not st.session_state.get('reasonable_value',''):
                    st.write(st.session_state.get('reasonable_value',''))
                    st.session_state.response_df = editable_df
                # with st.spinner("Training ongoing"):

                #     # favorite_command = editable_df["qbc"].to_list()
                #     # print(favorite_command)
                #     analyse_messages = []
                #     analyse_messages.append({"role": "assistant", "content": editable_df.to_json()})
                #     analyse_messages.append({"role": "user", "content": "Can you create a scatterplot for the each column for the above json"})
                #     # print(editable_df.head())
                #     completion = client.chat.completions.create(
                #     model="gpt-4-1106-preview",
                #     messages=analyse_messages,temperature = 0)
                #     response = completion.choices[0].message.content
                #     print('doe,response',response)
                #     st.write(response)
                    # fig, ax = plt.subplots()
                    # vars = [['red',1,10,0],['normal',100,500,0]]
                    # num_factors = 5
                print('response_df',st.session_state.response_df)
                ols_output,pred_output = prediction_profiler(st.session_state.response_df,st.session_state.num_response)
                for _,pred in pred_output.items():
                    st.write(pred.summary())
                st.session_state.ols_output = ols_output
                # st.write(ols_output)
                vars = {}
                columns = st.session_state.response_df.columns.to_list()
                for i in range(1,int(st.session_state.num_factors)+1):
                    session_values = st.session_state[f'values{i}'].split(',')
                    if not session_values:
                        session_values = st.session_state[f'values{i}'].split('to')

                    vars[columns[i-1]] =  session_values
                    st.session_state.vars = vars
                # vars['oventemp'] = ['345','350']
                # vars['FlourType'] = ['Butter',"Margarine"]
                # ols_output = {}
                st.session_state.analyseDoe = 'Done'
        # if st.session_state.analyseDoe =='Done':
            # st.session_state.analyseDoe = ''
        if st.session_state.get('analyseDoe','') == 'Done':
            print('brrrrrrrrrr',st.session_state.response_df)
            editabledf_row = st.session_state.response_df.shape[0]
            csv = st.session_state.response_df.to_csv().encode('utf-8')
            st.download_button(label="Download DOE table",data=csv,file_name='DOE_table.csv',mime='text/csv')

            ols_output = {}
            pred_output = {}
            num_response = int(st.session_state.num_response)
            # print(editable_df.info())
            # editable_df["FlourType"] = editable_df["FlourType"].map({"All-Purpose": 0, "Whole Wheat": 1})
            # editable_df["FatType"] = editable_df["FatType"].map({"Butter": 0, "Margarine": 1})
            # object_features = editable_df.select_dtypes(include=['object']).columns

            resp_variables = st.session_state.response_df.columns[-num_response:]

            fig, axs = plt.subplots(math.ceil(num_response/2),2,sharex=True,figsize=(17, 5))
            # fig = plt.figure(figsize=(20,3))
            st.title('Run order test')
            # st.write(axs.shape)
            fig.suptitle('Run order test')
            title = 0
            axs_columns = axs.shape[1] if len(axs.shape)>1 else 1 
            # st.write(axs_columns)
            xlim = range(1,editabledf_row,2)
            plt.xticks(xlim)
            for row in range(axs.shape[0]):
                if not axs_columns == 1:
                    for column in range(axs_columns):
                        if title < num_response:
                            # st.write('vebwe',editable_df[resp_variables[title]].to_list())
                            axs[row,column].plot(range(1,st.session_state.response_df.shape[0]+1), st.session_state.response_df[resp_variables[title]].to_list(),'-o')
                            axs[row,column].set_title(resp_variables[title])
                            title+=1
                        else:
                            title = 0
                else:
                    # st.write(resp_variables[title])
                    axs[row].plot(range(1,st.session_state.response_df.shape[0]+1), st.session_state.response_df[resp_variables[title]].to_list(),'-o')
                    axs[row].set_title(resp_variables[title])
                    title+=1
            st.pyplot(fig)
            st.session_state.runCodePlot = fig
            df = pd.DataFrame(st.session_state.t_values)
            print('pareto_columns',df)
            df.drop('Intercept',axis = 0,inplace=True)
            # st.write('pareto columns',df)
            parto_columns = df.columns.to_list()

            # df = df.abs()
            # st.write(type(df))
            # st.write(df)
            # df['pareto'] = df.Color.cumsum()

            fig, axs = plt.subplots(math.ceil(num_response/2),2,figsize=(17, 5))
            st.title('Pareto Chart')
            fig.suptitle('Pareto Chart')
            title = 0
            for row in range(axs.shape[0]):
                if not axs_columns == 1:
                    for column in range(axs_columns):
                        if title < len(parto_columns):
                            parto_column = parto_columns[title]
                            df[parto_column] = df[parto_column].abs()
                            # st.write('brnbrrrr')
                            # print('ddddddddddd',df)
                            df.sort_values(by = parto_column,inplace=True,ascending = False)
                            # st.write(df)
                            df['pareto'] = df[parto_column].cumsum()
                            # df = df[[parto_column,'pareto']]
                            ax1 = df.plot(use_index=True, y=parto_column,  kind='bar', ax=axs[row,column])
                            ax2 = df.plot(use_index=True, y='pareto', marker='D', color="C1", kind='line', ax=axs[row,column], secondary_y=True)
                            fig.autofmt_xdate()
                            # ax2.set_ylim([0,110])
                            title+=1
                        else:
                            title = 0

                else:
                    parto_column = parto_columns[title]
                    df[parto_column] = df[parto_column].abs()
                    df.sort_values(by = parto_column,inplace=True,ascending = False)
                    df['pareto'] = df[parto_column].cumsum()
                    # print('abccccc',type(df))
                    # print('abdddddddddd',df)
                    ax1 = df.plot(use_index=True, y=parto_column,  kind='bar', ax=axs[row])
                    ax2 = df.plot(use_index=True, y='pareto', marker='D', color="C1", kind='line', ax=axs[row], secondary_y=True)
                    fig.autofmt_xdate()
                    # ax2.set_ylim([0,110])
                    title+=1

            st.pyplot(fig)
            st.session_state.paretoPlot = fig
            factors = st.session_state.response_df.columns.to_list()[:-num_response]
            object_features = st.session_state.response_df.select_dtypes(include=['object']).columns.to_list()
            factors_num = [factor for factor in factors if factor not in object_features]
            # st.write(factors_num)
            # factors = factors[0:2]
            # fig, axs = plt.subplots(math.ceil(len(factors)/2),2,figsize=(20, 5))
            # axs_columns = axs.shape[1] if len(axs.shape)>1 else 1
            # st.write(axs_columns,axs.shape)
            save_fig = []
            for resp in resp_variables:
                st.title('Main effects for '+ resp)
                y = st.session_state.response_df[resp].to_list()
                fig, axs = plt.subplots(math.ceil(len(factors_num)/2),2,figsize=(20, 5),sharex=False)
                fig.suptitle(resp)
                axs_columns = axs.shape[1] if len(axs.shape)>1 else 1
                title = 0
                for row in range(axs.shape[0]):
                    if not axs_columns == 1:
                        for column in range(axs.shape[1]):
                            if title < len(factors_num):
                                x = st.session_state.response_df[factors_num[title]].to_list()
                                # st.write(f'{factors_num[title]}',x)
                                # st.write('qqqqqqqqqqqqqqqqq',x)
                                # print(type())
                                axs[row,column].plot(x, y,'o')
                                axs[row,column].set_xlabel(factors_num[title])
                                # This will fit the best line into the graph
                                axs[row,column].plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='red')
                                plt.tight_layout()
                                # fig.autofmt_xdate()
                                # ax2.set_ylim([0,110])
                                title+=1
                            else:
                                title = 0
                                # st.write(row,column)
                                # fig.delaxes(axs[row,column])
                    else:
                        x = st.session_state.response_df[factors_num[title]].to_list()
                        # st.write('qqqqqqqqqqqqq',x)
                        # st.write(f'{factors_num[title]} else',x)
                        # print('qqqqqqqqqqqqq',x)
                        # print(x,y)                    
                        axs[row].plot(x, y,'o')
                        axs[row].set_xlabel(factors_num[title])
                    # This will fit the best line into the graph
                        axs[row].plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='red')
                        # fig.autofmt_xdate()
                        title+=1
                        # ax2.set_ylim([0,110])
                        # title+=1
            # ax1 = df.plot(use_index=True, y='Color',  kind='bar', ax=axes)
            # ax2 = df.plot(use_index=True, y='pareto', marker='D', color="C1", kind='line', ax=axes, secondary_y=True)
            # fig.autofmt_xdate()
            # ax2.set_ylim([0,110])
                st.pyplot(fig)
                save_fig.append(fig)
            st.session_state.maineffect = save_fig
            # st.write(factors)
            print(factors)
            df = st.session_state.response_df.groupby(factors).max()
    # df = df.set_index(['OvenTemp','BakeTime','FlourType','SugarAmount','FatType',['Color','Texture','Taste']])[['Color','Texture','Taste']].unstack()
    # print(df)
            st.title('Variablity plot')
            fig,ax = plt.subplots(1,2)
            fig.suptitle('Variablity plot')
            # st.write(ax.shape)
            print(df)
            # st.write(df)
            ax = df.plot(marker='o', linestyle='none', xlim=(-1,df.shape[0]), ylim=(1,10),figsize=(15, 6))
            #Below 2 lines remove default labels
            # print(ax[0])
            ax.set_xticklabels('')
            ax.set_xlabel('')
            label_group_bar_table(ax, df)
            # you may need these lines, if not working interactive
            plt.tight_layout()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(plt.show())
            # st.pyplot(fig)
            plt.show()
            # var_buf = io.BytesIO()
            # plt.savefig(var_buf)
            # plt.close()
            st.session_state.images = [st.session_state.runCodePlot,st.session_state.paretoPlot,st.session_state.maineffect]
            file_names = ['runcodeplot','paretoplot',resp_variables]
            # st.write(st.session_state.images)
            zip_file_name = "export.zip"
            # print("Creating archive: {:s}".format(zip_file_name))
            with ZipFile(zip_file_name, mode="w") as zf:
                for index,i in enumerate(st.session_state.images):
                    # plt.plot([i, i])
                    # print()
                    if isinstance(i,list):
                        for j_index,j in enumerate(i):
                            buf = io.BytesIO()
                            j.savefig(buf)
                            # plt.close()
                            img_name = f"MainEffects_{file_names[index][j_index]}.png"
                            # print(" Writing image {:s} in the archive".format(img_name))
                            zf.writestr(img_name, buf.getvalue())


                    else:
                        buf = io.BytesIO()
                        i.savefig(buf)
                        # plt.close()
                        img_name = f"{file_names[index]}.png"
                        # print(" Writing image {:s} in the archive".format(img_name))
                        zf.writestr(img_name, buf.getvalue())

                # img_name = "variablity_plot.png"
                # # print("  Writing image {:s} in the archive".format(img_name))
                # zf.writestr(img_name, var_buf.getvalue())

            with open(zip_file_name, "rb") as fp:
                btn = st.download_button(
                    label="Download DOE statistics",
                    data=fp,
                    file_name=f"{st.session_state.doe_name.replace(' ','')}_doeStatistics.zip",
                    mime="application/zip"
                )
            form = st.form("my_form5")
            features = []

            for key1,value in st.session_state.vars.items():
                value = list(map(lambda x: x.strip(),value))
                # add_slider(i*0.03, var[0], var[1], var[2],var[3])
                features.append(key1)
                try:
                    # st.write('try')
                    # st.write(np.arange(float(value[0]),float(value[1])+0.5,0.5))
                    # float_values = [x/10.0]
                    if len(np.arange(float(value[0]),float(value[-1]),0.5)) == 1:
                        form.select_slider(key1, np.arange(float(value[0]),float(value[-1])+0.05,0.05),key = key1)
                    else:
                        form.select_slider(key1, np.arange(float(value[0]),float(value[-1])+0.5,0.5),key = key1)
                except Exception as e:
                    # print(e)
                    form.select_slider(key1, value,key = key1)
            st.session_state.features = features
            if form.form_submit_button("Submit slider response"):
                fig = add_slider(st.session_state.ols_output,st.session_state.pred_output)
                print(fig)
                st.pyplot(fig)
        # st.sidebar.button("Refresh",on_click=clear_cache)
        # st.sidebar.button("Refresh doe table response variable",on_click=response_clear_cache)
    except Exception as e:
        print(e)
        pass


    # editable_df = st.data_editor(dict_response)
    # messages.append({"role": "assistant", "content": editable_df.to_json()})
    # with st.form(key='my_form3'):
    #     if st.form_submit_button(label='Assume reasonable value for the response variables'):
    #         #   print('besnhsen')
    #         messages.append({"role": "user", "content": "Assume reasonable value for the response variables and return the dataframe. Don't return any unwanted text."})
    #         completion = client.chat.completions.create(
    #         model="gpt-4-1106-preview",
    #         messages=messages)
    #         response = completion.choices[0].message.content
    #         st.write(response)
    #         # print(editable_df.head())
    #     if st.form_submit_button(label='ANALYSE DOE'):
    #         with st.spinner("Training ongoing"):
    #             favorite_command = editable_df["qbc"].to_list()
    #             print(favorite_command)
    #             print(editable_df.head())
    # print('messages',messages)
    # response_value = st.button(label='Assume reasonable value for the response variables')
    # time.sleep(50)
    # if response_value:
        # messages.append({"role": "user", "content": "Assume reasonable value for the response variables and return the dataframe. Don't return any unwanted text."})
        # completion = client.chat.completions.create(
        # model="gpt-4-1106-preview",
        # messages=messages)
        # response = completion.choices[0].message.content
        # st.write(response)


    # messages.append({"role": "user", "content": "return the statistical analysis like fit model, factor effects, regression equation, graphical analysis using variability charts etc.."})

    # analyse_doe = st.button(label='ANALYSE DOE')
    # if analyse_doe:
    #     st.session_state['last_used'] ='ANALYSE DOE'
    # if st.session_state['last_used'] == 'ANALYSE DOE':
    #     completion = client.chat.completions.create(
    #         model="gpt-4-1106-preview",
    #         messages=messages)
    #     response = completion.choices[0].message.content
    #     print(response)
    #     st.write(response)

import pyecma376_2
import xmltodict
import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
import time
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score
import zipfile
import xmltodict
import glob
import shutil
"""
prepare the aas submodel template from the xlm file within a aasx-format zip file
"""
RELATIONSHIP_TYPE_AASX_ORIGIN = "http://www.admin-shell.io/aasx/relationships/aasx-origin"
RELATIONSHIP_TYPE_AAS_SPEC = "http://www.admin-shell.io/aasx/relationships/aas-spec"
RELATIONSHIP_TYPE_AAS_SPEC_SPLIT = "http://www.admin-shell.io/aasx/relationships/aas-spec-split"
RELATIONSHIP_TYPE_AAS_SUPL = "http://www.admin-shell.io/aasx/relationships/aas-suppl"

def get_text_by_language(description_dict, language='en'):
    if description_dict is None:
        return 'None'
    else:
        lang_list = description_dict.get('langStringTextType', [])
        if isinstance(lang_list, list):
            for item in lang_list:
                if item.get('language') == language:
                    try:
                        return item.get('text')
                    except:
                        return 'None'
            # If 'en' language text is not found or the list is empty, return the first item's text or None
            return lang_list[0].get('text') if lang_list else None
        elif isinstance(lang_list, dict):
            try:
                return lang_list.get('text') if lang_list else None
            except:
                return 'None'
        else:
            return 'None'



def get_text_by_language_old(description_dict, language='en'):
    if description_dict is None:
        return 'None'
    else:
        lang_list = description_dict.get('aas:langString', [])
        if isinstance(lang_list, list):
            for item in lang_list:
                if item.get('@lang') == language:
                    try:
                        return item.get('#text')
                    except:
                        return 'None'
            # If 'en' language text is not found or the list is empty, return the first item's text or None
            return lang_list[0].get('#text') if lang_list else None
        elif isinstance(lang_list, dict):
            try:
                return lang_list.get('#text') if lang_list else None
            except:
                return 'None'
        else:
            return 'None'

def smc_to_smes_fromXML(list_dict):
    extracted_data = []
    if isinstance(list_dict, list):
        pass
    else:
        list_dict = [list_dict]
    for index, item in enumerate(list_dict):
        nested_key = next(iter(item.keys()), None)
        nested_dict = item.get(nested_key, {})
        if nested_key == 'aas:submodelElementCollection':
            if nested_dict['aas:value'] != None:
                temp = nested_dict.get('aas:value', None)['aas:submodelElement']
                if isinstance(temp, list):
                    extracted_data.extend(smc_to_smes_fromXML(temp))
                else:
                    temp_key = next(iter(temp.keys()), None)
                    if temp_key == 'aas:submodelElementCollection':
                        extracted_data.extend(
                            smc_to_smes_fromXML(temp['aas:submodelElementCollection']['aas:value']['aas:submodelElement']))
                    else:
                        extracted_data.extend(temp)
        else:
            # Extract 'aas:idShort' and 'aas:value'
            id_short = nested_dict.get('aas:idShort', None)
            description = str(get_text_by_language_old(nested_dict.get('aas:description', None)))
            if 'aas:semanticId' in nested_dict and nested_dict.get('aas:semanticId', None) is not None and 'aas:keys' in nested_dict.get('aas:semanticId', None) and nested_dict.get('aas:semanticId', None)['aas:keys'] is not None:
                semanticId = str(nested_dict.get('aas:semanticId', None)['aas:keys']['aas:key'])
            else:
                semanticId = 'None'
            # Add the information to our list if both 'aas:idShort' and 'aas:value' are found
            extracted_data.append((str(index).zfill(2), id_short, description, semanticId))
    return extracted_data


def smc_to_smes_fromXML_new(dict):
    extracted_data = []
    for index, key in enumerate(dict):
        if key == 'submodelElementCollection' or key == 'submodelElementList':
            if isinstance(dict[key], list):
                for list_dict in dict[key]:
                    if 'value' in list_dict and list_dict['value'] != None:
                        extracted_data.extend(smc_to_smes_fromXML_new(list_dict['value']))
            else:
                if dict[key]['value'] != None:
                    extracted_data.extend(smc_to_smes_fromXML_new(dict[key]['value']))

        if isinstance(dict[key], list):
            for list_dict in dict[key]:
                id_short = list_dict['idShort']
                if 'description' in list_dict  and list_dict['description'] != None:
                    description = str(get_text_by_language(list_dict['description']))
                else:
                    description = 'None'
                if 'semanticId' in list_dict and list_dict['semanticId'] != None and 'keys' in list_dict['semanticId'] and list_dict['semanticId']['keys'] != None:
                    semanticId = str(list_dict['semanticId']['keys']['key'])
                else:
                    semanticId = 'None'
                extracted_data.append((str(index).zfill(2), id_short, description, semanticId))
        else:
            # Extract 'aas:idShort' and 'aas:value'
            if 'idShort' in dict[key]:
                id_short = dict[key]['idShort']
                if 'description' in dict[key] and dict[key]['description'] != None:
                    description = str(get_text_by_language(dict[key]['description']))
                else:
                    description = 'None'
                if 'semanticId' in dict[key] and dict[key]['semanticId'] != None:
                    semanticId = str(dict[key]['semanticId']['keys']['key'])
                else:
                    semanticId = 'None'
                # Add the information to our list if both 'aas:idShort' and 'aas:value' are found
                extracted_data.append((str(index).zfill(2), id_short, description, semanticId))
    return extracted_data




def process_aasx_file(aasx_path):
    """
    Process a single AASX file to extract AAS and Concept Description data.

    Args:
        aasx_path (str): Path to the AASX file.

    Returns:
        tuple: A tuple containing two DataFrames, one for AAS data and one for Concept Descriptions.
    """
    with pyecma376_2.ZipPackageReader(aasx_path) as reader:
        # core_rels = reader.get_related_parts_by_type()
        # aasx_origin_part = core_rels["http://www.admin-shell.io/aasx/relationships/aasx-origin"][0]
        aasx_origin_part = "/aasx/aasx-origin"
        for aas_part in reader.get_related_parts_by_type(aasx_origin_part)["http://www.admin-shell.io/aasx/relationships/aas-spec"]:
            with reader.open_part(aas_part) as p:
                dict_aas = xmltodict.parse(p.read())
    if 'aas:aasenv' in dict_aas:
        list_aas = dict_aas['aas:aasenv']['aas:submodels']['aas:submodel']['aas:submodelElements']['aas:submodelElement']
        extracted_data_aas = smc_to_smes_fromXML(list_aas)
    else:
        list_aas = dict_aas['environment']['submodels']['submodel']['submodelElements']
        extracted_data_aas = smc_to_smes_fromXML_new(list_aas)
    extracted_data_aas = [elem for elem in extracted_data_aas if isinstance(elem, tuple) and len(elem) == 4]
    df_aas = pd.DataFrame(extracted_data_aas, columns=['Index', 'idShort', 'description', 'semanticId'])
    # list_concept = dict_aas['aas:aasenv']['aas:conceptDescriptions']['aas:conceptDescription']
    return df_aas, dict_aas

def process_all_aasx_files(directory):
    """
    Process all AASX files in the specified directory and accumulate df_aas.

    Args:
        directory (str): Directory containing AASX files.

    Returns:
        DataFrame: A single DataFrame containing accumulated data from all df_aas of the AASX files.
        list: A list of DataFrames for each df_concept from the AASX files.
    """
    all_aas_data = []  # List to store all df_aas DataFrames

    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".aasx"):
            print('Processing:', filename, '...')
            aasx_path = os.path.join(directory, filename)
            df_aas, _ = process_aasx_file(aasx_path)
            all_aas_data.append(df_aas)  # Accumulate df_aas
            # all_concept_data.append(df_concept)  # Optionally accumulate df_concept if needed

    accumulated_df_aas = pd.concat(all_aas_data, ignore_index=True)

    return accumulated_df_aas  # Return accumulated df_aas and list of df_concept



def get_value(idShort, results, pdf_df):
    index = next((i for i, s in enumerate(results) if idShort in s), None)
    if index is not None:
        result = results[index]
        key = result.split(": ")[0]
        try:
            value = pdf_df.loc[pdf_df['Name'] == key, 'Value'].values[0]
        except:
            value = 'None'
        return value


def fill_value_toXML(dict_sme, df):
    for row in df.itertuples():
        if row.value is not None:
            print('populating ', row.idShort, ' with the value ', row.value)
            update_dict_recursively_new(dict_sme, row)

def update_dict_recursively_new(dict_sme, row):
    for index, key in enumerate(dict_sme):
        if key == 'submodelElementCollection' or key == 'submodelElementList':
            if isinstance(dict_sme[key], list):
                for list_dict in dict_sme[key]:
                    if 'value' in list_dict and list_dict['value'] != None:
                        # print(list_dict['value'])
                        update_dict_recursively_new(list_dict['value'], row)
            else:
                if dict_sme[key]['value'] != None:
                    update_dict_recursively_new(list_dict['value'], row)
        else:
            if isinstance(dict_sme[key], list):
                for list_dict in dict_sme[key]:
                    id_short = list_dict['idShort']
                    if id_short == row.idShort:
                        if key == 'multiLanguageProperty':
                            if list_dict.get('value') is not None:
                                if 'langStringTextType' not in list_dict['value'] or not isinstance(
                                        list_dict['value']['langStringTextType'], dict):
                                    list_dict['value']['langStringTextType'] = {}
                            else:
                                list_dict['value'] = {}
                                list_dict['value']['langStringTextType'] = {}
                            list_dict['value']['langStringTextType']['@lang'] = 'en'
                            list_dict['value']['langStringTextType']['#text'] = row.value
                        else:
                            list_dict['value'] = row.value
            else:
                if 'idShort' in dict_sme[key]:
                    id_short = dict_sme[key]['idShort']
                    if id_short == row.idShort:
                        if key == 'multiLanguageProperty':
                            if dict_sme[key].get('value') is not None:
                                if 'langStringTextType' not in dict_sme[key]['value'] or not isinstance(
                                        dict_sme[key]['value']['langStringTextType'], dict):
                                    dict_sme[key]['value']['langStringTextType'] = {}
                            else:
                                dict_sme[key]['value'] = {}
                                dict_sme[key]['value']['langStringTextType'] = {}
                            dict_sme[key]['value']['langStringTextType']['@lang'] = 'en'
                            dict_sme[key]['value']['langStringTextType']['#text'] = row.value
                        else:
                            dict_sme[key]['value'] = row.value


def fill_template(df_aasx, results, pdf_df, dict_aasx):
    df_aasx['value'] = df_aasx['idShort'].apply(
        lambda x: get_value(x, results, pdf_df))
    # dict_aasx_backup = dict_aasx.copy()
    fill_value_toXML(dict_aasx['environment']['submodels']['submodel']['submodelElements'], df_aasx)


def save_aasx(aasx_path, new_aasx_path, dict_aas):
    extracted_folder = "extracted_aasx"
    os.makedirs(extracted_folder, exist_ok=True)

    # Extract the original AASX package
    with zipfile.ZipFile(aasx_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder)

    # Convert the AAS dictionary to an XML string
    xml_string = xmltodict.unparse(dict_aas, pretty=True)

    # Find the XML file within the extracted directory
    pattern = os.path.join(extracted_folder, "**", "*.aas.xml")
    xml_files = glob.glob(pattern, recursive=True)

    if xml_files:
        # Write the modified XML string to the XML file
        xml_path = xml_files[0]
        with open(xml_path, 'w', encoding='utf-8') as xml_file:
            xml_file.write(xml_string)
        print(f"XML file updated at: {xml_path}")
    else:
        print("No XML file found. Unable to update.")

    # Repack the extracted files into a new AASX package
    with zipfile.ZipFile(new_aasx_path, 'w') as zip_ref:
        for folder_name, subfolders, filenames in os.walk(extracted_folder):
            for filename in filenames:
                file_path = os.path.join(folder_name, filename)
                zip_ref.write(file_path, os.path.relpath(file_path, extracted_folder))
        print(f"New AASX file created at: {new_aasx_path}")
    shutil.rmtree(extracted_folder)

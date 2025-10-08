
import os
import json
import pandas as pd
from collections import defaultdict

DIAGINOSIS_MAPPING = {
    '정상': 'normal',
    '하지부동': 'leg_length_discrepancy',
    'kneevarus': 'knee_varus',
    '퇴행성관절염' : 'degenerative_arthritis',
    'kneevalgus': 'knee_valgus',
    'heelvalgus': 'heel_valgus',
    '요족' : 'high_arch',
    '부주상골증후군': 'accessory_navicular_syndrome',
    '평발': 'flat_foot',
    'heelvarus': 'heel_varus',
    '족부관절염': 'foot_arthritis',
    '무지외반증': 'hallux_valgus',
    '발뒤꿈치통증증후군': 'heel_pain_syndrome',
    '족근골유합': 'tarsal_coalition',
    '외상후관절염' : 'post_traumatic_arthritis',
    '내족지보행': 'in_toeing',
    '외족지보행': 'out_toeing',
    '류마티스 관절염': 'rhematoid_arthritis'   
}

CATEGORY_MAPPING = {
    '정상': 'normal',
    '족부질환': 'foot_disease',
    '족관절관절염': 'ankle_arthritis',
    '보행장애': 'gait_disorder'
}

# CATEGORY_TO_DISEASE_MAPPING = {
#     'normal': ['정상'],
#     'foot_disease': ['족부관절염','무지외반증', '족근골유합', '발뒤꿈치 통증증후군','부주상골증후군'],
#     'ankle_arthritis': ['퇴행성 관절염', '류마티스 관절염', '외상후 관절염'],
#     'gait_disorder': ['평발','요족', 'knee valgus', 'heel valgus','knee varus','heel varus','내족지 보행','외족지 보행', '하지부동']
# }


CATEGORY_TO_DISEASE_MAPPING = {
    'normal': ['normal'],
    'foot_disease': ['foot_arthritis','hallux_valgus', 'tarsal_coalition', 'heel_pain_syndrome','accessory_navicular_syndrome'],
    'ankle_arthritis': ['degenerative_arthritis', 'rhematoid_arthritis', 'post_traumatic_arthritis'],
    'gait_disorder': ['flat_foot','high_arch', 'knee_valgus', 'heel_valgus','knee varus','heel_varus','in_toeing','out_toeing', 'leg_length_discrepancy']
}


def normalize_diagnosis_name(diagnosis):
    return diagnosis.replace(" ", "").lower()

def parse_vtp_filename(filename):
    filename = filename.replace('.vtp','')

    if "_R_" in filename:
        direction = "R"
        parts = filename.split("_R_")
    elif "_L_" in filename:
        direction = "L"
        parts = filename.split("_L_")
    else:
        raise ValueError("File name dose not contain direction")
    
    bone_type = parts[1].split("_LabelPolyLine")[0]

    patient_id = parts[0].split('_')[0]

    st_number = int([part for part in parts[0].split('_') if part.startswith('ST')][0][2:])   
    return patient_id, direction, bone_type, filename, st_number


def parse_clinical_info(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            clinical_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file {json_path}: {e}")
            return None, None, None, None, {}, []

    sex = clinical_data.get("Sex")
    weight = clinical_data.get("Weight")
    height = clinical_data.get("Height")
    age = clinical_data.get("Age")
    category_rl = clinical_data.get("Category_RL", {})
    diagnosis_names = clinical_data.get("Diagnosis_Names", [])
    category_info = clinical_data.get("Category_Info", {})

    print(f"Sex: {sex}, Weight: {weight}, Height: {height}, Age: {age}" )
    print(f"Category_RL: {category_rl}, Diagnosis_names: {diagnosis_names}, Category_info: {category_info}")

    normalized_diagnosis_names = [DIAGINOSIS_MAPPING.get(normalize_diagnosis_name(diagnosis), diagnosis) for diagnosis in diagnosis_names]

    return sex,  weight, height, age, category_rl, normalized_diagnosis_names, category_info

def initial_record():
    record = {}
    for category_eng in CATEGORY_MAPPING.values():
        record[f"group_category_{category_eng}"] = "other_catergory"
    for diagnosis_eng in DIAGINOSIS_MAPPING.values():
        record[f"group_disease_{diagnosis_eng}"] = "other_disease"
    return record    

def initialize_header(vtp_folder):
    headers = ["patient_id", "group_direction", "group_sex", "group_weight", "group_height", "group_age", "ST_number"]

    for filename in os.listdir(vtp_folder):
        if filename.endswith("LabelPolyLine.vtp"):
            _, _, bone_type, _, _ = parse_vtp_filename(filename)
            shape_col = f"shape_{bone_type}"
            landmarks_col = f"landmarks_file_{bone_type}"
            if shape_col not in headers:
                headers.append(shape_col)
            if landmarks_col not in headers:
                headers.append(landmarks_col)

    for category in CATEGORY_MAPPING.values():
        headers.append(f"group_category_{category}")
    for diagnosis in DIAGINOSIS_MAPPING.values():
        headers.append(f"group_disease_{diagnosis}")
    
    return headers

def create_csv(raw_data, output_path, headers, direction_filter="all",include_duplicates=True, exclude_age_0=False, 
                 age_filter=None, exclude_leg_length_discrepancy=False, remove_incomplete_shape=False, max_subjects=None):
    records = []  

    data = [record for record in raw_data if direction_filter == "all" or record.get("group_direction") == direction_filter]

    if age_filter == "20_above":
        data = [record for record in data if record.get("group_age") != "NULL" and float(record.get("group_age", 0)) >= 20]
    elif age_filter == "20_below":
        data = [record for record in data if record.get("group_age") != "NULL" and float(record.get("group_age", 0)) < 20]

    if exclude_leg_length_discrepancy:
        data = [record for record in data if record.get("group_disease_leg_discrepancy") != "leg_length_discrepancy"]       

    if exclude_age_0:
        data = [record for record in data if record.get("group_age") != "NULL"]
    if not include_duplicates:
        unique_records = {}
        for record in data:
            key = (record["patient_id"], record["group_direction"])
            if key not in unique_records or unique_records[key]["ST_number"] > record["ST_number"]:
                unique_records[key] = record
        records = list(unique_records.values())
    else:
        records  = data
           
    if remove_incomplete_shape:
        records = [record for record in records if all(record.get(col) for col in headers if col.startswith("shape_"))]   

    if max_subjects is not None:
        unique_patient = set()
        limited_data = []
        for record in data:
            patient_id = record.get('patient_id')
            if patient_id not in unique_patient:
                unique_patient.add(patient_id)
                limited_data.append(record)
            if len(unique_patient) >= max_subjects:
                break
        records = limited_data

    if not records:
        print("Warning: No records found to write to Excel.")

    
    df = pd.DataFrame(records, columns=headers)
    print(f"Generated DataFrame with {len(df)} records:") 
    print(df.head())
    df.to_csv(output_path, index=False, encoding='utf-8-sig')



def main(vtp_folder, clinical_folder, particles_folder, output_folder, direction_filter="all",max_files=None):
    try:

        
        patient_data = defaultdict(lambda: defaultdict(dict))
        

        folder_name = os.path.basename(os.path.normpath(vtp_folder))
        
        headers = initialize_header(vtp_folder)
        file_count = 0

        for filename in os.listdir(vtp_folder):
            if filename.endswith("LabelPolyLine.vtp"):
                print(f"Processing file: {filename}")
                file_count +=1

                try:
                    patient_id, direction, bone_type, vtp_file, st_number = parse_vtp_filename(filename)
                except Exception as e:
                    print(f"Skipping file {filename} due to error: {e}")
                    continue
                
                if direction_filter != "all" and direction != direction_filter:
                    continue

                clinical_info_file = f"{patient_id}_ClinicalInfo.json"
                clinical_info_path = os.path.join(clinical_folder, clinical_info_file)

                
                if os.path.exists(clinical_info_path):
                    try:
                        
                        sex, weight, height, age, category_rl, normalized_diagnosis_names, category_info = parse_clinical_info(clinical_info_path)
                    
                        sex = "NULL" if sex is None else sex
                        weight = "NULL" if weight is None else weight
                        height = "NULL" if height is None else height
                        age = "NULL" if age is None or age == 0 else age
                       

                        record = patient_data[patient_id][direction].get(st_number, initial_record())
                        
                        if  "patien_id" not in record:
                            record.update({
                                "patient_id": patient_id,
                                "group_direction":direction,
                                "group_sex": sex,
                                "group_weight": weight,
                                "group_height": height,
                                "group_age": age,
                                "ST_number": st_number,
                                
                            })

                        vtp_path = os.path.abspath(os.path.join(vtp_folder,vtp_file + ".vtp")) 
                        record[f"shape_{bone_type}"] = vtp_path
                        
                        particles_path = os.path.join(particles_folder, vtp_file + ".particles")
                        record[f"landmarks_file_{bone_type}"] = particles_path

                        for category, category_active in category_info.items():                           
                            
                            category_eng = CATEGORY_MAPPING.get(category, category)
                            print(f"Processing category: {category}, catergory_acitve: {category_active}, catergory_eng: {category_eng}")
                            rl_value = category_rl.get(category, "")
                            print(f"rl_values for {category}: {rl_value}")
                            current_key =f"group_category_{category_eng}"
                            

                            if category_active and (rl_value == direction or rl_value == "A"):
                                if category_eng == 'normal':
                                    for cat in CATEGORY_MAPPING.values():
                                        record[f"group_category_{cat}"] = 'normal'

                                    for disease in DIAGINOSIS_MAPPING.values():
                                        record[f"group_disease_{disease}"] = 'normal'
                                else:
                                    record[current_key] = category_eng
                                    print(f"Setting {current_key} to {category_eng}")

                                    for disease in CATEGORY_TO_DISEASE_MAPPING[category_eng]:
                                        print(f"disease: {disease}")
                                        disease_key = f"group_disease_{disease}"
                                        print(f"disease key: {disease_key}")
                                        print(f'normalized_diagnosis_names: {normalized_diagnosis_names}')
                                        if disease in normalized_diagnosis_names:
                                            record[disease_key] = disease
                                            print(f"Setting {disease_key} to {disease}")
                                        else:
                                            record[disease_key] = "other_disease"
                                            print(f"Setting {disease_key} to {disease}")
                            else:
                                print(f"Category {category} is not active or dose not match direction.")

                        
                        patient_data[patient_id][direction][st_number]= record
                        print(f"Record updated for patient_ID: {patient_id}, direction: {direction}, ST_number: {st_number}")        
                            
                    except Exception as e:
                        print(f"Error processing {clinical_info_file}: {e}")
                  
                    if max_files is not None and file_count >= max_files:
                        break
    
        
        flattened_data = []
    

        for patient_id, directions in patient_data.items():
            for direction, st_records in directions.items():
                for st_number, record in st_records.items():
                    flattened_data.append(record)


        if not flattened_data:
            print("Warining: No records found to write to Excel.")
        else:
            print(f"Patient data contains {len(flattened_data)} entries.")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        max_patient = None
        create_csv(flattened_data, os.path.join(output_folder, f"{folder_name}_output_{max_patient}.csv"), headers, include_duplicates=False, age_filter="20_above", exclude_leg_length_discrepancy=False, remove_incomplete_shape=True, direction_filter=direction_filter, max_subjects=max_patient)
        create_csv(flattened_data, os.path.join(output_folder, f"{folder_name}_output_with_duplicate_{max_patient}.csv"), headers, include_duplicates=True, remove_incomplete_shape=True, direction_filter=direction_filter, max_subjects=max_patient)

        # create_csv(flattened_data, os.path.join(output_folder, f"{folder_name}_output_no_age_0_{max_patient}.xlsx"), headers, include_duplicates=False, exclude_age_0=True, remove_incomplete_shape=True, direction_filter=direction_filter, max_subjects=max_patient)

        
    except Exception as e:
        print(f"Error in main process : {e}")
        raise     

if __name__ == "__main__":
    vtp_folder = "./test_data/vtp_aligned"
    clinical_folder = "./test_data/clinical_info"
    particles_folder = "./test_data/particles_files"
    output_folder = "./test_data/output"

    main(vtp_folder, clinical_folder, particles_folder, output_folder, max_files=None, direction_filter="all")
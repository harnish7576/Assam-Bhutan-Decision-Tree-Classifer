import math
import pandas as pd
import numpy as np

def find_theshold( data ):
    # Target is 'Assam'
    matrix = data.to_numpy()
    # print(matrix)

    attributes_hash = {'Age': 0,'Ht': 1,'TailLn': 2,'HairLn': 3,'BangLn': 4,'Reach': 5,'EarLobes': 6}
    class_id_column = 8

    # Results will be stored here
    best_thresholds = {}
    lowest_gini_indices = {}

    for key, column in attributes_hash.items():
        lowest_gini_index = 1

        for threshold in range(len(matrix)):
            group1_count = 0
            group2_count = 0
            group1_assam_count = 0
            group2_assam_count = 0
            gini_index_group1 = -1
            gini_index_group2 = -1
            
            for row in range(len(matrix)):
                # group1 counting number of classID as Assam to calculate Gini Index
                if matrix[row][column] < matrix[threshold][column]:
                    group1_count += 1
                    if matrix[row][class_id_column] == -1:
                        group1_assam_count += 1
                
                # group2 counting number of classID as Assam to calculate Gini Index
                if matrix[row][column] >= matrix[threshold][column]:
                    group2_count += 1
                    if matrix[row][class_id_column] == -1:
                        group2_assam_count += 1
            
            # In each group -> bhuttan count = total count - assam count
            if not group1_count < 1:
                gini_index_group1 = 1 - math.pow((group1_assam_count/group1_count),2) - math.pow(((group1_count-group1_assam_count)/group1_count),2)

            if not group2_count < 1:
                gini_index_group2 = 1 - math.pow((group2_assam_count/group2_count),2) - math.pow(((group2_count-group2_assam_count)/group2_count),2)   
            
            # If both groups in gini's are computed i.e. there are not the default values
            if gini_index_group1 != -1 and gini_index_group2 != -1:
                mixed_gini = (gini_index_group1*(group1_count/(group1_count+group2_count))) + (gini_index_group2*(group2_count/(group1_count+group2_count)))

            # If one group's gini is computed and other has the default
            elif gini_index_group1 != -1 :
                mixed_gini = gini_index_group1
            
            elif gini_index_group2 != -1 :
                mixed_gini = gini_index_group2

            else: mixed_gini = 1    


            if mixed_gini <= lowest_gini_index:
                best_threshold = matrix[threshold][column]
                lowest_gini_index = mixed_gini

        # Storing resultant values
        best_thresholds[key] = best_threshold
        lowest_gini_indices[key] = lowest_gini_index

    return best_thresholds, lowest_gini_indices


    




def main():
    # Reading File
    data = pd.read_csv('Abominable_Data_HW_LABELED_TRAINING_DATA__v770_2225.csv')

    # Quantizing data as required
    attributes = ['Age','Ht','TailLn','HairLn','BangLn','Reach','EarLobes']
    for attribute in attributes:
        if attribute == 'Age': 
            data[attribute] = round(data[attribute] / 2) * 2
        elif attribute == 'Ht': 
            data[attribute] = round(data[attribute] / 5) * 5
        else:
            data[attribute] = round(data[attribute])
    # print(data)
    
    # ------------------A small set of data, testing my code, with known results---------------------

    # Test_Suite_Each_Class_4.csv, which has 8 records, and the ClassName separates the two classes
    bhuttan_data = data[data['ClassName'] == 'Bhuttan']
    assam_data = data[data['ClassName'] == 'Assam']
    combined_data = pd.concat([bhuttan_data.head(4), assam_data.head(4)], axis=0, ignore_index=True)
    combined_data.to_csv('Test_Suite_Each_Class_4.csv', index=False)

    # -----------------------------------------------------------------------------------------------

    # Initializing used attributes
    used_attributes = []
    print()
    
    print("Computing new feature to be selected...")
    print()

    best_thresholds, lowest_gini_indices =  find_theshold(data)
    # print(best_thresholds)
    # print(lowest_gini_indices)

    # Discarding the used atrributes
    if used_attributes:
        for feature in used_attributes:
            del lowest_gini_indices[feature]

    # Computing the next feature but comparing Gini Indices
    feature = min(lowest_gini_indices, key=lambda k: lowest_gini_indices[k])
    used_attributes.append(feature)
    print("Feature selected for splitting: ", feature) 

    # Splitting at threshold selected of the feature selected
    group1_data = data[data[feature] < best_thresholds[feature]] # e.g. data[data['BangLn] < 6.0]
    group2_data = data[data[feature] >= best_thresholds[feature]]
    print("Split after separating 2 classes at threshold ", best_thresholds[feature] )
    print()

    print("group1 Data Assam", len(group1_data[group1_data["ClassName"] == "Assam"]))
    print("group1 Data Bhuttan", len(group1_data[group1_data["ClassName"] == "Bhuttan"]))
    print()

    print("group2 data Assam", len(group2_data[group2_data["ClassName"] == "Assam"]))
    print("group2 data Bhuttan", len(group2_data[group2_data["ClassName"] == "Bhuttan"]))

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # Group1 is splitting in Group1_1 and Group1_2

    print("Computing new feature to be selected...")
    print()

    best_thresholds, lowest_gini_indices =  find_theshold(group1_data)
    # print(best_thresholds)
    # print(lowest_gini_indices)

    # Discarding the used atrributes
    if used_attributes:
        for feature in used_attributes:
            del lowest_gini_indices[feature]

    # Computing the next feature but comparing Gini Indices
    feature = min(lowest_gini_indices, key=lambda k: lowest_gini_indices[k])
    used_attributes.append(feature)
    print("Feature selected for splitting: ", feature) 

    # Splitting at threshold selected of the feature selected
    group1_1_data = group1_data[group1_data[feature] < best_thresholds[feature]] # e.g. data[data['BangLn] < 6.0]
    group1_2_data = group1_data[group1_data[feature] >= best_thresholds[feature]]
    print("Split after separating 2 classes at threshold ", best_thresholds[feature] )
    print()

    print("group1_1 Data Assam", len(group1_1_data[group1_1_data["ClassName"] == "Assam"]))
    print("group1_1 Data Bhuttan", len(group1_1_data[group1_1_data["ClassName"] == "Bhuttan"]))
    print()

    print("group1_2 data Assam", len(group1_2_data[group1_2_data["ClassName"] == "Assam"]))
    print("group1_2 data Bhuttan", len(group1_2_data[group1_2_data["ClassName"] == "Bhuttan"]))

    # -----------------------------------------------------------------------------------------------
    # Group2 is splitting in Group2_1 and Group2_2  

    print("Computing new feature to be selected...")
    print()

    best_thresholds, lowest_gini_indices =  find_theshold(group2_data)
    # print(best_thresholds)
    # print(lowest_gini_indices)

    # Discarding the used atrributes
    if used_attributes:
        for feature in used_attributes:
            del lowest_gini_indices[feature]

    # Computing the next feature but comparing Gini Indices
    feature = min(lowest_gini_indices, key=lambda k: lowest_gini_indices[k])
    used_attributes.append(feature)
    print("Feature selected for splitting: ", feature) 

    # Splitting at threshold selected of the feature selected
    group2_1_data = group2_data[group2_data[feature] < best_thresholds[feature]] # e.g. data[data['BangLn] < 6.0]
    group2_2_data = group2_data[group2_data[feature] >= best_thresholds[feature]]
    print("Split after separating 2 classes at threshold ", best_thresholds[feature] )
    print()

    print("group2_1 Data Assam", len(group2_1_data[group2_1_data["ClassName"] == "Assam"]))
    print("group2_1 Data Bhuttan", len(group2_1_data[group2_1_data["ClassName"] == "Bhuttan"]))
    print()

    print("group2_2 data Assam", len(group2_2_data[group2_2_data["ClassName"] == "Assam"]))
    print("group2_2 data Bhuttan", len(group2_2_data[group2_2_data["ClassName"] == "Bhuttan"]))

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # Group1_1 is splitting in Group1_1_1 and Group1_1_2

    print("Computing new feature to be selected...")
    print()

    best_thresholds, lowest_gini_indices =  find_theshold(group1_1_data)
    # print(best_thresholds)
    # print(lowest_gini_indices)

    # Discarding the used atrributes
    if used_attributes:
        for feature in used_attributes:
            del lowest_gini_indices[feature]

    # Computing the next feature but comparing Gini Indices
    feature = min(lowest_gini_indices, key=lambda k: lowest_gini_indices[k])
    used_attributes.append(feature)
    print("Feature selected for splitting: ", feature) 

    # Splitting at threshold selected of the feature selected
    group1_1_1_data = group1_1_data[group1_1_data[feature] < best_thresholds[feature]] # e.g. data[data['BangLn] < 6.0]
    group1_1_2_data = group1_1_data[group1_1_data[feature] >= best_thresholds[feature]]
    print("Split after separating 2 classes at threshold ", best_thresholds[feature] )
    print()

    print("group1_1_1 Data Assam", len(group1_1_1_data[group1_1_1_data["ClassName"] == "Assam"]))
    print("group1_1_1 Data Bhuttan", len(group1_1_1_data[group1_1_1_data["ClassName"] == "Bhuttan"]))
    print()

    print("group1_1_2 data Assam", len(group1_1_2_data[group1_1_2_data["ClassName"] == "Assam"]))
    print("group1_1_2 data Bhuttan", len(group1_1_2_data[group1_1_2_data["ClassName"] == "Bhuttan"]))

    # -----------------------------------------------------------------------------------------------
    # Group1_2 is splitting in Group1_2_1 and Group1_2_2

    print("Computing new feature to be selected...")
    print()

    best_thresholds, lowest_gini_indices =  find_theshold(group1_2_data)
    # print(best_thresholds)
    # print(lowest_gini_indices)

    # Discarding the used atrributes
    if used_attributes:
        for feature in used_attributes:
            del lowest_gini_indices[feature]

    # Computing the next feature but comparing Gini Indices
    feature = min(lowest_gini_indices, key=lambda k: lowest_gini_indices[k])
    used_attributes.append(feature)
    print("Feature selected for splitting: ", feature) 

    # Splitting at threshold selected of the feature selected
    group1_2_1_data = group1_2_data[group1_2_data[feature] < best_thresholds[feature]] # e.g. data[data['BangLn] < 6.0]
    group1_2_2_data = group1_2_data[group1_2_data[feature] >= best_thresholds[feature]]
    print("Split after separating 2 classes at threshold ", best_thresholds[feature] )
    print()

    print("group1_2_1 Data Assam", len(group1_2_1_data[group1_2_1_data["ClassName"] == "Assam"]))
    print("group1_2_1 Data Bhuttan", len(group1_2_1_data[group1_2_1_data["ClassName"] == "Bhuttan"]))
    print()

    print("group1_2_2 data Assam", len(group1_2_2_data[group1_2_2_data["ClassName"] == "Assam"]))
    print("group1_2_2 data Bhuttan", len(group1_2_2_data[group1_2_2_data["ClassName"] == "Bhuttan"]))

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # Group2_1 is splitting in Group2_1_1 and Group2_1_2

    print("Computing new feature to be selected...")
    print()

    best_thresholds, lowest_gini_indices =  find_theshold(group2_1_data)
    # print(best_thresholds)
    # print(lowest_gini_indices)

    # Discarding the used atrributes
    if used_attributes:
        for feature in used_attributes:
            del lowest_gini_indices[feature]

    # Computing the next feature but comparing Gini Indices
    feature = min(lowest_gini_indices, key=lambda k: lowest_gini_indices[k])
    used_attributes.append(feature)
    print("Feature selected for splitting: ", feature) 

    # Splitting at threshold selected of the feature selected
    group2_1_1_data = group2_1_data[group2_1_data[feature] < best_thresholds[feature]] # e.g. data[data['BangLn] < 6.0]
    group2_1_2_data = group2_1_data[group2_1_data[feature] >= best_thresholds[feature]]
    print("Split after separating 2 classes at threshold ", best_thresholds[feature] )
    print()

    
    print("group2_1_1 Data Assam", len(group2_1_1_data[group2_1_1_data["ClassName"] == "Assam"]))
    print("group2_1_1 Data Bhuttan", len(group2_1_1_data[group2_1_1_data["ClassName"] == "Bhuttan"]))
    print()

    print("group2_1_2 data Assam", len(group2_1_2_data[group2_1_2_data["ClassName"] == "Assam"]))
    print("group2_1_2 data Bhuttan", len(group2_1_2_data[group2_1_2_data["ClassName"] == "Bhuttan"]))

    # -----------------------------------------------------------------------------------------------
    # Group2_2 is splitting in Group2_2_1 and Group2_2_2

    print("Computing new feature to be selected...")
    print()

    best_thresholds, lowest_gini_indices =  find_theshold(group2_2_data)
    # print(best_thresholds)
    # print(lowest_gini_indices)

    # Discarding the used atrributes
    if used_attributes:
        for feature in used_attributes:
            del lowest_gini_indices[feature]

    # Computing the next feature but comparing Gini Indices
    feature = min(lowest_gini_indices, key=lambda k: lowest_gini_indices[k])
    used_attributes.append(feature)
    print("Feature selected for splitting: ", feature) 

    # Splitting at threshold selected of the feature selected
    group2_2_1_data = group2_2_data[group2_2_data[feature] < best_thresholds[feature]] # e.g. data[data['BangLn] < 6.0]
    group2_2_2_data = group2_2_data[group2_2_data[feature] >= best_thresholds[feature]]
    print("Split after separating 2 classes at threshold ", best_thresholds[feature] )
    print()

    print("group2_2_1 Data Assam", len(group2_2_1_data[group2_2_1_data["ClassName"] == "Assam"]))
    print("group2_2_1 Data Bhuttan", len(group2_2_1_data[group2_2_1_data["ClassName"] == "Bhuttan"]))
    print()

    print("group2_2_2 data Assam", len(group2_2_2_data[group2_2_2_data["ClassName"] == "Assam"]))
    print("group2_2_2 data Bhuttan", len(group2_2_2_data[group2_2_2_data["ClassName"] == "Bhuttan"]))


if __name__ == "__main__":
    main()
import pandas as pd
import csv
from sklearn.metrics import confusion_matrix

def main():
    print()
    # Opening new file for writing predicted results
    with open('MyClassifications.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        print("Predicting for file Abominable_VALIDATION_Data_FOR_STUDENTS_v770_2225.csv")
        # Reading File
        data = pd.read_csv('Abominable_VALIDATION_Data_FOR_STUDENTS_v770_2225.csv')

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


        # Labelling Columns for Results
        writer.writerow(['ClassName', 'ClassID'])
        print("Predicting....")
        for index, line in data.iterrows():
            if line['BangLn'] < 6:
                if line['HairLn'] < 11:
                    if line['TailLn'] < 9.0: writer.writerow(["Assam", "-1"])
                    else: writer.writerow(["Assam", "-1"])
                else:
                    if line['EarLobes'] < 1: writer.writerow(["Assam", "-1"])
                    else: writer.writerow(["Bhuttan", "1"])
            else:
                if line['Ht'] < 180:
                    if line['Reach'] < 127: writer.writerow(["Assam", "-1"])
                    else: writer.writerow(["Bhuttan", "1"])
                else:
                    if line['Age']< 46: writer.writerow(["Assam", "-1"])
                    else: writer.writerow(["Assam", "-1"])
        print()
        print("Prediction done...check file created MyClassifications.csv")


    print()
    print()
    print()
    print("Creating Confusion Matrix of the Labelled Training Data")
    print()
    # Creating confusion Matrix
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

    predictions=[]
    for index, line in data.iterrows():
        if line['BangLn'] < 6:
            if line['HairLn'] < 11:
                if line['TailLn'] < 9.0: predictions.append("Assam")
                else: predictions.append("Assam")
            else:
                if line['EarLobes'] < 1: predictions.append("Assam")
                else: predictions.append("Bhuttan")
        else:
            if line['Ht'] < 180:
                if line['Reach'] < 127: predictions.append("Assam")
                else: predictions.append("Bhuttan")
            else:
                if line['Age']< 46: predictions.append("Assam")
                else: predictions.append("Assam")
    
    actual_classname = []
    for index, line in data.iterrows():
        actual_classname.append(line['ClassName']) 

    # Confusion Matrix
    confusionMatrix = confusion_matrix(actual_classname, predictions)
    print(confusionMatrix) 
    print()
    print("Accuracy achieved: ", ((confusionMatrix[0][0]+confusionMatrix[1][1])/(confusionMatrix[0][0]+confusionMatrix[1][1]+confusionMatrix[0][1]+confusionMatrix[1][0]))*100)
    


if __name__ == "__main__":
    main()
from datetime import datetime
import pandas as pd
import pathlib

#date = datetime.now().strftime("%d/%m/%Y")
data = pd.read_csv(str(pathlib.Path(__file__).parent.absolute()) + "/air.csv", sep=",")
print(data.head())
#for i in range(len(data.date)):
#  if data.date[i] == date:
#    print(data.tempbas[i]) 



#Poid(today)
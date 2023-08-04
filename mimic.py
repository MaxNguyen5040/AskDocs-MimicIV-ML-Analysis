import csv  
import time
#total rows: 98,972,025 / 98972025

start_time = time.time()


with open("discharge.csv") as csvfile:  
    data = csv.DictReader(csvfile)
    count2 = 0
    for row in data:
        if count2 > 1000:
            break

        # print(f"Row {count2} of 98972025")

        if "depression" in row["text"]:
            word_list = row["text"].split()
            for i in range(len(word_list)):
                if word_list[i] == "depression":
                    string1 = ""
                    for w in word_list[i-5:i+5]:
                        string1+= w 
                        string1+= " "
                    print(string1)
                    print("\n_____________________________ _")
        count2 +=1
        

print("--- %s seconds ---" % (time.time() - start_time))
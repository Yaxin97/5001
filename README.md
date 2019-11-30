#### language
The programming language I used is Python3
#### packages
os,numpy,pandas,re,sklearn,time,warnings
#### how to run it
run the only py file named 123
after import packages in the begining,there are codes like below:

os.chdir('/Users/yaxin/Google Drive/BDT/5001/Ind-project')
train = pd.read_csv("./data/train.csv", parse_dates=["purchase_date","release_date"])
test = pd.read_csv("./data/test.csv", parse_dates=["purchase_date","release_date"])

change the file path and names then run the code
get resulting test.csv file: named playtime.csv
